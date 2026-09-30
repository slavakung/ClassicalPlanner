// Reproducible benchmark entry point: JSONL, steady-clock phase timing, and
// independent replay against the grounded IR (without compiled action masks).
#include "ApproximateDPPlanner.h"
#include "FiniteStatePlanner.h"
#include "InformedPlanner.h"
#include "IterativePlanner.h"
#include "LocalSearchPlanner.h"
#include "PlanningHeuristics.h"
#include "pddl/Adapter.h"
#include "symbolic/ClassicalPlanners.h"
#include "symbolic/CSPPlanner.h"
#include "symbolic/PartialOrderPlanner.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace {
using namespace planning;
using Clock = std::chrono::steady_clock;
using Problem = pddl::PDDLPlanningProblem;
using Action = pddl::PDDLAction;
using State = pddl::PDDLState;
using ResultPlan = Plan<Action, float>;
using Atom = std::pair<ir::PredicateId, std::vector<ir::ObjectId>>;

struct Arguments {
    std::string domainFile, problemFile;
    std::string algorithm = "astar", heuristic = "hmax";
    std::uint64_t maxDepth = std::numeric_limits<std::uint64_t>::max();
    std::size_t repetitions = 1, beamWidth = 32;
    std::size_t rolloutDepth = 16, adpTrials = 32, trialDepth = 64;
    double weight = 1.5;
    bool depthExplicit = false, pruneCycles = true, help = false, prepareOnly = false;
};

bool Symbolic(const std::string& algorithm) {
    return algorithm == "graphplan" || algorithm == "satplan" || algorithm == "regression" ||
           algorithm == "csp" || algorithm == "partial-order";
}
bool Finite(const std::string& algorithm) {
    return algorithm == "backward-bfs" || algorithm == "bidirectional" ||
           algorithm == "forward-vi" || algorithm == "backward-vi";
}
bool UsesHeuristic(const std::string& algorithm) {
    return algorithm == "astar" || algorithm == "greedy" || algorithm == "weighted-astar" ||
           algorithm == "beam" || algorithm == "idastar" || algorithm == "rbfs" ||
           algorithm == "hill" || algorithm == "ehc" || algorithm == "rollout" ||
           algorithm == "adp" || algorithm == "adp-astar";
}

template<class T> T Number(std::string_view text) {
    T value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument("invalid number: " + std::string(text));
    return value;
}

void Parse(int argc, char** argv, Arguments& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + flag);
            return argv[i];
        };
        if (flag == "--help") a.help = true;
        else if (flag == "--prepare-only") a.prepareOnly = true;
        else if (flag == "--allow-cycles") a.pruneCycles = false;
        else if (flag == "--domain") a.domainFile = value();
        else if (flag == "--problem") a.problemFile = value();
        else if (flag == "--algorithm") a.algorithm = value();
        else if (flag == "--heuristic") a.heuristic = value();
        else if (flag == "--max-depth") { a.maxDepth = Number<std::uint64_t>(value()); a.depthExplicit = true; }
        else if (flag == "--repetitions") a.repetitions = Number<std::size_t>(value());
        else if (flag == "--weight") a.weight = Number<double>(value());
        else if (flag == "--beam-width") a.beamWidth = Number<std::size_t>(value());
        else if (flag == "--rollout-depth") a.rolloutDepth = Number<std::size_t>(value());
        else if (flag == "--adp-trials") a.adpTrials = Number<std::size_t>(value());
        else if (flag == "--trial-depth") a.trialDepth = Number<std::size_t>(value());
        else throw std::invalid_argument("unknown option: " + flag);
    }
    if (a.help) return;
    if (a.domainFile.empty() || a.problemFile.empty())
        throw std::invalid_argument("--domain and --problem are required");
    if (a.repetitions == 0) throw std::invalid_argument("--repetitions must be positive");
    const std::set<std::string> algorithms{"bfs", "ucs", "astar", "greedy", "weighted-astar", "beam",
        "iddfs", "idastar", "rbfs", "graphplan", "satplan", "regression", "csp", "partial-order",
        "dfs", "recursive-dfs", "bbdfs", "hill", "ehc", "backward-bfs", "bidirectional", "forward-vi", "backward-vi",
        "rollout", "adp", "adp-astar"};
    if (!algorithms.contains(a.algorithm)) throw std::invalid_argument("unknown algorithm: " + a.algorithm);
    const std::set<std::string> heuristics{"zero", "hmax", "hadd", "relaxed-plan"};
    if (!heuristics.contains(a.heuristic)) throw std::invalid_argument("unknown heuristic: " + a.heuristic);
    if ((a.algorithm == "astar" || a.algorithm == "idastar" || a.algorithm == "rbfs" || a.algorithm == "adp-astar") &&
        a.heuristic != "hmax" && a.heuristic != "zero")
        throw std::invalid_argument("optimal heuristic search requires hmax or zero");
    if (!std::isfinite(a.weight) || a.weight < 1 || a.beamWidth == 0)
        throw std::invalid_argument("weight must be finite and >= 1; beam width must be positive");
    if (Symbolic(a.algorithm)) {
        if (!a.depthExplicit && !a.prepareOnly) throw std::invalid_argument("symbolic planners require an explicit --max-depth horizon");
    }
    if (Finite(a.algorithm)) {
        if (!a.depthExplicit && !a.prepareOnly) throw std::invalid_argument("finite-state planners require an explicit --max-depth horizon");
    }
}

double Seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

std::string Quote(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

struct Record {
    std::string status = "error", error, phase = "configuration", domainName, problemName, engine;
    std::size_t repetition = 0, groundedActions = 0, groundedFacts = 0, initialApplicable = 0;
    double parseSeconds = 0, groundingSeconds = 0, metadataSeconds = 0;
    double minimumActionCost = 0, maximumActionCost = 0;
    bool unitCost = true;
    std::optional<double> planningSeconds, replaySeconds, peakRssKiB;
    std::optional<std::uint64_t> expansionCalls, candidateRows;
    std::optional<ResultPlan> plan;
    bool replayValid = false;
};

// Called from a handler after unwinding has released the failed phase's work.
// Keep allocation failures separate from invalid plans and implementation errors.
int RecordFailure(Record& record) {
    try { throw; }
    catch (const std::bad_alloc& error) {
        record.status = "resource_exhausted";
        record.error = error.what();
        return 3;
    } catch (const std::exception& error) {
        record.status = "error";
        record.error = error.what();
    } catch (...) {
        record.status = "error";
        record.error = "unknown exception";
    }
    return 1;
}

void Print(const Arguments& a, const Record& r) {
    std::ostringstream out;
    out << std::setprecision(17) << std::boolalpha;
    out << "{\"schema_version\":1,\"status\":" << Quote(r.status)
        << ",\"error\":" << (r.error.empty() ? "null" : Quote(r.error))
        << ",\"phase\":" << Quote(r.phase)
        << ",\"domain_file\":" << Quote(a.domainFile) << ",\"problem_file\":" << Quote(a.problemFile)
        << ",\"domain_name\":" << Quote(r.domainName) << ",\"problem_name\":" << Quote(r.problemName)
        << ",\"algorithm\":" << Quote(a.algorithm) << ",\"heuristic\":" << Quote(a.heuristic)
        << ",\"prepare_only\":" << a.prepareOnly
        << ",\"unit_action_costs\":" << r.unitCost
        << ",\"minimum_action_cost\":" << r.minimumActionCost
        << ",\"maximum_action_cost\":" << r.maximumActionCost
        << ",\"heuristic_used\":" << UsesHeuristic(a.algorithm)
        << ",\"scheduler\":\"sequential\",\"expansion\":\"native\""
        << ",\"expansion_engine\":" << Quote(r.engine)
        << ",\"measurement_kind\":\"performance\",\"workers\":1"
        << ",\"max_depth\":" << a.maxDepth << ",\"depth_bound_explicit\":" << a.depthExplicit
        << ",\"prune_path_cycles\":" << a.pruneCycles
        << ",\"bfs_duplicate_pruning\":" << (a.algorithm == "bfs")
        << ",\"weight\":" << (std::isfinite(a.weight) ? a.weight : 0.0)
        << ",\"beam_width\":" << a.beamWidth << ",\"repetition\":" << r.repetition
        << ",\"rollout_depth\":" << a.rolloutDepth << ",\"adp_trials\":" << a.adpTrials
        << ",\"trial_depth\":" << a.trialDepth
        << ",\"repetitions\":" << a.repetitions << ",\"first_in_process\":" << (r.repetition == 0)
        << ",\"fresh_planner\":true,\"parse_seconds\":" << r.parseSeconds
        << ",\"grounding_seconds\":" << r.groundingSeconds << ",\"metadata_seconds\":" << r.metadataSeconds;
    auto optionalNumber = [&](std::string_view key, const auto& number) {
        out << ',' << Quote(key) << ':';
        if (number) out << *number; else out << "null";
    };
    optionalNumber("planning_seconds", r.planningSeconds);
    optionalNumber("replay_seconds", r.replaySeconds);
    optionalNumber("peak_rss_kib", r.peakRssKiB);
    out << ",\"grounded_actions\":" << r.groundedActions << ",\"grounded_facts\":" << r.groundedFacts
        << ",\"initial_applicable_actions\":" << r.initialApplicable
;
    optionalNumber("expansion_calls", r.expansionCalls);
    optionalNumber("candidate_rows", r.candidateRows);
    out << ",\"plan_length\":";
    if (r.plan) out << r.plan->actions.size(); else out << "null";
    out << ",\"plan_cost\":";
    if (r.plan && r.plan->totalCost) out << *r.plan->totalCost; else out << "null";
    out << ",\"replay_valid\":" << r.replayValid << ",\"plan_actions\":[";
    if (r.plan) for (std::size_t i = 0; i < r.plan->actions.size(); ++i) {
        if (i) out << ',';
        out << Quote(pddl::ToString(r.plan->actions[i]));
    }
    out << "]}";
    std::cout << out.str() << std::endl;
}

Atom Key(const ir::GroundAtom& atom) { return {atom.predicate, atom.arguments}; }

std::size_t FactCount(const Problem& problem) {
    std::set<Atom> facts;
    const auto& source = problem.Source();
    const auto add = [&](const ir::GroundAtom& atom) {
        if (atom.predicate != ir::EqualityPredicate) facts.insert(Key(atom));
    };
    for (const auto& atom : source.source.initialFacts) add(atom);
    for (const auto& literal : source.source.goal) add(literal.atom);
    for (const auto& action : source.actions) {
        for (const auto& literal : action.preconditions) add(literal.atom);
        for (const auto& literal : action.effects) add(literal.atom);
    }
    // Equality is interpreted, not a propositional fluent; synthetic internal
    // contradiction bits are deliberately excluded from this domain fact count.
    return facts.size();
}

void Replay(const Problem& problem, const ResultPlan& plan) {
    if (plan.status != PlanStatus::success || plan.stageLen != plan.actions.size())
        throw std::runtime_error("invalid returned plan status or length");
    std::set<Atom> state;
    const auto& source = problem.Source();
    for (const auto& atom : source.source.initialFacts) state.insert(Key(atom));
    auto holds = [&](const ir::GroundLiteral& literal) {
        const bool positive = literal.atom.predicate == ir::EqualityPredicate
            ? literal.atom.arguments.at(0) == literal.atom.arguments.at(1)
            : state.contains(Key(literal.atom));
        return literal.negated ? !positive : positive;
    };
    double cost = source.source.initialTotalCost;
    for (const auto& returned : plan.actions) {
        const auto& compiled = problem.CompiledActions();
        const auto found = std::find_if(compiled.begin(), compiled.end(), [&](const auto& canonical) {
            return canonical.name == returned.name && canonical.arguments == returned.arguments;
        });
        if (found == compiled.end()) throw std::runtime_error("plan contains an unknown grounded action");
        const auto& action = source.actions.at(static_cast<std::size_t>(found - compiled.begin()));
        for (const auto& precondition : action.preconditions)
            if (!holds(precondition)) throw std::runtime_error("independent IR replay: action precondition fails");
        // STRIPS applies deletes before adds, including conflicting effects.
        for (const auto& effect : action.effects) if (effect.negated) state.erase(Key(effect.atom));
        for (const auto& effect : action.effects) if (!effect.negated) state.insert(Key(effect.atom));
        cost += action.cost;
    }
    for (const auto& goal : source.source.goal)
        if (!holds(goal)) throw std::runtime_error("independent IR replay: goal fails");
    if (!plan.totalCost || !std::isfinite(*plan.totalCost) ||
        std::abs(cost - *plan.totalCost) > 1e-5 * std::max(1.0, std::abs(cost)))
        throw std::runtime_error("independent IR replay: plan cost mismatch");
}

std::function<float(const State&)> Heuristic(const Problem& p, const Arguments& a) {
    if (a.heuristic == "zero" || a.algorithm == "ucs" || a.algorithm == "iddfs") return {};
    const auto kind = a.heuristic == "hadd" ? RelaxationHeuristic::additive :
        a.heuristic == "relaxed-plan" ? RelaxationHeuristic::relaxedPlan : RelaxationHeuristic::max;
    return DeleteRelaxationHeuristic(p, kind);
}

struct SerialReport { std::string engine = "native-serial"; };
template<class Factory, class P, class Consumer>
SerialReport SerialRun(Factory factory, const P& problem, Consumer consume) {
    auto planner = factory();
    planner.Search(problem, consume);
    return {};
}

SerialReport Solve(const Problem& problem, const Arguments& a, std::optional<ResultPlan>& plan) {
    const auto consume = [&](const ResultPlan& result) { plan = result; return false; };
    if (a.algorithm == "bfs") {
        BFSOptions options;
        options.maxDepth = a.maxDepth;
        options.maxPlans = 1;
        options.prunePathCycles = a.pruneCycles;
        options.pruneDuplicateStates = true;
        return SerialRun([=] { return BFSPlanner<Action, State, float, LogicStateKey>(options); },
                             problem, consume);
    }
    if (a.algorithm == "dfs" || a.algorithm == "recursive-dfs") {
        DFSOptions<float> options;
        options.maxDepth = a.maxDepth;
        options.maxPlans = 1;
        options.prunePathCycles = a.pruneCycles;
        if (a.algorithm == "dfs")
            return SerialRun([=] { return DFSPlanner<Action, State, float, LogicStateKey>(options); },
                                 problem, consume);
        return SerialRun([=] { return RecursiveDFSPlanner<Action, State, float, LogicStateKey>(options); },
                             problem, consume);
    }
    if (a.algorithm == "bbdfs") {
        BBDFSOptions<float> options;
        options.maxDepth = a.maxDepth;
        options.maxPlans = 1;
        options.prunePathCycles = a.pruneCycles;
        return SerialRun([=] { return BranchAndBoundDFSPlanner<Action, State, float, LogicStateKey>(options); },
                             problem, consume);
    }
    if (Finite(a.algorithm)) {
        finite::Options options;
        options.maxDepth = static_cast<std::size_t>(a.maxDepth);
        options.maxStates = options.maxEdges = options.maxTableEntries = std::numeric_limits<std::size_t>::max();
        finite::ReachableGraph graph(problem, options);
        plan = a.algorithm == "backward-bfs" ? finite::BackwardBreadthFirst(graph) :
               a.algorithm == "bidirectional" ? finite::BidirectionalBreadthFirst(graph) :
               a.algorithm == "forward-vi" ? finite::ForwardValueIteration(graph) :
               finite::BackwardValueIteration(graph);
        return {};
    }
    if (Symbolic(a.algorithm)) {
        const symbolic::SymbolicOptions options{static_cast<std::size_t>(a.maxDepth)};
        auto result = a.algorithm == "graphplan" ? symbolic::Graphplan(problem, options) :
                      a.algorithm == "satplan" ? symbolic::SATPlan(problem, options) :
                      a.algorithm == "regression" ? symbolic::Regression(problem, options) :
                      a.algorithm == "csp" ? symbolic::CSPPlan(problem, options) :
                      symbolic::PartialOrder(problem, {options.maxDepth, std::numeric_limits<std::size_t>::max()});
        if (result.status == PlanStatus::success) plan = std::move(result);
        return {"symbolic-serial"};
    }
    auto heuristic = Heuristic(problem, a);
    std::uint64_t heuristicId = 0;
    for (unsigned char c : a.heuristic) heuristicId = heuristicId * 131 + c;
    if (a.algorithm == "rollout" || a.algorithm == "adp" || a.algorithm == "adp-astar") {
        ApproximateDPOptions options;
        options.maxDepth = a.maxDepth;
        options.prunePathCycles = a.pruneCycles;
        options.heuristicId = heuristicId;
        options.rolloutDepth = a.rolloutDepth;
        options.trials = a.adpTrials;
        options.trialDepth = a.trialDepth;
        options.admissibleInitialization = a.heuristic == "zero" || a.heuristic == "hmax";
        options.mode = a.algorithm == "rollout" ? ApproximateDPMode::rollout :
            a.algorithm == "adp" ? ApproximateDPMode::rtdp : ApproximateDPMode::rtdpAStar;
        return SerialRun([=] { return ApproximateDPPlanner<Action, State, float, LogicStateKey>(options, heuristic); },
                             problem, consume);
    }
    if (a.algorithm == "hill" || a.algorithm == "ehc") {
        LocalSearchOptions options;
        options.maxDepth = a.maxDepth;
        options.prunePathCycles = a.pruneCycles;
        options.heuristicId = heuristicId;
        options.mode = a.algorithm == "hill" ? LocalSearch::hillClimbing : LocalSearch::enforcedHillClimbing;
        return SerialRun([=] { return LocalSearchPlanner<Action, State, float, LogicStateKey>(options, heuristic); },
                             problem, consume);
    }
    if (a.algorithm == "iddfs" || a.algorithm == "idastar" || a.algorithm == "rbfs") {
        IterativeOptions options;
        options.maxDepth = a.maxDepth;
        options.prunePathCycles = a.pruneCycles;
        options.heuristicId = heuristicId;
        if (a.algorithm == "idastar") options.mode = IterativeSearch::idaStar;
        if (a.algorithm == "rbfs") options.mode = IterativeSearch::recursiveBestFirst;
        return SerialRun([=] { return IterativePlanner<Action, State, float, LogicStateKey>(options, heuristic); },
                             problem, consume);
    }
    InformedOptions options;
    options.maxDepth = a.maxDepth;
    options.prunePathCycles = a.pruneCycles;
    options.weight = a.weight;
    options.beamWidth = a.beamWidth;
    options.heuristicId = heuristicId;
    if (a.algorithm == "ucs") options.mode = InformedSearch::uniformCost;
    else if (a.algorithm == "greedy") options.mode = InformedSearch::greedy;
    else if (a.algorithm == "weighted-astar") options.mode = InformedSearch::weightedAStar;
    else if (a.algorithm == "beam") options.mode = InformedSearch::beam;
    return SerialRun([=] { return InformedPlanner<Action, State, float, LogicStateKey>(options, heuristic); },
                         problem, consume);
}

int Benchmark(const Arguments& args, Record& common) {
    std::exception_ptr error;
    std::optional<pddl::ParsedFiles> parsed;
    std::unique_ptr<Problem> problem;
    common.engine = "native-serial";
    common.phase = "parse";
    auto start = Clock::now();
    try { parsed.emplace(pddl::ParseFiles(args.domainFile, args.problemFile)); } catch (...) { error = std::current_exception(); }
    common.parseSeconds = Seconds(start);
    if (error) std::rethrow_exception(error);
    common.phase = "grounding";
    start = Clock::now();
    try {
        auto analyzed = pddl::SemanticAnalyzer{}.Analyze(parsed->domain, parsed->problem);
        problem = std::make_unique<Problem>(pddl::Grounder{}.Ground(analyzed));
    } catch (...) { error = std::current_exception(); }
    common.groundingSeconds = Seconds(start);
    if (error) std::rethrow_exception(error);
    common.phase = "metadata";
    start = Clock::now();
    try {
        common.domainName = problem->Source().source.domainName;
        common.problemName = problem->Source().source.name;
        common.groundedActions = problem->GroundActionCount();
        common.groundedFacts = FactCount(*problem);
        if (!problem->Source().actions.empty()) {
            common.minimumActionCost = std::numeric_limits<double>::infinity();
            for (const auto& action : problem->Source().actions) {
                common.minimumActionCost = std::min(common.minimumActionCost, static_cast<double>(action.cost));
                common.maximumActionCost = std::max(common.maximumActionCost, static_cast<double>(action.cost));
                common.unitCost = common.unitCost && action.cost == 1;
            }
        }
        problem->ForEachInitialState([&](const State& state) {
            problem->ForEachApplicableAction(state, [&](const Action&) { ++common.initialApplicable; return true; });
            return true;
        });
    } catch (...) { error = std::current_exception(); }
    common.metadataSeconds = Seconds(start);
    if (error) std::rethrow_exception(error);
    if (args.prepareOnly) {
        common.status = "prepared";
        common.phase = "complete";
        struct rusage usage {};
        common.peakRssKiB = (getrusage(RUSAGE_SELF, &usage) == 0 ? static_cast<double>(usage.ru_maxrss) : 0.0);
        Print(args, common);
        return 0;
    }
    int exitCode = 0;
    for (std::size_t repeat = 0; repeat < args.repetitions; ++repeat) {
        Record record = common;
        record.repetition = repeat;
        record.phase = "planning";
        SerialReport report;
        error = nullptr;
        start = Clock::now();
        try { report = Solve(*problem, args, record.plan); } catch (...) { error = std::current_exception(); }
        record.planningSeconds = Seconds(start);
        try {
            if (error) std::rethrow_exception(error);
            record.engine = report.engine;
            record.phase = "replay";
            start = Clock::now();
            try {
                if (record.plan) { Replay(*problem, *record.plan); record.replayValid = true; }
            } catch (...) { error = std::current_exception(); }
            record.replaySeconds = Seconds(start);
            if (error) std::rethrow_exception(error);
            const auto solved = (record.plan ? 1 : 0);
            record.status = solved ? "solved" : "unsolved";
            record.phase = "complete";
            if (!solved) exitCode = 2;
        } catch (...) {
            exitCode = RecordFailure(record);
        }
        struct rusage usage {};
        const double localRss = getrusage(RUSAGE_SELF, &usage) == 0 ? static_cast<double>(usage.ru_maxrss) : 0.0;
        record.peakRssKiB = localRss;
        Print(args, record);
        if (exitCode == 1 || exitCode == 3) break;
    }
    return exitCode;
}
} // namespace

int main(int argc, char** argv) {
    Arguments args;
    Record record;
    try {
        Parse(argc, argv, args);
        if (args.help) {
            std::cout <<
                "Usage: benchmark_planner --domain FILE --problem FILE [options]\n"
                "  --algorithm bfs|ucs|astar|greedy|weighted-astar|beam|iddfs|idastar|rbfs\n"
                "              graphplan|satplan|regression|csp|partial-order\n"
                "              dfs|recursive-dfs|bbdfs|hill|ehc\n"
                "              backward-bfs|bidirectional|forward-vi|backward-vi|rollout|adp|adp-astar\n"
                "  --heuristic zero|hmax|hadd|relaxed-plan --weight W --beam-width N\n"
                "  --rollout-depth N --adp-trials N --trial-depth N --repetitions N\n"
                "  --max-depth N --allow-cycles --prepare-only\n"
                "All search executes directly on one CPU thread.\n"
                "Symbolic and finite-state planners require explicit --max-depth.\n"
                "Other default depth is UINT64_MAX; use the campaign runner for time and memory limits.\n"
                "Each JSON line uses a fresh planner with shared grounding; planning includes heuristic setup.\n"
                "Peak RSS is the process high-water mark. Expansion counters are null when uninstrumented.\n"
                "Exit status: solved=0, error=1, no plan within configured semantics=2, resource exhausted=3.\n";
            return 0;
        }
        return Benchmark(args, record);
    } catch (...) {
        const int exitCode = RecordFailure(record);
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) == 0) record.peakRssKiB = static_cast<double>(usage.ru_maxrss);
        Print(args, record);
        return exitCode;
    }
}
