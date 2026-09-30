#include "GridProblem.h"
#include "InformedPlanner.h"
#include "PlanningHeuristics.h"
#include "pddl/Adapter.h"
#include "tests/GraphProblem.h"
#include "tests/TestSupport.h"

#include <random>

using namespace planning;
using test::Require;
using test::Throws;

namespace {
template<typename P, typename PlanT>
void Replay(const P& p, const PlanT& plan) {
    bool valid = false;
    p.ForEachInitialState([&](const auto& initial) {
        auto state = initial;
        float cost = 0;
        for (const auto& a : plan.actions) {
            auto next = p.Apply(state, a);
            if (!next) return true;
            cost += p.StepCost(state, a, *next).value_or(0);
            state = std::move(*next);
        }
        cost += p.TerminalCost(state).value_or(0);
        valid = p.IsGoal(state) && plan.stageLen == plan.actions.size() &&
            (!plan.totalCost || std::abs(cost - *plan.totalCost) < 1e-5F);
        return !valid;
    });
    Require(valid, "independent plan replay");
}

void Oracles() {
    std::mt19937 rng(781238);
    for (int run = 0; run < 160; ++run) {
        test::GraphProblem p;
        p.goals = {5};
        p.hasCost = run % 3 != 0;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                if (rng() % 5 == 0) p.edges.push_back({i, j, static_cast<float>(rng() % 8)});
        InformedOptions o;
        o.maxDepth = static_cast<std::uint64_t>(run % 7);
        float best = std::numeric_limits<float>::infinity();
        std::function<void(int, unsigned, float)> enumerate = [&](int s, unsigned depth, float g) {
            if (p.IsGoal(s)) best = std::min(best, g);
            if (depth == o.maxDepth) return;
            for (const auto& edge : p.edges)
                if (edge.from == s) enumerate(edge.to, depth + 1, g + (p.hasCost ? edge.cost : 1));
        };
        enumerate(0, 0, 0);
        for (auto mode : {InformedSearch::uniformCost, InformedSearch::aStar}) {
            o.mode = mode;
            const auto plans = InformedPlanner<int,int>(o).Search(p);
            Require(plans.empty() == !std::isfinite(best), "bounded oracle reachability");
            if (!plans.empty()) {
                Replay(p, plans.front());
                Require((p.hasCost ? *plans.front().totalCost : plans.front().stageLen) == best,
                        "bounded oracle optimal objective");
                Require(plans.front().totalCost.has_value() == p.hasCost, "costless semantics");
            }
        }
    }
    test::GraphProblem bounded;
    bounded.edges = {{0,1,0},{1,2,0},{2,3,0},{0,3,5},{3,4,1}};
    InformedOptions o;
    o.maxDepth = 2;
    Require(*InformedPlanner<int,int>(o).Search(bounded).at(0).totalCost == 6,
            "deeper cheaper label does not discard feasible shallow label");
    test::GraphProblem reopen;
    reopen.goals = {3};
    reopen.edges = {{0,1,3},{0,2,1},{2,1,1},{1,3,2},{2,3,100}};
    o.maxDepth = 5;
    const auto result = InformedPlanner<int,int>(o, [](int s) { return s == 2 ? 3.0F : 0.0F; }).Search(reopen);
    Require(*result.at(0).totalCost == 4, "A* reopens under inconsistent admissible heuristic");
    struct TerminalProblem : test::GraphProblem {
        std::optional<float> TerminalCost(const int& s) const override { return s == 4 ? 20 : 0; }
    } terminal;
    terminal.goals = {4,5};
    terminal.edges = {{0,4,1},{4,5,1},{0,5,8}};
    Require(*InformedPlanner<int,int>(o).Search(terminal).at(0).totalCost == 2,
            "virtual terminal edge permits cheaper later goal");
    bounded.edges.front().cost = -1;
    Throws([&] { (void)InformedPlanner<int,int>(o).Search(bounded); }, "negative cost rejected");
    Throws([&] { (void)InformedPlanner<int,int>(o, [](int) { return std::nanf(""); }).Search(reopen); },
           "NaN heuristic rejected");
    o.maxPlans = 0;
    Require(InformedPlanner<int,int>(o).Search(reopen).empty(), "zero output cap");
}

void HashIndexOracles() {
    // Force collisions as well as a well-distributed index; both must keep
    // reopenings and depth/cost Pareto labels identical to the constant hash.
    struct HashedGraph : test::GraphProblem {
        int buckets = 1;
        std::size_t HistoryKeyHash(const int& key) const override {
            return static_cast<std::size_t>(key % buckets);
        }
    } graph;
    graph.goals = {5};
    graph.edges = {{0,1,4},{0,2,1},{2,1,1},{1,3,0},{3,4,0},{4,5,1},
                   {0,4,9},{2,3,8},{3,1,0}};
    for (const auto depth : {2U, 3U, 5U, 8U}) {
        InformedOptions options;
        options.maxDepth = depth;
        graph.buckets = 1;
        const auto reference = InformedPlanner<int,int>(options).Search(graph);
        for (const int buckets : {2, 7}) {
            graph.buckets = buckets;
            const auto result = InformedPlanner<int,int>(options).Search(graph);
            Require(result.size() == reference.size(), "hash collisions preserve reachability");
            if (!result.empty()) {
                Require(result.front().totalCost == reference.front().totalCost &&
                        result.front().actions == reference.front().actions,
                        "indexed duplicate detection preserves Pareto labels and deterministic tie breaks");
                Replay(graph, result.front());
            }
        }
    }
}

void PartitionOracles() {
    test::GraphProblem regression;
    regression.initial = {0,1};
    regression.goals = {3};
    regression.edges = {{0,2,2},{1,2,1},{1,3,0},{2,1,0}};
    InformedOptions options;
    options.maxDepth = 3;
    for (auto mode : {InformedSearch::uniformCost, InformedSearch::aStar}) {
        options.mode = mode;
        std::vector<Plan<int,float>> plans;
        InformedPlanner<int,int>(options).Search(regression, [&](const auto& plan) {
            plans.push_back(plan); return true;
        }, {0,2});
        Require(plans.size() == 1 && plans.front().totalCost == 2 &&
                plans.front().actions == std::vector<int>({0,3,2}),
                "another initial root cannot dominate a usable partition prefix");
    }

    std::mt19937 random(193041);
    for (unsigned trial = 0; trial < 80; ++trial) {
        struct Problem : test::GraphProblem {
            std::vector<float> terminalCosts = std::vector<float>(6);
            std::optional<float> TerminalCost(const int& state) const override {
                return hasCost ? std::optional<float>{terminalCosts[static_cast<std::size_t>(state)]}
                               : std::nullopt;
            }
        } problem;
        problem.initial = {0,1};
        problem.goals = trial % 5 == 0 ? std::vector<int>{0,4,5} : std::vector<int>{4,5};
        problem.hasCost = trial % 3 != 0;
        for (auto& cost : problem.terminalCosts) cost = static_cast<float>(random() % 5);
        for (int from = 0; from < 6; ++from)
            for (int to = 0; to < 6; ++to)
                if (random() % 4 == 0)
                    problem.edges.push_back({from,to,static_cast<float>(random() % 4)});
        options.maxDepth = trial % 6;
        options.prunePathCycles = trial % 2 == 0;
        std::vector<float> lower(6, std::numeric_limits<float>::infinity());
        for (auto goal : problem.goals) lower[static_cast<std::size_t>(goal)] = 0;
        for (unsigned pass = 0; pass < 6; ++pass)
            for (const auto& edge : problem.edges)
                lower[static_cast<std::size_t>(edge.from)] = std::min(lower[static_cast<std::size_t>(edge.from)],
                    (problem.hasCost ? edge.cost : 1) + lower[static_cast<std::size_t>(edge.to)]);
        // Lowering alternating entries preserves admissibility while allowing
        // inconsistency; unreachable states use the safe zero lower bound.
        const auto heuristic = [lower](int state) {
            const auto value = lower[static_cast<std::size_t>(state)];
            return state % 2 || !std::isfinite(value) ? 0.0F : value;
        };
        for (std::size_t partition = 0; partition < 3; ++partition) {
            std::optional<float> optimum;
            std::vector<int> history;
            const auto enumerate = [&](auto&& self, int state, unsigned depth, float cost) -> void {
                if (problem.IsGoal(state) && (depth != 0 || partition == 0)) {
                    const auto total = cost + problem.TerminalCost(state).value_or(0);
                    if (!optimum || total < *optimum) optimum = total;
                }
                if (depth == options.maxDepth) return;
                std::size_t rootAction = 0;
                for (const auto& edge : problem.edges) {
                    if (edge.from != state) continue;
                    if (depth == 0 && rootAction++ % 3 != partition) continue;
                    if (options.prunePathCycles &&
                        std::find(history.begin(), history.end(), edge.to) != history.end()) continue;
                    history.push_back(edge.to);
                    self(self, edge.to, depth + 1, cost + (problem.hasCost ? edge.cost : 1));
                    history.pop_back();
                }
            };
            for (auto root : problem.initial) { history = {root}; enumerate(enumerate,root,0,0); }
            for (auto mode : {InformedSearch::uniformCost, InformedSearch::aStar}) {
                options.mode = mode;
                std::vector<Plan<int,float>> plans;
                InformedPlanner<int,int>(options,heuristic).Search(problem, [&](const auto& plan) {
                    plans.push_back(plan); return true;
                }, {partition,3});
                Require(plans.empty() == !optimum, "each partition preserves multiroot oracle feasibility");
                if (!plans.empty()) {
                    Replay(problem,plans.front());
                    Require((problem.hasCost ? *plans.front().totalCost : plans.front().stageLen) == *optimum,
                            "each partition preserves bounded oracle optimum");
                }
            }
        }
    }
}

void NativeDomains() {
    GridProblem grid(4, 3, {0,0}, {3,2});
    for (auto mode : {InformedSearch::uniformCost, InformedSearch::aStar,
                     InformedSearch::greedy, InformedSearch::weightedAStar, InformedSearch::beam}) {
        InformedOptions o;
        o.mode = mode;
        o.maxDepth = 8;
        o.weight = 2;
        std::size_t count = 0;
        test::Run([=] { return InformedPlanner<GridMove,GridState>(o, [](const GridState& s) {
            return static_cast<float>((3 - s.x) + (2 - s.y));
        }); }, grid, [&](const auto& plan) {
            Replay(grid, plan);
            if (mode == InformedSearch::aStar || mode == InformedSearch::uniformCost)
                Require(plan.stageLen == 5, "native informed minimum");
            ++count;
            return true;
        });
        Require(count == 1, "one informed result");
    }
    const auto problem = pddl::LoadGroundedProblemFromText(R"((define (domain d)
        (:predicates (p) (q) (r))
        (:action both :precondition (and) :effect (and (p) (q)))
        (:action finish :precondition (and (p) (q)) :effect (r))))",
        "(define (problem p) (:domain d) (:init) (:goal (r)))");
    const auto initial = problem.InitialStates().front();
    DeleteRelaxationHeuristic hmax(problem);
    Require(hmax(initial) == 2, "hmax shared achiever");
    Require(DeleteRelaxationHeuristic(problem, RelaxationHeuristic::additive)(initial) == 3,
            "hadd counts shared support twice");
    Require(DeleteRelaxationHeuristic(problem, RelaxationHeuristic::relaxedPlan)(initial) == 2,
            "relaxed plan counts shared action once");
    InformedOptions options;
    options.maxDepth = 4;
    test::Run([&] {
        return InformedPlanner<pddl::PDDLAction,DynamicBitset,float,LogicStateKey>(options, hmax);
    }, problem, [&](const auto& plan) { Replay(problem, plan); Require(plan.stageLen == 2, "PDDL A*"); return true; });
    const auto negative = pddl::LoadGroundedProblemFromText(R"((define (domain d)
        (:requirements :strips :negative-preconditions) (:predicates (p) (q))
        (:action remove :precondition (p) :effect (not (p)))
        (:action finish :precondition (not (p)) :effect (q))))",
        "(define (problem p) (:domain d) (:init (p)) (:goal (and (not (p)) (q))))");
    Require(DeleteRelaxationHeuristic(negative)(negative.InitialStates().front()) == 2,
            "signed delete relaxation");
    const auto unreachable = pddl::LoadGroundedProblemFromText(
        "(define (domain d) (:predicates (p)))", "(define (problem p) (:domain d) (:init) (:goal (p)))");
    Require(std::isinf(DeleteRelaxationHeuristic(unreachable)(unreachable.InitialStates().front())),
            "relaxed dead end is infinite");
    Require(InformedPlanner<pddl::PDDLAction,DynamicBitset,float,LogicStateKey>(options,
        DeleteRelaxationHeuristic(unreachable)).Search(unreachable).empty(), "A* prunes proven relaxed dead end");
}
}

int main() {
    try {
        Oracles();
        HashIndexOracles();
        PartitionOracles();
        NativeDomains();
        std::cout << test::checks << " informed-search checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
