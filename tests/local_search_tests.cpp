#include "GridProblem.h"
#include "LocalSearchPlanner.h"
#include "GraphProblem.h"
#include "PlanningHeuristics.h"
#include "TestSupport.h"
#include "pddl/Adapter.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace planning;
using test::Require;
using test::Throws;

template<class Problem, class PlanType>
void Replay(const Problem& problem, const PlanType& plan) {
    bool valid = false;
    problem.ForEachInitialState([&](const auto& initial) {
        auto state = initial;
        float cost = 0;
        for (const auto& action : plan.actions) {
            auto next = problem.Apply(state, action);
            if (!next) return true;
            cost += problem.StepCost(state, action, *next).value_or(0);
            state = std::move(*next);
        }
        cost += problem.TerminalCost(state).value_or(0);
        valid = problem.IsGoal(state) && (!plan.totalCost || plan.totalCost == cost);
        return !valid;
    });
    Require(valid && plan.stageLen == plan.actions.size() && plan.status == PlanStatus::success,
            "local-search result replays independently");
}
void Examples() {
    test::GraphProblem plateau;
    plateau.goals = {3};
    plateau.edges = {{0,1,1},{1,2,1},{2,3,1}};
    const auto h = [](int state) { return state == 3 ? 0.0F : (state == 0 ? 3.0F : 2.0F); };
    LocalSearchOptions options;
    options.maxDepth = 3;
    Require(HillClimbingPlanner<int,int>(options,h).Search(plateau).empty(), "hill climbing gets stuck on plateau");
    const auto escaped = EnforcedHillClimbingPlanner<int,int>(options,h).Search(plateau);
    Require(escaped.size() == 1 && escaped.front().stageLen == 3, "EHC crosses plateau after committed prefix");
    Replay(plateau, escaped.front());
    options.maxDepth = 2;
    Require(EnforcedHillClimbingPlanner<int,int>(options,h).Search(plateau).empty(),
            "EHC total depth includes committed and escape actions");

    test::GraphProblem minimum;
    minimum.goals = {2};
    minimum.edges = {{0,1,1},{1,2,1}};
    const auto valley = [](int state) { return state == 0 ? 1.0F : state == 1 ? 2.0F : 0.0F; };
    Require(HillClimbingPlanner<int,int>(options,valley).Search(minimum).empty(), "hill climbing local minimum");
    Require(EnforcedHillClimbingPlanner<int,int>(options,valley).Search(minimum).size() == 1,
            "EHC can cross uphill state");

    test::GraphProblem trap;
    trap.edges = {{0,1,1},{0,2,1},{2,4,1}};
    const auto misleading = [](int state) { return state == 0 ? 3.0F : state == 1 ? 1.0F : state == 2 ? 2.0F : 0.0F; };
    Require(HillClimbingPlanner<int,int>(options,misleading).Search(trap).empty(), "hill climbing intentionally incomplete");
    Require(EnforcedHillClimbingPlanner<int,int>(options,misleading).Search(trap).empty(),
            "EHC intentionally commits without backtracking");
    const auto dead = [](int state) {
        return state == 1 ? std::numeric_limits<float>::infinity() : state == 0 ? 2.0F : state == 2 ? 1.0F : 0.0F;
    };
    for (auto algorithm : {LocalSearch::hillClimbing, LocalSearch::enforcedHillClimbing}) {
        options.mode = algorithm;
        const auto solution = LocalSearchPlanner<int,int>(options,dead).Search(trap);
        Require(solution.size() == 1, "proven dead-end neighbor pruned");
        Replay(trap, solution.front());
        Throws([&] { (void)LocalSearchPlanner<int,int>(options, [](int) { return -1.0F; }).Search(trap); },
               "negative heuristic rejected");
        Throws([&] { (void)LocalSearchPlanner<int,int>(options, [](int) {
            return std::numeric_limits<float>::quiet_NaN();
        }).Search(trap); }, "NaN heuristic rejected");
        trap.edges[1].cost = -1;
        Throws([&] { (void)LocalSearchPlanner<int,int>(options,dead).Search(trap); }, "negative accepted-path cost rejected");
        trap.edges[1].cost = 1;
        options.maxPlans = 0;
        Require(LocalSearchPlanner<int,int>(options,dead).Search(trap).empty(), "zero output cap");
        options.maxPlans = 1;
        trap.initial = {4};
        options.maxDepth = 0;
        const auto root = LocalSearchPlanner<int,int>(options,dead).Search(trap);
        Require(root.size() == 1 && root.front().stageLen == 0, "zero-depth initial goal");
        std::size_t count = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            LocalSearchPlanner<int,int>(options,dead).Search(trap,
                [&](const auto&) { ++count; return true; }, {i,3});
        }
        Require(count == 1, "zero-step goal assigned only to partition zero");
        trap.initial = {0};
        options.maxDepth = 2;
    }
    test::GraphProblem steepest;
    steepest.edges = {{0,1,1},{0,2,1},{1,4,1},{2,4,1}};
    const auto values = [](int state) { return state == 0 ? 3.0F : state == 1 ? 2.0F : state == 2 ? 1.0F : 0.0F; };
    const auto best = HillClimbingPlanner<int,int>(options,values).Search(steepest);
    Require(best.front().actions.front() == 1, "steepest hill climbing considers every successor");
}
std::optional<unsigned> ShortestDepth(const test::GraphProblem& graph, unsigned bound) {
    std::vector<std::pair<int,unsigned>> frontier{{0,0}};
    std::vector<int> seen{0};
    for (std::size_t i = 0; i < frontier.size(); ++i) {
        const auto [state, depth] = frontier[i];
        if (graph.IsGoal(state)) return depth;
        if (depth == bound) continue;
        for (const auto& edge : graph.edges) {
            if (edge.from != state || std::find(seen.begin(), seen.end(), edge.to) != seen.end()) continue;
            seen.push_back(edge.to);
            frontier.emplace_back(edge.to, depth + 1);
        }
    }
    return std::nullopt;
}
void RandomZeroHeuristic() {
    std::mt19937 random(3829);
    for (unsigned trial = 0; trial < 100; ++trial) {
        test::GraphProblem graph;
        graph.goals = {5};
        graph.hasCost = trial % 2 != 0;
        graph.terminal = 2;
        for (int from = 0; from < 6; ++from) {
            for (int to = 0; to < 6; ++to) {
                if (random() % 5 == 0) graph.edges.push_back({from,to,static_cast<float>(random() % 4)});
            }
        }
        LocalSearchOptions options;
        options.maxDepth = trial % 6;
        options.prunePathCycles = trial % 2 == 0;
        const auto expected = ShortestDepth(graph, static_cast<unsigned>(options.maxDepth));
        const auto found = EnforcedHillClimbingPlanner<int,int>(options).Search(graph);
        Require(found.empty() == !expected, "zero-heuristic EHC reachability equals bounded BFS");
        if (!found.empty()) {
            Require(found.front().stageLen == *expected, "zero-heuristic EHC escape chooses minimum depth");
            Require(found.front().totalCost.has_value() == graph.hasCost, "costless plan preserves absent cost");
            Replay(graph, found.front());
        }
    }
}
void NativeDomains() {
    GridProblem grid(3,2,{0,0},{2,1});
    const auto problem = pddl::LoadGroundedProblemFromText(R"((define (domain local-test)
        (:predicates (p) (q) (r))
        (:action both :precondition (and) :effect (and (p) (q)))
        (:action finish :precondition (and (p) (q)) :effect (r))))",
        "(define (problem p) (:domain local-test) (:init) (:goal (r)))");
    const DeleteRelaxationHeuristic hmax(problem);
    for (auto algorithm : {LocalSearch::hillClimbing, LocalSearch::enforcedHillClimbing}) {
        LocalSearchOptions options;
        options.mode = algorithm;
        options.maxDepth = 5;
        std::size_t count = 0;
        test::Run([=] { return LocalSearchPlanner<GridMove,GridState>(options, [](const GridState& state) {
            return static_cast<float>((2 - state.x) + (1 - state.y));
        }); }, grid, [&](const auto& plan) {
            Replay(grid,plan); ++count; return true;
        });
        Require(count == 1, "local grid native result count");
        count = 0;
        test::Run([=] {
            return LocalSearchPlanner<pddl::PDDLAction,DynamicBitset,float,LogicStateKey>(options,hmax);
        }, problem, [&](const auto& plan) { Replay(problem,plan); ++count; return true; });
        Require(count == 1, "local PDDL native result count");
    }
}
}
int main() {
    try {
        Examples();
        RandomZeroHeuristic();
        NativeDomains();
        std::cout << "Local-search checks passed: " << test::checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
