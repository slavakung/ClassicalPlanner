#include "GridProblem.h"
#include "IterativePlanner.h"
#include "GraphProblem.h"
#include "PlanningHeuristics.h"
#include "TestSupport.h"
#include "pddl/Adapter.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace planning;
std::size_t checks = 0;
void Require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class F>
void Throws(F&& function, const std::string& message) {
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    Require(threw, message);
}
struct Graph : test::GraphProblem {
    std::vector<float> terminalCosts = std::vector<float>(6, 0);
    std::optional<float> TerminalCost(const int& state) const override {
        return hasCost ? std::optional<float>{terminalCosts.at(static_cast<std::size_t>(state))}
                       : std::nullopt;
    }
};
struct Optimum { std::optional<float> cost; std::optional<std::uint64_t> depth; };
Optimum Oracle(const Graph& graph, unsigned maxDepth, bool pruneCycles) {
    Optimum result;
    std::vector<int> history;
    auto visit = [&](auto&& self, int state, float cost, unsigned depth) -> void {
        if (graph.IsGoal(state)) {
            const float total = cost + graph.TerminalCost(state).value_or(0);
            if (!result.cost || total < *result.cost) result.cost = total;
            if (!result.depth || depth < *result.depth) result.depth = depth;
        }
        if (depth == maxDepth) return;
        for (const auto& edge : graph.edges) {
            if (edge.from != state || (pruneCycles &&
                std::find(history.begin(), history.end(), edge.to) != history.end())) continue;
            history.push_back(edge.to);
            self(self, edge.to, cost + (graph.hasCost ? edge.cost : 1), depth + 1);
            history.pop_back();
        }
    };
    for (int root : graph.initial) {
        history = {root};
        visit(visit, root, 0, 0);
    }
    return result;
}
void Replay(const Graph& graph, const Plan<int,float>& plan) {
    Require(plan.status == PlanStatus::success && plan.stageLen == plan.actions.size(), "plan metadata");
    bool valid = false;
    for (int root : graph.initial) {
        int state = root;
        float total = 0;
        bool applicable = true;
        for (int action : plan.actions) {
            const auto next = graph.Apply(state, action);
            if (!next) { applicable = false; break; }
            total += graph.StepCost(state, action, *next).value_or(0);
            state = *next;
        }
        if (applicable && graph.IsGoal(state)) {
            total += graph.TerminalCost(state).value_or(0);
            valid = valid || (graph.hasCost ? plan.totalCost == total : !plan.totalCost);
        }
    }
    Require(valid, "plan independently replays with reported cost");
}
std::vector<float> AdmissibleHeuristic(const Graph& graph) {
    std::vector<float> h(6, 1000);
    for (int goal : graph.goals) h[static_cast<std::size_t>(goal)] = 0;
    for (unsigned iteration = 0; iteration < 6; ++iteration) {
        for (const auto& edge : graph.edges) {
            auto& from = h[static_cast<std::size_t>(edge.from)];
            from = std::min(from, (graph.hasCost ? edge.cost : 1) + h[static_cast<std::size_t>(edge.to)]);
        }
    }
    for (auto& value : h) if (value == 1000) value = 0;
    return h;
}
void RandomGraphs() {
    std::mt19937 random(9751);
    for (unsigned trial = 0; trial < 100; ++trial) {
        Graph graph;
        graph.hasCost = trial % 5 != 0;
        graph.initial = trial % 4 == 0 ? std::vector<int>{0,2} : std::vector<int>{0};
        graph.goals = trial % 7 == 0 ? std::vector<int>{0,4,5} : std::vector<int>{4,5};
        for (auto& terminal : graph.terminalCosts) terminal = static_cast<float>(random() % 5);
        for (int from = 0; from < 6; ++from) {
            for (int to = 0; to < 6; ++to) {
                if (random() % 5 == 0) graph.edges.push_back({from,to,static_cast<float>(random() % 4)});
            }
        }
        IterativeOptions options;
        options.maxDepth = trial % 6;
        options.prunePathCycles = trial % 2 == 0;
        const auto expected = Oracle(graph, static_cast<unsigned>(options.maxDepth), options.prunePathCycles);
        const auto h = AdmissibleHeuristic(graph);
        // Alternating exact/zero entries produces admissible but often
        // inconsistent heuristics, exercising RBFS pathmax and IDA* thresholds.
        const auto heuristic = [=](int state) { return state % 2 ? 0.0F : h[static_cast<std::size_t>(state)]; };
        for (const auto algorithm : {IterativeSearch::iterativeDeepening, IterativeSearch::idaStar,
                                     IterativeSearch::recursiveBestFirst}) {
            options.mode = algorithm;
            IterativePlanner<int,int> planner(options, heuristic);
            const auto result = planner.Search(graph);
            Require(result.empty() == !expected.cost, "random feasibility agrees with exhaustive oracle");
            if (!result.empty()) {
                Require(result.size() == 1, "one representative returned");
                Replay(graph, result.front());
                if (algorithm == IterativeSearch::iterativeDeepening) {
                    Require(result.front().stageLen == expected.depth, "IDDFS minimizes depth");
                } else {
                    const float actual = graph.hasCost ? *result.front().totalCost
                                                       : static_cast<float>(result.front().stageLen);
                    Require(actual == expected.cost, "IDA*/RBFS minimize total cost");
                }
            }

        }
    }
}
void Regressions() {
    Graph graph;
    graph.goals = {1,3};
    graph.edges = {{0,1,1},{1,3,1},{0,2,1},{2,3,3}};
    graph.terminalCosts[1] = 20;
    IterativeOptions options;
    options.maxDepth = 4;
    Require(IterativeDeepeningPlanner<int,int>(options).Search(graph).front().stageLen == 1,
            "IDDFS returns shallow goal");
    Require(IDAStarPlanner<int,int>(options).Search(graph).front().totalCost == 2,
            "IDA* continues through expensive terminal goal");
    Require(RBFSPlanner<int,int>(options).Search(graph).front().totalCost == 2,
            "RBFS continues through expensive terminal goal");
    for (auto algorithm : {IterativeSearch::idaStar, IterativeSearch::recursiveBestFirst}) {
        options.mode = algorithm;
        graph.edges[0].cost = -1;
        Throws([&] { (void)IterativePlanner<int,int>(options).Search(graph); }, "negative edge rejected");
        graph.edges[0].cost = 1;
        Throws([&] { (void)IterativePlanner<int,int>(options, [](int) {
            return std::numeric_limits<float>::quiet_NaN();
        }).Search(graph); }, "NaN heuristic rejected");
        Throws([&] { (void)IterativePlanner<int,int>(options, [](int) { return 1.0F; }).Search(graph); },
               "nonzero goal heuristic rejected");
        options.maxPlans = 0;
        Require(IterativePlanner<int,int>(options).Search(graph).empty(), "zero plan cap");
        options.maxPlans = 1;
        Graph dead;
        dead.goals = {1};
        dead.edges = {{0,2,0},{0,1,1}};
        const auto provenDead = [](int state) {
            return state == 2 ? std::numeric_limits<float>::infinity() : 0.0F;
        };
        Require(IterativePlanner<int,int>(options, provenDead).Search(dead).front().totalCost == 1,
                "positive infinite heuristic prunes a proven dead end");
        dead.initial = {2};
        Require(IterativePlanner<int,int>(options, provenDead).Search(dead).empty(),
                "infinite root heuristic terminates without numeric overflow");
    }
    Graph empty;
    empty.initial.clear();
    options.maxDepth = std::numeric_limits<std::uint64_t>::max();
    Require(IterativeDeepeningPlanner<int,int>(options).Search(empty).empty(), "empty root set terminates");
    empty.initial = {0};
    Require(IterativeDeepeningPlanner<int,int>(options).Search(empty).empty(), "exhausted dead end terminates");
}
template<class Problem>
void ReplayNative(const Problem& problem, const typename IterativePlanner<
        typename Problem::ActionType, typename Problem::StateType, typename Problem::CostType,
        typename Problem::HistoryKeyType>::PlanType& plan) {
    bool valid = false;
    problem.ForEachInitialState([&](const auto& initial) {
        auto state = initial;
        float total = 0;
        for (const auto& action : plan.actions) {
            auto next = problem.Apply(state, action);
            if (!next) return true;
            total += problem.StepCost(state, action, *next).value_or(0);
            state = std::move(*next);
        }
        total += problem.TerminalCost(state).value_or(0);
        valid = problem.IsGoal(state) && (!plan.totalCost || *plan.totalCost == total);
        return !valid;
    });
    Require(valid && plan.actions.size() == plan.stageLen, "native CPU replay of native problem");
}
void NativeDomains() {
    GridProblem grid(3, 2, {0,0}, {2,1});
    const auto problem = pddl::LoadGroundedProblemFromText(R"((define (domain iterative-test)
        (:predicates (p) (q) (r))
        (:action both :precondition (and) :effect (and (p) (q)))
        (:action finish :precondition (and (p) (q)) :effect (r))))",
        "(define (problem p) (:domain iterative-test) (:init) (:goal (r)))");
    const DeleteRelaxationHeuristic hmax(problem);
    for (auto algorithm : {IterativeSearch::iterativeDeepening, IterativeSearch::idaStar,
                           IterativeSearch::recursiveBestFirst}) {
        IterativeOptions options;
        options.mode = algorithm;
        options.maxDepth = 5;
        std::size_t count = 0;
        test::Run([=] { return IterativePlanner<GridMove,GridState>(options, [](const GridState& state) {
            return static_cast<float>((2 - state.x) + (1 - state.y));
        }); }, grid, [&](const auto& plan) {
            ReplayNative(grid, plan);
            Require(plan.stageLen == 3, "iterative grid optimum");
            ++count;
            return true;
        });
        Require(count == 1, "iterative grid native result count");
        count = 0;
        test::Run([=] {
            return IterativePlanner<pddl::PDDLAction,DynamicBitset,float,LogicStateKey>(options, hmax);
        }, problem, [&](const auto& plan) {
            ReplayNative(problem, plan);
            Require(plan.stageLen == 2, "iterative PDDL optimum");
            ++count;
            return true;
        });
        Require(count == 1, "iterative PDDL native result count");
    }
}
}
int main() {
    try {
        RandomGraphs();
        Regressions();
        NativeDomains();
        std::cout << "Iterative planning checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
