#include "FiniteStatePlanner.h"
#include "GridProblem.h"
#include "pddl/Adapter.h"
#include "tests/GraphProblem.h"
#include "tests/TestSupport.h"

#include <random>

using namespace planning;

namespace {
template<typename P, typename PlanType>
void Replay(const P& problem, const PlanType& plan) {
    bool valid = false;
    problem.ForEachInitialState([&](const auto& root) {
        auto state = root;
        float cost = 0;
        for (const auto& action : plan.actions) {
            auto successor = problem.Apply(state, action);
            if (!successor) { return true; }
            cost += problem.StepCost(state, action, *successor).value_or(0);
            state = *successor;
        }
        cost += problem.TerminalCost(state).value_or(0);
        valid = problem.IsGoal(state) && (!plan.totalCost || *plan.totalCost == cost);
        return !valid;
    });
    test::Require(valid && plan.stageLen == plan.actions.size(), "finite-state plan replays");
}

void RandomOracle() {
    std::mt19937 random(19937);
    for (unsigned trial = 0; trial < 120; ++trial) {
        test::GraphProblem problem;
        problem.initial = trial % 4 == 0 ? std::vector<int>{0, 2} : std::vector<int>{0};
        problem.goals = {5};
        problem.terminal = static_cast<float>(static_cast<int>(random() % 7) - 3);
        for (int from = 0; from < 6; ++from) {
            for (int to = 0; to < 6; ++to) {
                if (random() % 5 == 0) {
                    problem.edges.push_back({from, to, static_cast<float>(static_cast<int>(random() % 9) - 4)});
                }
            }
        }
        const finite::Options options{.maxDepth = trial % 6};
        std::optional<float> optimum;
        std::optional<std::size_t> shortest;
        const auto oracle = [&](auto&& self, int state, float cost, std::size_t depth) -> void {
            if (problem.IsGoal(state)) {
                const auto total = cost + problem.terminal;
                if (!optimum || total < *optimum) { optimum = total; }
                if (!shortest || depth < *shortest) { shortest = depth; }
                return;
            }
            if (depth == options.maxDepth) { return; }
            for (const auto& edge : problem.edges) {
                if (edge.from == state) { self(self, edge.to, cost + edge.cost, depth + 1); }
            }
        };
        for (const auto root : problem.initial) { oracle(oracle, root, 0, 0); }
        const finite::ReachableGraph graph(problem, options);
        for (const auto& solution : {finite::ForwardValueIteration(graph), finite::BackwardValueIteration(graph)}) {
            test::Require(solution.has_value() == optimum.has_value(), "DP feasibility agrees with exhaustive oracle");
            if (solution) {
                test::Require(solution->totalCost == optimum, "DP optimum with negative edges and bounded cycles");
                Replay(problem, *solution);
            }
        }
        for (const auto& solution : {finite::BackwardBreadthFirst(graph), finite::BidirectionalBreadthFirst(graph)}) {
            test::Require(solution.has_value() == shortest.has_value(), "reverse BFS feasibility agrees with oracle");
            if (solution) {
                test::Require(solution->stageLen == *shortest, "reverse BFS minimum depth");
                Replay(problem, *solution);
            }
        }
    }
}

void Boundaries() {
    test::GraphProblem problem;
    problem.edges = {{0, 1, 1}, {1, 1, -2}, {1, 4, 1}};
    auto plan = finite::ForwardValueIteration(problem, {.maxDepth = 5});
    test::Require(plan && plan->totalCost == -4 && plan->stageLen == 5, "finite horizon permits profitable repeated state");
    plan = finite::BackwardValueIteration(problem, {.maxDepth = 5});
    test::Require(plan && plan->totalCost == -4, "backward negative cycle is bounded by horizon");
    test::Throws([&] { finite::ReachableGraph graph(problem, {.maxDepth = 5, .maxStates = 1}); }, "graph state budget enforced");
    test::Throws([&] { finite::ReachableGraph graph(problem, {.maxDepth = 5, .maxStates = 10, .maxEdges = 1}); }, "graph edge budget enforced");
    test::Throws([&] { finite::ForwardValueIteration(problem, {.maxDepth = 5, .maxTableEntries = 2}); }, "DP table budget enforced");
    problem.initial = {4, 0};
    plan = finite::BackwardBreadthFirst(problem);
    test::Require(plan && plan->actions.empty(), "multiple initial states include empty plan");
    problem.initial = {0};
    problem.hasCost = false;
    plan = finite::BackwardValueIteration(problem, {.maxDepth = 5});
    test::Require(plan && plan->stageLen == 2 && !plan->totalCost, "costless DP uses unit steps and preserves absent cost");
    problem.initial.clear();
    test::Require(!finite::BidirectionalBreadthFirst(problem), "empty root set is unsolvable");
}

void Domains() {
    GridProblem grid(3, 2, {0, 0}, {2, 1});
    const finite::ReachableGraph graph(grid, {.maxDepth = 6});
    for (const auto& solution : {finite::BackwardBreadthFirst(graph), finite::BidirectionalBreadthFirst(graph),
                                 finite::ForwardValueIteration(graph), finite::BackwardValueIteration(graph)}) {
        test::Require(solution && solution->stageLen == 3, "finite-state grid shortest solution");
        Replay(grid, *solution);
    }
    const auto pddlProblem = pddl::LoadGroundedProblemFromText(
        "(define (domain one) (:requirements :strips) (:predicates (ready) (done)) "
        "(:action setup :parameters () :precondition (and) :effect (ready)) "
        "(:action finish :parameters () :precondition (ready) :effect (done)))",
        "(define (problem p) (:domain one) (:init) (:goal (done)))");
    const finite::ReachableGraph pddlGraph(pddlProblem, {.maxDepth = 4});
    for (const auto& solution : {finite::BackwardBreadthFirst(pddlGraph), finite::BidirectionalBreadthFirst(pddlGraph),
                                 finite::ForwardValueIteration(pddlGraph), finite::BackwardValueIteration(pddlGraph)}) {
        test::Require(solution && solution->stageLen == 2, "finite-state grounded PDDL solution");
        Replay(pddlProblem, *solution);
    }
}

} // namespace

int main() {
    try {
        RandomOracle();
        Boundaries();
        Domains();
        std::cout << "finite-state: " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
