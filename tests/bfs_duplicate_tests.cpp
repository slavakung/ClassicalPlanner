#include "Planner.h"
#include "GraphProblem.h"
#include "TestSupport.h"

#include <iostream>
#include <random>

namespace {
using namespace planning;
using test::Require;

class CountedGraph : public test::GraphProblem {
public:
    mutable std::size_t expansions = 0;
    void ForEachSuccessor(const int& state, const SuccessorConsumer<int, int>& consume) const override {
        ++expansions;
        test::GraphProblem::ForEachSuccessor(state, consume);
    }
    // Deliberate collisions exercise exact equality in the visited table.
    std::size_t HistoryKeyHash(const int&) const override { return 0; }
};

void DiamondLayers() {
    CountedGraph graph;
    graph.goals = {61};
    graph.edges = {{0, 1, 1}, {0, 2, 1}};
    for (int layer = 0; layer < 29; ++layer)
        for (int from = 1 + 2 * layer; from <= 2 + 2 * layer; ++from)
            for (int to = 3 + 2 * layer; to <= 4 + 2 * layer; ++to)
                graph.edges.push_back({from, to, 1});
    graph.edges.push_back({59, 61, 1});
    graph.edges.push_back({60, 61, 1});
    BFSOptions options;
    options.pruneDuplicateStates = true;
    options.maxDepth = 31;
    options.arenaChunkCapacity = 2;
    options.initialArenaBytes = 1;
    const auto plans = BFSPlanner<int, int>(options).Search(graph);
    Require(plans.size() == 1 && plans[0].stageLen == 31, "merging paths retain the shortest plan");
    Require(graph.expansions == 61, "each distinct non-goal state is expanded once, despite exponentially many paths");
    int state = 0;
    for (auto action : plans[0].actions) {
        const auto next = graph.Apply(state, action);
        Require(next.has_value(), "reconstructed action is applicable after path slot recycling");
        state = *next;
    }
    Require(graph.IsGoal(state), "reconstructed plan reaches goal");
}

void MatchTreeSearch() {
    std::mt19937 random(20260924);
    for (unsigned trial = 0; trial < 120; ++trial) {
        CountedGraph graph;
        graph.initial = trial % 3 ? std::vector<int>{0} : std::vector<int>{0, 2, 0};
        graph.goals = trial % 5 ? std::vector<int>{5} : std::vector<int>{0, 5};
        for (int from = 0; from < 6; ++from)
            for (int to = 0; to < 6; ++to)
                if (random() % 4 == 0) graph.edges.push_back({from, to, float(random() % 7)});
        BFSOptions options;
        options.maxDepth = trial % 6;
        options.prunePathCycles = trial % 2 == 0;
        const auto tree = BFSPlanner<int, int>(options).Search(graph);
        options.pruneDuplicateStates = true;
        const auto dedup = BFSPlanner<int, int>(options).Search(graph);
        Require(tree.size() == dedup.size(), "duplicate pruning preserves bounded reachability");
        if (!tree.empty()) {
            Require(tree[0].actions == dedup[0].actions, "duplicate pruning preserves first FIFO solution");
            Require(tree[0].totalCost == dedup[0].totalCost, "BFS still reports cost without optimizing it");
        }
    }
}

void PreserveEnumeration() {
    test::GraphProblem graph;
    graph.goals = {3};
    graph.edges = {{0, 1, 1}, {0, 2, 1}, {1, 3, 1}, {2, 3, 1}};
    BFSOptions options;
    options.returnMode = BFSReturnMode::allMinimalDepth;
    Require(BFSPlanner<int, int>(options).Search(graph).size() == 2,
            "default BFS enumeration keeps distinct paths to the same goal");
    options.pruneDuplicateStates = true;
    test::Throws([&] { (void)BFSPlanner<int, int>(options).Search(graph); },
                 "state pruning cannot silently suppress requested plan enumeration");
    options.returnMode = BFSReturnMode::firstGoal;
    test::Throws([&] {
        BFSPlanner<int, int>(options).Search(graph, [](const auto&) { return true; }, {0, 2});
    }, "root partitioning cannot silently change duplicate dominance");
}
}

int main() {
    try {
        DiamondLayers();
        MatchTreeSearch();
        PreserveEnumeration();
        std::cout << "BFS duplicate checks passed: " << test::checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
