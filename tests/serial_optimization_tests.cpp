#include "FiniteStatePlanner.h"
#include "InformedPlanner.h"
#include "tests/GraphProblem.h"

#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void PriorityCache() {
    test::GraphProblem problem;
    problem.goals = {3};
    problem.edges = {{0,1,1},{0,1,1},{0,2,1},{1,0,1},{1,3,1},{2,3,1}};
    planning::InformedOptions options;
    options.maxDepth = 4;
    std::size_t calls = 0;
    auto heuristic = [&](int) { ++calls; return 0.0F; };
    auto result = planning::InformedPlanner<int,int>(options, heuristic).Search(problem);
    Check(calls == 4, "A*: cycle and dominated labels never evaluate heuristics; accepted labels evaluate once");
    Check(result.at(0).actions == std::vector<int>({0,4}), "A*: stable returned action sequence");
    calls = 0;
    options.mode = planning::InformedSearch::beam;
    result = planning::InformedPlanner<int,int>(options, heuristic).Search(problem);
    Check(calls == 7, "beam: comparator reads cached priorities and retains duplicate candidate semantics");
    Check(result.at(0).actions == std::vector<int>({0,4}), "beam: stable tie ordering");
}
void ExactInterner() {
    struct Problem : test::GraphProblem {
        bool collide = false;
        mutable std::size_t comparisons = 0;
        std::size_t HistoryKeyHash(const int& key) const override {
            return collide ? 0 : static_cast<std::size_t>(key);
        }
        bool HistoryKeysEqual(const int& a, const int& b) const override {
            ++comparisons;
            return a == b;
        }
    } problem;
    problem.goals = {80};
    for (int i = 0; i < 80; ++i) {
        problem.edges.push_back({i,i+1,1});
        if (i) problem.edges.push_back({i,0,1});
    }
    planning::finite::Options options;
    options.maxDepth = 100;
    planning::finite::ReachableGraph reference(problem, options);
    Check(reference.states.size() == 81, "interner retains distinct states");
    Check(problem.comparisons < 200, "distributed hashes avoid scanning unrelated keys");
    const auto expected = planning::finite::BackwardBreadthFirst(reference);
    problem.collide = true;
    planning::finite::ReachableGraph collision(problem, options);
    Check(collision.states == reference.states, "hash collisions preserve exact discovery order");
    Check(collision.edges.size() == reference.edges.size(), "hash collisions preserve edges");
    Check(planning::finite::BackwardBreadthFirst(collision)->actions == expected->actions,
          "hash collisions preserve complete returned plan");
}
}
int main() {
    try { PriorityCache(); ExactInterner(); std::cout << "serial optimization regressions passed\n"; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
