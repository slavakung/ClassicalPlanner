#include "Planner.h"
#include "GraphProblem.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace planning;
std::size_t checks = 0;
void Require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class Exception = std::exception, class F>
void Throws(F&& function, const std::string& message) {
    bool threw = false;
    try { function(); } catch (const Exception&) { threw = true; }
    Require(threw, message);
}
using Plans = std::vector<Plan<int, float>>;
auto Signatures(const Plans& plans) {
    std::vector<std::tuple<std::vector<int>, std::optional<float>, std::uint64_t>> signatures;
    for (const auto& plan : plans) {
        Require(plan.status == PlanStatus::success, "returned plan must be successful");
        Require(plan.stageLen == plan.actions.size(), "reconstructed depth must match actions");
        signatures.emplace_back(plan.actions, plan.totalCost, plan.stageLen);
    }
    std::sort(signatures.begin(), signatures.end());
    return signatures;
}
Plans Oracle(const test::GraphProblem& graph, unsigned maxDepth, bool pruneCycles) {
    Plans result;
    std::vector<int> actions, history;
    auto visit = [&](auto&& self, int state, float cost) -> void {
        if (std::find(graph.goals.begin(), graph.goals.end(), state) != graph.goals.end()) {
            result.push_back({actions, graph.hasCost ? std::optional<float>{cost + graph.terminal}
                                                    : std::nullopt,
                              actions.size(), PlanStatus::success});
            return;
        }
        if (actions.size() == maxDepth) return;
        for (std::size_t index = 0; index < graph.edges.size(); ++index) {
            const auto& edge = graph.edges[index];
            if (edge.from != state || (pruneCycles &&
                std::find(history.begin(), history.end(), edge.to) != history.end())) continue;
            actions.push_back(static_cast<int>(index));
            history.push_back(edge.to);
            self(self, edge.to, cost + edge.cost);
            history.pop_back();
            actions.pop_back();
        }
    };
    for (int root : graph.initial) {
        history = {root};
        visit(visit, root, 0);
    }
    return result;
}
void RandomGraphs() {
    std::mt19937 random(441);
    for (unsigned trial = 0; trial < 100; ++trial) {
        test::GraphProblem graph;
        graph.initial = trial % 3 == 0 ? std::vector<int>{0, 2} : std::vector<int>{0};
        graph.goals = trial % 7 == 0 ? std::vector<int>{0, 4} : std::vector<int>{4};
        const bool negative = trial % 2 == 0;
        const bool pruneCycles = trial % 3 == 0;
        const unsigned maxDepth = trial % 5;
        graph.terminal = static_cast<float>(static_cast<int>(random() % 4) - (negative ? 2 : 0));
        for (int from = 0; from < 5; ++from) {
            for (int to = 0; to < 5; ++to) {
                if (random() % 4 == 0) {
                    graph.edges.push_back({from, to,
                        static_cast<float>(static_cast<int>(random() % 5) - (negative ? 2 : 0))});
                }
            }
        }
        auto expected = Oracle(graph, maxDepth, pruneCycles);
        const auto all = Signatures(expected);
        BFSOptions bfs;
        bfs.returnMode = BFSReturnMode::allPlansUpToDepth;
        bfs.maxDepth = maxDepth;
        bfs.maxPlans = 10000;
        bfs.prunePathCycles = pruneCycles;
        bfs.arenaChunkCapacity = 1 + trial % 3;
        bfs.initialArenaBytes = 1;
        DFSOptions<float> dfs;
        dfs.returnMode = DFSReturnMode::allPlansUpToDepth;
        dfs.maxDepth = maxDepth;
        dfs.maxPlans = 10000;
        dfs.prunePathCycles = pruneCycles;
        dfs.arenaChunkCapacity = bfs.arenaChunkCapacity;
        dfs.initialArenaBytes = 1;
        dfs.stackReserveHint = 0;
        Require(Signatures(BFSPlanner<int,int>(bfs).Search(graph)) == all, "BFS oracle enumeration");
        Require(Signatures(DFSPlanner<int,int>(dfs).Search(graph)) == all, "DFS oracle enumeration");
        Require(Signatures(RecursiveDFSPlanner<int,int>(dfs).Search(graph)) == all, "recursive oracle enumeration");
        bfs.returnMode = BFSReturnMode::allMinimalDepth;
        auto minimumDepth = expected;
        if (!minimumDepth.empty()) {
            const auto depth = std::min_element(minimumDepth.begin(), minimumDepth.end(),
                [](const auto& a, const auto& b) { return a.stageLen < b.stageLen; })->stageLen;
            std::erase_if(minimumDepth, [=](const auto& plan) { return plan.stageLen != depth; });
        }
        Require(Signatures(BFSPlanner<int,int>(bfs).Search(graph)) == Signatures(minimumDepth),
                "BFS minimum depth with multiple roots");
        for (int bound = -1; bound <= 3; bound += 2) {
            auto within = expected;
            std::erase_if(within, [=](const auto& plan) { return *plan.totalCost > bound; });
            dfs.returnMode = DFSReturnMode::allPlansWithinCost;
            dfs.assumeNonnegativeCosts = !negative;
            dfs.targetCost = static_cast<float>(bound);
            Require(Signatures(DFSPlanner<int,int>(dfs).Search(graph)) == Signatures(within),
                    "DFS inclusive cost bound");
            Require(Signatures(RecursiveDFSPlanner<int,int>(dfs).Search(graph)) == Signatures(within),
                    "recursive DFS inclusive cost bound");
            BBDFSOptions<float> bb;
            bb.maxDepth = maxDepth;
            bb.maxPlans = 10000;
            bb.prunePathCycles = pruneCycles;
            bb.assumeNonnegativeCosts = !negative;
            bb.initialCostBound = static_cast<float>(bound);
            bb.returnMode = BBDFSReturnMode::allWithinCost;
            bb.initialArenaBytes = 1;
            bb.arenaChunkCapacity = bfs.arenaChunkCapacity;
            bb.stackReserveHint = 0;
            Require(Signatures(BranchAndBoundDFSPlanner<int,int>(bb).Search(graph)) == Signatures(within),
                    "BBDFS cost bound");
            auto optimal = within;
            if (!optimal.empty()) {
                const auto cost = std::min_element(optimal.begin(), optimal.end(),
                    [](const auto& a, const auto& b) { return a.totalCost < b.totalCost; })->totalCost;
                std::erase_if(optimal, [=](const auto& plan) { return plan.totalCost != cost; });
            }
            bb.returnMode = BBDFSReturnMode::allOptimal;
            Require(Signatures(BranchAndBoundDFSPlanner<int,int>(bb).Search(graph)) == Signatures(optimal),
                    "BBDFS all optimal bounded ties");
            bb.maxPlans = 1;
            const auto single = BranchAndBoundDFSPlanner<int,int>(bb).Search(graph);
            Require(single.empty() == optimal.empty(), "BBDFS maxPlans feasibility");
            if (!single.empty()) Require(single.front().totalCost == optimal.front().totalCost,
                                          "BBDFS maxPlans must not truncate proof");
        }
    }
}
struct ThrowingValue {
    int value;
    static inline int movesBeforeThrow = -1;
    explicit ThrowingValue(int v) : value(v) {}
    ThrowingValue(const ThrowingValue&) = default;
    ThrowingValue(ThrowingValue&& other) : value(other.value) {
        if (movesBeforeThrow == 0) throw std::runtime_error("injected move failure");
        if (movesBeforeThrow > 0) --movesBeforeThrow;
    }
    ThrowingValue& operator=(const ThrowingValue&) = default;
    ThrowingValue& operator=(ThrowingValue&& other) {
        if (movesBeforeThrow == 0) throw std::runtime_error("injected assignment failure");
        if (movesBeforeThrow > 0) --movesBeforeThrow;
        value = other.value;
        return *this;
    }
};
class FailingResource final : public std::pmr::memory_resource {
public:
    bool fail = false;
private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (fail) throw std::bad_alloc();
        return std::pmr::new_delete_resource()->allocate(bytes, alignment);
    }
    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override {
        std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
};
void StorageFailures() {
    SearchMemory memory(1);
    PathArena<int, ThrowingValue> paths(memory.resource(), 1);
    ThrowingValue::movesBeforeThrow = 1;
    Throws([&] { (void)paths.AddRoot(ThrowingValue{8}); }, "injected history move must fail");
    Require(paths.size() == 0, "failed append must not count a path");
    ThrowingValue::movesBeforeThrow = -1;
    const auto root = paths.AddRoot(ThrowingValue{9});
    Require(root.value == 0 && paths.History(root).value == 9 && paths.Depth(root) == 0,
            "retry must reuse failed root chunk");
    ThrowingValue::movesBeforeThrow = 1;
    Throws([&] { (void)paths.AddChild(root, 3, ThrowingValue{10}); }, "child history move failure");
    ThrowingValue::movesBeforeThrow = -1;
    const auto child = paths.AddChild(root, 4, ThrowingValue{11});
    Require(child.value == 1 && paths.History(child).value == 11 && paths.Parent(child) == root &&
            paths.ActionFromParent(child) == 4, "append after exception must preserve SoA alignment");

    paths.Release(child);
    Require(paths.size() == 1, "completed child must be reclaimed");
    Throws<std::out_of_range>([&] { (void)paths.History(child); }, "released path access must be checked");
    ThrowingValue::movesBeforeThrow = 1;
    Throws([&] { (void)paths.AddChild(root, 5, ThrowingValue{12}); },
           "recycled history construction failure");
    ThrowingValue::movesBeforeThrow = -1;
    const auto retriedChild = paths.AddChild(root, 6, ThrowingValue{13});
    Require(retriedChild == child && paths.Parent(retriedChild) == root &&
            paths.History(retriedChild).value == 13 && paths.ActionFromParent(retriedChild) == 6,
            "failed recycled path must preserve free slot and aligned fields");
    paths.Release(root);
    Require(paths.History(root).value == 9, "child must retain released parent's ancestry");
    paths.Release(retriedChild);
    Require(paths.size() == 0, "last child must reclaim ancestors without phantom references");

    NodePool<ThrowingValue, int> nodes(memory.resource(), 1);
    ThrowingValue::movesBeforeThrow = 0;
    Throws([&] { (void)nodes.Allocate(ThrowingValue{1}, root, 0); }, "state construction failure");
    ThrowingValue::movesBeforeThrow = -1;
    const auto first = nodes.Allocate(ThrowingValue{2}, root, 0);
    Require(first.value == 0 && nodes.StateAt(first).value == 2, "failed fresh allocation must not consume handle");
    nodes.Release(first);
    Throws<std::out_of_range>([&] { (void)nodes.StateAt(first); }, "released state access must be checked");
    Throws<std::out_of_range>([&] { (void)nodes.CostAt(first); }, "released cost access must be checked");
    ThrowingValue::movesBeforeThrow = 0;
    Throws([&] { (void)nodes.Allocate(ThrowingValue{3}, root, 0); }, "recycled construction failure");
    ThrowingValue::movesBeforeThrow = -1;
    const auto recycled = nodes.Allocate(ThrowingValue{4}, child, 1);
    Require(recycled == first && nodes.StateAt(recycled).value == 4, "failed recycled allocation must retain free slot");
    nodes.Release(recycled);
    nodes.Release(recycled);
    const auto reused = nodes.Allocate(ThrowingValue{5}, child, 1);
    const auto distinct = nodes.Allocate(ThrowingValue{6}, child, 1);
    Require(reused != distinct, "double release must not duplicate free slot");

    NodePool<int, ThrowingValue> expensiveCosts(memory.resource(), 1);
    const std::optional<ThrowingValue> cost{ThrowingValue{5}};
    ThrowingValue::movesBeforeThrow = 0;
    Throws([&] { (void)expensiveCosts.Allocate(8, root, cost); }, "cost move can fail after state creation");
    ThrowingValue::movesBeforeThrow = -1;
    const auto costNode = expensiveCosts.Allocate(9, root, cost);
    Require(costNode.value == 0 && expensiveCosts.StateAt(costNode) == 9 &&
            expensiveCosts.CostAt(costNode)->value == 5, "cost failure must roll back state and handle");

    FailingResource resource;
    NodePool<int,int> allocationFailures(&resource, 8);
    resource.fail = true;
    Throws<std::bad_alloc>([&] { allocationFailures.Reserve(std::numeric_limits<std::size_t>::max()); },
                           "reserve ceiling arithmetic must not wrap into zero chunks");
    Throws<std::bad_alloc>([&] { (void)allocationFailures.Allocate(2, root, 0); },
                           "injected chunk allocation failure");
    resource.fail = false;
    const auto live = allocationFailures.Allocate(3, root, 0);
    Require(live.value == 0, "failed chunk allocation must not consume node handle");
    resource.fail = true;
    Throws<std::bad_alloc>([&] { allocationFailures.Release(live); }, "injected free-list allocation failure");
    Require(allocationFailures.StateAt(live) == 3, "failed release must preserve live state");
    resource.fail = false;
    allocationFailures.Release(live);
    Require(allocationFailures.Allocate(4, root, 0) == live, "release retry must recycle preserved slot");
}
void RecycledPathStorage() {
    FailingResource resource;
    PathArena<int, int> paths(&resource, 8);
    const auto root = paths.AddRoot(0);
    const auto left = paths.AddChild(root, 1, 1);
    const auto right = paths.AddChild(root, 2, 2);
    const auto leaf = paths.AddChild(left, 3, 3);
    paths.Release(root);
    paths.Release(left);
    paths.Release(right);
    Require(paths.size() == 3 && paths.Parent(leaf) == left && paths.Parent(left) == root,
            "release must preserve only the ancestry of remaining descendants");
    Require(detail::BuildPlan<int, int>(paths, leaf, 3).actions == std::vector<int>({1, 3}),
            "retained ancestors must reconstruct the original action order");
    resource.fail = true;
    paths.Release(leaf);
    Require(paths.size() == 0, "cascading path release must not allocate");
    for (int branch = 0; branch < 10000; ++branch) {
        const auto nextRoot = paths.AddRoot(branch);
        const auto nextChild = paths.AddChild(nextRoot, branch + 1, branch + 1);
        paths.Release(nextRoot);
        Require(paths.History(nextChild) == branch + 1 && paths.Depth(nextChild) == 1,
                "reused slot must not expose the previous path");
        paths.Release(nextChild);
    }
    Require(paths.size() == 0 && paths.capacity() == 8,
            "completed branches must reuse a fixed number of path slots");
}

struct CountedHistory {
    std::uint64_t value;
    static inline std::size_t live = 0;
    static inline std::size_t peak = 0;
    explicit CountedHistory(std::uint64_t v) : value(v) {
        peak = std::max(peak, ++live);
    }
    CountedHistory(const CountedHistory& other) : CountedHistory(other.value) {}
    CountedHistory(CountedHistory&& other) noexcept : CountedHistory(other.value) {}
    ~CountedHistory() { --live; }
    friend bool operator==(const CountedHistory&, const CountedHistory&) = default;
};

class ExhaustiveTree final : public PlanningProblem<int, std::uint64_t, float, CountedHistory> {
public:
    bool comb = false;
    mutable std::size_t visited = 0;
    void ForEachInitialState(const Consumer<std::uint64_t>& consume) const override {
        consume(1);
    }
    bool IsGoal(const std::uint64_t&) const override {
        ++visited;
        return false;
    }
    void ForEachApplicableAction(const std::uint64_t& state,
                                 const Consumer<int>& consume) const override {
        if (comb && state % 17 != 1) return;
        const int branching = comb ? 17 : 2;
        for (int action = 0; action < branching; ++action) {
            if (!consume(action)) break;
        }
    }
    std::optional<std::uint64_t> Apply(const std::uint64_t& state,
                                      const int& action) const override {
        return comb ? state + 17 + action : 2 * state + action;
    }
    CountedHistory MakeHistoryKey(const std::uint64_t& state) const override {
        return CountedHistory{state};
    }
    bool HasStepCost() const override { return true; }
    std::optional<float> StepCost(const std::uint64_t&, const int&,
                                  const std::uint64_t&) const override { return 1; }
};

void SearchReclaimsCompletedBranches() {
    ExhaustiveTree tree;
    DFSOptions<float> dfs;
    dfs.maxDepth = 12;
    dfs.initialArenaBytes = 1;
    dfs.arenaChunkCapacity = 2;
    dfs.stackReserveHint = 0;
    auto check = [&](auto planner, std::size_t expectedVisits, std::size_t maxHistory,
                     const std::string& name) {
        tree.visited = 0;
        CountedHistory::peak = 0;
        Require(planner.Search(tree).empty(), name + " must exhaust the tree");
        Require(tree.visited == expectedVisits, name + " must preserve exhaustive traversal");
        Require(CountedHistory::peak <= maxHistory, name + " retained completed branch history");
        Require(CountedHistory::live == 0, name + " must destroy all search history");
    };
    check(DFSPlanner<int, std::uint64_t, float, CountedHistory>(dfs), 8191, 64, "DFS");
    check(RecursiveDFSPlanner<int, std::uint64_t, float, CountedHistory>(dfs), 8191, 64,
          "recursive DFS");
    BBDFSOptions<float> bb;
    bb.maxDepth = dfs.maxDepth;
    bb.initialArenaBytes = 1;
    bb.arenaChunkCapacity = 2;
    bb.stackReserveHint = 0;
    check(BranchAndBoundDFSPlanner<int, std::uint64_t, float, CountedHistory>(bb), 8191, 64,
          "BBDFS");
    tree.comb = true;
    BFSOptions bfs;
    bfs.maxDepth = 40;
    bfs.initialArenaBytes = 1;
    bfs.arenaChunkCapacity = 2;
    check(BFSPlanner<int, std::uint64_t, float, CountedHistory>(bfs), 681, 80, "BFS");
}

void InvalidAndCancelledSearch() {
    test::GraphProblem graph;
    graph.edges = {{0, 4, 1}};
    BBDFSOptions<float> bb;
    bb.initialCostBound = std::numeric_limits<float>::quiet_NaN();
    Throws<std::invalid_argument>([&] { (void)BranchAndBoundDFSPlanner<int,int>(bb).Search(graph); },
                                 "NaN cost bound must fail explicitly");
    bb.initialCostBound = std::numeric_limits<float>::infinity();
    Require(BranchAndBoundDFSPlanner<int,int>(bb).Search(graph).size() == 1, "infinite bound means unbounded");
    auto control = std::make_shared<SearchControl<float>>();
    control->stopRequested = true;
    struct UntouchableProblem : test::GraphProblem {
        void ForEachInitialState(const Consumer<int>&) const override {
            throw std::runtime_error("cancelled search accessed problem");
        }
    } untouchable;
    BFSPlanner<int,int> bfs;
    bfs.SetControl(control);
    Require(bfs.Search(untouchable).empty(), "pre-cancelled search must not enumerate roots");
    Require(detail::CheckedAdd(std::numeric_limits<int>::max(), -1) == std::numeric_limits<int>::max() - 1,
            "checked integer addition boundary");
    Throws<std::overflow_error>([] { (void)detail::CheckedAdd(std::numeric_limits<int>::max(), 1); }, "signed overflow");
    Throws<std::overflow_error>([] { (void)detail::CheckedAdd(std::numeric_limits<int>::min(), -1); }, "signed underflow");
    Throws<std::overflow_error>([] { (void)detail::CheckedAdd(std::numeric_limits<unsigned>::max(), 1U); }, "unsigned overflow");
}
}
int main() {
    try {
        RandomGraphs();
        StorageFailures();
        RecycledPathStorage();
        SearchReclaimsCompletedBranches();
        InvalidAndCancelledSearch();
        std::cout << "Search regression checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
