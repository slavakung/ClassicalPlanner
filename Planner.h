#pragma once

#include "Plan.h"
#include "PlanningProblem.h"
#include "SearchStorage.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <unordered_set>
#include <vector>

namespace planning {

// A partition owns root successors with index % workerCount == workerIndex.
// Non-root successors stay in that subtree. All initial roots are considered;
// zero-action solutions belong ONLY to partition zero.
struct SearchPartition {
    std::size_t workerIndex = 0;
    std::size_t workerCount = 1;

    [[nodiscard]] bool OwnsRootAction(std::size_t index) const noexcept {
        return workerCount != 0 && index % workerCount == workerIndex;
    }
};

// Cancellation and incumbent state for serial search. Only BBDFS optimal
// modes use the incumbent bound. Search and bound access run on one thread.
template<typename Cost>
class SearchControl {
public:
    std::atomic<bool> stopRequested{false};

    void ImproveBound(const Cost& candidate) {
        if (!incumbent_ || candidate < *incumbent_) {
            incumbent_ = candidate;
        }
    }

    [[nodiscard]] std::optional<Cost> Bound() const {
        return incumbent_;
    }

private:
    std::optional<Cost> incumbent_;
};

enum class BFSReturnMode { firstGoal, allMinimalDepth, allPlansUpToDepth };
enum class DFSReturnMode { firstGoal, allPlansUpToDepth, allPlansWithinCost };
enum class BBDFSReturnMode {
    firstGoal, firstOptimalAfterExhaustion, allOptimal, allWithinCost
};

struct BFSOptions {
    BFSReturnMode returnMode = BFSReturnMode::firstGoal;
    std::uint64_t maxDepth = 100;
    std::size_t maxPlans = 1000;
    bool prunePathCycles = true;
    std::size_t frontierReserveHint = 0;
    std::size_t arenaChunkCapacity = 4096;
    std::size_t initialArenaBytes = 1024 * 1024;
    // Keep the first FIFO representative of each exact history key. Only
    // unpartitioned firstGoal search supports this; enumeration needs all paths.
    bool pruneDuplicateStates = false;
};

template<typename Cost = float>
struct DFSOptions {
    DFSReturnMode returnMode = DFSReturnMode::firstGoal;
    std::uint64_t maxDepth = 100;
    std::size_t maxPlans = 1000;
    std::optional<Cost> targetCost = std::nullopt;
    bool prunePathCycles = true;
    std::size_t stackReserveHint = 1024;
    std::size_t arenaChunkCapacity = 4096;
    std::size_t initialArenaBytes = 1024 * 1024;
    // A contract, not a proof inferred from sampled edges. Set false to use
    // cost filtering only at goals when negative costs are possible.
    bool assumeNonnegativeCosts = true;
};

template<typename Cost = float>
struct BBDFSOptions {
    BBDFSReturnMode returnMode = BBDFSReturnMode::firstOptimalAfterExhaustion;
    std::uint64_t maxDepth = 100;
    std::size_t maxPlans = 1000;
    std::optional<Cost> initialCostBound = std::nullopt;
    bool prunePathCycles = true;
    bool requireCostModel = true;
    std::size_t stackReserveHint = 1024;
    std::size_t arenaChunkCapacity = 4096;
    std::size_t initialArenaBytes = 1024 * 1024;
    bool assumeNonnegativeCosts = true;
};

// Public metadata is used by the executor to combine partition results without
// guessing what a planner's return mode means.
enum class SearchAlgorithm { bfs, dfs, recursiveDfs, branchAndBound };
enum class ResultSelection { first, minimumDepth, minimumCost, all };
struct SearchResultPolicy {
    ResultSelection selection = ResultSelection::first;
    bool keepTies = false;
    std::size_t maxPlans = 1000;
};

template<typename Action, typename State, typename Cost = float,
         typename HistoryKey = State>
class Planner {
public:
    using Problem = PlanningProblem<Action, State, Cost, HistoryKey>;
    using PlanType = Plan<Action, Cost>;
    using PlanConsumer = Consumer<PlanType>;
    using CostType = Cost;
    virtual ~Planner() = default;

    virtual void Search(const Problem&, const PlanConsumer&,
                        SearchPartition = {}) = 0;
    virtual SearchResultPolicy ResultPolicy() const = 0;

    // Attach serial search cancellation and incumbent state before Search.
    void SetControl(std::shared_ptr<SearchControl<Cost>> control) {
        control_ = std::move(control);
    }

    [[nodiscard]] std::vector<PlanType> Search(const Problem& problem) {
        std::vector<PlanType> result;
        Search(problem, [&](const PlanType& plan) {
            result.push_back(plan);
            return true;
        });
        return result;
    }


protected:
    std::shared_ptr<SearchControl<Cost>> control_;
};

namespace detail {

template<typename Action, typename Cost, typename HistoryKey>
[[nodiscard]] Plan<Action, Cost> BuildPlan(
    const PathArena<Action, HistoryKey>& paths, PathId goal,
    std::optional<Cost> cost) {
    Plan<Action, Cost> plan;
    plan.status = PlanStatus::success;
    plan.stageLen = paths.Depth(goal);
    plan.totalCost = cost;
    plan.actions.reserve(static_cast<std::size_t>(plan.stageLen));
    for (PathId id = goal; id.valid(); id = paths.Parent(id)) {
        if (const auto& action = paths.ActionFromParent(id); action) {
            plan.actions.push_back(*action);
        }
    }
    std::reverse(plan.actions.begin(), plan.actions.end());
    return plan;
}

template<typename Problem, typename Action, typename HistoryKey>
[[nodiscard]] bool AppearsInPath(
    const Problem& problem, const PathArena<Action, HistoryKey>& paths,
    PathId path, const HistoryKey& key) {
    for (PathId id = path; id.valid(); id = paths.Parent(id)) {
        if (problem.HistoryKeysEqual(paths.History(id), key)) {
            return true;
        }
    }
    return false;
}

// Arithmetic costs must never wrap: a wrapped incumbent can wrongly prune
// every remaining solution. Custom Cost types retain their own addition rules.
template<typename Cost>
Cost CheckedAdd(const Cost& a, const Cost& b) {
    if constexpr (std::is_integral_v<Cost>) {
        if constexpr (std::is_signed_v<Cost>) {
            if ((b > 0 && a > std::numeric_limits<Cost>::max() - b) ||
                (b < 0 && a < std::numeric_limits<Cost>::min() - b)) {
                throw std::overflow_error("planning cost overflow");
            }
        } else {
            if (a > std::numeric_limits<Cost>::max() - b) {
                throw std::overflow_error("planning cost overflow");
            }
        }
    }
    const Cost total = a + b;
    if constexpr (std::is_floating_point_v<Cost>) {
        if (!std::isfinite(total)) {
            throw std::overflow_error("planning cost overflow");
        }
    }
    return total;
}

template<typename Cost>
struct CoreOptions {
    SearchAlgorithm algorithm = SearchAlgorithm::bfs;
    SearchResultPolicy policy;
    std::uint64_t maxDepth = 100;
    bool pruneCycles = true;
    std::size_t reserve = 0;
    std::size_t chunkCapacity = 4096;
    std::size_t initialBytes = 1024 * 1024;
    std::optional<Cost> fixedCostBound;
    bool requireCost = false;
    bool monotoneCost = true;
    bool onlyMinimalDepth = false;
    bool pruneDuplicateStates = false;
};

// The four traversals use the same path/cost/goal logic. This avoids four subtly
// different cost-bound implementations. State expansion is a virtual problem
// operation supplied by the native planning problem.
template<typename Action, typename State, typename Cost, typename HistoryKey>
class SearchRun {
    using Problem = PlanningProblem<Action, State, Cost, HistoryKey>;
    using PlanType = Plan<Action, Cost>;
    using Successor = std::pair<Action, State>;

public:
    SearchRun(const Problem& problem, const Consumer<PlanType>& consumer,
              SearchPartition partition, CoreOptions<Cost> options,
              std::shared_ptr<SearchControl<Cost>> control)
        : problem_(problem), consumer_(consumer), partition_(partition),
          options_(std::move(options)), control_(std::move(control)),
          memory_(options_.initialBytes),
          paths_(memory_.resource(), options_.chunkCapacity),
          nodes_(memory_.resource(), options_.chunkCapacity),
          visited_(0, HistoryHash{&problem}, HistoryEqual{&problem}),
          hasCost_(problem.HasStepCost() || problem.HasTerminalCost()),
          incumbent_(options_.fixedCostBound) {
        if (partition_.workerCount == 0 ||
            partition_.workerIndex >= partition_.workerCount) {
            throw std::invalid_argument("invalid search partition");
        }
        if constexpr (std::is_floating_point_v<Cost>) {
            if (options_.fixedCostBound && std::isnan(*options_.fixedCostBound)) {
                throw std::invalid_argument("planning cost bound must not be NaN");
            }
        }
    }

    void Run() {
        if (Stopped() || options_.policy.maxPlans == 0 ||
            (options_.requireCost && !hasCost_)) {
            return;
        }
        // Recursive DFS keeps states in its C++ call frames. Other traversals
        // retain only active full states in recyclable NodePool slots.
        if (options_.algorithm == SearchAlgorithm::recursiveDfs) {
            problem_.ForEachInitialState([&](const State& state) {
                if (Stopped()) return false;
                const PathId path = paths_.AddRoot(problem_.MakeHistoryKey(state));
                VisitRecursive(state, path, InitialCost());
                return !Stopped();
            });
        } else {
            RunIterative();
        }
        // Optimal modes emit only after exhaustion. maxPlans limits retained
        // ties, NOT proof work: maxPlans=1 must still search for cheaper plans.
        if (Optimal() && !Stopped()) {
            for (const auto& plan : optimalPlans_) {
                if (!consumer_(plan)) {
                    break;
                }
            }
        }
    }

private:
    const Problem& problem_;
    const Consumer<PlanType>& consumer_;
    SearchPartition partition_;
    CoreOptions<Cost> options_;
    std::shared_ptr<SearchControl<Cost>> control_;
    // Declaration order ensures pools die before their backing resource.
    SearchMemory memory_;
    PathArena<Action, HistoryKey> paths_;
    NodePool<State, Cost> nodes_;
    struct HistoryHash {
        const Problem* problem;
        std::size_t operator()(const HistoryKey& key) const {
            return problem->HistoryKeyHash(key);
        }
    };
    struct HistoryEqual {
        const Problem* problem;
        bool operator()(const HistoryKey& left, const HistoryKey& right) const {
            return problem->HistoryKeysEqual(left, right);
        }
    };
    // Visited keys own their values: recycled PathIds cannot identify them.
    std::unordered_set<HistoryKey, HistoryHash, HistoryEqual> visited_;
    bool hasCost_;
    bool stopped_ = false;
    std::size_t emitted_ = 0;
    std::optional<std::uint64_t> solutionDepth_;
    std::optional<Cost> incumbent_;
    std::vector<PlanType> optimalPlans_;

    struct PathOwner {
        PathArena<Action, HistoryKey>& paths;
        PathId path;
        ~PathOwner() { paths.Release(path); }
    };

    [[nodiscard]] bool Optimal() const {
        return options_.policy.selection == ResultSelection::minimumCost;
    }
    [[nodiscard]] bool Stopped() const {
        return stopped_ || (control_ && control_->stopRequested.load());
    }
    [[nodiscard]] std::optional<Cost> InitialCost() const {
        return hasCost_ || options_.algorithm == SearchAlgorithm::branchAndBound
            ? std::optional<Cost>{Cost{}} : std::nullopt;
    }
    void CheckCost(const Cost& cost) const {
        if constexpr (std::is_floating_point_v<Cost>) {
            if (!std::isfinite(cost)) {
                throw std::domain_error("non-finite planning cost");
            }
        }
        if (options_.monotoneCost && cost < Cost{}) {
            throw std::domain_error(
                "negative cost violates assumeNonnegativeCosts; disable cost pruning");
        }
    }
    [[nodiscard]] std::optional<Cost> NextCost(
        const State& from, const Action& action, const State& to,
        std::optional<Cost> previous) const {
        if (!previous) {
            return std::nullopt;
        }
        const Cost step = problem_.StepCost(from, action, to).value_or(Cost{});
        CheckCost(step);
        const Cost total = CheckedAdd(*previous, step);
        if constexpr (std::is_floating_point_v<Cost>) {
            if (!std::isfinite(total)) {
                throw std::overflow_error("path cost overflow");
            }
        }
        return total;
    }
    [[nodiscard]] bool OverBound(std::optional<Cost> cost) const {
        if (!options_.monotoneCost || !cost) {
            return false;
        }
        if (options_.fixedCostBound && *cost > *options_.fixedCostBound) {
            return true;
        }
        if (Optimal()) {
            if (incumbent_ && *cost > *incumbent_) {
                return true;
            }
            if (control_) {
                const auto shared = control_->Bound();
                if (shared && *cost > *shared) {
                    return true;
                }
            }
        }
        return false;
    }
    bool Goal(const State& state, PathId path, std::optional<Cost> cost) {
        if (!problem_.IsGoal(state)) {
            return false;
        }
        const auto depth = paths_.Depth(path);
        if (depth == 0 && partition_.workerIndex != 0) {
            return true;
        }
        if (cost) {
            const Cost terminal = problem_.TerminalCost(state).value_or(Cost{});
            CheckCost(terminal);
            *cost = CheckedAdd(*cost, terminal);
            if constexpr (std::is_floating_point_v<Cost>) {
                if (!std::isfinite(*cost)) {
                    throw std::overflow_error("terminal cost overflow");
                }
            }
        }
        if (options_.fixedCostBound && cost && *cost > *options_.fixedCostBound) {
            return true;
        }
        auto plan = BuildPlan(paths_, path, cost);
        if (Optimal()) {
            const Cost total = cost.value_or(Cost{});
            if (!incumbent_ || total < *incumbent_) {
                incumbent_ = total;
                optimalPlans_.clear();
            }
            if (incumbent_ && total == *incumbent_ &&
                optimalPlans_.size() < options_.policy.maxPlans &&
                (options_.policy.keepTies || optimalPlans_.empty())) {
                optimalPlans_.push_back(std::move(plan));
            }
            if (control_) {
                control_->ImproveBound(total);
            }
        } else {
            if (options_.onlyMinimalDepth && solutionDepth_ && depth > *solutionDepth_) {
                return true;
            }
            solutionDepth_ = depth;
            ++emitted_;
            if (!consumer_(plan) || emitted_ >= options_.policy.maxPlans ||
                options_.policy.selection == ResultSelection::first ||
                (options_.policy.selection == ResultSelection::minimumDepth &&
                 !options_.policy.keepTies)) {
                stopped_ = true;
            }
        }
        return true; // Goal paths are terminal; do not extend after success.
    }
    [[nodiscard]] bool CanExpand(PathId path, std::optional<Cost> cost) const {
        const auto depth = paths_.Depth(path);
        return !Stopped() && depth < options_.maxDepth && !OverBound(cost) &&
               !(options_.onlyMinimalDepth && solutionDepth_ && depth >= *solutionDepth_);
    }
    // Do not allocate short-lived successor vectors in the monotonic resource:
    // deallocation there is a no-op and would retain scratch for every expansion.
    std::vector<Successor> Expand(const State& state, PathId path) {
        std::vector<Successor> successors;
        std::size_t actionIndex = 0;
        problem_.ForEachSuccessor(state, [&](const Action& action, State&& next) {
            if (Stopped()) {
                return false;
            }
            const bool owned = paths_.Depth(path) != 0 ||
                partition_.OwnsRootAction(actionIndex);
            ++actionIndex;
            if (!owned) {
                return true;
            }
            if (options_.pruneCycles || options_.pruneDuplicateStates) {
                auto key = problem_.MakeHistoryKey(next);
                if (options_.pruneCycles && AppearsInPath(problem_, paths_, path, key)) {
                    return true;
                }
                if (options_.pruneDuplicateStates && !visited_.insert(std::move(key)).second) {
                    return true;
                }
            }
            successors.emplace_back(action, std::move(next));
            return true;
        });
        return successors;
    }
    void RunIterative() {
        std::vector<NodeId> frontier;
        std::vector<NodeId> nextFrontier;
        frontier.reserve(options_.reserve);
        nextFrontier.reserve(options_.reserve);
        if (options_.reserve != 0) {
            nodes_.Reserve(options_.reserve);
        }
        problem_.ForEachInitialState([&](const State& state) {
            if (Stopped()) return false;
            auto key = problem_.MakeHistoryKey(state);
            if (options_.pruneDuplicateStates && !visited_.insert(key).second) {
                return true;
            }
            const PathId root = paths_.AddRoot(std::move(key));
            frontier.push_back(nodes_.Allocate(state, root, InitialCost()));
            return !Stopped();
        });
        const bool bfs = options_.algorithm == SearchAlgorithm::bfs;
        if (!bfs) {
            std::reverse(frontier.begin(), frontier.end());
        }
        while (!frontier.empty() && !Stopped()) {
            if (bfs) {
                nextFrontier.clear();
                for (const auto id : frontier) {
                    ExpandNode(id, nextFrontier, false);
                    if (Stopped()) {
                        break;
                    }
                }
                frontier.swap(nextFrontier);
            } else {
                const NodeId id = frontier.back();
                frontier.pop_back();
                ExpandNode(id, frontier, true);
            }
        }
        // Remaining occupied optionals are destroyed by NodePool. Never release
        // stale IDs from an old frontier: those slots might already be reused.
    }
    void ExpandNode(NodeId id, std::vector<NodeId>& destination, bool reverse) {
        const PathId path = nodes_.PathAt(id);
        const PathOwner owner{paths_, path};
        const auto cost = nodes_.CostAt(id);
        // Move the full state out before recycling its slot. Children may reuse
        // that slot immediately without invalidating this local state.
        State state = std::move(nodes_.StateAt(id));
        nodes_.Release(id);
        if (Goal(state, path, cost) || !CanExpand(path, cost)) {
            return;
        }
        auto children = Expand(state, path);
        if (reverse) {
            std::reverse(children.begin(), children.end());
        }
        for (auto& [action, next] : children) {
            const auto nextCost = NextCost(state, action, next, cost);
            if (OverBound(nextCost)) {
                continue;
            }
            const PathId child = paths_.AddChild(
                path, action, problem_.MakeHistoryKey(next));
            destination.push_back(nodes_.Allocate(std::move(next), child, nextCost));
        }
    }
    void VisitRecursive(const State& state, PathId path, std::optional<Cost> cost) {
        const PathOwner owner{paths_, path};
        if (Stopped() || Goal(state, path, cost) || !CanExpand(path, cost)) {
            return;
        }
        // The callback is consumed synchronously. Recursive descendants cannot
        // invalidate references into device output: PackedProblem materializes
        // each returned state as an owning value before invoking this callback.
        std::size_t actionIndex = 0;
        problem_.ForEachSuccessor(state, [&](const Action& action, State&& next) {
            const bool owned = paths_.Depth(path) != 0 ||
                partition_.OwnsRootAction(actionIndex);
            ++actionIndex;
            if (!owned) {
                return !Stopped();
            }
            const auto key = problem_.MakeHistoryKey(next);
            if (options_.pruneCycles && AppearsInPath(problem_, paths_, path, key)) {
                return !Stopped();
            }
            const auto nextCost = NextCost(state, action, next, cost);
            if (!OverBound(nextCost)) {
                const PathId child = paths_.AddChild(path, action, key);
                VisitRecursive(next, child, nextCost);
            }
            return !Stopped();
        });
    }
};
} // namespace detail

// Public classes deliberately retain their original names, options and overloads.
template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class BFSPlanner final : public Planner<Action, State, Cost, HistoryKey> {
public:
    using Base = Planner<Action, State, Cost, HistoryKey>;
    using typename Base::Problem;
    using typename Base::PlanConsumer;
    using typename Base::PlanType;
    using Base::Search;
    explicit BFSPlanner(BFSOptions options = {}) : options_(options) {}
    [[nodiscard]] const BFSOptions& Options() const noexcept {
        return options_;
    }
    [[nodiscard]] unsigned Algorithm() const noexcept {
        return 300U + static_cast<unsigned>(SearchAlgorithm::bfs);
    }
    SearchResultPolicy ResultPolicy() const override {
        return {options_.returnMode == BFSReturnMode::allPlansUpToDepth
                    ? ResultSelection::all : ResultSelection::minimumDepth,
                options_.returnMode == BFSReturnMode::allMinimalDepth, options_.maxPlans};
    }
    void Search(const Problem& p, const PlanConsumer& c, SearchPartition part = {}) override {
        if (options_.pruneDuplicateStates &&
                (options_.returnMode != BFSReturnMode::firstGoal || part.workerCount != 1)) {
            throw std::invalid_argument(
                "BFS duplicate pruning requires unpartitioned firstGoal search");
        }
        detail::CoreOptions<Cost> o;
        o.algorithm = SearchAlgorithm::bfs;
        o.pruneDuplicateStates = options_.pruneDuplicateStates;
        o.policy = ResultPolicy();
        o.maxDepth = options_.maxDepth;
        o.pruneCycles = options_.prunePathCycles;
        o.reserve = options_.frontierReserveHint != 0 ? options_.frontierReserveHint
            : p.EstimatedMaxFrontierSize(options_.maxDepth).value_or(0);
        o.chunkCapacity = options_.arenaChunkCapacity;
        o.initialBytes = options_.initialArenaBytes;
        o.monotoneCost = false; // BFS does not prune by accumulated cost.
        o.onlyMinimalDepth = options_.returnMode != BFSReturnMode::allPlansUpToDepth;
        detail::SearchRun<Action, State, Cost, HistoryKey>(p, c, part, o, this->control_).Run();
    }
private:
    BFSOptions options_;
};

template<typename Action, typename State, typename Cost, typename HistoryKey,
         bool Recursive>
class DepthFirstPlanner : public Planner<Action, State, Cost, HistoryKey> {
public:
    using Base = Planner<Action, State, Cost, HistoryKey>;
    using typename Base::Problem;
    using typename Base::PlanConsumer;
    using typename Base::PlanType;
    using Base::Search;
    explicit DepthFirstPlanner(DFSOptions<Cost> options = {}) : options_(options) {}
    [[nodiscard]] const DFSOptions<Cost>& Options() const noexcept {
        return options_;
    }
    [[nodiscard]] unsigned Algorithm() const noexcept {
        return 300U + static_cast<unsigned>(Recursive ? SearchAlgorithm::recursiveDfs
                                                      : SearchAlgorithm::dfs);
    }
    SearchResultPolicy ResultPolicy() const override {
        return {options_.returnMode == DFSReturnMode::firstGoal
                    ? ResultSelection::first : ResultSelection::all, false, options_.maxPlans};
    }
    void Search(const Problem& p, const PlanConsumer& c, SearchPartition part = {}) override {
        detail::CoreOptions<Cost> o;
        o.algorithm = Recursive ? SearchAlgorithm::recursiveDfs : SearchAlgorithm::dfs;
        o.policy = ResultPolicy();
        o.maxDepth = options_.maxDepth;
        o.pruneCycles = options_.prunePathCycles;
        o.reserve = options_.stackReserveHint;
        o.chunkCapacity = options_.arenaChunkCapacity;
        o.initialBytes = options_.initialArenaBytes;
        if (options_.returnMode == DFSReturnMode::allPlansWithinCost) {
            o.fixedCostBound = options_.targetCost;
        }
        o.monotoneCost = options_.assumeNonnegativeCosts && o.fixedCostBound.has_value();
        detail::SearchRun<Action, State, Cost, HistoryKey>(p, c, part, o, this->control_).Run();
    }
private:
    DFSOptions<Cost> options_;
};

template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class DFSPlanner final : public DepthFirstPlanner<Action, State, Cost, HistoryKey, false> {
public:
    using DepthFirstPlanner<Action, State, Cost, HistoryKey, false>::DepthFirstPlanner;
};

template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class RecursiveDFSPlanner final : public DepthFirstPlanner<Action, State, Cost, HistoryKey, true> {
public:
    using DepthFirstPlanner<Action, State, Cost, HistoryKey, true>::DepthFirstPlanner;
};

template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class BranchAndBoundDFSPlanner final : public Planner<Action, State, Cost, HistoryKey> {
public:
    using Base = Planner<Action, State, Cost, HistoryKey>;
    using typename Base::Problem;
    using typename Base::PlanConsumer;
    using typename Base::PlanType;
    using Base::Search;
    explicit BranchAndBoundDFSPlanner(BBDFSOptions<Cost> options = {}) : options_(options) {}
    [[nodiscard]] const BBDFSOptions<Cost>& Options() const noexcept {
        return options_;
    }
    [[nodiscard]] unsigned Algorithm() const noexcept {
        return 300U + static_cast<unsigned>(SearchAlgorithm::branchAndBound);
    }
    SearchResultPolicy ResultPolicy() const override {
        ResultSelection selection = ResultSelection::minimumCost;
        if (options_.returnMode == BBDFSReturnMode::firstGoal) {
            selection = ResultSelection::first;
        } else if (options_.returnMode == BBDFSReturnMode::allWithinCost) {
            selection = ResultSelection::all;
        }
        return {selection, options_.returnMode == BBDFSReturnMode::allOptimal, options_.maxPlans};
    }
    void Search(const Problem& p, const PlanConsumer& c, SearchPartition part = {}) override {
        detail::CoreOptions<Cost> o;
        o.algorithm = SearchAlgorithm::branchAndBound;
        o.policy = ResultPolicy();
        o.maxDepth = options_.maxDepth;
        o.pruneCycles = options_.prunePathCycles;
        o.reserve = options_.stackReserveHint;
        o.chunkCapacity = options_.arenaChunkCapacity;
        o.initialBytes = options_.initialArenaBytes;
        o.fixedCostBound = options_.initialCostBound;
        o.requireCost = options_.requireCostModel;
        o.monotoneCost = options_.assumeNonnegativeCosts;
        detail::SearchRun<Action, State, Cost, HistoryKey>(p, c, part, o, this->control_).Run();
    }
private:
    BBDFSOptions<Cost> options_;
};

// The original unimplemented names remain deliberately commented, not removed.
// template<...> class DijkstraPlanner;
// template<...> class AStarPlanner;
// template<...> class BestFirstPlanner;
// template<...> class IterativeDeepeningPlanner;
} // namespace planning
