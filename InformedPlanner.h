#pragma once

#include "Planner.h"

#include <functional>
#include <queue>
#include <unordered_map>

namespace planning {

enum class InformedSearch { uniformCost, aStar, greedy, weightedAStar, beam };

struct InformedOptions {
    InformedSearch mode = InformedSearch::aStar;
    std::uint64_t maxDepth = 100;
    std::size_t maxPlans = 1;
    bool prunePathCycles = true;
    double weight = 1.0;
    std::size_t beamWidth = 32;
    // Identifies the fixed heuristic used for this search.
    std::uint64_t heuristicId = 0;
};

// Priority search over exact state labels. A depth bound makes (g, depth) a
// Pareto label: a cheaper but deeper path cannot replace a shallower path that
// has more remaining steps. Labels are reopened when a better path is found.
// Heuristics for optimal A* must lower-bound remaining STEP cost, be nonnegative
// and zero at goals. Step and terminal costs must be nonnegative and finite.
// Costs absent from the model use unit steps for ordering but stay absent in
// the returned Plan. A terminal cost is treated as an edge to a virtual goal.
// The heuristic must be a pure, fixed function of state throughout Search.
// Priorities are evaluated once per accepted label and cached; mutating a
// heuristic during search violates this contract (learn first, then search).
template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class InformedPlanner : public Planner<Action, State, Cost, HistoryKey> {
public:
    using Base = Planner<Action, State, Cost, HistoryKey>;
    using typename Base::Problem;
    using typename Base::PlanConsumer;
    using typename Base::PlanType;
    using Base::Search;
    using Heuristic = std::function<Cost(const State&)>;

    explicit InformedPlanner(InformedOptions options = {}, Heuristic heuristic = {})
        : options_(options), heuristic_(std::move(heuristic)) {
        if (!std::isfinite(options.weight) || options.weight < 1.0) {
            throw std::invalid_argument("A* weight must be finite and at least one");
        }
        if (options.beamWidth == 0) throw std::invalid_argument("beam width must be positive");
    }

    const InformedOptions& Options() const noexcept { return options_; }
    InformedSearch Algorithm() const noexcept { return options_.mode; }
    SearchResultPolicy ResultPolicy() const override {
        const bool optimal = options_.mode == InformedSearch::uniformCost ||
                             options_.mode == InformedSearch::aStar;
        return {optimal ? ResultSelection::minimumCost : ResultSelection::first,
                false, options_.maxPlans == 0 ? 0U : 1U};
    }

    void Search(const Problem& problem, const PlanConsumer& consume,
                SearchPartition partition = {}) override {
        if (partition.workerCount == 0 || partition.workerIndex >= partition.workerCount)
            throw std::invalid_argument("invalid search partition");
        if (options_.maxPlans == 0 || (this->control_ && this->control_->stopRequested.load())) return;
        const bool hasCost = problem.HasStepCost() || problem.HasTerminalCost();
        struct Node {
            State state;
            HistoryKey key;
            Cost g;
            std::uint64_t depth;
            std::size_t parent;
            std::size_t initialRoot;
            std::optional<Action> action;
            long double priority;
            bool active = true;
        };
        constexpr auto none = std::numeric_limits<std::size_t>::max();
        struct Entry { long double priority; std::size_t node; bool goal; Cost total; };
        struct Later {
            bool operator()(const Entry& a, const Entry& b) const {
                if (a.priority != b.priority) return a.priority > b.priority;
                if (a.node != b.node) return a.node > b.node;
                return a.goal < b.goal;
            }
        };
        std::vector<Node> nodes;
        std::unordered_map<std::size_t, std::vector<std::size_t>> labelsByHash;
        std::priority_queue<Entry, std::vector<Entry>, Later> open;
        std::vector<std::size_t> layer;
        auto stopped = [&] { return this->control_ && this->control_->stopRequested.load(); };
        auto checked = [](Cost value) {
            if (value < Cost{}) throw std::invalid_argument("informed search needs nonnegative costs");
            if constexpr (std::is_floating_point_v<Cost>) {
                if (!std::isfinite(value)) throw std::invalid_argument("nonfinite cost or heuristic");
            }
            return value;
        };
        auto priority = [&](const State& state, Cost g) -> long double {
            const Cost h = options_.mode == InformedSearch::uniformCost || !heuristic_
                ? Cost{} : heuristic_(state);
            if constexpr (std::is_floating_point_v<Cost>) {
                if (h == std::numeric_limits<Cost>::infinity())
                    return std::numeric_limits<long double>::infinity();
            }
            checked(h);
            if (options_.mode == InformedSearch::aStar && problem.IsGoal(state) && h != Cost{})
                throw std::invalid_argument("A* heuristic must be zero at goals");
            long double result;
            if (options_.mode == InformedSearch::greedy || options_.mode == InformedSearch::beam)
                result = static_cast<long double>(h);
            else result = static_cast<long double>(g) + static_cast<long double>(h) *
                (options_.mode == InformedSearch::weightedAStar ? options_.weight : 1.0);
            if (!std::isfinite(result)) throw std::overflow_error("search priority overflow");
            return result;
        };
        auto add = [&](State state, Cost g, std::uint64_t depth, std::size_t parent,
                       std::optional<Action> action) -> std::optional<std::size_t> {
            auto key = problem.MakeHistoryKey(state);
            if (options_.prunePathCycles) {
                for (auto p = parent; p != none; p = nodes[p].parent)
                    if (problem.HistoryKeysEqual(nodes[p].key, key)) return std::nullopt;
            }
            // Root partitioning restricts outgoing root edges, so root labels
            // must not dominate labels reached inside another initial subtree.
            // With path-cycle pruning, a label from a different initial root
            // can forbid a return to that root even though another prefix can
            // legally pass through it and use an unrestricted outgoing edge.
            // Keep separate dominance tables for those partitioned subtrees.
            // Beam search intentionally discards labels and cannot use this
            // global dominance table to suppress later rediscovery.
            const auto initialRoot = parent == none ? nodes.size() : nodes[parent].initialRoot;
            const auto comparable = [&](const Node& previous) {
                return partition.workerCount == 1 || previous.initialRoot == initialRoot;
            };
            const auto hash = problem.HistoryKeyHash(key);
            auto& labels = labelsByHash[hash];
            if (options_.mode != InformedSearch::beam && depth != 0) {
                for (const auto previousId : labels) {
                    const auto& previous = nodes[previousId];
                    if (previous.active && previous.depth != 0 &&
                        comparable(previous) &&
                        problem.HistoryKeysEqual(previous.key, key) &&
                        previous.depth <= depth && previous.g <= g) return std::nullopt;
                }
            }
            // Evaluate only labels surviving cycle/dominance rejection. Do not
            // invalidate old labels until the candidate priority is validated.
            const auto cachedPriority = priority(state, g);
            if (!std::isfinite(cachedPriority)) return std::nullopt;
            if (options_.mode != InformedSearch::beam && depth != 0) {
                for (const auto previousId : labels) {
                    auto& previous = nodes[previousId];
                    if (previous.active && previous.depth != 0 &&
                        comparable(previous) &&
                        problem.HistoryKeysEqual(previous.key, key) &&
                        depth <= previous.depth && g <= previous.g) previous.active = false;
                }
            }
            const auto id = nodes.size();
            nodes.push_back({std::move(state), std::move(key), g, depth, parent,
                             initialRoot, std::move(action), cachedPriority});
            labels.push_back(id);
            return id;
        };
        problem.ForEachInitialState([&](const State& initial) {
            if (auto id = add(initial, Cost{}, 0, none, std::nullopt)) {
                open.push({nodes[*id].priority, *id, false, Cost{}});
                layer.push_back(*id);
            }
            return !stopped();
        });
        auto emit = [&](std::size_t id, Cost total) {
            PlanType plan;
            plan.status = PlanStatus::success;
            plan.stageLen = nodes[id].depth;
            if (hasCost) plan.totalCost = total;
            for (auto p = id; p != none; p = nodes[p].parent)
                if (nodes[p].action) plan.actions.push_back(*nodes[p].action);
            std::reverse(plan.actions.begin(), plan.actions.end());
            consume(plan);
        };
        auto goal = [&](std::size_t id) {
            return (nodes[id].depth != 0 || partition.workerIndex == 0) &&
                   problem.IsGoal(nodes[id].state);
        };
        auto expand = [&](std::size_t id, auto&& accept) {
            // Values are copied because adding successors can reallocate nodes.
            const State state = nodes[id].state;
            const auto depth = nodes[id].depth;
            const Cost g = nodes[id].g;
            if (depth == options_.maxDepth) return;
            std::size_t rootAction = 0;
            problem.ForEachSuccessor(state, [&](const Action& action, State&& next) {
                if (stopped()) return false;
                if (depth == 0 && !partition.OwnsRootAction(rootAction++)) return true;
                const auto step = checked(hasCost ? problem.StepCost(state, action, next).value_or(Cost{}) : Cost{1});
                const auto nextG = detail::CheckedAdd(g, step);
                if (auto child = add(std::move(next), nextG, depth + 1, id, action)) accept(*child);
                return true;
            });
        };
        if (options_.mode == InformedSearch::beam) {
            while (!layer.empty() && !stopped()) {
                std::vector<std::size_t> next;
                for (auto id : layer) {
                    if (goal(id)) {
                        emit(id, detail::CheckedAdd(nodes[id].g,
                            checked(problem.TerminalCost(nodes[id].state).value_or(Cost{}))));
                        return;
                    }
                    expand(id, [&](std::size_t child) { next.push_back(child); });
                }
                std::stable_sort(next.begin(), next.end(), [&](auto a, auto b) {
                    return nodes[a].priority < nodes[b].priority;
                });
                if (next.size() > options_.beamWidth) next.resize(options_.beamWidth);
                layer = std::move(next);
            }
            return;
        }
        while (!open.empty() && !stopped()) {
            const auto entry = open.top();
            open.pop();
            const auto id = entry.node;
            if (!nodes[id].active) continue;
            if (entry.goal) { emit(id, entry.total); return; }
            if (goal(id)) {
                const auto total = detail::CheckedAdd(nodes[id].g,
                    checked(problem.TerminalCost(nodes[id].state).value_or(Cost{})));
                if (options_.mode == InformedSearch::greedy) { emit(id, total); return; }
                open.push({static_cast<long double>(total), id, true, total});
                // Search can terminate at this goal or continue to a different
                // goal with a smaller terminal charge.
            }
            expand(id, [&](std::size_t child) {
                open.push({nodes[child].priority, child, false, Cost{}});
            });
        }
    }

private:
    InformedOptions options_;
    Heuristic heuristic_;
};

template<typename A, typename S, typename C = float, typename K = S>
class DijkstraPlanner final : public InformedPlanner<A,S,C,K> {
public:
    explicit DijkstraPlanner(InformedOptions options = {})
        : InformedPlanner<A,S,C,K>(Configure(options)) {}
private:
    static InformedOptions Configure(InformedOptions o) { o.mode = InformedSearch::uniformCost; return o; }
};

template<typename A, typename S, typename C = float, typename K = S>
using AStarPlanner = InformedPlanner<A,S,C,K>;

} // namespace planning
