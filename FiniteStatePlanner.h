#pragma once

#include "Planner.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <unordered_map>

namespace planning::finite {

// These algorithms explicitly materialize the reachable transition graph.
// Equality uses the domain's exact history keys; no hash collision can merge
// distinct states. Goal states are absorbing, as in the other planners.
struct Options {
    std::size_t maxDepth = 100;
    std::size_t maxStates = 10000;
    std::size_t maxEdges = 1000000;
    std::size_t maxTableEntries = 10000000;
};

template<typename P>
class ReachableGraph {
public:
    using Action = typename P::ActionType;
    using State = typename P::StateType;
    using Cost = typename P::CostType;
    using Key = typename P::HistoryKeyType;
    using PlanType = Plan<Action, Cost>;
    struct Edge { std::size_t from; std::size_t to; Action action; Cost cost; };

    explicit ReachableGraph(const P& problem, Options options = {})
        : options(options), hasCost(problem.HasStepCost() || problem.HasTerminalCost()) {
        if (!options.maxStates || !options.maxEdges || !options.maxTableEntries) {
            throw std::invalid_argument("finite graph memory limits must be positive");
        }
        std::unordered_map<std::size_t, std::vector<std::size_t>> indicesByHash;
        const auto intern = [&](const State& state, std::size_t depth) {
            const auto key = problem.MakeHistoryKey(state);
            auto& bucket = indicesByHash[problem.HistoryKeyHash(key)];
            // Hashes narrow the candidates; exact equality decides identity.
            // Appending new states preserves the original BFS discovery order.
            for (const auto i : bucket) {
                if (problem.HistoryKeysEqual(keys[i], key)) return i;
            }
            if (states.size() == options.maxStates) {
                throw std::length_error("reachable graph exceeds maxStates");
            }
            const auto index = states.size();
            states.push_back(state);
            keys.push_back(key);
            depths.push_back(depth);
            goals.push_back(problem.IsGoal(state));
            terminal.push_back(goals.back() && hasCost
                ? problem.TerminalCost(state).value_or(Cost{}) : Cost{});
            CheckCost(terminal.back());
            outgoing.emplace_back();
            incoming.emplace_back();
            bucket.push_back(index);
            return index;
        };
        problem.ForEachInitialState([&](const State& state) {
            const auto root = intern(state, 0);
            if (std::find(roots.begin(), roots.end(), root) == roots.end()) {
                roots.push_back(root);
            }
            return true;
        });
        // Appending successors gives a multi-source BFS discovery order. Copy
        // the expanded state because interning successors can reallocate states.
        for (std::size_t from = 0; from < states.size(); ++from) {
            if (goals[from] || depths[from] == options.maxDepth) {
                continue;
            }
            const auto state = states[from];
            problem.ForEachSuccessor(state, [&](const Action& action, State&& next) {
                if (edges.size() == options.maxEdges) {
                    throw std::length_error("reachable graph exceeds maxEdges");
                }
                const auto cost = hasCost
                    ? problem.StepCost(state, action, next).value_or(Cost{}) : UnitCost();
                CheckCost(cost);
                const auto to = intern(next, depths[from] + 1);
                const auto edge = edges.size();
                edges.push_back({from, to, action, cost});
                outgoing[from].push_back(edge);
                incoming[to].push_back(edge);
                return true;
            });
        }
    }

    [[nodiscard]] PlanType MakePlan(const std::vector<std::size_t>& path,
                                    std::size_t goal) const {
        PlanType result;
        result.status = PlanStatus::success;
        result.stageLen = path.size();
        Cost cost{};
        for (const auto edge : path) {
            result.actions.push_back(edges[edge].action);
            cost = Add(cost, edges[edge].cost);
        }
        if (hasCost) {
            result.totalCost = Add(cost, terminal[goal]);
        }
        return result;
    }

    [[nodiscard]] std::size_t TableRows() const {
        if (options.maxDepth == std::numeric_limits<std::size_t>::max() ||
            options.maxDepth + 1 > options.maxTableEntries ||
            (!states.empty() && options.maxDepth + 1 > options.maxTableEntries / states.size())) {
            throw std::length_error("value iteration exceeds maxTableEntries");
        }
        return options.maxDepth + 1;
    }
    static Cost Add(const Cost& a, const Cost& b) {
        const auto sum = planning::detail::CheckedAdd(a, b);
        CheckCost(sum);
        return sum;
    }
    static Cost UnitCost() {
        if constexpr (requires { Cost{1}; }) {
            return Cost{1};
        } else {
            throw std::invalid_argument("costless value iteration needs Cost constructible from one");
        }
    }
    static void CheckCost(const Cost& cost) {
        if constexpr (std::is_floating_point_v<Cost>) {
            if (!std::isfinite(cost)) {
                throw std::domain_error("finite-state search requires finite costs");
            }
        }
    }

    Options options;
    bool hasCost;
    std::vector<State> states;
    std::vector<Key> keys;
    std::vector<std::size_t> depths;
    std::vector<bool> goals;
    std::vector<Cost> terminal;
    std::vector<std::size_t> roots;
    std::vector<Edge> edges;
    std::vector<std::vector<std::size_t>> outgoing;
    std::vector<std::vector<std::size_t>> incoming;
};

// Multi-source backward BFS uses predecessor edges, then picks the shallowest
// initial state. The returned plan is minimum length, irrespective of cost.
template<typename P>
auto BackwardBreadthFirst(const ReachableGraph<P>& graph)
    -> std::optional<typename ReachableGraph<P>::PlanType> {
    constexpr auto unseen = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> distance(graph.states.size(), unseen);
    std::vector<std::size_t> next(graph.states.size(), unseen), queue;
    for (std::size_t s = 0; s < graph.states.size(); ++s) {
        if (graph.goals[s]) { distance[s] = 0; queue.push_back(s); }
    }
    for (std::size_t i = 0; i < queue.size(); ++i) {
        const auto state = queue[i];
        if (distance[state] == graph.options.maxDepth) { continue; }
        for (const auto edge : graph.incoming[state]) {
            const auto from = graph.edges[edge].from;
            if (distance[from] != unseen) { continue; }
            distance[from] = distance[state] + 1;
            next[from] = edge;
            queue.push_back(from);
        }
    }
    std::size_t root = unseen;
    for (const auto candidate : graph.roots) {
        if (distance[candidate] != unseen &&
            (root == unseen || distance[candidate] < distance[root])) { root = candidate; }
    }
    if (root == unseen) { return std::nullopt; }
    std::vector<std::size_t> path;
    auto state = root;
    while (!graph.goals[state]) {
        path.push_back(next[state]);
        state = graph.edges[next[state]].to;
    }
    return graph.MakePlan(path, state);
}

// Alternate complete BFS layers in both directions. Stop only once the sum of
// explored layer depths proves that no shorter meeting path remains.
template<typename P>
auto BidirectionalBreadthFirst(const ReachableGraph<P>& graph)
    -> std::optional<typename ReachableGraph<P>::PlanType> {
    constexpr auto unseen = std::numeric_limits<std::size_t>::max();
    const auto n = graph.states.size();
    std::vector<std::size_t> forward(n, unseen), backward(n, unseen);
    std::vector<std::size_t> previous(n, unseen), next(n, unseen), front, back;
    for (const auto root : graph.roots) { forward[root] = 0; front.push_back(root); }
    for (std::size_t s = 0; s < n; ++s) {
        if (graph.goals[s]) { backward[s] = 0; back.push_back(s); }
    }
    std::size_t meeting = unseen, best = unseen, forwardDepth = 0, backwardDepth = 0;
    const auto consider = [&](std::size_t state) {
        if (forward[state] == unseen || backward[state] == unseen) { return; }
        const auto length = forward[state] + backward[state];
        if (length <= graph.options.maxDepth && length < best) {
            best = length; meeting = state;
        }
    };
    for (const auto root : graph.roots) { consider(root); }
    while (!front.empty() && !back.empty() &&
           (best == unseen || forwardDepth + backwardDepth < best)) {
        const bool expandForward = forwardDepth <= backwardDepth;
        auto& frontier = expandForward ? front : back;
        auto& distances = expandForward ? forward : backward;
        std::vector<std::size_t> fresh;
        for (const auto state : frontier) {
            if (distances[state] == graph.options.maxDepth) { continue; }
            const auto& adjacent = expandForward ? graph.outgoing[state] : graph.incoming[state];
            for (const auto edge : adjacent) {
                const auto destination = expandForward ? graph.edges[edge].to : graph.edges[edge].from;
                if (distances[destination] == unseen) {
                    distances[destination] = distances[state] + 1;
                    (expandForward ? previous : next)[destination] = edge;
                    fresh.push_back(destination);
                }
                consider(destination);
            }
        }
        frontier = std::move(fresh);
        ++(expandForward ? forwardDepth : backwardDepth);
    }
    if (meeting == unseen) { return std::nullopt; }
    std::vector<std::size_t> path;
    auto state = meeting;
    while (previous[state] != unseen) {
        path.push_back(previous[state]); state = graph.edges[previous[state]].from;
    }
    std::reverse(path.begin(), path.end());
    state = meeting;
    while (next[state] != unseen) {
        path.push_back(next[state]); state = graph.edges[next[state]].to;
    }
    return graph.MakePlan(path, state);
}

// Finite-horizon Bellman recurrences allow negative edges and cycles. They
// optimize over plans of at most maxDepth actions, with first-goal termination.
template<typename P>
auto ForwardValueIteration(const ReachableGraph<P>& graph)
    -> std::optional<typename ReachableGraph<P>::PlanType> {
    using Graph = ReachableGraph<P>;
    using Cost = typename Graph::Cost;
    constexpr auto unseen = std::numeric_limits<std::size_t>::max();
    const auto rows = graph.TableRows(), n = graph.states.size();
    std::vector<std::vector<std::size_t>> previous(rows, std::vector<std::size_t>(n, unseen));
    std::vector<std::optional<Cost>> current(n), fresh(n);
    for (const auto root : graph.roots) { current[root] = Cost{}; }
    std::optional<Cost> best;
    std::size_t bestGoal = unseen, bestDepth = 0;
    for (std::size_t depth = 0; depth < rows; ++depth) {
        for (std::size_t state = 0; state < n; ++state) {
            if (current[state] && graph.goals[state]) {
                const auto total = Graph::Add(*current[state], graph.terminal[state]);
                if (!best || total < *best) { best = total; bestGoal = state; bestDepth = depth; }
            }
        }
        if (depth + 1 == rows) { break; }
        std::fill(fresh.begin(), fresh.end(), std::nullopt);
        for (std::size_t i = 0; i < graph.edges.size(); ++i) {
            const auto& edge = graph.edges[i];
            if (!current[edge.from]) { continue; }
            const auto candidate = Graph::Add(*current[edge.from], edge.cost);
            if (!fresh[edge.to] || candidate < *fresh[edge.to]) {
                fresh[edge.to] = candidate; previous[depth + 1][edge.to] = i;
            }
        }
        current.swap(fresh);
    }
    if (!best) { return std::nullopt; }
    std::vector<std::size_t> path;
    auto state = bestGoal;
    for (auto depth = bestDepth; depth > 0; --depth) {
        const auto edge = previous[depth][state]; path.push_back(edge); state = graph.edges[edge].from;
    }
    std::reverse(path.begin(), path.end());
    return graph.MakePlan(path, bestGoal);
}

template<typename P>
auto BackwardValueIteration(const ReachableGraph<P>& graph)
    -> std::optional<typename ReachableGraph<P>::PlanType> {
    using Graph = ReachableGraph<P>;
    using Cost = typename Graph::Cost;
    constexpr auto unseen = std::numeric_limits<std::size_t>::max();
    const auto rows = graph.TableRows(), n = graph.states.size();
    std::vector<std::vector<std::size_t>> choice(rows, std::vector<std::size_t>(n, unseen));
    std::vector<std::optional<Cost>> value(n), fresh(n);
    for (std::size_t s = 0; s < n; ++s) { if (graph.goals[s]) { value[s] = graph.terminal[s]; } }
    for (std::size_t remaining = 1; remaining < rows; ++remaining) {
        std::fill(fresh.begin(), fresh.end(), std::nullopt);
        for (std::size_t s = 0; s < n; ++s) { if (graph.goals[s]) { fresh[s] = graph.terminal[s]; } }
        for (std::size_t i = 0; i < graph.edges.size(); ++i) {
            const auto& edge = graph.edges[i];
            if (!value[edge.to]) { continue; }
            const auto candidate = Graph::Add(edge.cost, *value[edge.to]);
            if (!fresh[edge.from] || candidate < *fresh[edge.from]) {
                fresh[edge.from] = candidate; choice[remaining][edge.from] = i;
            }
        }
        value.swap(fresh);
    }
    std::size_t root = unseen;
    for (const auto candidate : graph.roots) {
        if (value[candidate] && (root == unseen || *value[candidate] < *value[root])) { root = candidate; }
    }
    if (root == unseen) { return std::nullopt; }
    std::vector<std::size_t> path;
    auto state = root;
    for (auto remaining = graph.options.maxDepth; !graph.goals[state]; --remaining) {
        const auto edge = choice[remaining][state]; path.push_back(edge); state = graph.edges[edge].to;
    }
    return graph.MakePlan(path, state);
}

// Convenience overloads; callers comparing methods can reuse ReachableGraph.
template<typename P> auto BackwardBreadthFirst(const P& p, Options options = {}) {
    return BackwardBreadthFirst(ReachableGraph<P>(p, options));
}
template<typename P> auto BidirectionalBreadthFirst(const P& p, Options options = {}) {
    return BidirectionalBreadthFirst(ReachableGraph<P>(p, options));
}
template<typename P> auto ForwardValueIteration(const P& p, Options options = {}) {
    return ForwardValueIteration(ReachableGraph<P>(p, options));
}
template<typename P> auto BackwardValueIteration(const P& p, Options options = {}) {
    return BackwardValueIteration(ReachableGraph<P>(p, options));
}

} // namespace planning::finite
