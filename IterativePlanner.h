#pragma once

#include "Planner.h"

#include <functional>
#include <type_traits>

namespace planning {

enum class IterativeSearch { iterativeDeepening = 100, idaStar, recursiveBestFirst };

struct IterativeOptions {
    IterativeSearch mode = IterativeSearch::iterativeDeepening;
    std::uint64_t maxDepth = 100;
    std::size_t maxPlans = 1;
    bool prunePathCycles = true;
    // Identifies the fixed heuristic used for this search.
    std::uint64_t heuristicId = 0;
};

// IDDFS repeats depth-limited DFS. IDA* repeats DFS at increasing g+h bounds.
// RBFS retains successors only along its active recursion path and backs up
// their best remaining f values before revisiting a discarded subtree.
//
// IDA*/RBFS require nonnegative, finite step/terminal costs and an admissible,
// nonnegative heuristic for remaining step cost, with h(goal)=0. Positive
// infinity denotes a proven dead end. Omitting the
// callback uses h=0. There is no global closed set; exact path cycles may be
// pruned. Goals carry virtual terminal edges, allowing another goal with a
// cheaper terminal charge to win. Costless problems use unit search costs but
// preserve nullopt in returned plans. Each mode returns at most one plan.
// Recursion uses the host stack: use an iterative traversal for very deep trees.
template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class IterativePlanner : public Planner<Action, State, Cost, HistoryKey> {
public:
    using Base = Planner<Action, State, Cost, HistoryKey>;
    using typename Base::Problem;
    using typename Base::PlanConsumer;
    using typename Base::PlanType;
    using Base::Search;
    using Heuristic = std::function<Cost(const State&)>;

    explicit IterativePlanner(IterativeOptions options = {}, Heuristic heuristic = {})
        : options_(options), heuristic_(std::move(heuristic)) {}

    const IterativeOptions& Options() const noexcept { return options_; }
    IterativeSearch Algorithm() const noexcept { return options_.mode; }
    SearchResultPolicy ResultPolicy() const override {
        return {options_.mode == IterativeSearch::iterativeDeepening
                    ? ResultSelection::minimumDepth : ResultSelection::minimumCost,
                false, options_.maxPlans == 0 ? 0U : 1U};
    }

    void Search(const Problem& problem, const PlanConsumer& consume,
                SearchPartition partition = {}) override {
        if (partition.workerCount == 0 || partition.workerIndex >= partition.workerCount)
            throw std::invalid_argument("invalid search partition");
        const auto stopped = [&] { return this->control_ && this->control_->stopRequested.load(); };
        if (options_.maxPlans == 0 || stopped()) return;
        const bool hasCost = problem.HasStepCost() || problem.HasTerminalCost();
        const bool depthOnly = options_.mode == IterativeSearch::iterativeDeepening;
        const auto checked = [&](Cost value) {
            if (!depthOnly && value < Cost{})
                throw std::domain_error("IDA*/RBFS require nonnegative costs and heuristics");
            if constexpr (std::is_floating_point_v<Cost>) {
                if (!std::isfinite(value)) throw std::domain_error("nonfinite planning cost or heuristic");
            }
            return value;
        };
        const auto heuristic = [&](const State& state) {
            const Cost h = !depthOnly && heuristic_ ? heuristic_(state) : Cost{};
            if (h < Cost{}) throw std::domain_error("negative planning heuristic");
            if constexpr (std::is_floating_point_v<Cost>) {
                if (std::isnan(h)) throw std::domain_error("NaN planning heuristic");
            }
            if (!depthOnly && problem.IsGoal(state) && h != Cost{})
                throw std::domain_error("IDA*/RBFS heuristic must be zero at goals");
            return h;
        };
        const auto deadEnd = [](const Cost& h) {
            if constexpr (std::is_floating_point_v<Cost>) return std::isinf(h);
            else return false;
        };
        const auto stepCost = [&](const State& from, const Action& action, const State& to) {
            if (hasCost) return checked(problem.StepCost(from, action, to).value_or(Cost{}));
            if constexpr (std::is_constructible_v<Cost, int>) {
                return Cost{1};
            } else {
                throw std::invalid_argument("costless search requires a Cost constructible from one");
            }
        };
        std::vector<State> roots;
        problem.ForEachInitialState([&](const State& state) {
            if (stopped()) return false;
            roots.push_back(state);
            return true;
        });
        std::vector<Action> actions;
        std::vector<HistoryKey> history;
        const auto cyclic = [&](const HistoryKey& key) {
            return options_.prunePathCycles && std::any_of(history.begin(), history.end(),
                [&](const auto& ancestor) { return problem.HistoryKeysEqual(ancestor, key); });
        };
        const auto ownsGoal = [&](const State& state, std::uint64_t depth) {
            return (depth != 0 || partition.workerIndex == 0) && problem.IsGoal(state);
        };
        const auto terminalCost = [&](const State& state, const Cost& g) {
            return detail::CheckedAdd(g, checked(problem.TerminalCost(state).value_or(Cost{})));
        };
        const auto emit = [&](const Cost& total) {
            PlanType result;
            result.actions = actions;
            result.stageLen = actions.size();
            result.status = PlanStatus::success;
            if (hasCost) result.totalCost = total;
            consume(result);
        };

        if (depthOnly) {
            struct Outcome { bool found = false; bool cutoff = false; };
            std::function<Outcome(const State&, Cost, std::uint64_t, std::uint64_t)> visit;
            visit = [&](const State& state, Cost g, std::uint64_t depth, std::uint64_t limit) -> Outcome {
                if (stopped()) return {};
                if (ownsGoal(state, depth)) { emit(terminalCost(state, g)); return {true, false}; }
                if (depth == limit) return {false, true};
                Outcome result;
                std::size_t rootAction = 0;
                problem.ForEachSuccessor(state, [&](const Action& action, State&& next) {
                    if (stopped()) return false;
                    if (depth == 0 && !partition.OwnsRootAction(rootAction++)) return true;
                    auto key = problem.MakeHistoryKey(next);
                    if (cyclic(key)) return true;
                    const auto nextG = detail::CheckedAdd(g, stepCost(state, action, next));
                    actions.push_back(action);
                    history.push_back(std::move(key));
                    const auto child = visit(next, nextG, depth + 1, limit);
                    history.pop_back();
                    actions.pop_back();
                    result.found = child.found;
                    result.cutoff = result.cutoff || child.cutoff;
                    return !result.found && !stopped();
                });
                return result;
            };
            for (std::uint64_t limit = 0; !stopped(); ++limit) {
                bool cutoff = false;
                for (const auto& root : roots) {
                    history = {problem.MakeHistoryKey(root)};
                    const auto result = visit(root, Cost{}, 0, limit);
                    if (result.found) return;
                    cutoff = cutoff || result.cutoff;
                }
                if (!cutoff || limit == options_.maxDepth) return;
            }
            return;
        }

        if (options_.mode == IterativeSearch::idaStar) {
            struct Outcome { bool found = false; std::optional<Cost> nextBound; };
            const auto improve = [](std::optional<Cost>& bound, const std::optional<Cost>& candidate) {
                if (candidate && (!bound || *candidate < *bound)) bound = candidate;
            };
            std::function<Outcome(const State&, Cost, std::uint64_t, const Cost&)> visit;
            visit = [&](const State& state, Cost g, std::uint64_t depth, const Cost& bound) -> Outcome {
                if (stopped()) return {};
                const Cost h = heuristic(state);
                if (deadEnd(h)) return {};
                const Cost f = detail::CheckedAdd(g, h);
                if (bound < f) return {false, f};
                Outcome result;
                if (ownsGoal(state, depth)) {
                    const Cost total = terminalCost(state, g);
                    if (!(bound < total)) { emit(total); return {true, std::nullopt}; }
                    result.nextBound = total;
                }
                if (depth == options_.maxDepth) return result;
                std::size_t rootAction = 0;
                problem.ForEachSuccessor(state, [&](const Action& action, State&& next) {
                    if (stopped()) return false;
                    if (depth == 0 && !partition.OwnsRootAction(rootAction++)) return true;
                    auto key = problem.MakeHistoryKey(next);
                    if (cyclic(key)) return true;
                    const Cost nextG = detail::CheckedAdd(g, stepCost(state, action, next));
                    actions.push_back(action);
                    history.push_back(std::move(key));
                    const auto child = visit(next, nextG, depth + 1, bound);
                    history.pop_back();
                    actions.pop_back();
                    result.found = child.found;
                    improve(result.nextBound, child.nextBound);
                    return !result.found && !stopped();
                });
                return result;
            };
            std::optional<Cost> bound;
            for (const auto& root : roots) {
                const Cost h = heuristic(root);
                if (!deadEnd(h)) improve(bound, h);
            }
            while (bound && !stopped()) {
                std::optional<Cost> nextBound;
                for (const auto& root : roots) {
                    history = {problem.MakeHistoryKey(root)};
                    const auto result = visit(root, Cost{}, 0, *bound);
                    if (result.found) return;
                    improve(nextBound, result.nextBound);
                }
                bound = nextBound;
            }
            return;
        }

        // nullopt is an exhausted subtree / infinite bound, so no numeric
        // infinity or conversion to floating point is required for generic Cost.
        struct Candidate {
            std::optional<State> state; // absent for a virtual terminal edge
            std::optional<Action> action;
            Cost g;
            std::optional<Cost> f;
            std::uint64_t depth;
        };
        struct Outcome { bool found = false; std::optional<Cost> backup; };
        std::function<Outcome(const State&, Cost, Cost, std::uint64_t, std::optional<Cost>)> visit;
        std::function<Outcome(std::vector<Candidate>&, std::optional<Cost>)> choose;
        choose = [&](std::vector<Candidate>& children, std::optional<Cost> limit) -> Outcome {
            while (!stopped()) {
                std::optional<std::size_t> best;
                std::optional<Cost> alternative;
                for (std::size_t i = 0; i < children.size(); ++i) {
                    if (!children[i].f) continue;
                    if (!best || *children[i].f < *children[*best].f) {
                        if (best) alternative = children[*best].f;
                        best = i;
                    } else if (!alternative || *children[i].f < *alternative) {
                        alternative = children[i].f;
                    }
                }
                if (!best) return {};
                auto& child = children[*best];
                if (limit && *limit < *child.f) return {false, child.f};
                if (!child.state) { emit(child.g); return {true, std::nullopt}; }
                std::optional<Cost> nextLimit = limit;
                if (alternative && (!nextLimit || *alternative < *nextLimit)) nextLimit = alternative;
                if (child.action) actions.push_back(*child.action);
                history.push_back(problem.MakeHistoryKey(*child.state));
                const auto result = visit(*child.state, child.g, *child.f, child.depth, nextLimit);
                history.pop_back();
                if (child.action) actions.pop_back();
                if (result.found) return result;
                child.f = result.backup;
            }
            return {};
        };
        visit = [&](const State& state, Cost g, Cost f, std::uint64_t depth,
                    std::optional<Cost> limit) -> Outcome {
            if (stopped()) return {};
            std::vector<Candidate> children;
            if (ownsGoal(state, depth)) {
                const Cost total = terminalCost(state, g);
                children.push_back({std::nullopt, std::nullopt, total, std::max(f, total), depth});
            }
            if (depth < options_.maxDepth) {
                std::size_t rootAction = 0;
                problem.ForEachSuccessor(state, [&](const Action& action, State&& next) {
                    if (stopped()) return false;
                    if (depth == 0 && !partition.OwnsRootAction(rootAction++)) return true;
                    if (cyclic(problem.MakeHistoryKey(next))) return true;
                    const Cost nextG = detail::CheckedAdd(g, stepCost(state, action, next));
                    const Cost h = heuristic(next);
                    if (deadEnd(h)) return true;
                    const Cost nextF = std::max(f, detail::CheckedAdd(nextG, h));
                    children.push_back({std::move(next), action, nextG, nextF, depth + 1});
                    return true;
                });
            }
            return choose(children, limit);
        };
        std::vector<Candidate> candidates;
        for (auto& root : roots) {
            const auto f = heuristic(root);
            if (deadEnd(f)) continue;
            candidates.push_back({std::move(root), std::nullopt, Cost{}, f, 0});
        }
        (void)choose(candidates, std::nullopt);
    }

private:
    IterativeOptions options_;
    Heuristic heuristic_;
};

template<typename A, typename S, typename C = float, typename K = S>
class IterativeDeepeningPlanner final : public IterativePlanner<A, S, C, K> {
public:
    explicit IterativeDeepeningPlanner(IterativeOptions options = {})
        : IterativePlanner<A, S, C, K>(Configure(options)) {}
private:
    static IterativeOptions Configure(IterativeOptions o) { o.mode = IterativeSearch::iterativeDeepening; return o; }
};

template<typename A, typename S, typename C = float, typename K = S>
class IDAStarPlanner final : public IterativePlanner<A, S, C, K> {
public:
    using Heuristic = typename IterativePlanner<A, S, C, K>::Heuristic;
    explicit IDAStarPlanner(IterativeOptions options = {}, Heuristic heuristic = {})
        : IterativePlanner<A, S, C, K>(Configure(options), std::move(heuristic)) {}
private:
    static IterativeOptions Configure(IterativeOptions o) { o.mode = IterativeSearch::idaStar; return o; }
};

template<typename A, typename S, typename C = float, typename K = S>
class RBFSPlanner final : public IterativePlanner<A, S, C, K> {
public:
    using Heuristic = typename IterativePlanner<A, S, C, K>::Heuristic;
    explicit RBFSPlanner(IterativeOptions options = {}, Heuristic heuristic = {})
        : IterativePlanner<A, S, C, K>(Configure(options), std::move(heuristic)) {}
private:
    static IterativeOptions Configure(IterativeOptions o) { o.mode = IterativeSearch::recursiveBestFirst; return o; }
};

} // namespace planning
