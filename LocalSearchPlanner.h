#pragma once

#include "Planner.h"

#include <functional>

namespace planning {

enum class LocalSearch { hillClimbing = 200, enforcedHillClimbing };

struct LocalSearchOptions {
    LocalSearch mode = LocalSearch::enforcedHillClimbing;
    std::uint64_t maxDepth = 100;
    std::size_t maxPlans = 1;
    bool prunePathCycles = true;
    std::uint64_t heuristicId = 0;
};

// Both methods are incomplete: committing to an improving state can enter a
// dead end even when another branch has a solution. Steepest-descent hill
// climbing accepts a strictly lower heuristic successor (or an immediate goal).
// Enforced hill climbing uses BFS to find the first strictly improving state or
// goal, commits that escape path, and repeats. This is the EHC search strategy,
// not a full FF planner. Neither method promises minimum depth or cost.
//
// Heuristics and costs must be nonnegative; +infinity in a heuristic means a
// proven dead end. A missing heuristic is zero (EHC then performs bounded BFS).
// EHC keeps an exact duplicate table within each escape search; the cycle option
// additionally excludes states on the committed prefix. maxDepth includes both
// committed actions and escape actions. At most one executable plan is returned.
template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class LocalSearchPlanner : public Planner<Action, State, Cost, HistoryKey> {
public:
    using Base = Planner<Action, State, Cost, HistoryKey>;
    using typename Base::Problem;
    using typename Base::PlanConsumer;
    using typename Base::PlanType;
    using Base::Search;
    using Heuristic = std::function<Cost(const State&)>;

    explicit LocalSearchPlanner(LocalSearchOptions options = {}, Heuristic heuristic = {})
        : options_(options), heuristic_(std::move(heuristic)) {}
    const LocalSearchOptions& Options() const noexcept { return options_; }
    LocalSearch Algorithm() const noexcept { return options_.mode; }
    SearchResultPolicy ResultPolicy() const override {
        return {ResultSelection::first, false, options_.maxPlans == 0 ? 0U : 1U};
    }

    void Search(const Problem& problem, const PlanConsumer& consume,
                SearchPartition partition = {}) override {
        if (partition.workerCount == 0 || partition.workerIndex >= partition.workerCount)
            throw std::invalid_argument("invalid search partition");
        const auto stopped = [&] { return this->control_ && this->control_->stopRequested.load(); };
        if (options_.maxPlans == 0 || stopped()) return;
        const bool hasCost = problem.HasStepCost() || problem.HasTerminalCost();
        const auto checkedCost = [](Cost value) {
            if (value < Cost{}) throw std::domain_error("local search requires nonnegative costs");
            if constexpr (std::is_floating_point_v<Cost>) {
                if (!std::isfinite(value)) throw std::domain_error("nonfinite planning cost");
            }
            return value;
        };
        const auto heuristic = [&](const State& state) {
            const Cost h = heuristic_ ? heuristic_(state) : Cost{};
            if (h < Cost{}) throw std::domain_error("negative planning heuristic");
            if constexpr (std::is_floating_point_v<Cost>) {
                if (std::isnan(h)) throw std::domain_error("NaN planning heuristic");
            }
            if (problem.IsGoal(state) && h != Cost{})
                throw std::domain_error("local-search heuristic must be zero at goals");
            return h;
        };
        const auto deadEnd = [](const Cost& h) {
            if constexpr (std::is_floating_point_v<Cost>) return std::isinf(h);
            else return false;
        };
        const auto ownsGoal = [&](const State& state, std::uint64_t depth) {
            return (depth != 0 || partition.workerIndex == 0) && problem.IsGoal(state);
        };
        const auto nextCost = [&](const State& from, const Action& action, const State& to,
                                  std::optional<Cost> g) -> std::optional<Cost> {
            if (!g) return std::nullopt;
            return detail::CheckedAdd(*g, checkedCost(problem.StepCost(from, action, to).value_or(Cost{})));
        };
        const auto runRoot = [&](const State& initial) {
            State current = initial;
            std::vector<Action> actions;
            std::vector<HistoryKey> prefix{problem.MakeHistoryKey(initial)};
            std::optional<Cost> g = hasCost ? std::optional<Cost>{Cost{}} : std::nullopt;
            const auto onPrefix = [&](const HistoryKey& key) {
                return options_.prunePathCycles && std::any_of(prefix.begin(), prefix.end(),
                    [&](const auto& previous) { return problem.HistoryKeysEqual(previous, key); });
            };
            while (!stopped()) {
                const Cost h = heuristic(current);
                if (ownsGoal(current, actions.size())) {
                    PlanType plan;
                    plan.actions = actions;
                    plan.stageLen = actions.size();
                    plan.status = PlanStatus::success;
                    if (g) plan.totalCost = detail::CheckedAdd(*g,
                        checkedCost(problem.TerminalCost(current).value_or(Cost{})));
                    consume(plan);
                    return true;
                }
                if (deadEnd(h) || actions.size() >= options_.maxDepth) return false;
                if (options_.mode == LocalSearch::hillClimbing) {
                    struct Choice { State state; Action action; HistoryKey key; Cost h;
                                    std::optional<Cost> g; bool goal; };
                    std::optional<Choice> best;
                    std::size_t rootAction = 0;
                    problem.ForEachSuccessor(current, [&](const Action& action, State&& next) {
                        if (stopped()) return false;
                        if (actions.empty() && !partition.OwnsRootAction(rootAction++)) return true;
                        auto key = problem.MakeHistoryKey(next);
                        if (onPrefix(key)) return true;
                        const Cost nextH = heuristic(next);
                        if (deadEnd(nextH)) return true;
                        const bool goal = ownsGoal(next, actions.size() + 1);
                        if (!goal && !(nextH < h)) return true;
                        if (best && !(nextH < best->h) && !(goal && !best->goal)) return true;
                        auto nextG = nextCost(current, action, next, g);
                        best.emplace(Choice{std::move(next), action, std::move(key), nextH, nextG, goal});
                        return true;
                    });
                    if (!best || stopped()) return false;
                    actions.push_back(std::move(best->action));
                    prefix.push_back(std::move(best->key));
                    current = std::move(best->state);
                    g = best->g;
                    continue;
                }

                constexpr auto none = std::numeric_limits<std::size_t>::max();
                struct Node { State state; HistoryKey key; std::optional<Action> action;
                              std::size_t parent; std::uint64_t depth; std::optional<Cost> g; };
                std::vector<Node> queue;
                queue.push_back({current, problem.MakeHistoryKey(current), std::nullopt,
                                 none, static_cast<std::uint64_t>(actions.size()), g});
                std::optional<std::size_t> improved;
                for (std::size_t head = 0; head < queue.size() && !improved && !stopped(); ++head) {
                    // Successor insertion can reallocate queue; keep an owning
                    // parent copy alive for the entire enumeration callback.
                    const State state = queue[head].state;
                    const auto depth = queue[head].depth;
                    const auto cost = queue[head].g;
                    if (depth == options_.maxDepth) continue;
                    std::size_t rootAction = 0;
                    problem.ForEachSuccessor(state, [&](const Action& action, State&& next) {
                        if (stopped()) return false;
                        if (depth == 0 && !partition.OwnsRootAction(rootAction++)) return true;
                        auto key = problem.MakeHistoryKey(next);
                        if (onPrefix(key) || std::any_of(queue.begin(), queue.end(), [&](const auto& previous) {
                                return problem.HistoryKeysEqual(previous.key, key);
                            })) return true;
                        const Cost nextH = heuristic(next);
                        if (deadEnd(nextH)) return true;
                        const bool better = nextH < h || ownsGoal(next, depth + 1);
                        auto nextG = nextCost(state, action, next, cost);
                        queue.push_back({std::move(next), std::move(key), action, head, depth + 1, nextG});
                        if (better) improved = queue.size() - 1;
                        return !improved;
                    });
                }
                if (!improved || stopped()) return false;
                std::vector<std::size_t> escape;
                for (auto id = *improved; queue[id].parent != none; id = queue[id].parent)
                    escape.push_back(id);
                for (auto iterator = escape.rbegin(); iterator != escape.rend(); ++iterator) {
                    actions.push_back(*queue[*iterator].action);
                    prefix.push_back(queue[*iterator].key);
                }
                current = std::move(queue[*improved].state);
                g = queue[*improved].g;
            }
            return false;
        };
        problem.ForEachInitialState([&](const State& initial) {
            return !stopped() && !runRoot(initial);
        });
    }

private:
    LocalSearchOptions options_;
    Heuristic heuristic_;
};

template<typename A, typename S, typename C = float, typename K = S>
class HillClimbingPlanner final : public LocalSearchPlanner<A,S,C,K> {
public:
    using Heuristic = typename LocalSearchPlanner<A,S,C,K>::Heuristic;
    explicit HillClimbingPlanner(LocalSearchOptions options = {}, Heuristic heuristic = {})
        : LocalSearchPlanner<A,S,C,K>(Configure(options), std::move(heuristic)) {}
private:
    static LocalSearchOptions Configure(LocalSearchOptions o) { o.mode = LocalSearch::hillClimbing; return o; }
};

template<typename A, typename S, typename C = float, typename K = S>
class EnforcedHillClimbingPlanner final : public LocalSearchPlanner<A,S,C,K> {
public:
    using Heuristic = typename LocalSearchPlanner<A,S,C,K>::Heuristic;
    explicit EnforcedHillClimbingPlanner(LocalSearchOptions options = {}, Heuristic heuristic = {})
        : LocalSearchPlanner<A,S,C,K>(Configure(options), std::move(heuristic)) {}
private:
    static LocalSearchOptions Configure(LocalSearchOptions o) { o.mode = LocalSearch::enforcedHillClimbing; return o; }
};

} // namespace planning
