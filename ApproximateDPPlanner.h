#pragma once

#include "InformedPlanner.h"

#include <functional>
#include <unordered_map>

namespace planning {

enum class ApproximateDPMode { rollout = 300, rtdp, rtdpAStar };

struct ApproximateDPOptions {
    ApproximateDPMode mode = ApproximateDPMode::rtdp;
    std::uint64_t maxDepth = 100;
    std::size_t maxPlans = 1;
    bool prunePathCycles = true;
    std::uint64_t heuristicId = 0;
    std::size_t rolloutDepth = 16; // includes the candidate first action
    std::size_t trials = 32;       // per initial state; RTDP modes only
    std::size_t trialDepth = 64;   // trial steps, or committed rollout decisions
    // Caller contract: a supplied heuristic lower-bounds unrestricted remaining
    // STEP cost. Required for rtdpAStar; a missing heuristic is certified zero.
    bool admissibleInitialization = false;
};

struct ApproximateDPStatistics {
    std::size_t trialsStarted = 0;
    std::size_t bellmanBackups = 0;
    std::size_t successorEvaluations = 0;
    std::size_t rolloutEvaluations = 0;
    std::size_t committedDecisions = 0;
    std::size_t learnedStates = 0;
};

// Deterministic rollout and asynchronous, trial-based real-time DP. The two
// approximate modes return the best complete trajectory actually encountered;
// exhausting a budget, encountering a loop, or returning no plan proves nothing
// about feasibility or optimality. Rollout simulates a deterministic c+h greedy
// base policy for EVERY candidate first action, retains feasible incumbents, and
// replans after committing one action. Complete candidate trajectories are
// preferred to truncated estimates. There is no general dominance claim over
// an untruncated base policy that has not been evaluated.
//
// RTDP applies V(s) <- max(V(s), min_a[c(s,a)+V(s')]), with V(goal)=0.
// Backups always enumerate ALL successors; path cycles, depth budgets and root
// ownership affect only trajectory selection. Thus, with admissible initial V,
// the table is a lower bound on the UNRESTRICTED remaining step cost, also a
// relaxation lower bound for each restricted search. A trial leaf is never
// declared a dead end merely because its budget or permitted choices ran out.
// Dead-end infinity is inferred only from a non-goal with no original successors
// (or all successors already carrying certified infinity).
//
// rtdpAStar freezes that table after bounded warm-up and delegates to existing
// A*: no rollout upper bound enters its heuristic. Terminal charges stay outside
// the learned step bounds and are handled by A*'s virtual terminal edge. The
// approximate modes stop a trajectory at its first goal; they may miss a cheaper
// goal reachable through another goal. Costs must be nonnegative and finite.
// Floating-point rtdpAStar additionally requires INTEGER-valued step/terminal
// charges strictly below 2^numeric_limits<Cost>::digits, and only returns totals
// strictly below that limit. Every prefix of a cheaper competitor then accumulates
// exactly, so Bellman bounds and forward A* optimize the same numerical objective.
// Other numeric regimes throw; rollout/rtdp retain fractional-cost support.
// Costless problems use implicit unit steps for decisions and return no cost.
// Each worker owns its table; no process-global learning or random policy exists.
template<typename Action, typename State, typename Cost = float, typename HistoryKey = State>
class ApproximateDPPlanner : public Planner<Action, State, Cost, HistoryKey> {
    static_assert(std::is_arithmetic_v<Cost>, "approximate DP requires an arithmetic Cost type");
public:
    using Base = Planner<Action, State, Cost, HistoryKey>;
    using typename Base::Problem;
    using typename Base::PlanConsumer;
    using typename Base::PlanType;
    using Base::Search;
    using Heuristic = std::function<Cost(const State&)>;

    explicit ApproximateDPPlanner(ApproximateDPOptions options = {}, Heuristic heuristic = {})
        : options_(options), heuristic_(std::move(heuristic)) {
        if (options_.rolloutDepth == 0) throw std::invalid_argument("rolloutDepth must be positive");
        if (options_.mode == ApproximateDPMode::rtdpAStar && heuristic_ && !options_.admissibleInitialization)
            throw std::invalid_argument("ADP-A* requires a certified admissible initialization");
    }

    const ApproximateDPOptions& Options() const noexcept { return options_; }
    ApproximateDPMode Algorithm() const noexcept { return options_.mode; }
    const ApproximateDPStatistics& Statistics() const noexcept { return statistics_; }
    SearchResultPolicy ResultPolicy() const override {
        // Selecting the cheapest observed feasible result across workers is not
        // an optimality assertion for the approximate modes.
        return {ResultSelection::minimumCost, false, options_.maxPlans == 0 ? 0U : 1U};
    }

    // nullopt means unseen; +infinity is a stored dead-end estimate. Certification
    // depends on admissible initialization, not on the approximate-mode name.
    std::optional<long double> LearnedValue(const Problem& problem, const State& state) const {
        const auto key = problem.MakeHistoryKey(state);
        const auto found = byHash_.find(problem.HistoryKeyHash(key));
        if (found != byHash_.end()) {
            for (auto index : found->second)
                if (problem.HistoryKeysEqual(values_[index].key, key)) return values_[index].value;
        }
        return std::nullopt;
    }

    void Search(const Problem& problem, const PlanConsumer& consume,
                SearchPartition partition = {}) override {
        if (partition.workerCount == 0 || partition.workerIndex >= partition.workerCount)
            throw std::invalid_argument("invalid search partition");
        values_.clear(); byHash_.clear(); statistics_ = {};
        const auto stopped = [&] { return this->control_ && this->control_->stopRequested.load(); };
        if (options_.maxPlans == 0 || stopped()) return;
        const bool hasCost = problem.HasStepCost() || problem.HasTerminalCost();
        constexpr long double infinity = std::numeric_limits<long double>::infinity();
        const long double exactIntegerLimit = std::ldexp(1.0L, std::numeric_limits<Cost>::digits);
        auto checked = [&](Cost value) {
            if (value < Cost{} || !std::isfinite(static_cast<long double>(value)))
                throw std::domain_error("approximate DP needs finite nonnegative costs");
            if constexpr (std::is_floating_point_v<Cost>) {
                if (options_.mode == ApproximateDPMode::rtdpAStar &&
                    (std::trunc(static_cast<long double>(value)) != static_cast<long double>(value) ||
                     static_cast<long double>(value) >= exactIntegerLimit))
                    throw std::domain_error("ADP-A* floating costs must be nonnegative integers strictly below 2^Cost::digits");
            }
            return value;
        };
        auto initialValue = [&](const State& state) -> long double {
            const auto value = heuristic_ ? heuristic_(state) : Cost{};
            const auto converted = static_cast<long double>(value);
            if (converted < 0 || std::isnan(converted)) throw std::domain_error("invalid DP heuristic");
            if (problem.IsGoal(state) && converted != 0) throw std::domain_error("DP heuristic must be zero at goals");
            return converted;
        };
        auto ensure = [&](const State& state) {
            auto key = problem.MakeHistoryKey(state);
            auto& bucket = byHash_[problem.HistoryKeyHash(key)];
            for (auto index : bucket)
                if (problem.HistoryKeysEqual(values_[index].key, key)) return index;
            const auto index = values_.size();
            values_.push_back({std::move(key), initialValue(state)});
            bucket.push_back(index);
            statistics_.learnedStates = values_.size();
            return index;
        };
        auto value = [&](const State& state) { return values_[ensure(state)].value; };
        auto terminal = [&](const State& state) { return checked(problem.TerminalCost(state).value_or(Cost{})); };
        auto stepCost = [&](const State& from, const Action& action, const State& to) {
            return checked(hasCost ? problem.StepCost(from, action, to).value_or(Cost{}) : Cost{1});
        };
        auto onPath = [&](const std::vector<HistoryKey>& prefix, const HistoryKey& key) {
            return options_.prunePathCycles && std::any_of(prefix.begin(), prefix.end(), [&](const auto& previous) {
                return problem.HistoryKeysEqual(previous, key);
            });
        };
        struct Edge { Action action; State state; HistoryKey key; Cost cost; std::size_t ordinal; };
        auto successors = [&](const State& state, bool& complete) {
            std::vector<Edge> edges;
            complete = !stopped();
            if (!complete) return edges;
            std::size_t ordinal = 0;
            problem.ForEachSuccessor(state, [&](const Action& action, State&& next) {
                if (stopped()) { complete = false; return false; }
                auto key = problem.MakeHistoryKey(next);
                auto cost = stepCost(state, action, next);
                edges.push_back({action, std::move(next), std::move(key), cost, ordinal++});
                ++statistics_.successorEvaluations;
                return true;
            });
            complete = complete && !stopped();
            return edges;
        };
        // One rounding step toward -infinity avoids rounding a certified lower
        // bound upward. Infinity propagates only from certified successor values.
        const auto lowerSum = [&](Cost edge, long double tail) {
            if (std::isinf(tail)) return infinity;
            const long double sum = static_cast<long double>(edge) + tail;
            return sum == 0 ? 0.0L : std::nextafter(sum, -infinity);
        };
        auto backup = [&](const State& state, const std::vector<Edge>& edges) {
            const auto index = ensure(state);
            long double backed = problem.IsGoal(state) ? 0.0L : infinity;
            if (!problem.IsGoal(state)) {
                for (const auto& edge : edges) backed = std::min(backed, lowerSum(edge.cost, value(edge.state)));
            }
            values_[index].value = std::max(values_[index].value, backed);
            ++statistics_.bellmanBackups;
        };
        auto policyScore = [&](const Edge& edge) {
            const auto tail = value(edge.state);
            return static_cast<long double>(edge.cost) + tail +
                   (problem.IsGoal(edge.state) ? static_cast<long double>(terminal(edge.state)) : 0.0L);
        };
        std::optional<PlanType> incumbent;
        std::optional<Cost> incumbentScore;
        auto remember = [&](const std::vector<Action>& actions, Cost pathCost, const State& goal) {
            const auto total = detail::CheckedAdd(pathCost, terminal(goal));
            if (incumbentScore && ! (total < *incumbentScore)) return;
            PlanType plan;
            plan.status = PlanStatus::success;
            plan.actions = actions;
            plan.stageLen = actions.size();
            if (hasCost) plan.totalCost = total;
            incumbent = std::move(plan);
            incumbentScore = total;
        };
        auto allowed = [&](const Edge& edge, std::uint64_t depth, const std::vector<HistoryKey>& prefix) {
            return (depth != 0 || partition.OwnsRootAction(edge.ordinal)) && !onPath(prefix, edge.key);
        };
        auto train = [&](const State& initial) {
            for (std::size_t trial = 0; trial < options_.trials && !stopped(); ++trial) {
                ++statistics_.trialsStarted;
                State state = initial;
                std::vector<State> visited;
                std::vector<Action> actions;
                std::vector<HistoryKey> prefix{problem.MakeHistoryKey(initial)};
                Cost g{};
                for (std::size_t depth = 0; !stopped(); ++depth) {
                    (void)value(state);
                    if (problem.IsGoal(state)) {
                        if (depth != 0 || partition.workerIndex == 0) remember(actions, g, state);
                        break;
                    }
                    if (depth >= options_.trialDepth || depth >= options_.maxDepth) break;
                    bool complete;
                    auto edges = successors(state, complete);
                    if (!complete) break;
                    backup(state, edges); // BEFORE any trajectory restrictions
                    visited.push_back(state);
                    std::optional<std::size_t> best;
                    long double bestScore = infinity;
                    for (std::size_t i = 0; i < edges.size(); ++i) {
                        if (!allowed(edges[i], depth, prefix)) continue;
                        const auto score = policyScore(edges[i]);
                        if (score < bestScore) { best = i; bestScore = score; }
                    }
                    if (!best) break;
                    const auto& edge = edges[*best];
                    g = detail::CheckedAdd(g, edge.cost);
                    actions.push_back(edge.action);
                    prefix.push_back(edge.key);
                    state = edge.state;
                    ++statistics_.committedDecisions;
                }
                // Reverse sweeps reuse no filtered action list; every backup
                // again considers the original transition relation.
                for (auto it = visited.rbegin(); it != visited.rend() && !stopped(); ++it) {
                    bool complete;
                    const auto edges = successors(*it, complete);
                    if (complete) backup(*it, edges);
                }
            }
        };
        struct Evaluation { std::vector<Action> actions; Cost steps{};
                            long double score = std::numeric_limits<long double>::infinity();
                            std::optional<State> goal; };
        auto simulate = [&](const State& initial, std::vector<HistoryKey> prefix,
                            std::uint64_t remaining, Cost startingCost) {
            ++statistics_.rolloutEvaluations;
            Evaluation result;
            // Carry the actual prefix cost through each step in replay order;
            // summing a separately accumulated suffix can change float totals.
            result.steps = startingCost;
            State state = initial;
            const auto limit = std::min<std::uint64_t>(options_.rolloutDepth - 1, remaining);
            for (std::uint64_t depth = 0; !stopped(); ++depth) {
                const auto h = value(state);
                if (problem.IsGoal(state)) {
                    result.goal = state;
                    result.score = static_cast<long double>(result.steps) + terminal(state);
                    return result;
                }
                if (depth == limit) {
                    result.score = static_cast<long double>(result.steps) + h;
                    return result;
                }
                bool complete;
                const auto edges = successors(state, complete);
                if (!complete) return result;
                std::optional<std::size_t> best;
                long double bestScore = infinity;
                for (std::size_t i = 0; i < edges.size(); ++i) {
                    if (onPath(prefix, edges[i].key)) continue;
                    const auto score = policyScore(edges[i]);
                    if (score < bestScore) { best = i; bestScore = score; }
                }
                if (!best) return result;
                const auto& edge = edges[*best];
                result.steps = detail::CheckedAdd(result.steps, edge.cost);
                result.actions.push_back(edge.action);
                prefix.push_back(edge.key);
                state = edge.state;
            }
            return result;
        };
        auto rollout = [&](const State& initial) {
            State state = initial;
            std::vector<Action> actions;
            std::vector<HistoryKey> prefix{problem.MakeHistoryKey(initial)};
            Cost g{};
            for (std::uint64_t depth = 0; !stopped(); ++depth) {
                (void)value(state);
                if (problem.IsGoal(state)) {
                    if (depth != 0 || partition.workerIndex == 0) remember(actions, g, state);
                    return;
                }
                if (depth >= options_.trialDepth || depth >= options_.maxDepth) return;
                bool complete;
                const auto edges = successors(state, complete);
                if (!complete) return;
                std::optional<std::size_t> best;
                bool bestComplete = false;
                long double bestScore = infinity;
                for (std::size_t i = 0; i < edges.size() && !stopped(); ++i) {
                    const auto& edge = edges[i];
                    if (!allowed(edge, depth, prefix)) continue;
                    auto trialPrefix = prefix;
                    trialPrefix.push_back(edge.key);
                    auto evaluation = simulate(edge.state, std::move(trialPrefix), options_.maxDepth - depth - 1,
                                               detail::CheckedAdd(g, edge.cost));
                    const auto score = evaluation.score;
                    if (evaluation.goal) {
                        auto candidate = actions;
                        candidate.push_back(edge.action);
                        candidate.insert(candidate.end(), evaluation.actions.begin(), evaluation.actions.end());
                        remember(candidate, evaluation.steps, *evaluation.goal);
                    }
                    const bool solved = evaluation.goal.has_value();
                    if ((solved && !bestComplete) || (solved == bestComplete && score < bestScore)) {
                        best = i; bestScore = score; bestComplete = solved;
                    }
                }
                if (!best || stopped()) return;
                const auto& edge = edges[*best];
                g = detail::CheckedAdd(g, edge.cost);
                actions.push_back(edge.action);
                prefix.push_back(edge.key);
                state = edge.state;
                ++statistics_.committedDecisions;
            }
        };
        problem.ForEachInitialState([&](const State& initial) {
            if (stopped()) return false;
            if (problem.IsGoal(initial)) {
                (void)value(initial);
                if (partition.workerIndex == 0) remember({}, Cost{}, initial);
                return true;
            }
            if (options_.mode == ApproximateDPMode::rollout) rollout(initial);
            else train(initial);
            return !stopped();
        });
        if (stopped()) return;
        if (options_.mode == ApproximateDPMode::rtdpAStar) {
            // The table is frozen: this closure neither inserts nor backs up.
            auto lowerBound = [&](const State& state) -> Cost {
                const auto learned = LearnedValue(problem, state);
                const auto h = learned ? *learned : initialValue(state);
                if constexpr (std::is_floating_point_v<Cost>) {
                    if (std::isinf(h)) return std::numeric_limits<Cost>::infinity();
                    if (h > static_cast<long double>(std::numeric_limits<Cost>::max()))
                        return std::numeric_limits<Cost>::max();
                    auto converted = static_cast<Cost>(h);
                    if (static_cast<long double>(converted) > h)
                        converted = std::nextafter(converted, -std::numeric_limits<Cost>::infinity());
                    return converted;
                } else {
                    return h >= static_cast<long double>(std::numeric_limits<Cost>::max())
                        ? std::numeric_limits<Cost>::max() : static_cast<Cost>(h);
                }
            };
            InformedOptions exact;
            exact.mode = InformedSearch::aStar;
            exact.maxDepth = options_.maxDepth;
            exact.maxPlans = options_.maxPlans;
            exact.prunePathCycles = options_.prunePathCycles;
            exact.heuristicId = options_.heuristicId;
            InformedPlanner<Action, State, Cost, HistoryKey> astar(exact, lowerBound);
            astar.SetControl(this->control_);
            // Validate costs in states never seen during warm-up as well. This
            // transparent wrapper retains fused/PackedProblem expansion and
            // exact history-key semantics; it owns no states or expansion engine.
            class ValidatedProblem final : public Problem {
            public:
                ValidatedProblem(const Problem& source, std::function<Cost(Cost)> check)
                    : source_(source), check_(std::move(check)) {}
                void ForEachInitialState(const Consumer<State>& c) const override { source_.ForEachInitialState(c); }
                bool IsGoal(const State& state) const override { return source_.IsGoal(state); }
                void ForEachApplicableAction(const State& state, const Consumer<Action>& c) const override {
                    source_.ForEachApplicableAction(state,c);
                }
                std::optional<State> Apply(const State& state, const Action& action) const override {
                    return source_.Apply(state,action);
                }
                void ForEachSuccessor(const State& state, const SuccessorConsumer<Action,State>& c) const override {
                    source_.ForEachSuccessor(state,[&](const Action& action, State&& next) {
                        if (HasStepCost() || HasTerminalCost())
                            check_(source_.StepCost(state,action,next).value_or(Cost{}));
                        return c(action,std::move(next));
                    });
                }
                HistoryKey MakeHistoryKey(const State& state) const override { return source_.MakeHistoryKey(state); }
                bool HistoryKeysEqual(const HistoryKey& a,const HistoryKey& b) const override {
                    return source_.HistoryKeysEqual(a,b);
                }
                std::size_t HistoryKeyHash(const HistoryKey& key) const override { return source_.HistoryKeyHash(key); }
                bool HasStepCost() const override { return source_.HasStepCost(); }
                bool HasTerminalCost() const override { return source_.HasTerminalCost(); }
                std::optional<Cost> StepCost(const State& from,const Action& action,const State& to) const override {
                    auto cost = source_.StepCost(from,action,to);
                    if (cost) check_(*cost);
                    return cost;
                }
                std::optional<Cost> TerminalCost(const State& state) const override {
                    auto cost = source_.TerminalCost(state);
                    if (cost) check_(*cost);
                    return cost;
                }
            private:
                const Problem& source_;
                std::function<Cost(Cost)> check_;
            } validated(problem,checked);
            astar.Search(validated,[&](const PlanType& plan) {
                if constexpr (std::is_floating_point_v<Cost>) {
                    const auto total = plan.totalCost ? static_cast<long double>(*plan.totalCost)
                                                     : static_cast<long double>(plan.stageLen);
                    if (total >= exactIntegerLimit)
                        throw std::domain_error("ADP-A* result is outside its exact integer accumulation range");
                }
                return consume(plan);
            }, partition);
        } else if (incumbent) consume(*incumbent);
    }

private:
    struct Value { HistoryKey key; long double value; };
    ApproximateDPOptions options_;
    Heuristic heuristic_;
    ApproximateDPStatistics statistics_;
    std::vector<Value> values_;
    std::unordered_map<std::size_t, std::vector<std::size_t>> byHash_;
};

} // namespace planning
