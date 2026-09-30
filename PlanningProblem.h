#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace planning
{
    // A deliberately simple type-erased callback. Returning false asks the
    // producer to stop enumerating. This mirrors the "consumer" style that is
    // common in Java while remaining natural for lambdas in modern C++.
    template<typename T>
    using Consumer = std::function<bool(const T&)>;

    // A successor callback receives both the action and the successor state.
    // The state is passed as an rvalue so a planner can move it directly into
    // frontier storage instead of copying it.
    template<typename Action, typename State>
    using SuccessorConsumer = std::function<bool(const Action&, State&&)>;

    // HistoryKey is intentionally separate from State. A domain may retain a
    // much smaller exact cycle key than its complete search state. The default
    // keeps the complete state for domains that do not provide a specialization.
    template<
        typename Action,
        typename State,
        typename Cost = float,
        typename HistoryKey = State
    >
    class PlanningProblem {
    public:
        using ActionType = Action;
        using StateType = State;
        using CostType = Cost;
        using HistoryKeyType = HistoryKey;

        virtual ~PlanningProblem() = default;

        virtual void ForEachInitialState(
            const Consumer<State>& consumer
        ) const = 0;

        virtual bool IsGoal(const State& state) const = 0;

        virtual void ForEachApplicableAction(
            const State& state,
            const Consumer<Action>& consumer
        ) const = 0;

        virtual std::optional<State> Apply(
            const State& state,
            const Action& action
        ) const = 0;

        // Optional fused enumeration hook. The default implementation preserves
        // the old ApplicableActions + Apply contract, but a concrete problem can
        // override this to avoid computing a successor twice.
        virtual void ForEachSuccessor(
            const State& state,
            const SuccessorConsumer<Action, State>& consumer
        ) const {
            ForEachApplicableAction(
                state,
                [&](const Action& action) {
                    auto next = Apply(state, action);
                    if (!next.has_value()) {
                        return true;
                    }
                    return consumer(action, std::move(*next));
                }
            );
        }

        virtual HistoryKey MakeHistoryKey(
            const State& state
        ) const = 0;

        virtual bool HistoryKeysEqual(
            const HistoryKey& left,
            const HistoryKey& right
        ) const {
            return left == right;
        }

        // Equal keys must have equal hashes. A constant default preserves
        // custom equality semantics for domains without a hash implementation.
        // Hash collisions are always resolved with HistoryKeysEqual.
        virtual std::size_t HistoryKeyHash(const HistoryKey&) const {
            return 0;
        }

        virtual std::optional<Cost> StepCost(
            const State&,
            const Action&,
            const State&
        ) const {
            return std::nullopt;
        }

        virtual bool HasStepCost() const {
            return false;
        }

        virtual bool HasTerminalCost() const {
            return false;
        }

        virtual std::optional<Cost> TerminalCost(
            const State&
        ) const {
            return std::nullopt;
        }

        // Optional capacity hints. These do not alter semantics. They allow the
        // generic planners to reserve contiguous frontier storage when a domain
        // can estimate width cheaply (for example, a bounded grid).
        virtual std::optional<std::size_t> EstimatedInitialStateCount() const {
            return std::nullopt;
        }

        virtual std::optional<std::size_t> EstimatedMaxFrontierSize(
            std::uint64_t /*maxDepth*/
        ) const {
            return std::nullopt;
        }

        // Compatibility adapters. New performance-sensitive code should prefer
        // callbacks so it can stream values without mandatory temporary vectors.
        [[nodiscard]] std::vector<State> InitialStates() const {
            std::vector<State> result;
            if (auto n = EstimatedInitialStateCount(); n.has_value()) {
                result.reserve(*n);
            }
            ForEachInitialState([&result](const State& state) {
                result.push_back(state);
                return true;
            });
            return result;
        }

        [[nodiscard]] std::vector<Action> ApplicableActions(
            const State& state
        ) const {
            std::vector<Action> result;
            ForEachApplicableAction(state, [&result](const Action& action) {
                result.push_back(action);
                return true;
            });
            return result;
        }
    };
}
