#pragma once

#include "LogicBitset.h"
#include "PlanningProblem.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace planning
{
    using PropState = DynamicBitset;

    // Human-readable construction form. It is compiled once into integer-bitset
    // actions below; runtime search does not perform set<string> operations.
    struct PropActionSpec {
        std::string name;
        std::vector<std::string> preconditions;
        std::vector<std::string> addEffects;
        std::vector<std::string> deleteEffects;
    };

    struct PropAction {
        std::string name;
        DynamicBitset preconditions;
        DynamicBitset addEffects;
        DynamicBitset deleteEffects;
    };

    inline std::string ToString(const PropAction& action) {
        return action.name;
    }

    class PropLogicProblem final
        : public PlanningProblem<
              PropAction,
              PropState,
              float,
              LogicStateKey
          > {
    public:
        PropLogicProblem(
            std::vector<std::string> initialFacts,
            std::vector<std::string> goalFacts,
            std::vector<PropActionSpec> actionSpecs
        ) {
            // Pass one: establish a fixed vocabulary so all bitsets have the
            // same word layout. Pass two: compile states and action masks.
            InternAll(initialFacts, goalFacts, actionSpecs);
            initial_ = CompileFacts(initialFacts);
            goal_ = CompileFacts(goalFacts);

            actions_.reserve(actionSpecs.size());
            for (const PropActionSpec& spec : actionSpecs) {
                actions_.push_back(CompileAction(spec));
            }
        }

        void ForEachInitialState(
            const Consumer<PropState>& consumer
        ) const override {
            consumer(initial_);
        }

        bool IsGoal(const PropState& state) const override {
            return state.ContainsAll(goal_);
        }

        void ForEachApplicableAction(
            const PropState& state,
            const Consumer<PropAction>& consumer
        ) const override {
            for (const PropAction& action : actions_) {
                if (state.ContainsAll(action.preconditions) &&
                    !consumer(action)) {
                    return;
                }
            }
        }

        void ForEachSuccessor(
            const PropState& state,
            const SuccessorConsumer<PropAction, PropState>& consumer
        ) const override {
            for (const PropAction& action : actions_) {
                if (!state.ContainsAll(action.preconditions)) {
                    continue;
                }

                PropState next = state;
                next.DifferenceWith(action.deleteEffects);
                next.UnionWith(action.addEffects);

                if (!consumer(action, std::move(next))) {
                    return;
                }
            }
        }

        std::optional<PropState> Apply(
            const PropState& state,
            const PropAction& action
        ) const override {
            if (!state.ContainsAll(action.preconditions)) {
                return std::nullopt;
            }

            PropState next = state;
            next.DifferenceWith(action.deleteEffects);
            next.UnionWith(action.addEffects);
            return next;
        }

        LogicStateKey MakeHistoryKey(
            const PropState& state
        ) const override {
            return MakeLogicStateKey(state);
        }

        bool HasStepCost() const override {
            return true;
        }

        std::optional<float> StepCost(
            const PropState&,
            const PropAction&,
            const PropState&
        ) const override {
            return 1.0F;
        }

        std::optional<std::size_t> EstimatedInitialStateCount()
            const override {
            return 1;
        }

    // Read-only compiled catalogue for native successor enumeration.
        // No mutable device cache is stored in a problem shared by workers.
        [[nodiscard]] const std::vector<PropAction>& CompiledActions() const noexcept {
            return actions_;
        }

        [[nodiscard]] const DynamicBitset& PositiveGoal() const noexcept {
            return goal_;
        }

    private:
        void InternAll(
            const std::vector<std::string>& initialFacts,
            const std::vector<std::string>& goalFacts,
            const std::vector<PropActionSpec>& actionSpecs
        ) {
            for (const std::string& fact : initialFacts) {
                symbols_.Intern(fact);
            }
            for (const std::string& fact : goalFacts) {
                symbols_.Intern(fact);
            }
            for (const PropActionSpec& spec : actionSpecs) {
                for (const std::string& fact : spec.preconditions) {
                    symbols_.Intern(fact);
                }
                for (const std::string& fact : spec.addEffects) {
                    symbols_.Intern(fact);
                }
                for (const std::string& fact : spec.deleteEffects) {
                    symbols_.Intern(fact);
                }
            }
        }

        DynamicBitset CompileFacts(
            const std::vector<std::string>& facts
        ) {
            DynamicBitset result(symbols_.Size());
            for (const std::string& fact : facts) {
                result.Set(symbols_.Intern(fact));
            }
            return result;
        }

        PropAction CompileAction(const PropActionSpec& spec) {
            return PropAction{
                spec.name,
                CompileFacts(spec.preconditions),
                CompileFacts(spec.addEffects),
                CompileFacts(spec.deleteEffects)
            };
        }

        SymbolTable symbols_;
        PropState initial_;
        PropState goal_;
        std::vector<PropAction> actions_;
    };

    // Small deterministic Wumpus-world planning abstraction. As in the previous
    // revision, safety facts are supplied to the classical planner; this is not
    // the full partially observable knowledge-based Wumpus agent.
    inline PropLogicProblem MakePropositionalWumpusProblem() {
        std::vector<std::string> initial = {
            "At_1_1",
            "Safe_1_1",
            "Safe_2_1",
            "Safe_2_2",
            "GoldAt_2_2"
        };

        std::vector<std::string> goal = {
            "HaveGold",
            "At_1_1",
            "ClimbedOut"
        };

        std::vector<PropActionSpec> actions = {
            {
                "Move_1_1_to_2_1",
                {"At_1_1", "Safe_2_1"},
                {"At_2_1"},
                {"At_1_1"}
            },
            {
                "Move_2_1_to_1_1",
                {"At_2_1", "Safe_1_1"},
                {"At_1_1"},
                {"At_2_1"}
            },
            {
                "Move_2_1_to_2_2",
                {"At_2_1", "Safe_2_2"},
                {"At_2_2"},
                {"At_2_1"}
            },
            {
                "Move_2_2_to_2_1",
                {"At_2_2", "Safe_2_1"},
                {"At_2_1"},
                {"At_2_2"}
            },
            {
                "GrabGold",
                {"At_2_2", "GoldAt_2_2"},
                {"HaveGold"},
                {"GoldAt_2_2"}
            },
            {
                "ClimbOut",
                {"At_1_1", "HaveGold"},
                {"ClimbedOut"},
                {}
            }
        };

        return PropLogicProblem(
            std::move(initial),
            std::move(goal),
            std::move(actions)
        );
    }
}
