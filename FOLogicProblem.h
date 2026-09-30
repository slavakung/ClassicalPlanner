#pragma once

#include "LogicBitset.h"
#include "PlanningProblem.h"

#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace planning
{
    // Construction-time structured atom. Runtime search uses interned bit IDs.
    struct FOAtom {
        std::string predicate;
        std::vector<std::string> arguments;

        [[nodiscard]] std::string Canonical() const {
            std::ostringstream output;
            output << predicate << '(';

            for (std::size_t i = 0; i < arguments.size(); ++i) {
                if (i != 0) {
                    output << ',';
                }
                output << arguments[i];
            }

            output << ')';
            return output.str();
        }
    };

    inline FOAtom A(
        std::string predicate,
        std::initializer_list<std::string> arguments
    ) {
        return FOAtom{
            std::move(predicate),
            std::vector<std::string>(arguments)
        };
    }

    // These examples are grounded first-order planning actions. The structured
    // predicate/argument form is preserved for model construction and display,
    // but general unification/quantifier inference is intentionally not claimed.
    struct FOActionSpec {
        std::string name;
        std::vector<std::string> parameters;
        std::vector<FOAtom> preconditions;
        std::vector<FOAtom> addEffects;
        std::vector<FOAtom> deleteEffects;
    };

    struct FOAction {
        std::string name;
        std::vector<std::string> parameters;
        DynamicBitset preconditions;
        DynamicBitset addEffects;
        DynamicBitset deleteEffects;
    };

    inline std::string ToString(const FOAction& action) {
        std::ostringstream output;
        output << action.name;

        if (!action.parameters.empty()) {
            output << '(';
            for (std::size_t i = 0; i < action.parameters.size(); ++i) {
                if (i != 0) {
                    output << ',';
                }
                output << action.parameters[i];
            }
            output << ')';
        }

        return output.str();
    }

    using FOState = DynamicBitset;

    class FOLogicProblem
        : public PlanningProblem<
              FOAction,
              FOState,
              float,
              LogicStateKey
          > {
    public:
        FOLogicProblem(
            std::vector<FOAtom> initialFacts,
            std::vector<FOAtom> goalFacts,
            std::vector<FOActionSpec> actionSpecs
        ) {
            InternAll(initialFacts, goalFacts, actionSpecs);
            initial_ = CompileFacts(initialFacts);
            goal_ = CompileFacts(goalFacts);

            actions_.reserve(actionSpecs.size());
            for (const FOActionSpec& spec : actionSpecs) {
                actions_.push_back(CompileAction(spec));
            }
        }

        void ForEachInitialState(
            const Consumer<FOState>& consumer
        ) const override {
            consumer(initial_);
        }

        bool IsGoal(const FOState& state) const override {
            return state.ContainsAll(goal_);
        }

        void ForEachApplicableAction(
            const FOState& state,
            const Consumer<FOAction>& consumer
        ) const override {
            for (const FOAction& action : actions_) {
                if (state.ContainsAll(action.preconditions) &&
                    !consumer(action)) {
                    return;
                }
            }
        }

        void ForEachSuccessor(
            const FOState& state,
            const SuccessorConsumer<FOAction, FOState>& consumer
        ) const override {
            for (const FOAction& action : actions_) {
                if (!state.ContainsAll(action.preconditions)) {
                    continue;
                }

                FOState next = state;
                next.DifferenceWith(action.deleteEffects);
                next.UnionWith(action.addEffects);

                if (!consumer(action, std::move(next))) {
                    return;
                }
            }
        }

        std::optional<FOState> Apply(
            const FOState& state,
            const FOAction& action
        ) const override {
            if (!state.ContainsAll(action.preconditions)) {
                return std::nullopt;
            }

            FOState next = state;
            next.DifferenceWith(action.deleteEffects);
            next.UnionWith(action.addEffects);
            return next;
        }

        LogicStateKey MakeHistoryKey(
            const FOState& state
        ) const override {
            return MakeLogicStateKey(state);
        }

        bool HasStepCost() const override {
            return true;
        }

        std::optional<float> StepCost(
            const FOState&,
            const FOAction&,
            const FOState&
        ) const override {
            return 1.0F;
        }

        std::optional<std::size_t> EstimatedInitialStateCount()
            const override {
            return 1;
        }

        // Immutable compiled masks for native successor enumeration.
        [[nodiscard]] const std::vector<FOAction>& CompiledActions() const noexcept {
            return actions_;
        }
        [[nodiscard]] const DynamicBitset& PositiveGoal() const noexcept {
            return goal_;
        }

    protected:
        void InternAll(
            const std::vector<FOAtom>& initialFacts,
            const std::vector<FOAtom>& goalFacts,
            const std::vector<FOActionSpec>& actionSpecs
        ) {
            for (const FOAtom& atom : initialFacts) {
                symbols_.Intern(atom.Canonical());
            }
            for (const FOAtom& atom : goalFacts) {
                symbols_.Intern(atom.Canonical());
            }
            for (const FOActionSpec& action : actionSpecs) {
                for (const FOAtom& atom : action.preconditions) {
                    symbols_.Intern(atom.Canonical());
                }
                for (const FOAtom& atom : action.addEffects) {
                    symbols_.Intern(atom.Canonical());
                }
                for (const FOAtom& atom : action.deleteEffects) {
                    symbols_.Intern(atom.Canonical());
                }
            }
        }

        DynamicBitset CompileFacts(
            const std::vector<FOAtom>& facts
        ) {
            DynamicBitset result(symbols_.Size());
            for (const FOAtom& atom : facts) {
                result.Set(symbols_.Intern(atom.Canonical()));
            }
            return result;
        }

        FOAction CompileAction(const FOActionSpec& spec) {
            return FOAction{
                spec.name,
                spec.parameters,
                CompileFacts(spec.preconditions),
                CompileFacts(spec.addEffects),
                CompileFacts(spec.deleteEffects)
            };
        }

        SymbolTable symbols_;
        FOState initial_;
        FOState goal_;
        std::vector<FOAction> actions_;
    };

    inline FOActionSpec MoveFO(
        const std::string& from,
        const std::string& to
    ) {
        return FOActionSpec{
            "Move",
            {from, to},
            {
                A("At", {"Agent", from}),
                A("Safe", {to}),
                A("Adjacent", {from, to})
            },
            {A("At", {"Agent", to})},
            {A("At", {"Agent", from})}
        };
    }

    inline FOLogicProblem MakeFOWumpusProblem() {
        std::vector<FOAtom> initial = {
            A("At", {"Agent", "S11"}),
            A("Safe", {"S11"}),
            A("Safe", {"S21"}),
            A("Safe", {"S22"}),
            A("Adjacent", {"S11", "S21"}),
            A("Adjacent", {"S21", "S11"}),
            A("Adjacent", {"S21", "S22"}),
            A("Adjacent", {"S22", "S21"}),
            A("GoldAt", {"Gold", "S22"})
        };

        std::vector<FOAtom> goal = {
            A("Have", {"Agent", "Gold"}),
            A("At", {"Agent", "S11"}),
            A("ClimbedOut", {"Agent"})
        };

        std::vector<FOActionSpec> actions = {
            MoveFO("S11", "S21"),
            MoveFO("S21", "S11"),
            MoveFO("S21", "S22"),
            MoveFO("S22", "S21"),
            {
                "Grab",
                {"Agent", "Gold", "S22"},
                {
                    A("At", {"Agent", "S22"}),
                    A("GoldAt", {"Gold", "S22"})
                },
                {A("Have", {"Agent", "Gold"})},
                {A("GoldAt", {"Gold", "S22"})}
            },
            {
                "Climb",
                {"Agent", "S11"},
                {
                    A("At", {"Agent", "S11"}),
                    A("Have", {"Agent", "Gold"})
                },
                {A("ClimbedOut", {"Agent"})},
                {}
            }
        };

        return FOLogicProblem(
            std::move(initial),
            std::move(goal),
            std::move(actions)
        );
    }
}
