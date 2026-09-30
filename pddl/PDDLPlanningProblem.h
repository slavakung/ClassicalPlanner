#pragma once

#include "Grounder.h"
#include "LogicBitset.h"
#include "PlanningProblem.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace planning::pddl
{
    struct PDDLAction {
        std::string name;
        std::vector<std::string> arguments;
        DynamicBitset positivePreconditions;
        DynamicBitset negativePreconditions;
        DynamicBitset addEffects;
        DynamicBitset deleteEffects;
        float cost = 1.0F;
    };

    inline std::string ToString(const PDDLAction& action) {
        std::ostringstream output;
        output << action.name;
        if (!action.arguments.empty()) {
            output << '(';
            for (std::size_t i = 0; i < action.arguments.size(); ++i) {
                if (i != 0) {
                    output << ',';
                }
                output << action.arguments[i];
            }
            output << ')';
        }
        return output.str();
    }

    using PDDLState = DynamicBitset;

    // Grounded PDDL is compiled once into the same compact bit operations used
    // by the native logic examples. Search algorithms therefore do not depend on
    // the parser or on PDDL syntax at all.
    class PDDLPlanningProblem final
        : public PlanningProblem<PDDLAction, PDDLState, float, LogicStateKey> {
    public:
        explicit PDDLPlanningProblem(GroundProblem groundProblem);

        void ForEachInitialState(
            const Consumer<PDDLState>& consumer
        ) const override;

        [[nodiscard]] bool IsGoal(const PDDLState& state) const override;

        void ForEachApplicableAction(
            const PDDLState& state,
            const Consumer<PDDLAction>& consumer
        ) const override;

        void ForEachSuccessor(
            const PDDLState& state,
            const SuccessorConsumer<PDDLAction, PDDLState>& consumer
        ) const override;

        [[nodiscard]] std::optional<PDDLState> Apply(
            const PDDLState& state,
            const PDDLAction& action
        ) const override;

        [[nodiscard]] LogicStateKey MakeHistoryKey(
            const PDDLState& state
        ) const override;

        [[nodiscard]] std::size_t HistoryKeyHash(const LogicStateKey& key) const override {
            return static_cast<std::size_t>(key.hash);
        }

        [[nodiscard]] bool HasStepCost() const override { return true; }

        [[nodiscard]] bool HasTerminalCost() const override {
            return source_.source.initialTotalCost != 0.0;
        }

        [[nodiscard]] std::optional<float> TerminalCost(
            const PDDLState&
        ) const override {
            if (!HasTerminalCost()) {
                return std::nullopt;
            }
            return static_cast<float>(source_.source.initialTotalCost);
        }

        [[nodiscard]] std::optional<float> StepCost(
            const PDDLState&,
            const PDDLAction& action,
            const PDDLState&
        ) const override {
            return action.cost;
        }

        [[nodiscard]] std::optional<std::size_t> EstimatedInitialStateCount()
            const override {
            return 1;
        }

        [[nodiscard]] const GroundProblem& Source() const noexcept {
            return source_;
        }

        [[nodiscard]] std::size_t GroundActionCount() const noexcept {
            return actions_.size();
        }

        // Immutable compiled masks for native successor enumeration.
        [[nodiscard]] const std::vector<PDDLAction>& CompiledActions() const noexcept {
            return actions_;
        }
        [[nodiscard]] const DynamicBitset& PositiveGoal() const noexcept {
            return positiveGoal_;
        }
        [[nodiscard]] const DynamicBitset& NegativeGoal() const noexcept {
            return negativeGoal_;
        }

    private:
        [[nodiscard]] std::string AtomName(const ir::GroundAtom& atom) const;
        void InternAtom(const ir::GroundAtom& atom);
        [[nodiscard]] DynamicBitset CompileAtoms(
            const std::vector<ir::GroundAtom>& atoms
        ) const;

        GroundProblem source_;
        SymbolTable symbols_;
        PDDLState initial_;
        DynamicBitset positiveGoal_;
        DynamicBitset negativeGoal_;
        std::vector<PDDLAction> actions_;
        std::vector<ir::Object> objectsById_;
    };
}
