#include "pddl/PDDLPlanningProblem.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace planning::pddl
{
    PDDLPlanningProblem::PDDLPlanningProblem(GroundProblem groundProblem)
        : source_(std::move(groundProblem))
    {
        const auto validCost = [](double cost) {
            return std::isfinite(cost) && cost >= 0.0 &&
                cost <= static_cast<double>(std::numeric_limits<float>::max()) &&
                (cost == 0.0 || static_cast<float>(cost) > 0.0F);
        };
        if (!validCost(source_.source.initialTotalCost)) {
            throw std::invalid_argument("invalid PDDL initial total-cost");
        }
        for (const GroundAction& action : source_.actions) {
            if (!validCost(action.cost)) {
                throw std::invalid_argument("invalid grounded PDDL action cost");
            }
        }
        objectsById_.reserve(
            source_.source.domain.constants.size() +
            source_.source.objects.size()
        );
        objectsById_.insert(
            objectsById_.end(),
            source_.source.domain.constants.begin(),
            source_.source.domain.constants.end()
        );
        objectsById_.insert(
            objectsById_.end(),
            source_.source.objects.begin(),
            source_.source.objects.end()
        );

        for (const ir::GroundAtom& atom : source_.source.initialFacts) {
            InternAtom(atom);
        }
        for (const ir::GroundLiteral& goal : source_.source.goal) {
            if (goal.atom.predicate != ir::EqualityPredicate) {
                InternAtom(goal.atom);
            }
        }
        for (const GroundAction& action : source_.actions) {
            for (const ir::GroundLiteral& literal : action.preconditions) {
                InternAtom(literal.atom);
            }
            for (const ir::GroundLiteral& literal : action.effects) {
                InternAtom(literal.atom);
            }
        }

        initial_ = CompileAtoms(source_.source.initialFacts);
        positiveGoal_.Resize(symbols_.Size());
        negativeGoal_.Resize(symbols_.Size());

        for (const ir::GroundLiteral& goal : source_.source.goal) {
            if (goal.atom.predicate == ir::EqualityPredicate) {
                const bool equal = goal.atom.arguments.at(0) == goal.atom.arguments.at(1);
                const bool satisfied = goal.negated ? !equal : equal;
                if (!satisfied) {
                    // A false interpreted goal is a valid unsolvable problem.
                    // Encode contradiction in the masks so symbolic planners
                    // and packed adapters share the same goal semantics.
                    const auto impossible = symbols_.Intern("@false-equality-goal");
                    positiveGoal_.Set(impossible);
                    negativeGoal_.Set(impossible);
                }
                continue;
            }

            const auto id = symbols_.Intern(AtomName(goal.atom));
            (goal.negated ? negativeGoal_ : positiveGoal_).Set(id);
        }

        actions_.reserve(source_.actions.size());
        for (const GroundAction& sourceAction : source_.actions) {
            PDDLAction action;
            action.name = sourceAction.name;
            action.cost = static_cast<float>(sourceAction.cost);
            action.positivePreconditions.Resize(symbols_.Size());
            action.negativePreconditions.Resize(symbols_.Size());
            action.addEffects.Resize(symbols_.Size());
            action.deleteEffects.Resize(symbols_.Size());

            for (ir::ObjectId object : sourceAction.arguments) {
                action.arguments.push_back(objectsById_.at(object).name);
            }

            for (const ir::GroundLiteral& literal : sourceAction.preconditions) {
                const auto id = symbols_.Intern(AtomName(literal.atom));
                (literal.negated
                    ? action.negativePreconditions
                    : action.positivePreconditions).Set(id);
            }
            for (const ir::GroundLiteral& literal : sourceAction.effects) {
                const auto id = symbols_.Intern(AtomName(literal.atom));
                (literal.negated
                    ? action.deleteEffects
                    : action.addEffects).Set(id);
            }
            actions_.push_back(std::move(action));
        }
    }

    void PDDLPlanningProblem::ForEachInitialState(
        const Consumer<PDDLState>& consumer
    ) const {
        consumer(initial_);
    }

    bool PDDLPlanningProblem::IsGoal(const PDDLState& state) const {
        return state.ContainsAll(positiveGoal_) &&
               !state.Intersects(negativeGoal_);
    }

    void PDDLPlanningProblem::ForEachApplicableAction(
        const PDDLState& state,
        const Consumer<PDDLAction>& consumer
    ) const {
        for (const PDDLAction& action : actions_) {
            if (state.ContainsAll(action.positivePreconditions) &&
                !state.Intersects(action.negativePreconditions) &&
                !consumer(action)) {
                return;
            }
        }
    }

    void PDDLPlanningProblem::ForEachSuccessor(
        const PDDLState& state,
        const SuccessorConsumer<PDDLAction, PDDLState>& consumer
    ) const {
        for (const PDDLAction& action : actions_) {
            if (!state.ContainsAll(action.positivePreconditions) ||
                state.Intersects(action.negativePreconditions)) {
                continue;
            }

            PDDLState next = state;
            next.DifferenceWith(action.deleteEffects);
            next.UnionWith(action.addEffects);
            if (!consumer(action, std::move(next))) {
                return;
            }
        }
    }

    std::optional<PDDLState> PDDLPlanningProblem::Apply(
        const PDDLState& state,
        const PDDLAction& action
    ) const {
        if (!state.ContainsAll(action.positivePreconditions) ||
            state.Intersects(action.negativePreconditions)) {
            return std::nullopt;
        }

        PDDLState next = state;
        next.DifferenceWith(action.deleteEffects);
        next.UnionWith(action.addEffects);
        return next;
    }

    LogicStateKey PDDLPlanningProblem::MakeHistoryKey(
        const PDDLState& state
    ) const {
        return MakeLogicStateKey(state);
    }

    std::string PDDLPlanningProblem::AtomName(
        const ir::GroundAtom& atom
    ) const {
        const ir::Predicate& predicate =
            source_.source.domain.predicates.at(atom.predicate);

        std::ostringstream output;
        output << predicate.name << '(';
        for (std::size_t i = 0; i < atom.arguments.size(); ++i) {
            if (i != 0) {
                output << ',';
            }
            output << objectsById_.at(atom.arguments[i]).name;
        }
        output << ')';
        return output.str();
    }

    void PDDLPlanningProblem::InternAtom(const ir::GroundAtom& atom) {
        symbols_.Intern(AtomName(atom));
    }

    DynamicBitset PDDLPlanningProblem::CompileAtoms(
        const std::vector<ir::GroundAtom>& atoms
    ) const {
        DynamicBitset result(symbols_.Size());
        // All atoms were interned during the constructor's first compilation
        // pass. Find() is non-mutating, so compiling a state cannot accidentally
        // change the bit numbering used by existing actions.
        for (const ir::GroundAtom& atom : atoms) {
            const auto id = symbols_.Find(AtomName(atom));
            if (!id.has_value()) {
                throw std::logic_error("internal PDDL symbol-table mismatch");
            }
            result.Set(*id);
        }
        return result;
    }
}
