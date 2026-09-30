#pragma once

#include "FOLogicProblem.h"

namespace planning
{
    // A small reified-predicate extension: predicates/rules can themselves be
    // mentioned as objects in facts. This demonstrates higher-order-flavoured
    // representation without claiming a complete higher-order theorem prover.
    class FOPlusLogicProblem final : public FOLogicProblem {
    public:
        using FOLogicProblem::FOLogicProblem;
    };

    inline FOPlusLogicProblem MakeFOPlusToyProblem() {
        std::vector<FOAtom> initial = {
            A("At", {"Robot", "A"}),
            A("Connected", {"A", "B"}),
            A("PredicateObject", {"Safe"}),
            A("Holds", {"Safe", "A"}),
            A("Holds", {"Safe", "B"})
        };

        std::vector<FOAtom> goal = {
            A("At", {"Robot", "B"}),
            A("UsedRule", {"SafeTransitRule"})
        };

        std::vector<FOActionSpec> actions = {
            {
                "InstantiateRule",
                {"SafeTransitRule", "Safe"},
                {A("PredicateObject", {"Safe"})},
                {A("RuleReady", {"SafeTransitRule", "Safe"})},
                {}
            },
            {
                "MoveByPredicateRule",
                {"Robot", "A", "B", "Safe"},
                {
                    A("At", {"Robot", "A"}),
                    A("Connected", {"A", "B"}),
                    A("Holds", {"Safe", "A"}),
                    A("Holds", {"Safe", "B"}),
                    A("RuleReady", {"SafeTransitRule", "Safe"})
                },
                {
                    A("At", {"Robot", "B"}),
                    A("UsedRule", {"SafeTransitRule"})
                },
                {A("At", {"Robot", "A"})}
            }
        };

        return FOPlusLogicProblem(
            std::move(initial),
            std::move(goal),
            std::move(actions)
        );
    }
}
