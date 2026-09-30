#pragma once

#include "Ast.h"

#include <concepts>
#include <utility>

namespace planning::pddl
{
    struct SerializablePair {
        Domain domain;
        Problem problem;
    };

    // The concept deliberately relies on an unqualified ToPDDL(value) call so
    // domain adapters can be defined beside their native problem types and found
    // by argument-dependent lookup (ADL). Generic exporter code need not know
    // about GridProblem or future scheduling domains.
    template<typename ProblemType>
    concept PDDLSerializableProblem = requires(const ProblemType& problem) {
        { ToPDDL(problem) } -> std::same_as<SerializablePair>;
    };

    template<PDDLSerializableProblem ProblemType>
    [[nodiscard]] SerializablePair Export(const ProblemType& problem) {
        return ToPDDL(problem);
    }
}
