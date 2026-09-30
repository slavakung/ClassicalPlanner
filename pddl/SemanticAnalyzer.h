#pragma once

#include "Ast.h"
#include "ir/PlanningIR.h"

#include <string>

namespace planning::pddl
{
    class SemanticAnalyzer {
    public:
        // Converts the symbolic parser AST into a typed, ID-based intermediate
        // representation. Unsupported PDDL requirements are rejected here rather
        // than being silently approximated.
        [[nodiscard]] ir::Problem Analyze(
            const Domain& domain,
            const Problem& problem
        ) const;
    };
}
