#pragma once

#include "GridProblem.h"
#include "Serializable.h"

namespace planning
{
    // ADL-visible native GridProblem -> PDDL AST adapter. It lives in namespace
    // planning (beside GridProblem), while the return type remains pddl-specific.
    [[nodiscard]] pddl::SerializablePair ToPDDL(const GridProblem& grid);
}
