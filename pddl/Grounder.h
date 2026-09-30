#pragma once

#include "ir/PlanningIR.h"

#include <string>
#include <vector>

namespace planning::pddl
{
    enum class ExecutionMode {
        groundAll,
        lazyGround,
        lifted
    };

    struct GroundAction {
        std::string name;
        std::vector<ir::ObjectId> arguments;
        std::vector<ir::GroundLiteral> preconditions;
        std::vector<ir::GroundLiteral> effects;
        double cost = 1.0;
    };

    struct GroundProblem {
        ir::Problem source;
        std::vector<GroundAction> actions;
    };

    class Grounder {
    public:
        [[nodiscard]] GroundProblem Ground(
            const ir::Problem& problem,
            ExecutionMode mode = ExecutionMode::groundAll
        ) const;
    };
}
