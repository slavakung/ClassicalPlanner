#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace planning
{
    // The status belongs to the returned plan, not to an internal search node.
    // Keeping this in Plan.h makes the public result type independent of any
    // particular search implementation.
    enum class PlanStatus {
        inactive,
        inProgress,
        fail,
        success
    };

    template<typename Action, typename Cost = float>
    struct Plan {
        // A successful plan intentionally keeps the complete action sequence:
        // unlike temporary search state, the final result must be executable.
        std::vector<Action> actions;

        // std::optional distinguishes "this problem has no cost model" from a
        // perfectly valid plan whose numeric cost happens to be zero.
        std::optional<Cost> totalCost;

        std::uint64_t stageLen = 0;
        PlanStatus status = PlanStatus::inactive;
    };
}
