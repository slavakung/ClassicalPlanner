#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace planning
{
    // constexpr functions may be evaluated either at compile time or runtime.
    // This helper is intentionally iterative: modern constexpr no longer needs
    // C++11-style recursive formulations for loops and local mutation.
    [[nodiscard]] constexpr std::size_t SaturatingGeometricNodeBound(
        std::size_t branchingFactor,
        std::uint64_t maxDepth
    ) noexcept {
        if (branchingFactor == 0) {
            return 1;
        }

        std::size_t level = 1;
        std::size_t total = 1;
        constexpr std::size_t max =
            std::numeric_limits<std::size_t>::max();

        for (std::uint64_t depth = 1;
             depth <= maxDepth;
             ++depth) {
            if (level > max / branchingFactor) {
                return max;
            }
            level *= branchingFactor;

            if (total > max - level) {
                return max;
            }
            total += level;
        }

        return total;
    }
}
