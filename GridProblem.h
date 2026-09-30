#pragma once

#include "PlanningProblem.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace planning
{
    enum class GridMove {
        up,
        down,
        left,
        right
    };

    struct GridState {
        std::uint32_t x = 0;
        std::uint32_t y = 0;

        friend bool operator==(
            const GridState&,
            const GridState&
        ) = default;
    };

    inline std::string ToString(GridMove move) {
        switch (move) {
            case GridMove::up:
                return "Up";
            case GridMove::down:
                return "Down";
            case GridMove::left:
                return "Left";
            case GridMove::right:
                return "Right";
        }
        return "Unknown";
    }

    // GridState is already a tiny exact history representation, so the history
    // key type is simply GridState rather than a larger complete-world object.
    class GridProblem final
        : public PlanningProblem<
              GridMove,
              GridState,
              float,
              GridState
          > {
    public:
        GridProblem(
            std::uint32_t width,
            std::uint32_t height,
            GridState start,
            GridState goal,
            std::unordered_set<std::uint64_t> obstacles = {}
        )
            : width_(width),
              height_(height),
              start_(start),
              goal_(goal),
              obstacles_(std::move(obstacles))
        {
            if (width_ == 0 || height_ == 0) {
                throw std::invalid_argument(
                    "Grid dimensions must be positive"
                );
            }
            if (!Inside(start_) || !Inside(goal_)) {
                throw std::invalid_argument(
                    "Grid start and goal must be inside the grid"
                );
            }
            if (IsObstacle(start_) || IsObstacle(goal_)) {
                throw std::invalid_argument(
                    "Grid start and goal cannot be obstacles"
                );
            }
        }

        void ForEachInitialState(
            const Consumer<GridState>& consumer
        ) const override {
            consumer(start_);
        }

        bool IsGoal(const GridState& state) const override {
            return state == goal_;
        }

        void ForEachApplicableAction(
            const GridState& state,
            const Consumer<GridMove>& consumer
        ) const override {
            for (const GridMove move : Moves()) {
                if (Apply(state, move).has_value() &&
                    !consumer(move)) {
                    return;
                }
            }
        }

        // Fused successor generation avoids the Apply-twice pattern in the
        // generic default implementation.
        void ForEachSuccessor(
            const GridState& state,
            const SuccessorConsumer<GridMove, GridState>& consumer
        ) const override {
            for (const GridMove move : Moves()) {
                auto next = Apply(state, move);
                if (next.has_value() &&
                    !consumer(move, std::move(*next))) {
                    return;
                }
            }
        }

        std::optional<GridState> Apply(
            const GridState& state,
            const GridMove& move
        ) const override {
            GridState next = state;

            switch (move) {
                case GridMove::up:
                    if (next.y == 0) {
                        return std::nullopt;
                    }
                    --next.y;
                    break;

                case GridMove::down:
                    ++next.y;
                    break;

                case GridMove::left:
                    if (next.x == 0) {
                        return std::nullopt;
                    }
                    --next.x;
                    break;

                case GridMove::right:
                    ++next.x;
                    break;
            }

            if (!Inside(next) || IsObstacle(next)) {
                return std::nullopt;
            }

            return next;
        }

        GridState MakeHistoryKey(
            const GridState& state
        ) const override {
            return state;
        }

        bool HasStepCost() const override {
            return true;
        }

        std::optional<float> StepCost(
            const GridState&,
            const GridMove&,
            const GridState&
        ) const override {
            return 1.0F;
        }

        std::optional<std::size_t> EstimatedInitialStateCount()
            const override {
            return 1;
        }

        std::optional<std::size_t> EstimatedMaxFrontierSize(
            std::uint64_t
        ) const override {
            // A conservative finite-state upper bound. Path-based cycle pruning
            // can revisit a cell through different branches, so this is only a
            // reservation hint; vector growth remains correct if exceeded.
            return static_cast<std::size_t>(width_) * height_ -
                   obstacles_.size();
        }

        [[nodiscard]] std::uint32_t Width() const noexcept {
            return width_;
        }

        [[nodiscard]] std::uint32_t Height() const noexcept {
            return height_;
        }

        [[nodiscard]] GridState Start() const noexcept {
            return start_;
        }

        [[nodiscard]] GridState Goal() const noexcept {
            return goal_;
        }

        [[nodiscard]] bool IsBlocked(const GridState& state) const {
            return IsObstacle(state);
        }

        [[nodiscard]] std::string Render() const {
            std::ostringstream output;

            for (std::uint32_t y = 0; y < height_; ++y) {
                for (std::uint32_t x = 0; x < width_; ++x) {
                    const GridState cell{x, y};
                    char marker = '.';

                    if (cell == start_) {
                        marker = 'S';
                    } else if (cell == goal_) {
                        marker = 'G';
                    } else if (IsObstacle(cell)) {
                        marker = '#';
                    }

                    output << marker;
                }
                output << '\n';
            }

            return output.str();
        }

        static GridProblem Random(
            std::uint32_t width,
            std::uint32_t height,
            std::size_t obstacleCount,
            std::uint32_t seed
        ) {
            ValidateRandomDimensions(width, height);

            const GridState start{0, 0};
            const GridState goal{width - 1, height - 1};
            const std::size_t maxObstacles =
                static_cast<std::size_t>(width) * height - 2;
            obstacleCount = std::min(obstacleCount, maxObstacles);

            std::mt19937 generator(seed);
            const auto obstacles = RandomObstacles(
                width,
                height,
                obstacleCount,
                generator,
                {Flatten(start, width), Flatten(goal, width)}
            );

            return GridProblem(
                width,
                height,
                start,
                goal,
                obstacles
            );
        }

        // Generates an instance with a guaranteed monotone start-to-goal path.
        // This is useful for demonstrations where a random unsolvable instance
        // would be distracting. Obstacles are placed only outside the corridor.
        static GridProblem RandomSolvable(
            std::uint32_t width,
            std::uint32_t height,
            std::size_t obstacleCount,
            std::uint32_t seed
        ) {
            ValidateRandomDimensions(width, height);

            const GridState start{0, 0};
            const GridState goal{width - 1, height - 1};

            std::unordered_set<std::uint64_t> protectedCells;
            for (std::uint32_t x = 0; x < width; ++x) {
                protectedCells.insert(Flatten(GridState{x, 0}, width));
            }
            for (std::uint32_t y = 0; y < height; ++y) {
                protectedCells.insert(
                    Flatten(GridState{width - 1, y}, width)
                );
            }

            const std::size_t totalCells =
                static_cast<std::size_t>(width) * height;
            const std::size_t maxObstacles =
                totalCells - protectedCells.size();
            obstacleCount = std::min(obstacleCount, maxObstacles);

            std::mt19937 generator(seed);
            const auto obstacles = RandomObstacles(
                width,
                height,
                obstacleCount,
                generator,
                std::move(protectedCells)
            );

            return GridProblem(
                width,
                height,
                start,
                goal,
                obstacles
            );
        }

    private:
        [[nodiscard]] static constexpr std::array<GridMove, 4> Moves() {
            // constexpr creates the fixed action vocabulary at compile time.
            return {
                GridMove::up,
                GridMove::down,
                GridMove::left,
                GridMove::right
            };
        }

        static void ValidateRandomDimensions(
            std::uint32_t width,
            std::uint32_t height
        ) {
            if (width == 0 || height == 0 ||
                static_cast<std::uint64_t>(width) * height < 2) {
                throw std::invalid_argument(
                    "Random grid needs at least two cells"
                );
            }
        }

        static std::unordered_set<std::uint64_t> RandomObstacles(
            std::uint32_t width,
            std::uint32_t height,
            std::size_t obstacleCount,
            std::mt19937& generator,
            std::unordered_set<std::uint64_t> protectedCells
        ) {
            std::uniform_int_distribution<std::uint32_t> xDist(
                0,
                width - 1
            );
            std::uniform_int_distribution<std::uint32_t> yDist(
                0,
                height - 1
            );

            std::unordered_set<std::uint64_t> obstacles;
            while (obstacles.size() < obstacleCount) {
                const GridState candidate{
                    xDist(generator),
                    yDist(generator)
                };
                const std::uint64_t flat = Flatten(candidate, width);

                if (!protectedCells.contains(flat)) {
                    obstacles.insert(flat);
                }
            }

            return obstacles;
        }

        static std::uint64_t Flatten(
            const GridState& state,
            std::uint32_t width
        ) {
            return static_cast<std::uint64_t>(state.y) * width + state.x;
        }

        [[nodiscard]] bool Inside(const GridState& state) const {
            return state.x < width_ && state.y < height_;
        }

        [[nodiscard]] bool IsObstacle(const GridState& state) const {
            return obstacles_.contains(Flatten(state, width_));
        }

        std::uint32_t width_;
        std::uint32_t height_;
        GridState start_;
        GridState goal_;
        std::unordered_set<std::uint64_t> obstacles_;
    };
}
