#include "pddl/GridPDDL.h"

#include <string>

namespace planning
{
    namespace
    {
        std::string CellName(std::uint32_t x, std::uint32_t y) {
            return "c" + std::to_string(x) + "_" + std::to_string(y);
        }
    }

    pddl::SerializablePair ToPDDL(const GridProblem& grid) {
        using namespace pddl;

        Domain domain;
        domain.name = "native-grid";
        domain.requirements = {
            Requirement::strips,
            Requirement::typing,
            Requirement::negativePreconditions,
            Requirement::actionCosts
        };
        domain.types = {
            TypedName{"object", ""},
            TypedName{"location", "object"}
        };
        domain.predicates = {
            PredicateDecl{"at", {TypedName{"?x", "location"}}},
            PredicateDecl{
                "adjacent",
                {
                    TypedName{"?from", "location"},
                    TypedName{"?to", "location"}
                }
            },
            PredicateDecl{"blocked", {TypedName{"?x", "location"}}}
        };
        domain.actions = {
            ActionSchema{
                .name = "move",
                .parameters = {
                    TypedName{"?from", "location"},
                    TypedName{"?to", "location"}
                },
                .preconditions = {
                    Literal{"at", {"?from"}, false},
                    Literal{"adjacent", {"?from", "?to"}, false},
                    Literal{"blocked", {"?to"}, true}
                },
                .effects = {
                    Literal{"at", {"?from"}, true},
                    Literal{"at", {"?to"}, false}
                },
                .actionCost = 1.0,
                .hasExplicitActionCost = true,
                .staticActionCosts = {}
            }
        };

        Problem problem;
        problem.name = "native-grid-instance";
        problem.domainName = domain.name;
        problem.minimizeTotalCost = true;

        for (std::uint32_t y = 0; y < grid.Height(); ++y) {
            for (std::uint32_t x = 0; x < grid.Width(); ++x) {
                problem.objects.push_back(
                    TypedName{CellName(x, y), "location"}
                );

                const GridState cell{x, y};
                if (grid.IsBlocked(cell)) {
                    problem.initialFacts.push_back(
                        Literal{"blocked", {CellName(x, y)}, false}
                    );
                }

                constexpr int dx[] = {1, -1, 0, 0};
                constexpr int dy[] = {0, 0, 1, -1};
                for (int direction = 0; direction < 4; ++direction) {
                    const int nx = static_cast<int>(x) + dx[direction];
                    const int ny = static_cast<int>(y) + dy[direction];
                    if (nx >= 0 && ny >= 0 &&
                        nx < static_cast<int>(grid.Width()) &&
                        ny < static_cast<int>(grid.Height())) {
                        problem.initialFacts.push_back(
                            Literal{
                                "adjacent",
                                {
                                    CellName(x, y),
                                    CellName(
                                        static_cast<std::uint32_t>(nx),
                                        static_cast<std::uint32_t>(ny)
                                    )
                                },
                                false
                            }
                        );
                    }
                }
            }
        }

        problem.initialFacts.push_back(
            Literal{
                "at",
                {CellName(grid.Start().x, grid.Start().y)},
                false
            }
        );
        problem.goal.push_back(
            Literal{
                "at",
                {CellName(grid.Goal().x, grid.Goal().y)},
                false
            }
        );

        return SerializablePair{
            std::move(domain),
            std::move(problem)
        };
    }
}
