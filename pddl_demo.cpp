#include "Planner.h"
#include "pddl/Adapter.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string_view>

namespace
{
    bool PrintPlan(
        const planning::Plan<planning::pddl::PDDLAction, float>& plan
    ) {
        std::cout << "Plan found: depth=" << plan.stageLen;
        if (plan.totalCost.has_value()) {
            std::cout << ", cost=" << *plan.totalCost;
        }
        std::cout << "\n";

        for (std::size_t i = 0; i < plan.actions.size(); ++i) {
            std::cout << "  " << i + 1 << ". "
                      << planning::pddl::ToString(plan.actions[i])
                      << "\n";
        }
        return false;
    }
}

int main(int argc, char** argv) {
    using namespace planning;
    using namespace planning::pddl;

    if (argc != 3) {
        std::cerr
            << "Usage: pddl_planner_demo <domain.pddl> <problem.pddl>\n";
        return EXIT_FAILURE;
    }

    try {
        PDDLPlanningProblem problem = LoadGroundedProblem(argv[1], argv[2]);

        std::cout << "Grounded actions: "
                  << problem.GroundActionCount()
                  << "\n";

        BFSPlanner<PDDLAction, PDDLState, float, LogicStateKey> planner(
            BFSOptions{
                .returnMode = BFSReturnMode::firstGoal,
                .maxDepth = 100,
                .maxPlans = 1,
                .prunePathCycles = true,
                .frontierReserveHint = 1024,
                .arenaChunkCapacity = 1024,
                .initialArenaBytes = 1024 * 1024
            }
        );

        bool emitted = false;
        planner.Search(problem, [&](const Plan<PDDLAction, float>& plan) {
            emitted = true;
            return PrintPlan(plan);
        });

        if (!emitted) {
            std::cout << "No plan found within the configured depth bound.\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
