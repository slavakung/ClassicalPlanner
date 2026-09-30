#include "FOLogicProblem.h"
#include "FOPlusLogicProblem.h"
#include "GridProblem.h"
#include "Planner.h"
#include "PropLogicProblem.h"
#include "SearchStorage.h"
#include "pddl/Adapter.h"
#include "pddl/GridPDDL.h"
#include "pddl/Serializable.h"
#include "pddl/Parser.h"
#include "pddl/SemanticAnalyzer.h"
#include "pddl/Writer.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory_resource>
#include <optional>
#include <vector>

namespace
{
    template<typename PlanType>
    const PlanType& First(const std::vector<PlanType>& plans) {
        assert(!plans.empty());
        return plans.front();
    }
}

int main() {
    using namespace planning;

    // Storage test: NodeId slot recycling and persistent PathId history are
    // intentionally separate concepts.
    SearchMemory memory(64 * 1024);
    PathArena<int, int> paths(memory.resource(), 8);
    NodePool<int, float> nodes(memory.resource(), 8);

    const PathId root = paths.AddRoot(10);
    const PathId child = paths.AddChild(root, 7, 11);
    assert(paths.Depth(root) == 0);
    assert(paths.Depth(child) == 1);
    assert(paths.Parent(child) == root);
    assert(paths.ActionFromParent(child).value() == 7);

    const NodeId firstNode = nodes.Allocate(123, root, 0.0F);
    nodes.Release(firstNode);
    const NodeId recycled = nodes.Allocate(456, child, 1.0F);
    assert(recycled == firstNode);
    assert(nodes.StateAt(recycled) == 456);
    nodes.Release(recycled);

    const GridProblem grid(
        4,
        4,
        GridState{0, 0},
        GridState{3, 3}
    );

    const BFSOptions bfsOptions{
        .returnMode = BFSReturnMode::firstGoal,
        .maxDepth = 20,
        .maxPlans = 1,
        .prunePathCycles = true,
        .frontierReserveHint = 32,
        .arenaChunkCapacity = 16,
        .initialArenaBytes = 64 * 1024
    };

    BFSPlanner<GridMove, GridState, float> bfs(bfsOptions);
    const auto bfsPlans = bfs.Search(grid);
    assert(First(bfsPlans).status == PlanStatus::success);
    assert(First(bfsPlans).stageLen == 6);
    assert(std::fabs(First(bfsPlans).totalCost.value() - 6.0F) < 1e-6F);

    const DFSOptions<float> dfsOptions{
        .returnMode = DFSReturnMode::firstGoal,
        .maxDepth = 20,
        .maxPlans = 1,
        .targetCost = std::nullopt,
        .prunePathCycles = true,
        .stackReserveHint = 32,
        .arenaChunkCapacity = 16,
        .initialArenaBytes = 64 * 1024
    };

    DFSPlanner<GridMove, GridState, float> dfs(dfsOptions);
    const auto dfsPlans = dfs.Search(grid);
    assert(First(dfsPlans).status == PlanStatus::success);

    RecursiveDFSPlanner<GridMove, GridState, float> recursive(dfsOptions);
    const auto recursivePlans = recursive.Search(grid);
    assert(First(recursivePlans).status == PlanStatus::success);

    const BBDFSOptions<float> bbOptions{
        .returnMode = BBDFSReturnMode::firstOptimalAfterExhaustion,
        .maxDepth = 20,
        .maxPlans = 10,
        .initialCostBound = std::nullopt,
        .prunePathCycles = true,
        .requireCostModel = true,
        .stackReserveHint = 32,
        .arenaChunkCapacity = 16,
        .initialArenaBytes = 64 * 1024
    };

    BranchAndBoundDFSPlanner<GridMove, GridState, float> bb(bbOptions);
    const auto bbPlans = bb.Search(grid);
    assert(First(bbPlans).stageLen == 6);
    assert(std::fabs(First(bbPlans).totalCost.value() - 6.0F) < 1e-6F);

    auto prop = MakePropositionalWumpusProblem();
    BFSPlanner<PropAction, PropState, float, LogicStateKey> propPlanner(
        bfsOptions
    );
    const auto propPlans = propPlanner.Search(prop);
    assert(First(propPlans).stageLen == 6);

    auto fo = MakeFOWumpusProblem();
    BFSPlanner<FOAction, FOState, float, LogicStateKey> foPlanner(bfsOptions);
    const auto foPlans = foPlanner.Search(fo);
    assert(First(foPlans).stageLen == 6);

    auto foPlus = MakeFOPlusToyProblem();
    BFSPlanner<FOAction, FOState, float, LogicStateKey> foPlusPlanner(
        bfsOptions
    );
    const auto foPlusPlans = foPlusPlanner.Search(foPlus);
    assert(First(foPlusPlans).stageLen == 2);


    // PDDL interoperability test. The parser keeps a symbolic AST, semantic
    // analysis converts it to typed IDs, the grounder instantiates action
    // schemas, and the adapter compiles the result to bitsets for the unchanged
    // generic BFS planner.
    const std::string pddlDomain = R"PDDL(
(define (domain transport-test)
  (:requirements :strips :typing :negative-preconditions :equality :action-costs)
  (:types location)
  (:predicates (at ?x - location) (connected ?x - location ?y - location) (blocked ?x - location))
  (:functions (total-cost))
  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and (at ?from) (connected ?from ?to) (not (blocked ?to)) (not (= ?from ?to)))
    :effect (and (not (at ?from)) (at ?to) (increase (total-cost) 2))))
)PDDL";

    const std::string pddlProblem = R"PDDL(
(define (problem transport-test-instance)
  (:domain transport-test)
  (:objects a b c - location)
  (:init (at a) (connected a b) (connected b c) (= (total-cost) 0))
  (:goal (and (at c) (not (blocked c))))
  (:metric minimize (total-cost)))
)PDDL";

    pddl::Parser pddlParser;
    const auto parsedDomain = pddlParser.ParseDomainText(pddlDomain);
    const auto parsedProblem = pddlParser.ParseProblemText(pddlProblem);
    assert(parsedDomain.name == "transport-test");
    assert(parsedProblem.domainName == "transport-test");

    const auto pddlIR = pddl::SemanticAnalyzer{}.Analyze(
        parsedDomain,
        parsedProblem
    );
    assert(!pddlIR.domain.actions.empty());

    auto pddlPlanningProblem = pddl::LoadGroundedProblemFromText(
        pddlDomain,
        pddlProblem
    );
    BFSPlanner<
        pddl::PDDLAction,
        pddl::PDDLState,
        float,
        LogicStateKey
    > pddlPlanner(bfsOptions);
    const auto pddlPlans = pddlPlanner.Search(pddlPlanningProblem);
    assert(First(pddlPlans).stageLen == 2);
    assert(std::fabs(First(pddlPlans).totalCost.value() - 4.0F) < 1e-6F);

    // Writer round trip: canonicalized PDDL text must parse and preserve the
    // essential domain/problem identity and solve to the same depth.
    pddl::Writer pddlWriter;
    const std::string rewrittenDomain = pddlWriter.DomainText(parsedDomain);
    const std::string rewrittenProblem = pddlWriter.ProblemText(parsedProblem);
    auto roundTripProblem = pddl::LoadGroundedProblemFromText(
        rewrittenDomain,
        rewrittenProblem
    );
    const auto roundTripPlans = pddlPlanner.Search(roundTripProblem);
    assert(First(roundTripPlans).stageLen == 2);

    // Native -> PDDL -> parser/IR/grounder -> generic planner regression test.
    // This verifies actual interoperability rather than testing only hand-written
    // PDDL input files.
    const GridProblem exportGrid(
        3,
        2,
        GridState{0, 0},
        GridState{2, 1},
        {3} // flattened cell (0,1) is blocked
    );
    const auto exportedGrid = pddl::Export(exportGrid);
    auto importedGrid = pddl::LoadGroundedProblemFromText(
        pddlWriter.DomainText(exportedGrid.domain),
        pddlWriter.ProblemText(exportedGrid.problem)
    );
    const auto importedGridPlans = pddlPlanner.Search(importedGrid);
    assert(First(importedGridPlans).stageLen == 3);
    assert(std::fabs(First(importedGridPlans).totalCost.value() - 3.0F) < 1e-6F);

    std::cout << "All planner tests passed.\n";
    return 0;
}
