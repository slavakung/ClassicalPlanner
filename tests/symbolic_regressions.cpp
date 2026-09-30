#include "Planner.h"
#include "pddl/Adapter.h"
#include "symbolic/ClassicalPlanners.h"
#include "symbolic/CSPPlanner.h"
#include "symbolic/PartialOrderPlanner.h"
#include "tests/TestSupport.h"

#include <iostream>
#include <optional>
#include <random>

using namespace planning;
using namespace planning::pddl;
using namespace planning::symbolic;
using test::Require;

namespace {
std::optional<std::size_t> BFSLength(const PDDLPlanningProblem& problem, std::size_t depth) {
    BFSOptions options;
    options.maxDepth = depth;
    BFSPlanner<PDDLAction, PDDLState, float, LogicStateKey> planner(options);
    std::optional<std::size_t> length;
    planner.Search(problem, [&](const auto& plan) { length = plan.actions.size(); return false; });
    return length;
}

void Compare(const PDDLPlanningProblem& problem, std::size_t depth, const std::string& name) {
    const auto reference = BFSLength(problem, depth);
    const auto sat = SATPlan(problem, {depth});
    const auto regression = Regression(problem, {depth});
    const auto graph = Graphplan(problem, {depth});
    const auto csp = CSPPlan(problem, {depth});
    Require((sat.status == PlanStatus::success) == reference.has_value(), name + " SAT agrees with BFS");
    Require((regression.status == PlanStatus::success) == reference.has_value(), name + " regression agrees with BFS");
    Require((csp.status == PlanStatus::success) == reference.has_value(), name + " CSP agrees with BFS");
    if (reference) {
        Require(sat.actions.size() == *reference, name + " SAT finds minimum sequential length");
        Require(regression.actions.size() == *reference, name + " regression finds minimum sequential length");
        Require(csp.actions.size() == *reference, name + " CSP finds minimum sequential length");
        Require(graph.status == PlanStatus::success, name + " Graphplan reaches BFS-reachable goals");
    } else if (depth >= 7) {
        // Random tasks have 3 facts: every reachable state has a simple path of
        // at most 7 actions, so exhaustive BFS failure proves unreachable here.
        Require(graph.status == PlanStatus::fail, name + " Graphplan rejects unreachable goal");
    }
}

GroundProblem RandomTask(std::mt19937& rng, std::size_t facts = 3, std::size_t actions = 5) {
    GroundProblem task;
    task.source.domain.types.push_back({0, "object", ir::InvalidType});
    for (std::size_t f = 0; f < facts; ++f) {
        const auto id = static_cast<ir::PredicateId>(f);
        task.source.domain.predicates.push_back({id, "p" + std::to_string(f), {}});
        if (rng() % 2) task.source.initialFacts.push_back({id, {}});
        const auto requirement = rng() % 3;
        if (requirement) task.source.goal.push_back({{id, {}}, requirement == 2});
    }
    for (std::size_t a = 0; a < actions; ++a) {
        GroundAction action;
        action.name = "a" + std::to_string(a);
        action.cost = static_cast<double>(rng() % 4);
        for (std::size_t f = 0; f < facts; ++f) {
            const auto id = static_cast<ir::PredicateId>(f);
            const auto pre = rng() % 3;
            if (pre) action.preconditions.push_back({{id, {}}, pre == 2});
            const auto effect = rng() % 4;
            if (effect == 1 || effect == 3) action.effects.push_back({{id, {}}, false});
            if (effect == 2 || effect == 3) action.effects.push_back({{id, {}}, true});
        }
        task.actions.push_back(std::move(action));
    }
    return task;
}

void RegressionInvariants() {
    const auto hanoi = LoadGroundedProblemFromText(R"((define (domain hanoi)
        (:requirements :strips :typing)
        (:types support - object disk peg - support)
        (:predicates (on ?d - disk ?s - support) (clear ?s - support)
            (smaller ?d - disk ?s - support))
        (:action move :parameters (?d - disk ?from - support ?to - support)
            :precondition (and (on ?d ?from) (clear ?d) (clear ?to) (smaller ?d ?to))
            :effect (and (on ?d ?to) (clear ?from)
                (not (on ?d ?from)) (not (clear ?to))))))",
        R"((define (problem hanoi-3) (:domain hanoi)
        (:objects d1 d2 d3 - disk pa pb pc - peg)
        (:init (on d1 d2) (on d2 d3) (on d3 pa) (clear d1) (clear pb) (clear pc)
            (smaller d1 d2) (smaller d1 d3) (smaller d1 pa) (smaller d1 pb) (smaller d1 pc)
            (smaller d2 d3) (smaller d2 pa) (smaller d2 pb) (smaller d2 pc)
            (smaller d3 pa) (smaller d3 pb) (smaller d3 pc))
        (:goal (and (on d1 d2) (on d2 d3) (on d3 pc)))))");
    const auto reference = BFSLength(hanoi, 7);
    const auto plan = Regression(hanoi, {7});
    Require(reference == 7 && plan.status == PlanStatus::success && plan.actions.size() == 7,
            "regression handles Hanoi location invariants and finds the BFS optimum");
    Require(Regression(hanoi, {6}).status == PlanStatus::fail,
            "regression invariant pruning preserves the sequential depth bound");

    const auto exclusive = LoadGroundedProblemFromText(R"((define (domain exclusive)
        (:predicates (left) (right))
        (:action go-right :precondition (left) :effect (and (right) (not (left))))
        (:action go-left :precondition (right) :effect (and (left) (not (right))))))",
        "(define (problem p) (:domain exclusive) (:init (left)) (:goal (and (left) (right))))");
    Require(Regression(exclusive).status == PlanStatus::fail,
            "regression rejects individually reachable but mutually exclusive goals");

    // The pair table is deliberately capped. Large fact catalogues must fall
    // back to complete regression rather than using an unfinished relaxation.
    GroundProblem large;
    large.source.domain.types.push_back({0, "object", ir::InvalidType});
    for (std::size_t f = 0; f < 1025; ++f) {
        large.source.domain.predicates.push_back({static_cast<ir::PredicateId>(f),
                                                  "p" + std::to_string(f), {}});
        if (f < 1024) large.source.initialFacts.push_back({static_cast<ir::PredicateId>(f), {}});
    }
    large.source.goal.push_back({{1024, {}}, false});
    GroundAction finish;
    finish.name = "finish";
    finish.preconditions.push_back({{0, {}}, false});
    finish.effects.push_back({{1024, {}}, false});
    large.actions.push_back(std::move(finish));
    const auto fallback = Regression(PDDLPlanningProblem(large), {1});
    Require(fallback.status == PlanStatus::success && fallback.actions.size() == 1,
            "regression remains complete when pair preprocessing is skipped");
}

void Examples() {
    const std::string flashlight = R"((define (domain d) (:predicates (closed) (b1) (b2))
        (:action open :precondition (closed) :effect (not (closed)))
        (:action insert1 :precondition (not (closed)) :effect (b1))
        (:action insert2 :precondition (not (closed)) :effect (b2))
        (:action close :precondition (and (not (closed)) (b1) (b2)) :effect (closed))))";
    auto p = LoadGroundedProblemFromText(flashlight,
        "(define (problem p) (:domain d) (:init (closed)) (:goal (and (closed) (b1) (b2))))");
    Compare(p, 4, "flashlight");
    Require(Graphplan(p, {2}).status == PlanStatus::fail, "Graphplan respects interference mutex");
    Require(Graphplan(p, {3}).actions.size() == 4, "Graphplan combines independent actions in one layer");
    Require(SATPlan(p, {3}).status == PlanStatus::fail, "SAT serial horizon excludes parallel shortcuts");
    Require(CSPPlan(p, {3}).status == PlanStatus::fail, "CSP serial horizon excludes parallel shortcuts");
    Require(Regression(p, {3}).status == PlanStatus::fail, "regression obeys sequential depth bound");
    Require(PartialOrder(p, {.maxActions = 4}).status == PlanStatus::success,
            "partial-order planner resolves cap threats");
    Require(PartialOrder(p, {.maxActions = 3}).status == PlanStatus::fail,
            "partial-order planner enforces occurrence bound");

    const auto threats = LoadGroundedProblemFromText(R"((define (domain d) (:predicates (ready) (p) (q))
        (:action make-p :precondition (ready) :effect (and (p) (not (q))))
        (:action make-q :precondition (ready) :effect (and (q) (not (p))))))",
        "(define (problem p) (:domain d) (:init (ready)) (:goal (and (p) (q))))");
    Compare(threats, 7, "inconsistent effects");

    const auto negatives = LoadGroundedProblemFromText(R"((define (domain d) (:predicates (p) (q))
        (:action clear :precondition (p) :effect (not (p)))
        (:action set :precondition (not (p)) :effect (q))))",
        "(define (problem p) (:domain d) (:init (p)) (:goal (and (not (p)) (q))))");
    Compare(negatives, 3, "negative goals");

    const auto noop = LoadGroundedProblemFromText("(define (domain d) (:predicates (p)))",
        "(define (problem p) (:domain d) (:init) (:goal (and)))");
    Compare(noop, 0, "empty plan");
    const auto impossible = LoadGroundedProblemFromText("(define (domain d) (:predicates (p)))",
        "(define (problem p) (:domain d) (:init) (:goal (and (p) (not (p)))))");
    Compare(impossible, 7, "contradictory goals");

    const auto costs = LoadGroundedProblemFromText(R"((define (domain d)
        (:requirements :action-costs) (:predicates (p))
        (:action a :effect (and (p) (increase (total-cost) 2)))))",
        "(define (problem p) (:domain d) (:init (= (total-cost) 5)) (:goal (p)) (:metric minimize (total-cost)))");
    for (const auto& plan : {SATPlan(costs), Graphplan(costs), Regression(costs), CSPPlan(costs)}) {
        Require(plan.totalCost == 7.0F && plan.stageLen == 1, "plan cost includes initial total-cost once");
    }
}
}

int main() {
    try {
        Examples();
        RegressionInvariants();
        std::mt19937 rng(87491);
        for (std::size_t trial = 0; trial < 80; ++trial) {
            const PDDLPlanningProblem task(RandomTask(rng));
            Compare(task, 7, "random " + std::to_string(trial));
            if (trial < 20) {
                const auto reference = BFSLength(task, 4);
                const auto partial = PartialOrder(task, {.maxActions = 4});
                Require((partial.status == PlanStatus::success) == reference.has_value(),
                        "bounded partial-order search agrees with BFS");
            }
        }
        for (std::size_t trial = 0; trial < 200; ++trial) {
            const PDDLPlanningProblem task(RandomTask(rng, 5, 8));
            const auto reference = BFSLength(task, 6);
            const auto regression = Regression(task, {6});
            Require((regression.status == PlanStatus::success) == reference.has_value(),
                    "pair pruning agrees with BFS on signed five-fact tasks");
            if (reference) Require(regression.actions.size() == *reference,
                                   "parent-linked regression preserves BFS minimum length");
        }
        std::cout << "Symbolic regressions: " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
