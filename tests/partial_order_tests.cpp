#include "symbolic/PartialOrderPlanner.h"
#include "pddl/Adapter.h"
#include "tests/TestSupport.h"

using namespace planning;
using test::Require;
using test::Throws;

int main() {
    try {
        const std::string domain = R"((define (domain d)
            (:requirements :strips :negative-preconditions)
            (:predicates (p) (q) (r))
            (:action a :precondition (p) :effect (and (q) (not (p))))
            (:action b :precondition (q) :effect (and (p) (r)))))";
        const auto p = pddl::LoadGroundedProblemFromText(domain,
            "(define (problem p) (:domain d) (:init (p)) (:goal (and (p) (q) (r))))");
        auto plan = symbolic::PartialOrder(p, {4, 10000});
        Require(plan.status == PlanStatus::success && plan.stageLen == 2, "causal threat resolved by ordering");
        auto state = p.InitialStates().front();
        for (const auto& action : plan.actions) {
            auto next = p.Apply(state, action);
            Require(next.has_value(), "partial order replay applicability");
            state = *next;
        }
        Require(p.IsGoal(state), "partial order replay goal");
        Require(symbolic::PartialOrder(p, {1, 10000}).status == PlanStatus::fail, "action budget bounds feasibility");
        Throws([&] { (void)symbolic::PartialOrder(p, {4, 1}); }, "resource exhaustion explicit");
        const auto negative = pddl::LoadGroundedProblemFromText(domain,
            "(define (problem p) (:domain d) (:init (p)) (:goal (and (q) (not (p)))))");
        Require(symbolic::PartialOrder(negative, {2, 10000}).stageLen == 1, "negative causal conditions");
        const auto empty = pddl::LoadGroundedProblemFromText(domain,
            "(define (problem p) (:domain d) (:init (p)) (:goal (p)))");
        Require(symbolic::PartialOrder(empty, {0, 10000}).status == PlanStatus::success, "zero action goal");
        const auto impossible = pddl::LoadGroundedProblemFromText(domain,
            "(define (problem p) (:domain d) (:init) (:goal (r)))");
        Require(symbolic::PartialOrder(impossible, {4, 10000}).status == PlanStatus::fail, "unsupported condition fails");
        for (const std::string name : {"blocks", "gripper"}) {
            const auto example = pddl::LoadGroundedProblem("examples/pddl/" + name + "/domain.pddl",
                                                         "examples/pddl/" + name + "/problem.pddl");
            Require(symbolic::PartialOrder(example, {8, 1000000}).status == PlanStatus::success,
                    "example plan-space solution " + name);
        }
        std::cout << test::checks << " partial-order checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
