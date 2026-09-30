#pragma once

#include "Plan.h"
#include "pddl/PDDLPlanningProblem.h"

#include <cstddef>

namespace planning::symbolic {

// These reference implementations operate on finite, grounded, deterministic
// STRIPS tasks with negative preconditions/goals and nonnegative action costs.
// They return executable plans; they optimize length (SAT/regression) or parallel
// layers (Graphplan), not weighted action cost. Failure is relative to maxDepth.
struct SymbolicOptions {
    std::size_t maxDepth = 64;
};

using SymbolicPlan = Plan<pddl::PDDLAction, float>;

// maxDepth bounds planning-graph layers; independent actions may share a layer.
// The returned linearization can consequently contain more than maxDepth actions.
[[nodiscard]] SymbolicPlan Graphplan(
    const pddl::PDDLPlanningProblem& problem, SymbolicOptions options = {});

// Bounded sequential SAT encoding with explanatory frame axioms, action
// exclusion and an internal complete DPLL solver with unit propagation.
[[nodiscard]] SymbolicPlan SATPlan(
    const pddl::PDDLPlanningProblem& problem, SymbolicOptions options = {});

// Breadth-first backward regression with bounded pairwise reachability pruning
// and parent-linked paths. The preprocessing cap never limits completeness.
[[nodiscard]] SymbolicPlan Regression(
    const pddl::PDDLPlanningProblem& problem, SymbolicOptions options = {});

} // namespace planning::symbolic
