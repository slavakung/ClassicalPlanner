#pragma once

#include "Plan.h"
#include "pddl/PDDLPlanningProblem.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace planning::symbolic {

struct PartialOrderOptions {
    std::size_t maxActions = 32;
    std::size_t maxRefinements = 1000000;
};

// Grounded plan-space search (LaValle 2.5.1): repair open conditions by
// inserting/reusing action occurrences, protect causal links by promotion or
// demotion, then topologically linearize the consistent partial order.
// This is a feasibility planner; the action bound is not a cost objective.
inline Plan<pddl::PDDLAction, float> PartialOrder(
        const pddl::PDDLPlanningProblem& problem, PartialOrderOptions options = {}) {
    using Literal = std::size_t; // 2*p positive, 2*p+1 negative
    struct Step { int action; std::vector<Literal> pre, effects; };
    struct Link { std::size_t from; Literal literal; std::size_t to; };
    struct Open { std::size_t step; Literal literal; };
    struct Partial {
        std::vector<Step> steps;
        std::vector<std::vector<bool>> before;
        std::vector<Link> links;
        std::vector<Open> open;
    };
    if (options.maxRefinements == 0) throw std::invalid_argument("maxRefinements must be positive");
    if (problem.CompiledActions().size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("too many actions for partial-order planner");
    auto initial = problem.InitialStates().at(0);
    std::size_t facts = initial.Words().size() * 64;
    auto width = [&](const DynamicBitset& b) { facts = std::max(facts, b.Words().size() * 64); };
    width(problem.PositiveGoal()); width(problem.NegativeGoal());
    for (const auto& a : problem.CompiledActions()) {
        width(a.positivePreconditions); width(a.negativePreconditions);
        width(a.addEffects); width(a.deleteEffects);
    }
    auto literals = [&](const DynamicBitset& pos, const DynamicBitset& neg) {
        std::vector<Literal> result;
        for (std::size_t i = 0; i < facts; ++i) {
            if (pos.Test(i)) result.push_back(2 * i);
            if (neg.Test(i)) result.push_back(2 * i + 1);
        }
        return result;
    };
    std::vector<Step> operators;
    for (std::size_t i = 0; i < problem.CompiledActions().size(); ++i) {
        const auto& a = problem.CompiledActions()[i];
        auto deletes = a.deleteEffects;
        deletes.DifferenceWith(a.addEffects);
        operators.push_back({static_cast<int>(i), literals(a.positivePreconditions, a.negativePreconditions),
                             literals(a.addEffects, deletes)});
    }
    Partial root;
    root.steps = {{-1, {}, {}}, {-1, literals(problem.PositiveGoal(), problem.NegativeGoal()), {}}};
    for (std::size_t i = 0; i < facts; ++i)
        root.steps[0].effects.push_back(2 * i + (initial.Test(i) ? 0 : 1));
    root.before = {{false, true}, {false, false}};
    for (auto g : root.steps[1].pre) root.open.push_back({1, g});
    auto contains = [](const auto& xs, auto value) {
        return std::find(xs.begin(), xs.end(), value) != xs.end();
    };
    auto order = [](Partial& p, std::size_t from, std::size_t to) {
        if (from == to || p.before[to][from]) return false;
        p.before[from][to] = true;
        // Existing relation is transitively closed; one new edge joins every
        // predecessor of from to every successor of to.
        for (std::size_t i = 0; i < p.steps.size(); ++i)
            for (std::size_t j = 0; j < p.steps.size(); ++j)
                if ((i == from || p.before[i][from]) && (j == to || p.before[to][j]))
                    p.before[i][j] = true;
        return true;
    };
    std::size_t refinements = 0;
    std::optional<Partial> solution;
    std::function<bool(Partial)> refine = [&](Partial p) {
        if (++refinements > options.maxRefinements)
            throw std::length_error("partial-order refinement budget exhausted");
        for (const auto& link : p.links) {
            for (std::size_t t = 2; t < p.steps.size(); ++t) {
                if (t == link.from || t == link.to || p.before[t][link.from] || p.before[link.to][t] ||
                    !contains(p.steps[t].effects, link.literal ^ 1U)) continue;
                auto earlier = p;
                if (order(earlier, t, link.from) && refine(std::move(earlier))) return true;
                if (order(p, link.to, t) && refine(std::move(p))) return true;
                return false;
            }
        }
        if (p.open.empty()) { solution = std::move(p); return true; }
        const auto need = p.open.back();
        p.open.pop_back();
        for (std::size_t s = 0; s < p.steps.size(); ++s) {
            if (s == need.step || !contains(p.steps[s].effects, need.literal)) continue;
            auto child = p;
            if (!order(child, s, need.step)) continue;
            child.links.push_back({s, need.literal, need.step});
            if (refine(std::move(child))) return true;
        }
        if (p.steps.size() - 2 >= options.maxActions) return false;
        for (const auto& op : operators) {
            if (!contains(op.effects, need.literal)) continue;
            auto child = p;
            const auto id = child.steps.size();
            child.steps.push_back(op);
            for (auto& row : child.before) row.resize(id + 1);
            child.before.emplace_back(id + 1, false);
            if (!order(child, 0, id) || !order(child, id, 1) || !order(child, id, need.step)) continue;
            child.links.push_back({id, need.literal, need.step});
            for (auto pre : op.pre) child.open.push_back({id, pre});
            if (refine(std::move(child))) return true;
        }
        return false;
    };
    Plan<pddl::PDDLAction, float> result;
    result.status = PlanStatus::fail;
    if (!refine(std::move(root))) return result;
    const auto& p = *solution;
    std::vector<bool> done(p.steps.size(), false);
    auto state = initial;
    double cost = 0;
    for (std::size_t count = 0; count < p.steps.size(); ++count) {
        std::size_t chosen = p.steps.size();
        for (std::size_t i = 0; i < p.steps.size(); ++i) {
            if (done[i]) continue;
            bool ready = true;
            for (std::size_t j = 0; j < p.steps.size(); ++j)
                if (p.before[j][i] && !done[j]) ready = false;
            if (ready) { chosen = i; break; }
        }
        if (chosen == p.steps.size()) throw std::logic_error("cyclic partial-order solution");
        done[chosen] = true;
        if (p.steps[chosen].action < 0) continue;
        const auto& action = problem.CompiledActions()[static_cast<std::size_t>(p.steps[chosen].action)];
        auto next = problem.Apply(state, action);
        if (!next) throw std::logic_error("partial-order plan failed replay");
        cost += problem.StepCost(state, action, *next).value_or(0);
        state = std::move(*next);
        result.actions.push_back(action);
    }
    if (!problem.IsGoal(state)) throw std::logic_error("partial-order plan missed goal");
    cost += problem.TerminalCost(state).value_or(0);
    if (!std::isfinite(cost) || std::abs(cost) > std::numeric_limits<float>::max())
        throw std::overflow_error("partial-order plan cost overflow");
    result.status = PlanStatus::success;
    result.stageLen = result.actions.size();
    result.totalCost = static_cast<float>(cost);
    return result;
}

} // namespace planning::symbolic
