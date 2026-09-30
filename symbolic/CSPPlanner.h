#pragma once

#include "symbolic/ClassicalPlanners.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace planning::symbolic {

// Bounded finite-domain planning CSP (AIMA 11.2.4). Each time has one action
// variable, each fluent/time has a Boolean variable. Ternary transition
// constraints enforce preconditions, effects and persistence. Generalized arc
// consistency removes unsupported values before minimum-domain backtracking.
// Increasing horizons produce a minimum-length sequential plan, not min cost.
inline SymbolicPlan CSPPlan(const pddl::PDDLPlanningProblem& problem,
                           SymbolicOptions options = {}) {
    const auto& actions = problem.CompiledActions();
    const auto noop = actions.size();
    const auto initial = problem.InitialStates().at(0);
    std::size_t facts = initial.Words().size() * 64;
    auto width = [&](const DynamicBitset& b) { facts = std::max(facts, b.Words().size() * 64); };
    width(problem.PositiveGoal()); width(problem.NegativeGoal());
    for (const auto& a : actions) {
        width(a.positivePreconditions); width(a.negativePreconditions);
        width(a.addEffects); width(a.deleteEffects);
    }
    // Bit 0 permits false, bit 1 permits true.
    struct Domains {
        std::vector<std::vector<unsigned char>> fluent;
        std::vector<std::vector<std::size_t>> action;
    };
    auto permits = [&](std::size_t a, std::size_t f, unsigned before, unsigned after) {
        if (a == noop) return before == after;
        const auto& op = actions[a];
        if ((op.positivePreconditions.Test(f) && !before) ||
            (op.negativePreconditions.Test(f) && before)) return false;
        const unsigned expected = op.addEffects.Test(f) ? 1 : op.deleteEffects.Test(f) ? 0 : before;
        return after == expected;
    };
    auto supports = [&](const Domains& d, std::size_t t, std::size_t a, std::size_t f) {
        unsigned char pre = 0, post = 0;
        for (unsigned x = 0; x < 2; ++x) for (unsigned y = 0; y < 2; ++y)
            if ((d.fluent[t][f] & (1U << x)) && (d.fluent[t + 1][f] & (1U << y)) && permits(a, f, x, y)) {
                pre |= static_cast<unsigned char>(1U << x);
                post |= static_cast<unsigned char>(1U << y);
            }
        return std::pair{pre, post};
    };
    auto propagate = [&](Domains& d) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& time : d.fluent)
                if (std::find(time.begin(), time.end(), 0) != time.end()) return false;
            for (std::size_t t = 0; t < d.action.size(); ++t) {
                auto& domain = d.action[t];
                const auto oldSize = domain.size();
                std::erase_if(domain, [&](auto a) {
                    for (std::size_t f = 0; f < facts; ++f)
                        if (supports(d, t, a, f).first == 0) return true;
                    return false;
                });
                if (domain.empty()) return false;
                changed |= oldSize != domain.size();
                for (std::size_t f = 0; f < facts; ++f) {
                    unsigned char pre = 0, post = 0;
                    for (auto a : domain) {
                        const auto [x, y] = supports(d, t, a, f);
                        pre |= x; post |= y;
                    }
                    if (!pre || !post) return false;
                    changed |= pre != d.fluent[t][f] || post != d.fluent[t + 1][f];
                    d.fluent[t][f] = pre;
                    d.fluent[t + 1][f] = post;
                }
            }
        }
        return true;
    };
    std::optional<Domains> solution;
    std::function<bool(Domains)> search = [&](Domains d) {
        if (!propagate(d)) return false;
        auto chosen = d.action.size();
        auto smallest = std::numeric_limits<std::size_t>::max();
        for (std::size_t t = 0; t < d.action.size(); ++t)
            if (d.action[t].size() > 1 && d.action[t].size() < smallest) {
                chosen = t; smallest = d.action[t].size();
            }
        if (chosen == d.action.size()) { solution = std::move(d); return true; }
        for (auto a : d.action[chosen]) {
            auto child = d;
            child.action[chosen] = {a};
            if (search(std::move(child))) return true;
        }
        return false;
    };
    for (std::size_t horizon = 0;; ++horizon) {
        if (horizon == std::numeric_limits<std::size_t>::max()) throw std::length_error("CSP horizon overflow");
        Domains d;
        d.fluent.assign(horizon + 1, std::vector<unsigned char>(facts, 3));
        for (std::size_t f = 0; f < facts; ++f) {
            d.fluent[0][f] = initial.Test(f) ? 2 : 1;
            if (problem.PositiveGoal().Test(f)) d.fluent[horizon][f] &= 2;
            if (problem.NegativeGoal().Test(f)) d.fluent[horizon][f] &= 1;
        }
        d.action.assign(horizon, std::vector<std::size_t>(actions.size() + 1));
        for (auto& domain : d.action) std::iota(domain.begin(), domain.end(), 0);
        if (search(std::move(d))) {
            SymbolicPlan plan;
            plan.status = PlanStatus::success;
            auto state = initial;
            double cost = 0;
            for (const auto& domain : solution->action) {
                if (domain.front() == noop) continue;
                const auto& action = actions[domain.front()];
                auto next = problem.Apply(state, action);
                if (!next) throw std::logic_error("CSP plan failed replay");
                cost += problem.StepCost(state, action, *next).value_or(0);
                state = std::move(*next);
                plan.actions.push_back(action);
            }
            if (!problem.IsGoal(state)) throw std::logic_error("CSP plan missed goal");
            cost += problem.TerminalCost(state).value_or(0);
            if (!std::isfinite(cost) || std::abs(cost) > std::numeric_limits<float>::max())
                throw std::overflow_error("CSP plan cost overflow");
            plan.totalCost = static_cast<float>(cost);
            plan.stageLen = plan.actions.size();
            return plan;
        }
        if (horizon == options.maxDepth) break;
    }
    SymbolicPlan failed;
    failed.status = PlanStatus::fail;
    return failed;
}

} // namespace planning::symbolic
