#pragma once

#include "LogicBitset.h"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace planning {

enum class RelaxationHeuristic { max, additive, relaxedPlan };

// Delete relaxation over signed facts: p and not-p are separate facts, so an
// action can establish either without removing the other. This handles negative
// preconditions/goals without treating the closed-world complement as permanent.
// h_max is admissible for nonnegative action costs. h_add and the extracted
// relaxed-plan estimate are guidance only, and need not be admissible.
class DeleteRelaxationHeuristic {
public:
    template<typename Problem>
    explicit DeleteRelaxationHeuristic(const Problem& problem,
            RelaxationHeuristic kind = RelaxationHeuristic::max) : kind_(kind) {
        auto width = [&](const DynamicBitset& b) { facts_ = std::max(facts_, b.Words().size() * 64); };
        width(problem.PositiveGoal());
        if constexpr (requires { problem.NegativeGoal(); }) width(problem.NegativeGoal());
        for (const auto& a : problem.CompiledActions()) {
            if constexpr (requires { a.positivePreconditions; }) {
                width(a.positivePreconditions); width(a.negativePreconditions);
            } else width(a.preconditions);
            width(a.addEffects); width(a.deleteEffects);
        }
        problem.ForEachInitialState([&](const auto& s) { width(s); return true; });
        auto literals = [&](const DynamicBitset& pos, const DynamicBitset& neg) {
            std::vector<std::size_t> result;
            for (std::size_t i = 0; i < facts_; ++i) {
                if (pos.Test(i)) result.push_back(2 * i);
                if (neg.Test(i)) result.push_back(2 * i + 1);
            }
            return result;
        };
        DynamicBitset negatives;
        if constexpr (requires { problem.NegativeGoal(); }) negatives = problem.NegativeGoal();
        goals_ = literals(problem.PositiveGoal(), negatives);
        for (const auto& a : problem.CompiledActions()) {
            const float cost = [&] {
                if constexpr (requires { a.cost; }) return a.cost;
                else return 1.0F;
            }();
            if (!std::isfinite(cost) || cost < 0)
                throw std::invalid_argument("relaxation requires finite nonnegative costs");
            auto deletes = a.deleteEffects;
            deletes.DifferenceWith(a.addEffects); // application uses add-wins semantics
            auto pre = [&] {
                if constexpr (requires { a.positivePreconditions; })
                    return literals(a.positivePreconditions, a.negativePreconditions);
                else return literals(a.preconditions, DynamicBitset{});
            }();
            actions_.push_back({std::move(pre), literals(a.addEffects, deletes), cost});
        }
    }

    float operator()(const DynamicBitset& state) const {
        const auto infinity = std::numeric_limits<float>::infinity();
        const auto none = std::numeric_limits<std::size_t>::max();
        std::vector<float> costs(2 * facts_, infinity);
        std::vector<std::size_t> achiever(2 * facts_, none);
        for (std::size_t f = 0; f < facts_; ++f) costs[2 * f + (state.Test(f) ? 0 : 1)] = 0;
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t i = 0; i < actions_.size(); ++i) {
                const auto& a = actions_[i];
                float pre = 0;
                for (auto p : a.preconditions) {
                    if (kind_ == RelaxationHeuristic::max) pre = std::max(pre, costs[p]);
                    else pre += costs[p];
                }
                const auto candidate = pre + a.cost;
                for (auto e : a.effects) if (candidate < costs[e]) {
                    costs[e] = candidate;
                    achiever[e] = i;
                    changed = true;
                }
            }
        }
        float estimate = 0;
        for (auto g : goals_) {
            if (!std::isfinite(costs[g])) return infinity;
            if (kind_ == RelaxationHeuristic::max) estimate = std::max(estimate, costs[g]);
            else estimate += costs[g];
        }
        if (kind_ != RelaxationHeuristic::relaxedPlan) return estimate;
        std::vector<bool> selected(actions_.size(), false);
        std::vector<bool> visited(2 * facts_, false);
        estimate = 0;
        std::function<void(std::size_t)> support = [&](std::size_t f) {
            if (visited[f]) return;
            visited[f] = true;
            const auto a = achiever[f];
            if (a == none || selected[a]) return;
            selected[a] = true;
            estimate += actions_[a].cost;
            for (auto p : actions_[a].preconditions) support(p);
        };
        for (auto g : goals_) support(g);
        return estimate;
    }

private:
    struct Action { std::vector<std::size_t> preconditions, effects; float cost; };
    RelaxationHeuristic kind_;
    std::size_t facts_ = 0;
    std::vector<std::size_t> goals_;
    std::vector<Action> actions_;
};

} // namespace planning
