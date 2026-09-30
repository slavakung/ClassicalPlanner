#pragma once

#include "PlanningProblem.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace test {
// Tiny immutable weighted graph. Actions are edge indices, so two different
// edges to the same state remain distinguishable plans.
class GraphProblem : public planning::PlanningProblem<int, int, float> {
public:
    struct Edge {
        int from;
        int to;
        float cost;
    };
    std::vector<Edge> edges;
    std::vector<int> initial{0};
    std::vector<int> goals{4};
    float terminal = 0;
    bool hasCost = true;
    int throwOnState = -1;
    std::uint64_t fingerprint = 101;

    void ForEachInitialState(const planning::Consumer<int>& consumer) const override {
        for (const auto state : initial) {
            if (!consumer(state)) {
                return;
            }
        }
    }
    bool IsGoal(const int& state) const override {
        return std::find(goals.begin(), goals.end(), state) != goals.end();
    }
    void ForEachApplicableAction(const int& state,
                                 const planning::Consumer<int>& consumer) const override {
        if (state == throwOnState) {
            throw std::runtime_error("injected expansion failure");
        }
        for (std::size_t i = 0; i < edges.size(); ++i) {
            if (edges[i].from == state && !consumer(static_cast<int>(i))) {
                return;
            }
        }
    }
    std::optional<int> Apply(const int& state, const int& action) const override {
        if (action < 0 || static_cast<std::size_t>(action) >= edges.size() ||
            edges[static_cast<std::size_t>(action)].from != state) {
            return std::nullopt;
        }
        return edges[static_cast<std::size_t>(action)].to;
    }
    int MakeHistoryKey(const int& state) const override {
        return state;
    }
    bool HasStepCost() const override {
        return hasCost;
    }
    bool HasTerminalCost() const override {
        return hasCost;
    }
    std::optional<float> StepCost(const int&, const int& action, const int&) const override {
        return hasCost ? std::optional<float>{edges.at(static_cast<std::size_t>(action)).cost}
                       : std::nullopt;
    }
    std::optional<float> TerminalCost(const int&) const override {
        return hasCost ? std::optional<float>{terminal} : std::nullopt;
    }
    std::uint64_t MpiFingerprint() const {
        return fingerprint;
    }
};
} // namespace test
