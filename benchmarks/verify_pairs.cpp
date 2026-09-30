// Standalone relational/propositional transition-system equivalence check.
// Build instructions are in benchmarks/SOURCES.md.
#include "pddl/Adapter.h"

#include <deque>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using namespace planning;
using namespace planning::pddl;

namespace {
using Key = std::vector<std::uint64_t>;

PDDLState Initial(const PDDLPlanningProblem& problem) {
    PDDLState state;
    problem.ForEachInitialState([&](const auto& s) { state = s; return false; });
    return state;
}

std::map<std::string, PDDLAction> Actions(const PDDLPlanningProblem& problem,
                                         const PDDLState& state) {
    std::map<std::string, PDDLAction> actions;
    problem.ForEachApplicableAction(state, [&](const auto& action) {
        std::string name = action.name;
        for (const auto& argument : action.arguments) name += "-" + argument;
        if (!actions.emplace(name, action).second)
            throw std::runtime_error("duplicate normalized action: " + name);
        return true;
    });
    return actions;
}

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: verify_pairs typed-domain typed-problem prop-domain prop-problem max-depth\n"
                     "Use -1 for complete reachable transition graph.\n";
        return 2;
    }
    try {
        auto typed = LoadGroundedProblem(argv[1], argv[2]);
        auto prop = LoadGroundedProblem(argv[3], argv[4]);
        const int maxDepth = std::stoi(argv[5]);
        struct Item { PDDLState typed, prop; int depth; };
        std::deque<Item> frontier;
        std::map<Key, Key> forward, reverse;
        auto enqueue = [&](PDDLState a, PDDLState b, int depth) {
            const Key ka = MakeLogicStateKey(a).words, kb = MakeLogicStateKey(b).words;
            const auto [fa, freshA] = forward.emplace(ka, kb);
            const auto [fb, freshB] = reverse.emplace(kb, ka);
            Require(fa->second == kb && fb->second == ka && freshA == freshB,
                    "state correspondence is not a bijection");
            if (freshA) frontier.push_back({std::move(a), std::move(b), depth});
        };
        enqueue(Initial(typed), Initial(prop), 0);
        std::size_t expanded = 0, edges = 0;
        bool truncated = false;
        while (!frontier.empty()) {
            Item current = std::move(frontier.front());
            frontier.pop_front();
            Require(typed.IsGoal(current.typed) == prop.IsGoal(current.prop), "goal truth differs");
            auto a = Actions(typed, current.typed), b = Actions(prop, current.prop);
            Require(a.size() == b.size(), "applicable action counts differ");
            ++expanded;
            for (const auto& [name, action] : a) {
                Require(b.contains(name), "missing corresponding action: " + name);
                Require(action.cost == b.at(name).cost, "edge costs differ");
                if (maxDepth >= 0 && current.depth == maxDepth) {
                    truncated = true;
                    continue;
                }
                auto nextA = typed.Apply(current.typed, action);
                auto nextB = prop.Apply(current.prop, b.at(name));
                Require(nextA.has_value() && nextB.has_value(), "applicable action failed");
                ++edges;
                enqueue(std::move(*nextA), std::move(*nextB), current.depth + 1);
            }
        }
        std::cout << "Equivalent " << (truncated ? "reachable prefix" : "complete reachable graph")
                  << ": " << expanded << " states, " << edges << " transitions checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
