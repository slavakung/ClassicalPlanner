#include "symbolic/ClassicalPlanners.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace planning::symbolic {
namespace {
using Literals = std::vector<std::size_t>;
using Mutex = std::vector<std::vector<bool>>;
constexpr auto noAction = std::numeric_limits<std::size_t>::max();

struct Operator {
    Literals preconditions;
    Literals effects;
    std::size_t original = noAction;
};

void Canonicalize(Literals& literals) {
    std::sort(literals.begin(), literals.end());
    literals.erase(std::unique(literals.begin(), literals.end()), literals.end());
}

bool Has(const Literals& literals, std::size_t literal) {
    return std::binary_search(literals.begin(), literals.end(), literal);
}

bool Consistent(const Literals& literals) {
    for (const auto l : literals) {
        if (Has(literals, l ^ 1U)) return false;
    }
    return true;
}

std::string Key(const Literals& literals) {
    std::string result;
    for (const auto l : literals) {
        result += std::to_string(l);
        result += ',';
    }
    return result;
}

struct Task {
    const pddl::PDDLPlanningProblem& problem;
    pddl::PDDLState initial;
    std::size_t facts = 0;
    Literals goals;
    std::vector<Operator> operators;

    explicit Task(const pddl::PDDLPlanningProblem& p) : problem(p) {
        p.ForEachInitialState([&](const auto& s) { initial = s; return false; });
        auto extend = [&](const DynamicBitset& bits) {
            for (std::size_t w = 0; w < bits.Words().size(); ++w) {
                if (bits.Words()[w]) {
                    facts = std::max(facts, 64 * w + std::bit_width(bits.Words()[w]));
                }
            }
        };
        extend(initial);
        extend(p.PositiveGoal());
        extend(p.NegativeGoal());
        for (const auto& a : p.CompiledActions()) {
            extend(a.positivePreconditions); extend(a.negativePreconditions);
            extend(a.addEffects); extend(a.deleteEffects);
        }
        auto literals = [&](const DynamicBitset& positive, const DynamicBitset& negative,
                            bool effects) {
            Literals result;
            for (std::size_t i = 0; i < facts; ++i) {
                if (positive.Test(i)) result.push_back(2 * i);
                // Match the runtime's delete-then-add convention.
                if (negative.Test(i) && (!effects || !positive.Test(i))) result.push_back(2 * i + 1);
            }
            return result;
        };
        goals = literals(p.PositiveGoal(), p.NegativeGoal(), false);
        for (std::size_t i = 0; i < p.CompiledActions().size(); ++i) {
            const auto& a = p.CompiledActions()[i];
            operators.push_back({literals(a.positivePreconditions, a.negativePreconditions, false),
                                 literals(a.addEffects, a.deleteEffects, true), i});
        }
    }

    bool Initially(const Literals& literals) const {
        return std::all_of(literals.begin(), literals.end(), [&](auto l) {
            return initial.Test(l / 2) == (l % 2 == 0);
        });
    }

    SymbolicPlan Replay(const std::vector<std::size_t>& sequence) const {
        SymbolicPlan plan;
        auto state = initial;
        float cost = 0.0F;
        for (const auto index : sequence) {
            const auto& action = problem.CompiledActions().at(index);
            const auto next = problem.Apply(state, action);
            if (!next) throw std::logic_error("symbolic planner produced an inapplicable action");
            cost += action.cost;
            if (!std::isfinite(cost)) throw std::overflow_error("symbolic plan cost overflow");
            state = *next;
            plan.actions.push_back(action);
        }
        if (!problem.IsGoal(state)) throw std::logic_error("symbolic planner produced a non-goal plan");
        if (problem.HasTerminalCost()) cost += problem.TerminalCost(state).value();
        if (!std::isfinite(cost)) throw std::overflow_error("symbolic plan cost overflow");
        plan.totalCost = cost;
        plan.stageLen = plan.actions.size();
        plan.status = PlanStatus::success;
        return plan;
    }
};

// An h^2 relaxation: every reachable state has all its literal pairs in this
// table, although pairs in the table need not belong to a reachable state.
// This rules out domain invariants such as an object occupying two locations
// without constructing the forward state graph. An empty table means unknown.
Mutex RegressionReachability(const Task& task) {
    constexpr std::size_t maxLiterals = 2048;
    constexpr std::size_t maxWork = 20'000'000;
    if (task.facts > maxLiterals / 2) return {};
    const auto size = 2 * task.facts;
    Mutex reachable(size, std::vector<bool>(size));
    for (std::size_t f = 0; f < task.facts; ++f) {
        const auto p = 2 * f + (task.initial.Test(f) ? 0 : 1);
        for (std::size_t g = 0; g <= f; ++g) {
            const auto q = 2 * g + (task.initial.Test(g) ? 0 : 1);
            reachable[p][q] = reachable[q][p] = true;
        }
    }
    std::size_t work = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& action : task.operators) {
            if (++work >= maxWork) return {};
            bool applicable = true;
            for (const auto p : action.preconditions) {
                for (const auto q : action.preconditions) {
                    if (++work >= maxWork) return {};
                    if (!reachable[p][q]) { applicable = false; break; }
                }
                if (!applicable) break;
            }
            if (!applicable) continue;
            auto add = [&](std::size_t p, std::size_t q) {
                if (!reachable[p][q]) {
                    reachable[p][q] = reachable[q][p] = true;
                    changed = true;
                }
            };
            for (const auto p : action.effects) {
                for (const auto q : action.effects) {
                    if (++work >= maxWork) return {};
                    add(p, q);
                }
            }
            for (std::size_t p = 0; p < size; ++p) {
                if (++work >= maxWork) return {};
                if (!reachable[p][p] || Has(action.effects, p ^ 1U)) continue;
                bool persists = true;
                for (const auto q : action.preconditions) {
                    if (++work >= maxWork) return {};
                    if (!reachable[p][q]) { persists = false; break; }
                }
                if (!persists) continue;
                for (const auto q : action.effects) {
                    if (++work >= maxWork) return {};
                    add(p, q);
                }
            }
        }
    }
    // Never prune from an unfinished table: hitting either resource cap above
    // discards the relaxation and leaves the complete regression search intact.
    return reachable;
}

bool PairsReachable(const Mutex& reachable, const Literals& goals) {
    if (reachable.empty()) return true;
    for (const auto p : goals) {
        for (const auto q : goals) {
            if (!reachable[p][q]) return false;
        }
    }
    return true;
}

struct LiteralHash {
    std::size_t operator()(const Literals& literals) const noexcept {
        std::size_t hash = literals.size();
        for (const auto literal : literals) {
            hash ^= literal + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        }
        return hash;
    }
};

SymbolicPlan Failure() {
    SymbolicPlan plan;
    plan.status = PlanStatus::fail;
    return plan;
}

struct LiteralLayer {
    std::vector<bool> present;
    Mutex mutex;
};
struct ActionLayer {
    std::vector<std::size_t> actions;
    Mutex mutex;
    // Support entries index this layer's actions, not the global catalogue.
    std::vector<std::vector<std::size_t>> supports;
};

bool Possible(const LiteralLayer& layer, const Literals& goals) {
    for (const auto a : goals) {
        if (!layer.present[a]) return false;
        for (const auto b : goals) {
            if (layer.mutex[a][b]) return false;
        }
    }
    return true;
}

class PlanningGraph {
public:
    explicit PlanningGraph(const Task& task) : task_(task), operators_(task.operators) {
        LiteralLayer initial{std::vector<bool>(2 * task.facts),
                             Mutex(2 * task.facts, std::vector<bool>(2 * task.facts))};
        for (std::size_t f = 0; f < task.facts; ++f) {
            initial.present[2 * f + (task.initial.Test(f) ? 0 : 1)] = true;
        }
        literals_.push_back(std::move(initial));
        nogoods_.emplace_back();
        for (std::size_t l = 0; l < 2 * task.facts; ++l) {
            operators_.push_back({{l}, {l}, noAction});
        }
    }

    void Extend() {
        const auto& previous = literals_.back();
        ActionLayer actions;
        for (std::size_t a = 0; a < operators_.size(); ++a) {
            if (Consistent(operators_[a].preconditions) &&
                Possible(previous, operators_[a].preconditions)) actions.actions.push_back(a);
        }
        const auto count = actions.actions.size();
        actions.mutex.assign(count, std::vector<bool>(count));
        for (std::size_t i = 0; i < count; ++i) {
            const auto& a = operators_[actions.actions[i]];
            for (std::size_t j = 0; j < i; ++j) {
                const auto& b = operators_[actions.actions[j]];
                bool mutex = false;
                for (const auto e : a.effects) {
                    mutex = mutex || Has(b.effects, e ^ 1U) || Has(b.preconditions, e ^ 1U);
                }
                for (const auto e : b.effects) mutex = mutex || Has(a.preconditions, e ^ 1U);
                for (const auto p : a.preconditions) {
                    for (const auto q : b.preconditions) mutex = mutex || previous.mutex[p][q];
                }
                actions.mutex[i][j] = actions.mutex[j][i] = mutex;
            }
        }
        const auto size = 2 * task_.facts;
        LiteralLayer next{std::vector<bool>(size), Mutex(size, std::vector<bool>(size))};
        actions.supports.resize(size);
        for (std::size_t i = 0; i < count; ++i) {
            for (const auto e : operators_[actions.actions[i]].effects) {
                next.present[e] = true;
                actions.supports[e].push_back(i);
            }
        }
        for (std::size_t p = 0; p < size; ++p) {
            if (!next.present[p]) continue;
            for (std::size_t q = 0; q < p; ++q) {
                if (!next.present[q]) continue;
                bool mutex = true;
                if ((p ^ 1U) != q) {
                    for (const auto a : actions.supports[p]) {
                        for (const auto b : actions.supports[q]) {
                            if (!actions.mutex[a][b]) mutex = false;
                        }
                    }
                }
                next.mutex[p][q] = next.mutex[q][p] = mutex;
            }
        }
        actions_.push_back(std::move(actions));
        literals_.push_back(std::move(next));
        nogoods_.emplace_back();
    }

    bool Extract(std::size_t level, const Literals& goals,
                 std::vector<std::size_t>& sequence) {
        if (!Possible(literals_[level], goals)) return false;
        if (level == 0) return true;
        const auto key = Key(goals);
        if (nogoods_[level].contains(key)) return false;
        const auto& layer = actions_[level - 1];
        std::vector<std::size_t> selected;
        std::function<bool()> choose = [&]() {
            std::size_t uncovered = noAction;
            for (const auto goal : goals) {
                bool covered = false;
                for (const auto a : selected) {
                    covered = covered || Has(operators_[layer.actions[a]].effects, goal);
                }
                if (!covered && (uncovered == noAction ||
                    layer.supports[goal].size() < layer.supports[uncovered].size())) uncovered = goal;
            }
            if (uncovered == noAction) {
                Literals preceding;
                for (const auto a : selected) {
                    const auto& pre = operators_[layer.actions[a]].preconditions;
                    preceding.insert(preceding.end(), pre.begin(), pre.end());
                }
                Canonicalize(preceding);
                if (!Extract(level - 1, preceding, sequence)) return false;
                for (const auto a : selected) {
                    const auto original = operators_[layer.actions[a]].original;
                    if (original != noAction) sequence.push_back(original);
                }
                return true;
            }
            for (const auto a : layer.supports[uncovered]) {
                if (std::any_of(selected.begin(), selected.end(),
                    [&](auto b) { return layer.mutex[a][b]; })) continue;
                selected.push_back(a);
                if (choose()) return true;
                selected.pop_back();
            }
            return false;
        };
        if (choose()) return true;
        nogoods_[level].insert(key);
        return false;
    }

    bool UnreachableAtFixedPoint(const Literals& goals) const {
        if (literals_.size() < 2) return false;
        const auto& a = literals_[literals_.size() - 2];
        const auto& b = literals_.back();
        return a.present == b.present && a.mutex == b.mutex && !Possible(b, goals);
    }

private:
    const Task& task_;
    std::vector<Operator> operators_;
    std::vector<LiteralLayer> literals_;
    std::vector<ActionLayer> actions_;
    std::vector<std::unordered_set<std::string>> nogoods_;
};

using Clause = std::vector<int>;
using CNF = std::vector<Clause>;

// Unit propagation plus a shortest-unresolved-clause branching heuristic.
// Both truth values are explored, so UNSAT is a proof for the encoded horizon.
bool DPLL(const CNF& clauses, std::vector<std::int8_t>& assignment) {
    for (;;) {
        bool changed = false;
        for (const auto& clause : clauses) {
            bool satisfied = false;
            int unit = 0;
            std::size_t unknown = 0;
            for (const auto literal : clause) {
                const auto value = assignment[static_cast<std::size_t>(std::abs(literal))];
                if (value == 0) { unit = literal; ++unknown; }
                else if ((value > 0) == (literal > 0)) { satisfied = true; break; }
            }
            if (satisfied) continue;
            if (unknown == 0) return false;
            if (unknown == 1) {
                assignment[static_cast<std::size_t>(std::abs(unit))] = unit > 0 ? 1 : -1;
                changed = true;
            }
        }
        if (!changed) break;
    }
    int choice = 0;
    auto shortest = std::numeric_limits<std::size_t>::max();
    for (const auto& clause : clauses) {
        bool satisfied = false;
        std::size_t unknown = 0;
        int candidate = 0;
        for (const auto literal : clause) {
            const auto value = assignment[static_cast<std::size_t>(std::abs(literal))];
            if (value != 0 && ((value > 0) == (literal > 0))) { satisfied = true; break; }
            if (value == 0) { candidate = literal; ++unknown; }
        }
        if (!satisfied && unknown < shortest) { shortest = unknown; choice = candidate; }
    }
    if (choice == 0) return true;
    for (const auto value : {std::int8_t{1}, std::int8_t{-1}}) {
        auto branch = assignment;
        branch[static_cast<std::size_t>(std::abs(choice))] = value;
        if (DPLL(clauses, branch)) { assignment = std::move(branch); return true; }
    }
    return false;
}
} // namespace

SymbolicPlan Graphplan(const pddl::PDDLPlanningProblem& problem, SymbolicOptions options) {
    const Task task(problem);
    if (!Consistent(task.goals)) return Failure();
    if (task.Initially(task.goals)) return task.Replay({});
    PlanningGraph graph(task);
    for (std::size_t level = 1; level <= options.maxDepth; ++level) {
        graph.Extend();
        std::vector<std::size_t> sequence;
        if (graph.Extract(level, task.goals, sequence)) return task.Replay(sequence);
        if (graph.UnreachableAtFixedPoint(task.goals)) break;
    }
    return Failure();
}

SymbolicPlan SATPlan(const pddl::PDDLPlanningProblem& problem, SymbolicOptions options) {
    const Task task(problem);
    if (!Consistent(task.goals)) return Failure();
    if (task.Initially(task.goals)) return task.Replay({});
    const auto actionCount = task.operators.size();
    for (std::size_t horizon = 1; horizon <= options.maxDepth; ++horizon) {
        const auto intMax = static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (horizon >= intMax || task.facts > intMax / (horizon + 1) ||
            actionCount > (intMax - task.facts * (horizon + 1)) / horizon) {
            throw std::length_error("SATPlan encoding exceeds integer variable space");
        }
        const auto fluentVariables = task.facts * (horizon + 1);
        const auto variables = fluentVariables + horizon * actionCount;
        auto fact = [&](std::size_t t, std::size_t f) { return static_cast<int>(1 + t * task.facts + f); };
        auto action = [&](std::size_t t, std::size_t a) {
            return static_cast<int>(1 + fluentVariables + t * actionCount + a);
        };
        auto literal = [&](std::size_t t, std::size_t l) {
            return (l % 2 == 0 ? 1 : -1) * fact(t, l / 2);
        };
        CNF clauses;
        for (std::size_t f = 0; f < task.facts; ++f) {
            clauses.push_back({task.initial.Test(f) ? fact(0, f) : -fact(0, f)});
        }
        for (const auto g : task.goals) clauses.push_back({literal(horizon, g)});
        for (std::size_t t = 0; t < horizon; ++t) {
            for (std::size_t a = 0; a < actionCount; ++a) {
                for (const auto p : task.operators[a].preconditions) clauses.push_back({-action(t, a), literal(t, p)});
                for (const auto e : task.operators[a].effects) clauses.push_back({-action(t, a), literal(t + 1, e)});
                for (std::size_t b = 0; b < a; ++b) clauses.push_back({-action(t, a), -action(t, b)});
            }
            // A fluent may change only when an action explicitly causes it.
            // At-most-one permits idle steps, whose frame clauses preserve state.
            for (std::size_t f = 0; f < task.facts; ++f) {
                Clause becomesTrue{fact(t, f), -fact(t + 1, f)};
                Clause becomesFalse{-fact(t, f), fact(t + 1, f)};
                for (std::size_t a = 0; a < actionCount; ++a) {
                    if (Has(task.operators[a].effects, 2 * f)) becomesTrue.push_back(action(t, a));
                    if (Has(task.operators[a].effects, 2 * f + 1)) becomesFalse.push_back(action(t, a));
                }
                clauses.push_back(std::move(becomesTrue));
                clauses.push_back(std::move(becomesFalse));
            }
        }
        std::vector<std::int8_t> assignment(variables + 1);
        if (!DPLL(clauses, assignment)) continue;
        std::vector<std::size_t> sequence;
        for (std::size_t t = 0; t < horizon; ++t) {
            for (std::size_t a = 0; a < actionCount; ++a) {
                if (assignment[static_cast<std::size_t>(action(t, a))] > 0) sequence.push_back(a);
            }
        }
        return task.Replay(sequence);
    }
    return Failure();
}

SymbolicPlan Regression(const pddl::PDDLPlanningProblem& problem, SymbolicOptions options) {
    const Task task(problem);
    if (!Consistent(task.goals)) return Failure();
    if (task.Initially(task.goals)) return task.Replay({});
    if (options.maxDepth == 0) return Failure();
    const auto reachable = RegressionReachability(task);
    if (!PairsReachable(reachable, task.goals)) return Failure();
    // Hash-set elements have stable addresses across rehashes. Store goals once
    // and reconstruct a solution through parent links, instead of copying the
    // full path and a decimal serialization into every queued node.
    std::unordered_set<Literals, LiteralHash> visited{task.goals};
    struct Node {
        const Literals* goals;
        std::size_t parent;
        std::size_t action;
        std::size_t depth;
    };
    std::vector<Node> nodes{{&*visited.begin(), noAction, noAction, 0}};
    for (std::size_t current = 0; current < nodes.size(); ++current) {
        const auto node = nodes[current];
        if (node.depth >= options.maxDepth) continue;
        for (const auto& action : task.operators) {
            bool relevant = false, conflict = false;
            for (const auto goal : *node.goals) {
                relevant = relevant || Has(action.effects, goal);
                conflict = conflict || Has(action.effects, goal ^ 1U);
            }
            if (!relevant || conflict) continue;
            Literals preceding = action.preconditions;
            for (const auto goal : *node.goals) {
                if (!Has(action.effects, goal)) preceding.push_back(goal);
            }
            Canonicalize(preceding);
            if (!Consistent(preceding) || !PairsReachable(reachable, preceding)) continue;
            // Every shallower generated node was already checked, so checking
            // here preserves BFS's minimum action count and avoids queuing the
            // rest of a potentially enormous next layer after finding a plan.
            if (task.Initially(preceding)) {
                std::vector<std::size_t> sequence{action.original};
                for (auto ancestor = current; nodes[ancestor].parent != noAction;
                     ancestor = nodes[ancestor].parent) {
                    sequence.push_back(nodes[ancestor].action);
                }
                return task.Replay(sequence);
            }
            // At the depth bound an unsatisfied conjunction cannot contribute
            // a solution, so do not retain an otherwise unused terminal layer.
            if (node.depth + 1 >= options.maxDepth) continue;
            const auto [entry, inserted] = visited.insert(std::move(preceding));
            if (inserted) nodes.push_back({&*entry, current, action.original, node.depth + 1});
        }
    }
    return Failure();
}
} // namespace planning::symbolic
