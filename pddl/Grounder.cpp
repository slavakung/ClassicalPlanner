#include "pddl/Grounder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace planning::pddl
{
    namespace
    {
        bool IsSubtype(
            ir::TypeId actual,
            ir::TypeId required,
            const std::vector<ir::Type>& types
        ) {
            if (actual == required) {
                return true;
            }
            while (actual != ir::InvalidType) {
                actual = types.at(actual).parent;
                if (actual == required) {
                    return true;
                }
            }
            return false;
        }

        std::vector<ir::Object> AllObjects(const ir::Problem& problem) {
            std::vector<ir::Object> result;
            result.reserve(problem.domain.constants.size() + problem.objects.size());
            result.insert(
                result.end(),
                problem.domain.constants.begin(),
                problem.domain.constants.end()
            );
            result.insert(result.end(), problem.objects.begin(), problem.objects.end());
            return result;
        }

        ir::ObjectId ResolveTerm(
            const ir::Term& term,
            const std::vector<ir::ObjectId>& binding
        ) {
            if (term.kind == ir::TermKind::object) {
                return static_cast<ir::ObjectId>(term.id);
            }
            return binding.at(term.id);
        }

        bool EqualitySatisfied(
            const ir::LiteralSchema& literal,
            const std::vector<ir::ObjectId>& binding
        ) {
            const auto left = ResolveTerm(literal.terms.at(0), binding);
            const auto right = ResolveTerm(literal.terms.at(1), binding);
            const bool equal = left == right;
            return literal.negated ? !equal : equal;
        }

        ir::GroundLiteral InstantiateLiteral(
            const ir::LiteralSchema& literal,
            const std::vector<ir::ObjectId>& binding
        ) {
            ir::GroundLiteral result;
            result.negated = literal.negated;
            result.atom.predicate = literal.predicate;
            result.atom.arguments.reserve(literal.terms.size());
            for (const ir::Term& term : literal.terms) {
                result.atom.arguments.push_back(ResolveTerm(term, binding));
            }
            return result;
        }

        struct AtomHash {
            std::size_t operator()(const ir::GroundAtom& atom) const noexcept {
                std::size_t seed = atom.predicate;
                for (const auto object : atom.arguments) {
                    seed ^= static_cast<std::size_t>(object) + 0x9e3779b9U +
                        (seed << 6U) + (seed >> 2U);
                }
                return seed;
            }
        };

        struct StaticRelation {
            std::vector<ir::GroundAtom> facts;
            std::vector<std::size_t> allRows;
            std::vector<std::unordered_map<ir::ObjectId, std::vector<std::size_t>>> columns;
        };
    }

    GroundProblem Grounder::Ground(
        const ir::Problem& problem,
        ExecutionMode mode
    ) const {
        if (mode != ExecutionMode::groundAll) {
            throw std::runtime_error(
                "PDDL execution mode is architecturally reserved but not yet "
                "implemented. Use ExecutionMode::groundAll in this revision."
            );
        }

        GroundProblem result;
        result.source = problem;
        const std::vector<ir::Object> objects = AllObjects(problem);

        std::vector<bool> dynamicPredicate(problem.domain.predicates.size(), false);
        for (const auto& schema : problem.domain.actions) {
            for (const auto& effect : schema.effects) {
                dynamicPredicate.at(effect.predicate) = true;
            }
        }
        std::unordered_set<ir::GroundAtom, AtomHash> initialFacts;
        std::vector<StaticRelation> relations(problem.domain.predicates.size());
        for (const auto& fact : problem.initialFacts) {
            if (initialFacts.insert(fact).second && !dynamicPredicate.at(fact.predicate)) {
                relations.at(fact.predicate).facts.push_back(fact);
            }
        }
        for (std::size_t predicate = 0; predicate < relations.size(); ++predicate) {
            auto& relation = relations[predicate];
            relation.columns.resize(problem.domain.predicates[predicate].parameterTypes.size());
            for (std::size_t row = 0; row < relation.facts.size(); ++row) {
                relation.allRows.push_back(row);
                for (std::size_t column = 0; column < relation.columns.size(); ++column) {
                    relation.columns[column][relation.facts[row].arguments[column]].push_back(row);
                }
            }
        }
        std::unordered_map<ir::GroundAtom, double, AtomHash> staticValues;
        for (const auto& [application, value] : problem.staticFunctionValues) {
            if (!staticValues.emplace(application, value).second) {
                throw std::runtime_error("PDDL grounding error: duplicate numeric function initialization");
            }
        }

        for (const ir::ActionSchema& schema : problem.domain.actions) {
            std::vector<std::vector<ir::ObjectId>> candidates;
            candidates.resize(schema.parameters.size());

            for (std::size_t p = 0; p < schema.parameters.size(); ++p) {
                for (const ir::Object& object : objects) {
                    if (IsSubtype(
                            object.type,
                            schema.parameters[p].type,
                            problem.domain.types
                        )) {
                        candidates[p].push_back(object.id);
                    }
                }
            }

            // Join static positive relations before enumerating unconstrained
            // typed parameters. Sparse connectivity predicates otherwise produce
            // a huge Cartesian product of bindings that can never be applied.
            std::vector<const ir::LiteralSchema*> staticPositive;
            std::vector<const ir::LiteralSchema*> staticChecks;
            for (const auto& precondition : schema.preconditions) {
                if (precondition.predicate == ir::EqualityPredicate ||
                    !dynamicPredicate.at(precondition.predicate)) {
                    staticChecks.push_back(&precondition);
                    if (precondition.predicate != ir::EqualityPredicate && !precondition.negated) {
                        staticPositive.push_back(&precondition);
                    }
                }
            }
            std::vector<ir::ObjectId> binding(schema.parameters.size(), ir::InvalidObject);
            std::vector<bool> joined(staticPositive.size(), false);
            const auto fullyBound = [&](const ir::LiteralSchema& literal) {
                return std::all_of(literal.terms.begin(), literal.terms.end(), [&](const ir::Term& term) {
                    return term.kind == ir::TermKind::object || binding.at(term.id) != ir::InvalidObject;
                });
            };
            const auto consistent = [&] {
                for (const auto* literal : staticChecks) {
                    if (!fullyBound(*literal)) {
                        continue;
                    }
                    if (literal->predicate == ir::EqualityPredicate) {
                        if (!EqualitySatisfied(*literal, binding)) {
                            return false;
                        }
                    } else {
                        const bool present = initialFacts.contains(InstantiateLiteral(*literal, binding).atom);
                        if (present == literal->negated) {
                            return false;
                        }
                    }
                }
                return true;
            };
            const auto appendAction = [&] {
                GroundAction action;
                action.name = schema.name;
                action.arguments = binding;
                action.cost = schema.cost;
                for (const auto& function : schema.staticActionCosts) {
                    const auto atom = InstantiateLiteral(function, binding).atom;
                    const auto found = staticValues.find(atom);
                    if (found == staticValues.end()) {
                        // An undefined value is not zero in PDDL. Rejecting it
                        // avoids silently solving a different cost problem.
                        throw std::runtime_error("PDDL grounding error: uninitialized static cost function '" +
                            problem.domain.staticFunctions.at(function.predicate).name + "' in action '" + schema.name + "'");
                    }
                    if (!std::isfinite(found->second) || found->second < 0.0) {
                        throw std::runtime_error("PDDL grounding error: action cost increases must be finite and nonnegative");
                    }
                    action.cost += found->second;
                }
                if (!std::isfinite(action.cost) || action.cost < 0.0 ||
                    action.cost > static_cast<double>(std::numeric_limits<float>::max()) ||
                    (action.cost > 0.0 && static_cast<float>(action.cost) == 0.0F)) {
                    throw std::runtime_error("PDDL grounding error: action cost must be finite, nonnegative and representable as float");
                }
                for (const auto& literal : schema.preconditions) {
                    if (literal.predicate != ir::EqualityPredicate) {
                        action.preconditions.push_back(InstantiateLiteral(literal, binding));
                    }
                }
                for (const auto& literal : schema.effects) {
                    action.effects.push_back(InstantiateLiteral(literal, binding));
                }
                result.actions.push_back(std::move(action));
            };

            std::function<void()> enumerate;
            enumerate = [&] {
                if (!consistent()) {
                    return;
                }
                std::size_t selected = staticPositive.size();
                const std::vector<std::size_t>* selectedRows = nullptr;
                for (std::size_t i = 0; i < staticPositive.size(); ++i) {
                    if (joined[i]) {
                        continue;
                    }
                    const auto& literal = *staticPositive[i];
                    const auto& relation = relations.at(literal.predicate);
                    const auto* rows = &relation.allRows;
                    for (std::size_t column = 0; column < literal.terms.size(); ++column) {
                        const auto& term = literal.terms[column];
                        const auto value = ResolveTerm(term, binding);
                        if (value == ir::InvalidObject) {
                            continue;
                        }
                        const auto found = relation.columns[column].find(value);
                        if (found == relation.columns[column].end()) {
                            return;
                        }
                        if (found->second.size() < rows->size()) {
                            rows = &found->second;
                        }
                    }
                    if (selectedRows == nullptr || rows->size() < selectedRows->size()) {
                        selected = i;
                        selectedRows = rows;
                    }
                }
                if (selectedRows != nullptr) {
                    const auto& literal = *staticPositive[selected];
                    const auto& relation = relations.at(literal.predicate);
                    joined[selected] = true;
                    for (const auto row : *selectedRows) {
                        const auto& fact = relation.facts[row];
                        std::vector<ir::VariableId> assigned;
                        bool matches = true;
                        for (std::size_t column = 0; column < literal.terms.size(); ++column) {
                            const auto& term = literal.terms[column];
                            const auto value = fact.arguments[column];
                            const auto current = ResolveTerm(term, binding);
                            if (current != ir::InvalidObject) {
                                if (current != value) {
                                    matches = false;
                                    break;
                                }
                            } else if (IsSubtype(objects.at(value).type,
                                                 schema.parameters.at(term.id).type,
                                                 problem.domain.types)) {
                                binding[term.id] = value;
                                assigned.push_back(term.id);
                            } else {
                                matches = false;
                                break;
                            }
                        }
                        if (matches) {
                            enumerate();
                        }
                        for (const auto variable : assigned) {
                            binding[variable] = ir::InvalidObject;
                        }
                    }
                    joined[selected] = false;
                    return;
                }
                for (std::size_t parameter = 0; parameter < binding.size(); ++parameter) {
                    if (binding[parameter] != ir::InvalidObject) {
                        continue;
                    }
                    for (const auto object : candidates[parameter]) {
                        binding[parameter] = object;
                        enumerate();
                    }
                    binding[parameter] = ir::InvalidObject;
                    return;
                }
                appendAction();
            };
            const auto schemaBegin = result.actions.size();
            enumerate();
            // Joins may enumerate a different variable order. Preserve stable
            // action identities/order used by replay and deterministic searches.
            std::sort(result.actions.begin() + static_cast<std::ptrdiff_t>(schemaBegin), result.actions.end(),
                      [](const GroundAction& left, const GroundAction& right) {
                          return left.arguments < right.arguments;
                      });
        }

        return result;
    }
}
