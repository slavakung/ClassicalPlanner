#include "pddl/SemanticAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace planning::pddl
{
    namespace
    {
        [[nodiscard]] bool SupportedRequirement(Requirement requirement) {
            switch (requirement) {
                case Requirement::strips:
                case Requirement::typing:
                case Requirement::negativePreconditions:
                case Requirement::equality:
                case Requirement::actionCosts:
                // A bundled declaration is accepted; the parser still rejects
                // every unsupported ADL construct instead of approximating it.
                case Requirement::adl:
                    return true;
                default:
                    return false;
            }
        }

        [[nodiscard]] bool IsVariable(const std::string& term) {
            return !term.empty() && term.front() == '?';
        }

        [[nodiscard]] bool IsSubtype(
            ir::TypeId actual,
            ir::TypeId required,
            const std::vector<ir::Type>& types
        ) {
            if (actual == required) {
                return true;
            }

            while (actual != ir::InvalidType) {
                const ir::Type& type = types.at(actual);
                actual = type.parent;
                if (actual == required) {
                    return true;
                }
            }
            return false;
        }

        [[noreturn]] void SemanticError(const std::string& message) {
            throw std::runtime_error("PDDL semantic error: " + message);
        }

        void ValidateCost(double cost, const std::string& context) {
            if (!std::isfinite(cost) || cost < 0.0 ||
                cost > static_cast<double>(std::numeric_limits<float>::max()) ||
                (cost > 0.0 && static_cast<float>(cost) == 0.0F)) {
                SemanticError(context + " must be finite, nonnegative and representable as float");
            }
        }
    }

    ir::Problem SemanticAnalyzer::Analyze(
        const Domain& domain,
        const Problem& problem
    ) const {
        if (domain.name != problem.domainName) {
            SemanticError(
                "problem references domain '" + problem.domainName +
                "' but parsed domain is '" + domain.name + "'"
            );
        }

        bool hasActionCosts = false;
        for (Requirement requirement : domain.requirements) {
            hasActionCosts = hasActionCosts || requirement == Requirement::actionCosts;
            if (!SupportedRequirement(requirement)) {
                SemanticError(
                    "unsupported requirement " + ToString(requirement) +
                    "; supported subset is :strips, :typing, "
                    ":negative-preconditions, :equality, :action-costs; :adl is "
                    "accepted only when the actual syntax stays in this subset"
                );
            }
        }

        if (problem.minimizeTotalCost && !hasActionCosts) {
            SemanticError(
                "problem minimizes total-cost but domain does not declare :action-costs"
            );
        }
        ValidateCost(problem.initialTotalCost, "initial total-cost");
        if (problem.initialTotalCost != 0.0 && !hasActionCosts) {
            SemanticError("initial total-cost requires :action-costs");
        }

        ir::Problem result;
        result.name = problem.name;
        result.domainName = domain.name;
        result.domain.name = domain.name;
        result.minimizeTotalCost = problem.minimizeTotalCost;
        result.initialTotalCost = problem.initialTotalCost;

        std::unordered_map<std::string, ir::TypeId> typeIds;
        auto addType = [&](const std::string& name) -> ir::TypeId {
            if (typeIds.contains(name)) {
                SemanticError("duplicate type '" + name + "'");
            }
            const auto id = static_cast<ir::TypeId>(result.domain.types.size());
            typeIds.emplace(name, id);
            result.domain.types.push_back(ir::Type{id, name, ir::InvalidType});
            return id;
        };

        // Object is the distinguished root type even if not explicitly listed.
        addType("object");
        bool sawRootType = false;
        for (const TypedName& type : domain.types) {
            if (type.name == "object") {
                if (sawRootType || !type.type.empty()) {
                    SemanticError("the root type 'object' cannot be redeclared");
                }
                sawRootType = true;
                continue;
            }
            addType(type.name);
        }

        for (const TypedName& type : domain.types) {
            if (type.name == "object") {
                continue;
            }
            const auto child = typeIds.at(type.name);
            const std::string parentName = type.type.empty() ? "object" : type.type;
            const auto parentIt = typeIds.find(parentName);
            if (parentIt == typeIds.end()) {
                SemanticError("unknown parent type '" + parentName + "'");
            }
            result.domain.types[child].parent = parentIt->second;
        }

        // Validate before any subtype traversal; cyclic declarations otherwise
        // make both semantic analysis and grounding loop forever.
        for (const ir::Type& type : result.domain.types) {
            auto ancestor = type.id;
            std::unordered_set<ir::TypeId> visited;
            while (ancestor != ir::InvalidType) {
                if (!visited.insert(ancestor).second) {
                    SemanticError("cyclic type hierarchy involving '" + type.name + "'");
                }
                ancestor = result.domain.types.at(ancestor).parent;
            }
        }

        const auto typeOf = [&](const std::string& name) -> ir::TypeId {
            const auto it = typeIds.find(name);
            if (it == typeIds.end()) {
                SemanticError("unknown type '" + name + "'");
            }
            return it->second;
        };

        std::unordered_map<std::string, ir::ObjectId> objectIds;
        auto addObject = [&](const TypedName& object, bool constant) {
            if (object.name.empty() || IsVariable(object.name)) {
                SemanticError("object/constant names cannot be variables");
            }
            if (objectIds.contains(object.name)) {
                SemanticError("duplicate object/constant '" + object.name + "'");
            }
            const auto id = static_cast<ir::ObjectId>(
                result.domain.constants.size() + result.objects.size()
            );
            const ir::Object converted{id, object.name, typeOf(object.type)};
            objectIds.emplace(object.name, id);
            if (constant) {
                result.domain.constants.push_back(converted);
            } else {
                result.objects.push_back(converted);
            }
        };

        for (const TypedName& constant : domain.constants) {
            addObject(constant, true);
        }
        for (const TypedName& object : problem.objects) {
            addObject(object, false);
        }

        // IDs assigned above used constants.size()+objects.size(), which is only
        // contiguous if constants are all inserted before objects (they are). Build
        // a dense lookup vector once so later validation is constant-time.
        std::vector<ir::Object> allObjects;
        allObjects.reserve(result.domain.constants.size() + result.objects.size());
        allObjects.insert(
            allObjects.end(),
            result.domain.constants.begin(),
            result.domain.constants.end()
        );
        allObjects.insert(allObjects.end(), result.objects.begin(), result.objects.end());

        std::unordered_map<std::string, ir::PredicateId> predicateIds;
        for (const PredicateDecl& predicate : domain.predicates) {
            if (predicate.name == "=") {
                SemanticError("'=' is reserved for PDDL equality");
            }
            if (predicateIds.contains(predicate.name)) {
                SemanticError("duplicate predicate '" + predicate.name + "'");
            }

            ir::Predicate converted;
            converted.id = static_cast<ir::PredicateId>(result.domain.predicates.size());
            converted.name = predicate.name;
            for (const TypedName& parameter : predicate.parameters) {
                // Predicate declarations describe positional argument types;
                // these names introduce no bindings. IPC Logistics, for example,
                // declares (in ?obj ?obj). Repeated placeholders therefore do
                // not impose equality, unlike variables in action expressions.
                if (!IsVariable(parameter.name)) {
                    SemanticError("predicate '" + predicate.name + "' requires variable parameters");
                }
                converted.parameterTypes.push_back(typeOf(parameter.type));
            }
            predicateIds.emplace(predicate.name, converted.id);
            result.domain.predicates.push_back(std::move(converted));
        }

        const auto predicateOf = [&](const std::string& name) -> ir::PredicateId {
            if (name == "=") {
                return ir::EqualityPredicate;
            }
            const auto it = predicateIds.find(name);
            if (it == predicateIds.end()) {
                SemanticError("unknown predicate '" + name + "'");
            }
            return it->second;
        };

        std::unordered_map<std::string, ir::PredicateId> functionIds;
        for (const auto& function : domain.staticFunctions) {
            if (!hasActionCosts) {
                SemanticError("static numeric cost functions require :action-costs");
            }
            if (function.name == "total-cost" || function.name == "=" ||
                functionIds.contains(function.name)) {
                SemanticError("invalid or duplicate static function '" + function.name + "'");
            }
            ir::Predicate converted;
            converted.id = static_cast<ir::PredicateId>(result.domain.staticFunctions.size());
            converted.name = function.name;
            for (const auto& parameter : function.parameters) {
                if (!IsVariable(parameter.name)) {
                    SemanticError("function '" + function.name + "' requires variable parameters");
                }
                converted.parameterTypes.push_back(typeOf(parameter.type));
            }
            functionIds.emplace(function.name, converted.id);
            result.domain.staticFunctions.push_back(std::move(converted));
        }

        // Function and predicate namespaces are separate, but their arguments
        // obey the same arity, binding and subtype rules.
        auto staticFunctionOf = [&](const Literal& application) -> const ir::Predicate& {
            const auto found = functionIds.find(application.predicate);
            if (found == functionIds.end() || application.negated) {
                SemanticError("unknown or invalid static numeric function '" + application.predicate + "'");
            }
            const auto& function = result.domain.staticFunctions.at(found->second);
            if (function.parameterTypes.size() != application.terms.size()) {
                SemanticError("arity mismatch for numeric function '" + application.predicate + "'");
            }
            return function;
        };

        for (const auto& [application, value] : problem.staticFunctionValues) {
            const auto& function = staticFunctionOf(application);
            // Static values can be negative in PDDL; each resulting action cost
            // is checked separately after grounding. Numeric conditions remain
            // unsupported, so these values cannot otherwise affect the state.
            if (!std::isfinite(value)) {
                SemanticError("static numeric values must be finite");
            }
            ir::GroundAtom atom;
            atom.predicate = function.id;
            for (std::size_t i = 0; i < application.terms.size(); ++i) {
                const auto& term = application.terms[i];
                const auto found = objectIds.find(term);
                if (found == objectIds.end() || IsVariable(term)) {
                    SemanticError("unknown object in numeric function initialization '" + term + "'");
                }
                if (!IsSubtype(allObjects.at(found->second).type, function.parameterTypes[i], result.domain.types)) {
                    SemanticError("incompatible object type in numeric function initialization");
                }
                atom.arguments.push_back(found->second);
            }
            result.staticFunctionValues.emplace_back(std::move(atom), value);
        }

        auto convertGroundLiteral = [&](const Literal& literal) -> ir::GroundLiteral {
            ir::GroundLiteral converted;
            converted.negated = literal.negated;
            converted.atom.predicate = predicateOf(literal.predicate);

            if (converted.atom.predicate == ir::EqualityPredicate) {
                if (literal.terms.size() != 2) {
                    SemanticError("equality requires exactly two terms");
                }
            } else {
                const auto& predicate = result.domain.predicates.at(converted.atom.predicate);
                if (literal.terms.size() != predicate.parameterTypes.size()) {
                    SemanticError("arity mismatch for predicate '" + literal.predicate + "'");
                }
            }

            for (std::size_t i = 0; i < literal.terms.size(); ++i) {
                const std::string& term = literal.terms[i];
                if (IsVariable(term)) {
                    SemanticError("variable '" + term + "' is not allowed in a ground problem fact/goal");
                }
                const auto objectIt = objectIds.find(term);
                if (objectIt == objectIds.end()) {
                    SemanticError("unknown object/constant '" + term + "'");
                }

                const ir::ObjectId object = objectIt->second;
                if (converted.atom.predicate != ir::EqualityPredicate) {
                    const ir::TypeId required = result.domain.predicates
                        .at(converted.atom.predicate).parameterTypes.at(i);
                    if (!IsSubtype(allObjects.at(object).type, required, result.domain.types)) {
                        SemanticError("object '" + term + "' has incompatible type in predicate '" + literal.predicate + "'");
                    }
                }
                converted.atom.arguments.push_back(object);
            }
            return converted;
        };

        for (const Literal& fact : problem.initialFacts) {
            ir::GroundLiteral converted = convertGroundLiteral(fact);
            if (converted.negated) {
                SemanticError("initial state must contain positive facts only");
            }
            if (converted.atom.predicate == ir::EqualityPredicate) {
                SemanticError("equality is interpreted, not stored as an initial fact");
            }
            result.initialFacts.push_back(std::move(converted.atom));
        }

        for (const Literal& goal : problem.goal) {
            result.goal.push_back(convertGroundLiteral(goal));
        }

        std::unordered_set<std::string> actionNames;
        for (const pddl::ActionSchema& action : domain.actions) {
            if (!actionNames.insert(action.name).second) {
                SemanticError("duplicate action '" + action.name + "'");
            }
            if (action.hasExplicitActionCost && !hasActionCosts) {
                SemanticError("action-cost effects require :action-costs");
            }
            if (action.hasExplicitActionCost) {
                ValidateCost(action.actionCost, "cost of action '" + action.name + "'");
            }
            ir::ActionSchema converted;
            converted.name = action.name;
            converted.cost = hasActionCosts
                ? (action.hasExplicitActionCost ? action.actionCost : 0.0)
                : 1.0;

            std::unordered_map<std::string, ir::VariableId> variableIds;
            for (const TypedName& parameter : action.parameters) {
                if (!IsVariable(parameter.name)) {
                    SemanticError("action parameter '" + parameter.name + "' must begin with '?'");
                }
                if (variableIds.contains(parameter.name)) {
                    SemanticError("duplicate parameter '" + parameter.name + "' in action '" + action.name + "'");
                }
                const auto id = static_cast<ir::VariableId>(converted.parameters.size());
                variableIds.emplace(parameter.name, id);
                converted.parameters.push_back(ir::Variable{id, parameter.name, typeOf(parameter.type)});
            }

            for (const auto& application : action.staticActionCosts) {
                const auto& function = staticFunctionOf(application);
                ir::LiteralSchema costTerm;
                costTerm.predicate = function.id;
                for (std::size_t i = 0; i < application.terms.size(); ++i) {
                    const auto& term = application.terms[i];
                    if (IsVariable(term)) {
                        const auto found = variableIds.find(term);
                        if (found == variableIds.end()) {
                            SemanticError("undeclared variable '" + term + "' in action cost");
                        }
                        if (!IsSubtype(converted.parameters.at(found->second).type,
                                       function.parameterTypes[i], result.domain.types)) {
                            SemanticError("incompatible variable type in action cost");
                        }
                        costTerm.terms.push_back(ir::Term{ir::TermKind::variable, found->second});
                    } else {
                        const auto found = objectIds.find(term);
                        if (found == objectIds.end() || found->second >= result.domain.constants.size()) {
                            SemanticError("undeclared domain constant '" + term + "' in action cost");
                        }
                        if (!IsSubtype(allObjects.at(found->second).type,
                                       function.parameterTypes[i], result.domain.types)) {
                            SemanticError("incompatible constant type in action cost");
                        }
                        costTerm.terms.push_back(ir::Term{ir::TermKind::object, found->second});
                    }
                }
                converted.staticActionCosts.push_back(std::move(costTerm));
            }

            auto convertSchemaLiteral = [&](const Literal& literal) -> ir::LiteralSchema {
                ir::LiteralSchema out;
                out.negated = literal.negated;
                out.predicate = predicateOf(literal.predicate);

                if (out.predicate == ir::EqualityPredicate) {
                    if (literal.terms.size() != 2) {
                        SemanticError("equality in action '" + action.name + "' requires two terms");
                    }
                } else {
                    const auto& predicate = result.domain.predicates.at(out.predicate);
                    if (literal.terms.size() != predicate.parameterTypes.size()) {
                        SemanticError("arity mismatch for predicate '" + literal.predicate + "' in action '" + action.name + "'");
                    }
                }

                for (std::size_t i = 0; i < literal.terms.size(); ++i) {
                    const std::string& term = literal.terms[i];
                    if (IsVariable(term)) {
                        const auto variableIt = variableIds.find(term);
                        if (variableIt == variableIds.end()) {
                            SemanticError("undeclared variable '" + term + "' in action '" + action.name + "'");
                        }
                        const auto variable = variableIt->second;
                        if (out.predicate != ir::EqualityPredicate) {
                            const ir::TypeId required = result.domain.predicates
                                .at(out.predicate).parameterTypes.at(i);
                            const ir::TypeId actual = converted.parameters.at(variable).type;
                            if (!IsSubtype(actual, required, result.domain.types)) {
                                SemanticError("variable '" + term + "' has incompatible type in action '" + action.name + "'");
                            }
                        }
                        out.terms.push_back(ir::Term{ir::TermKind::variable, variable});
                    } else {
                        const auto objectIt = objectIds.find(term);
                        if (objectIt == objectIds.end()) {
                            SemanticError("unknown constant/object '" + term + "' in action '" + action.name + "'");
                        }
                        const ir::ObjectId object = objectIt->second;
                        if (object >= result.domain.constants.size()) {
                            SemanticError("action '" + action.name + "' references problem object '" + term + "'; declare it as a domain constant");
                        }
                        if (out.predicate != ir::EqualityPredicate) {
                            const ir::TypeId required = result.domain.predicates
                                .at(out.predicate).parameterTypes.at(i);
                            if (!IsSubtype(allObjects.at(object).type, required, result.domain.types)) {
                                SemanticError("constant/object '" + term + "' has incompatible type in action '" + action.name + "'");
                            }
                        }
                        out.terms.push_back(ir::Term{ir::TermKind::object, object});
                    }
                }
                return out;
            };

            for (const Literal& literal : action.preconditions) {
                converted.preconditions.push_back(convertSchemaLiteral(literal));
            }
            for (const Literal& literal : action.effects) {
                if (literal.predicate == "=") {
                    SemanticError("equality effects are not allowed");
                }
                converted.effects.push_back(convertSchemaLiteral(literal));
            }

            result.domain.actions.push_back(std::move(converted));
        }

        return result;
    }
}
