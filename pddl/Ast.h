#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace planning::pddl
{
    struct SourceLocation {
        std::size_t line = 1;
        std::size_t column = 1;
    };

    enum class Requirement {
        strips,
        typing,
        negativePreconditions,
        equality,
        actionCosts,
        adl,
        disjunctivePreconditions,
        existentialPreconditions,
        universalPreconditions,
        conditionalEffects,
        numericFluents,
        durativeActions,
        unknown
    };

    struct TypedName {
        std::string name;
        std::string type = "object";
    };

    struct PredicateDecl {
        std::string name;
        std::vector<TypedName> parameters;
    };

    // A Literal is deliberately symbolic at the parser boundary. Variable terms
    // retain their leading '?' so semantic analysis can distinguish them from
    // object/constant names without a second side table.
    struct Literal {
        std::string predicate;
        std::vector<std::string> terms;
        bool negated = false;
        SourceLocation location{};
    };

    struct ActionSchema {
        std::string name;
        std::vector<TypedName> parameters;
        std::vector<Literal> preconditions;
        std::vector<Literal> effects;
        double actionCost = 1.0;
        bool hasExplicitActionCost = false;
        SourceLocation location{};
        // Static numeric function applications whose values are added to cost.
        std::vector<Literal> staticActionCosts;
    };

    struct Domain {
        std::string name;
        std::vector<Requirement> requirements;
        std::vector<TypedName> types;
        std::vector<TypedName> constants;
        std::vector<PredicateDecl> predicates;
        std::vector<ActionSchema> actions;
        // total-cost is handled separately; these functions are immutable.
        std::vector<PredicateDecl> staticFunctions;
    };

    // Reserved AST nodes for future PDDL 2.1 work. They are intentionally not
    // interpreted by the classical semantic analyzer in this revision. Keeping
    // them at the syntax boundary prevents the parser/IR design from assuming
    // that every future expression is a simple propositional literal.
    struct NumericExpression {
        std::string function;
        std::vector<std::string> terms;
        std::optional<double> constant;
    };

    struct DurationConstraint {
        std::string relation;
        NumericExpression expression;
    };

    enum class TemporalQualifier {
        atStart,
        overAll,
        atEnd
    };

    struct TimedCondition {
        TemporalQualifier qualifier = TemporalQualifier::atStart;
        Literal literal;
    };

    struct TimedEffect {
        TemporalQualifier qualifier = TemporalQualifier::atEnd;
        Literal literal;
    };

    struct Problem {
        std::string name;
        std::string domainName;
        std::vector<TypedName> objects;
        std::vector<Literal> initialFacts;
        std::vector<Literal> goal;
        bool minimizeTotalCost = false;
        double initialTotalCost = 0.0;
        std::vector<std::pair<Literal, double>> staticFunctionValues;
    };

    [[nodiscard]] std::string ToString(Requirement requirement);
    [[nodiscard]] Requirement RequirementFromString(const std::string& text);
}
