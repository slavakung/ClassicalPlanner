#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace planning::ir
{
    using TypeId = std::uint32_t;
    using ObjectId = std::uint32_t;
    using PredicateId = std::uint32_t;
    using VariableId = std::uint32_t;

    inline constexpr TypeId InvalidType = static_cast<TypeId>(-1);
    inline constexpr ObjectId InvalidObject = static_cast<ObjectId>(-1);
    inline constexpr PredicateId EqualityPredicate = static_cast<PredicateId>(-1);

    struct Type {
        TypeId id = InvalidType;
        std::string name;
        TypeId parent = InvalidType;
    };

    struct Object {
        ObjectId id = InvalidObject;
        std::string name;
        TypeId type = InvalidType;
    };

    struct Predicate {
        PredicateId id = EqualityPredicate;
        std::string name;
        std::vector<TypeId> parameterTypes;
    };

    enum class TermKind {
        variable,
        object
    };

    struct Term {
        TermKind kind = TermKind::object;
        std::uint32_t id = 0;
    };

    struct LiteralSchema {
        PredicateId predicate = EqualityPredicate;
        std::vector<Term> terms;
        bool negated = false;
    };

    struct Variable {
        VariableId id = 0;
        std::string name;
        TypeId type = InvalidType;
    };

    struct ActionSchema {
        std::string name;
        std::vector<Variable> parameters;
        std::vector<LiteralSchema> preconditions;
        std::vector<LiteralSchema> effects;
        double cost = 1.0;
        // The predicate field indexes Domain::staticFunctions, not predicates.
        std::vector<LiteralSchema> staticActionCosts;
    };

    struct GroundAtom {
        PredicateId predicate = EqualityPredicate;
        std::vector<ObjectId> arguments;

        [[nodiscard]] bool operator==(const GroundAtom&) const = default;
    };

    struct GroundLiteral {
        GroundAtom atom;
        bool negated = false;
    };

    struct Domain {
        std::string name;
        std::vector<Type> types;
        std::vector<Object> constants;
        std::vector<Predicate> predicates;
        std::vector<ActionSchema> actions;
        std::vector<Predicate> staticFunctions;
    };

    struct Problem {
        std::string name;
        std::string domainName;
        Domain domain;
        std::vector<Object> objects;
        std::vector<GroundAtom> initialFacts;
        std::vector<GroundLiteral> goal;
        bool minimizeTotalCost = false;
        double initialTotalCost = 0.0;
        // GroundAtom::predicate indexes Domain::staticFunctions here.
        std::vector<std::pair<GroundAtom, double>> staticFunctionValues;
    };
}
