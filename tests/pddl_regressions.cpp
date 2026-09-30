#include "pddl/Adapter.h"
#include "pddl/Writer.h"
#include "tests/TestSupport.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

using namespace planning;
using namespace planning::pddl;
using test::Require;
using test::Throws;

namespace {
const std::string simpleProblem = "(define (problem p) (:domain d) (:init) (:goal (p)))";
const std::string simpleDomain = "(define (domain d) (:predicates (p)))";
const std::string costDomain = R"((define (domain d) (:requirements :strips :action-costs)
    (:predicates (p)) (:functions (total-cost) - number)
    (:action a :parameters () :precondition (and)
      :effect (and (p) (increase (total-cost) 2) (and (increase (total-cost) 3))))))";

PDDLState Initial(const PDDLPlanningProblem& p) {
    PDDLState result;
    p.ForEachInitialState([&](const auto& state) { result = state; return false; });
    return result;
}

void ParserErrors() {
    Parser parser;
    for (const std::string text : {
        "", "(", "(define (problem))", "(define (problem p) ())",
        "(define (problem p) garbage)",
        "(define (problem p extra) (:domain d) (:init) (:goal (p)))",
        "(define (problem p) (:domain d) (:init))",
        "(define (problem p) (:domain d) (:init) (:goal (p)) (:goal (and)))"}) {
        Throws([&] { (void)parser.ParseProblemText(text); }, "malformed problem is rejected");
    }
    Throws([&] { (void)parser.ParseDomain({}); }, "empty token stream is rejected");
    Throws([&] { (void)parser.ParseDomainText(
        "(define (domain d) (:predicates (p)) (:action a :parameters bad :effect (p)))");
    }, "nonlist parameters rejected");
    Throws([&] { (void)parser.ParseDomainText(
        "(define (domain d) (:predicates (p)) (:action a :effect (p) :effect (p)))");
    }, "duplicate action fields rejected");
    Throws([&] { (void)parser.ParseDomainText(
        "(define (domain d) (:functions (total-cost ?x)))");
    }, "total-cost must have zero arguments");
}

void SemanticErrors() {
    Throws([&] { (void)LoadGroundedProblemFromText(
        "(define (domain d) (:types a - b b - a) (:predicates (p)))", simpleProblem);
    }, "type cycles rejected before subtype traversal");
    Throws([&] { (void)LoadGroundedProblemFromText(
        "(define (domain d) (:types object) (:predicates (p)))", simpleProblem);
    }, "root type redeclaration rejected");
    Throws([&] { (void)LoadGroundedProblemFromText(
        "(define (domain d) (:predicates (p)) (:action a :effect (p)) (:action a :effect (p)))", simpleProblem);
    }, "duplicate action names rejected");
    Throws([&] { (void)LoadGroundedProblemFromText(
        "(define (domain d) (:predicates (p ?x)) (:action a :effect (p o)))",
        "(define (problem p) (:domain d) (:objects o) (:init) (:goal (p o)))");
    }, "domain cannot reference problem-only objects");
    for (const std::string cost : {"-1", "nan", "inf", "1e100", "1e-100", "1e-1000"}) {
        const std::string domain = "(define (domain d) (:requirements :action-costs) (:predicates (p)) "
            "(:action a :effect (and (p) (increase (total-cost) " + cost + "))))";
        Throws([&] { (void)LoadGroundedProblemFromText(domain, simpleProblem); }, "invalid action cost rejected");
    }
    Throws([&] { (void)LoadGroundedProblemFromText(
        "(define (domain d) (:predicates (p)) (:action a :effect (increase (total-cost) 2)))", simpleProblem);
    }, "cost effects cannot silently become unit costs");
    Throws([&] { (void)LoadGroundedProblemFromText(
        "(define (domain d) (:predicates (p)) (:action a :parameters (?x ?x) :effect (p)))",
        simpleProblem);
    }, "action parameters still require unique binding names");
    Throws([&] { (void)LoadGroundedProblemFromText(
        "(define (domain d) (:predicates (p) (q object)))", simpleProblem);
    }, "predicate declaration placeholders must still be variables");
}

void PredicatePlaceholders() {
    // Authentic IPC Logistics declares (in ?obj ?obj). The same placeholder
    // may even appear in positions with different types: it is not a binding.
    const std::string domain = R"((define (domain d) (:requirements :strips :typing)
        (:types package truck)
        (:predicates (in ?obj - package ?obj - truck))
        (:action load :parameters (?p - package ?t - truck)
          :precondition (and) :effect (in ?p ?t))))";
    const std::string problem = R"((define (problem p) (:domain d)
        (:objects parcel - package vehicle - truck)
        (:init) (:goal (in parcel vehicle))))";
    const auto compiled = LoadGroundedProblemFromText(domain, problem);
    Require(compiled.GroundActionCount() == 1, "repeated predicate placeholders preserve positional types");
    auto final = compiled.Apply(Initial(compiled), compiled.CompiledActions().front());
    Require(final && compiled.IsGoal(*final), "repeated predicate placeholders do not impose argument equality");
    Throws([&] { (void)LoadGroundedProblemFromText(domain,
        "(define (problem p) (:domain d) (:objects parcel - package vehicle - truck) "
        "(:init) (:goal (in vehicle parcel)))");
    }, "repeated predicate placeholders still check each argument's type");
    Throws([&] { (void)LoadGroundedProblemFromText(domain,
        "(define (problem p) (:domain d) (:objects parcel - package) "
        "(:init) (:goal (in parcel)))");
    }, "repeated predicate placeholders still enforce arity");
}

void RoundTripsAndSemantics() {
    const auto costProblem = LoadGroundedProblemFromText(costDomain, simpleProblem);
    Require(costProblem.CompiledActions().front().cost == 5.0F, "multiple increases sum");
    auto final = costProblem.Apply(Initial(costProblem), costProblem.CompiledActions().front());
    Require(final && costProblem.IsGoal(*final), "nested conjunction effects execute");

    Parser parser;
    Writer writer;
    auto domain = parser.ParseDomainText(R"((define (domain d) (:requirements :strips :typing :action-costs)
        (:types vehicle - object car - vehicle)
        (:constants thing - object small - car)
        (:predicates (p) (q ?x - object ?y - vehicle))
        (:action free :parameters (?x - object ?y - vehicle) :effect (p))
        (:action paid :parameters () :effect (and (p) (increase (total-cost) 0.12345678901234567)))))");
    const auto restored = parser.ParseDomainText(writer.DomainText(domain));
    Require(restored.types[1].type == "object" && restored.types[2].type == "vehicle", "type hierarchy round trip");
    Require(restored.constants[0].type == "object", "mixed constant types round trip");
    Require(restored.predicates[1].parameters[0].type == "object", "mixed predicate types round trip");
    Require(restored.actions[0].parameters[0].type == "object", "mixed action parameter types round trip");
    Require(!restored.actions[0].hasExplicitActionCost, "unspecified zero action cost preserved");
    Require(restored.actions[1].actionCost == domain.actions[1].actionCost, "double costs round trip exactly");
    const auto compiled = LoadGroundedProblemFromText(writer.DomainText(domain), simpleProblem);
    Require(compiled.CompiledActions()[0].cost == 0.0F, "no increase means zero with action costs");

    auto problem = parser.ParseProblemText(
        "(define (problem p) (:domain d) (:objects plain - object car1 - car) (:init) (:goal (p)))");
    const auto restoredProblem = parser.ParseProblemText(writer.ProblemText(problem));
    Require(restoredProblem.objects[0].type == "object", "mixed object types round trip");

    const auto unsolvable = LoadGroundedProblemFromText(
        "(define (domain d) (:requirements :equality) (:predicates (p)))",
        "(define (problem p) (:domain d) (:objects a b) (:init) (:goal (= a b)))");
    Require(!unsolvable.IsGoal(Initial(unsolvable)), "false equality is unsolvable without load failure");
    Require(unsolvable.PositiveGoal().Intersects(unsolvable.NegativeGoal()), "false equality visible in compiled masks");

    const auto typed = LoadGroundedProblemFromText(R"((define (domain d)
        (:requirements :strips :typing :negative-preconditions :equality)
        (:types vehicle - object car - vehicle)
        (:predicates (p ?x - vehicle))
        (:action mark :parameters (?x - vehicle ?y - vehicle)
          :precondition (and (not (= ?x ?y)) (and (not (p ?x))))
          :effect (and (not (p ?x)) (p ?x)))))",
        "(define (problem p) (:domain d) (:objects a b - car plain - object) (:init) (:goal (and (p a) (not (p b)))))");
    Require(typed.GroundActionCount() == 2, "subtypes and inequality prune action bindings");
    auto state = Initial(typed);
    Require(!typed.IsGoal(state), "positive goal absent initially");
    auto marked = typed.Apply(state, typed.CompiledActions().front());
    Require(marked && typed.IsGoal(*marked), "negative preconditions/goals and simultaneous add/delete semantics");
    Require(!typed.Apply(*marked, typed.CompiledActions().front()), "negative precondition disables repeat");

    DynamicBitset bits(128);
    bits.Set(63); bits.Set(64);
    bits.Resize(63);
    Require(!bits.Test(63) && !bits.Test(64), "resize clears all truncated bits");
    DynamicBitset padded(256), empty;
    Require(MakeLogicStateKey(padded) == MakeLogicStateKey(empty), "padding preserves exact history identity");
}

void StaticNumericCosts() {
    const std::string domain = R"((define (domain d)
      (:requirements :adl :action-costs)
      (:types place)
      (:predicates (at ?p - place) (edge ?a ?b - place))
      (:functions (total-cost) (distance ?from ?to - place) - number (fee) - number)
      (:action move :parameters (?a ?b - place)
        :precondition (and (at ?a) (edge ?a ?b))
        :effect (and (not (at ?a)) (at ?b)
          (increase (total-cost) (distance ?a ?b))
          (increase (total-cost) (fee)) (increase (total-cost) 1)))))";
    const std::string problem = R"((define (problem p) (:domain d)
      (:objects a b c - place)
      (:init (at a) (edge a b) (edge b c)
        (= (distance a b) 2) (= (distance b c) 4) (= (fee) 0.5) (= (total-cost) 0))
      (:goal (at c)) (:metric minimize (total-cost))))";
    const auto compiled = LoadGroundedProblemFromText(domain, problem);
    Require(compiled.GroundActionCount() == 2, "static relations prune cost-undefined impossible bindings");
    Require(compiled.CompiledActions()[0].cost == 3.5F && compiled.CompiledActions()[1].cost == 5.5F,
            "static functions and numeric increases sum for each grounded binding");
    auto state = Initial(compiled);
    for (const auto& action : compiled.CompiledActions()) {
        const auto next = compiled.Apply(state, action);
        Require(next.has_value(), "static numeric cost action remains applicable");
        state = *next;
    }
    Require(compiled.IsGoal(state), "numeric action cost does not change logical effects");
    Parser parser;
    Writer writer;
    const auto roundTrip = LoadGroundedProblemFromText(
        writer.DomainText(parser.ParseDomainText(domain)),
        writer.ProblemText(parser.ParseProblemText(problem)));
    Require(roundTrip.GroundActionCount() == 2 && roundTrip.CompiledActions()[1].cost == 5.5F,
            "static function declarations, applications and values round trip");
    for (const auto& init : {std::string(""), std::string("(= (distance a b) 2) (= (distance a b) 3)"),
                            std::string("(= (distance a b) -2)"), std::string("(= (distance a b) 1e100)")}) {
        Throws([&] { (void)LoadGroundedProblemFromText(domain,
            "(define (problem p) (:domain d) (:objects a b - place) "
            "(:init (at a) (edge a b) (= (fee) 0) " + init + ") (:goal (at b)))");
        }, "undefined, duplicate, negative and overflowing numeric costs rejected");
    }
    for (const auto& application : {"(missing ?a)", "(distance ?a)", "(distance ?a ?unknown)", "(total-cost)"}) {
        const std::string invalid = "(define (domain d) (:requirements :action-costs) "
            "(:predicates (p)) (:functions (distance ?x ?y)) "
            "(:action a :parameters (?a) :effect (increase (total-cost) " + std::string(application) + ")))";
        Throws([&] { (void)LoadGroundedProblemFromText(invalid,
            "(define (problem p) (:domain d) (:objects obj) (:init) (:goal (p)))");
        }, "numeric application requires a declared static function and valid arguments");
    }
    for (const auto& expression : {"(when (p) (p))", "(forall (?x) (p))", "(or (p) (p))", "(assign (fee) 1)"}) {
        Throws([&] { (void)LoadGroundedProblemFromText(
            "(define (domain d) (:requirements :adl) (:predicates (p)) "
            "(:action a :effect " + std::string(expression) + "))", simpleProblem);
        }, "ADL declaration does not silently admit unsupported syntax");
    }
}

void StaticJoinGrounding() {
    const std::string domain = R"((define (domain d) (:requirements :strips :typing :negative-preconditions :equality)
      (:types node - object special - node)
      (:constants anchor - special)
      (:predicates (edge ?x ?y - node) (blocked ?x - node) (done ?x ?y - node) (enabled))
      (:action join :parameters (?a - special ?b - node ?c - node)
        :precondition (and (enabled) (edge anchor ?a) (edge ?a ?b) (edge ?b ?c)
          (edge ?b ?b) (not (blocked ?c)) (not (= ?a ?c)))
        :effect (done ?a ?c))))";
    const std::string problem = R"((define (problem p) (:domain d)
      (:objects a - special b c z - node)
      (:init (enabled) (edge anchor a) (edge anchor b) (edge a b)
        (edge b b) (edge b c) (edge b z) (edge a b) (blocked z))
      (:goal (done a c))))";
    const auto ground = LoadGroundedProblemFromText(domain, problem);
    Require(ground.GroundActionCount() == 2, "indexed joins enforce constants, repeated variables, subtypes and negative facts");
    auto initial = Initial(ground);
    for (const auto& action : ground.CompiledActions()) {
        Require(ground.Apply(initial, action).has_value(), "every joined binding satisfies all static preconditions");
    }
    const auto empty = LoadGroundedProblemFromText(domain,
        "(define (problem p) (:domain d) (:objects a - special b c - node) (:init) (:goal (done a c)))");
    Require(empty.GroundActionCount() == 0, "false zero-arity static predicate removes all bindings");

    // 200^6 naive bindings versus 195 rows from five joins. This regression
    // checks the sparse relational grounding path without a timing assertion.
    const std::string chainDomain = R"((define (domain d) (:requirements :typing)
      (:types node) (:predicates (edge ?x ?y - node) (done ?x - node))
      (:action walk :parameters (?a ?b ?c ?d ?e ?f - node)
        :precondition (and (edge ?a ?b) (edge ?b ?c) (edge ?c ?d) (edge ?d ?e) (edge ?e ?f))
        :effect (done ?f))))";
    std::ostringstream chain;
    chain << "(define (problem p) (:domain d) (:objects";
    for (int i = 0; i < 200; ++i) chain << " n" << i;
    chain << " - node) (:init";
    for (int i = 0; i < 199; ++i) chain << " (edge n" << i << " n" << (i + 1) << ')';
    chain << ") (:goal (done n199)))";
    const auto sparse = LoadGroundedProblemFromText(chainDomain, chain.str());
    Require(sparse.GroundActionCount() == 195, "sparse static joins avoid the full typed Cartesian product");
}
}

int main() {
    try {
        ParserErrors();
        SemanticErrors();
        PredicatePlaceholders();
        RoundTripsAndSemantics();
        StaticNumericCosts();
        StaticJoinGrounding();
        std::cout << "PDDL regressions: " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
