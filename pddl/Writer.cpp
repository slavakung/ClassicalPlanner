#include "pddl/Writer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace planning::pddl
{
    namespace
    {
        void WriteTypedList(
            std::ostringstream& out,
            const std::vector<TypedName>& values
        ) {
            // Write one name/type pair per item. It is slightly more verbose than
            // grouping equal types but much easier to inspect and round-trip.
            const bool typed = std::any_of(values.begin(), values.end(),
                [](const TypedName& value) { return !value.type.empty() && value.type != "object"; });
            for (const TypedName& value : values) {
                out << ' ' << value.name;
                if (typed) {
                    out << " - " << (value.type.empty() ? "object" : value.type);
                }
            }
        }

        void WriteLiteral(std::ostringstream& out, const Literal& literal) {
            if (literal.negated) {
                out << "(not ";
            }
            out << '(' << literal.predicate;
            for (const std::string& term : literal.terms) {
                out << ' ' << term;
            }
            out << ')';
            if (literal.negated) {
                out << ')';
            }
        }

        void WriteConjunction(
            std::ostringstream& out,
            const std::vector<Literal>& literals,
            std::string_view indent
        ) {
            if (literals.empty()) {
                out << "(and)";
                return;
            }
            if (literals.size() == 1) {
                WriteLiteral(out, literals.front());
                return;
            }

            out << "(and\n";
            for (const Literal& literal : literals) {
                out << indent;
                WriteLiteral(out, literal);
                out << '\n';
            }
            out << std::string(indent.size() >= 2 ? indent.size() - 2 : 0, ' ')
                << ')';
        }
    }

    std::string Writer::DomainText(const Domain& domain) const {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "(define (domain " << domain.name << ")\n";

        if (!domain.requirements.empty()) {
            out << "  (:requirements";
            for (Requirement requirement : domain.requirements) {
                out << ' ' << ToString(requirement);
            }
            out << ")\n";
        }

        if (std::any_of(domain.types.begin(), domain.types.end(),
                        [](const TypedName& t) { return t.name != "object"; })) {
            out << "  (:types";
            for (const TypedName& type : domain.types) {
                if (type.name == "object") {
                    continue;
                }
                out << ' ' << type.name;
                out << " - " << (type.type.empty() ? "object" : type.type);
            }
            out << ")\n";
        }

        if (!domain.constants.empty()) {
            out << "  (:constants";
            WriteTypedList(out, domain.constants);
            out << ")\n";
        }

        out << "  (:predicates\n";
        for (const PredicateDecl& predicate : domain.predicates) {
            out << "    (" << predicate.name;
            WriteTypedList(out, predicate.parameters);
            out << ")\n";
        }
        out << "  )\n";

        bool usesActionCosts = false;
        for (Requirement requirement : domain.requirements) {
            usesActionCosts = usesActionCosts || requirement == Requirement::actionCosts;
        }
        if (usesActionCosts) {
            out << "  (:functions (total-cost)";
            for (const auto& function : domain.staticFunctions) {
                out << " (" << function.name;
                WriteTypedList(out, function.parameters);
                out << ')';
            }
            out << ")\n";
        }

        for (const ActionSchema& action : domain.actions) {
            out << "  (:action " << action.name << "\n";
            out << "    :parameters (";
            WriteTypedList(out, action.parameters);
            out << ")\n";
            out << "    :precondition ";
            WriteConjunction(out, action.preconditions, "      ");
            out << "\n    :effect ";

            if (usesActionCosts) {
                out << "(and\n";
                for (const Literal& literal : action.effects) {
                    out << "      ";
                    WriteLiteral(out, literal);
                    out << '\n';
                }
                if (action.hasExplicitActionCost) {
                    out << "      (increase (total-cost) " << action.actionCost << ")\n";
                    for (const auto& function : action.staticActionCosts) {
                        out << "      (increase (total-cost) ";
                        WriteLiteral(out, function);
                        out << ")\n";
                    }
                }
                out << "    )\n";
            } else {
                WriteConjunction(out, action.effects, "      ");
                out << '\n';
            }
            out << "  )\n";
        }
        out << ")\n";
        return out.str();
    }

    std::string Writer::ProblemText(const Problem& problem) const {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "(define (problem " << problem.name << ")\n";
        out << "  (:domain " << problem.domainName << ")\n";
        if (!problem.objects.empty()) {
            out << "  (:objects";
            WriteTypedList(out, problem.objects);
            out << ")\n";
        }
        out << "  (:init\n";
        for (const Literal& fact : problem.initialFacts) {
            out << "    ";
            WriteLiteral(out, fact);
            out << '\n';
        }
        for (const auto& [function, value] : problem.staticFunctionValues) {
            out << "    (= ";
            WriteLiteral(out, function);
            out << ' ' << value << ")\n";
        }
        if (problem.minimizeTotalCost || problem.initialTotalCost != 0.0) {
            out << "    (= (total-cost) " << problem.initialTotalCost << ")\n";
        }
        out << "  )\n";
        out << "  (:goal ";
        WriteConjunction(out, problem.goal, "    ");
        out << ")\n";
        if (problem.minimizeTotalCost) {
            out << "  (:metric minimize (total-cost))\n";
        }
        out << ")\n";
        return out.str();
    }

    void Writer::WriteDomain(
        const Domain& domain,
        const std::filesystem::path& path
    ) const {
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("unable to write PDDL domain: " + path.string());
        }
        output << DomainText(domain);
    }

    void Writer::WriteProblem(
        const Problem& problem,
        const std::filesystem::path& path
    ) const {
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("unable to write PDDL problem: " + path.string());
        }
        output << ProblemText(problem);
    }
}
