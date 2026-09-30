#include "pddl/Parser.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace planning::pddl
{
    namespace
    {
        struct SExpr {
            std::string atom;
            std::vector<SExpr> list;
            SourceLocation location{};

            [[nodiscard]] bool IsAtom() const noexcept {
                return list.empty() && !atom.empty();
            }
        };

        [[noreturn]] void ParseError(
            const SourceLocation& location,
            const std::string& message
        ) {
            throw std::runtime_error(
                "PDDL parse error at line " + std::to_string(location.line) +
                ", column " + std::to_string(location.column) + ": " + message
            );
        }

        SExpr ParseSExpr(const std::vector<Token>& tokens, std::size_t& index) {
            if (index >= tokens.size()) {
                ParseError({}, "unexpected end of token stream");
            }
            if (tokens[index].kind == TokenKind::atom) {
                SExpr result;
                result.atom = tokens[index].text;
                result.location = tokens[index].location;
                ++index;
                return result;
            }

            if (tokens[index].kind != TokenKind::leftParen) {
                ParseError(tokens[index].location, "expected '(' or atom");
            }

            SExpr result;
            result.location = tokens[index].location;
            ++index;

            while (index < tokens.size() && tokens[index].kind != TokenKind::rightParen) {
                if (tokens[index].kind == TokenKind::end) {
                    ParseError(tokens[index].location, "unterminated list");
                }
                result.list.push_back(ParseSExpr(tokens, index));
            }

            if (index >= tokens.size()) {
                ParseError(result.location, "unterminated list");
            }

            ++index;
            return result;
        }

        const std::string& Atom(const SExpr& expression, std::string_view context) {
            if (!expression.IsAtom()) {
                ParseError(expression.location, "expected atom in " + std::string(context));
            }
            return expression.atom;
        }

        bool IsHead(const SExpr& expression, std::string_view head) {
            return !expression.list.empty() &&
                   expression.list.front().IsAtom() &&
                   expression.list.front().atom == head;
        }

        bool IsTotalCost(const SExpr& expression) {
            return IsHead(expression, "total-cost") && expression.list.size() == 1;
        }

        std::vector<TypedName> ParseTypedList(
            const std::vector<SExpr>& values,
            std::size_t begin
        ) {
            std::vector<TypedName> result;
            std::vector<std::string> pending;

            for (std::size_t i = begin; i < values.size(); ++i) {
                const std::string& token = Atom(values[i], "typed list");
                if (token == "-") {
                    if (pending.empty() || i + 1 >= values.size()) {
                        ParseError(values[i].location, "malformed typed list");
                    }

                    const std::string type = Atom(values[++i], "type name");
                    if (type == "-") {
                        ParseError(values[i].location, "malformed type name");
                    }
                    for (std::string& name : pending) {
                        result.push_back(TypedName{std::move(name), type});
                    }
                    pending.clear();
                } else {
                    pending.push_back(token);
                }
            }

            for (std::string& name : pending) {
                result.push_back(TypedName{std::move(name), "object"});
            }
            return result;
        }

        Literal ParseLiteral(const SExpr& expression, bool effectContext = false) {
            if (IsHead(expression, "not")) {
                if (expression.list.size() != 2) {
                    ParseError(expression.location, "(not ...) must have one operand");
                }
                Literal literal = ParseLiteral(expression.list[1], effectContext);
                literal.negated = !literal.negated;
                return literal;
            }

            if (expression.list.empty()) {
                ParseError(expression.location, "expected predicate application");
            }

            Literal literal;
            literal.location = expression.location;
            literal.predicate = Atom(expression.list.front(), "predicate name");
            if (literal.predicate == "or" || literal.predicate == "imply" ||
                literal.predicate == "forall" || literal.predicate == "exists" ||
                literal.predicate == "when" || literal.predicate == "assign" ||
                literal.predicate == "decrease" || literal.predicate == "scale-up" ||
                literal.predicate == "scale-down") {
                ParseError(expression.location, "unsupported construct '" + literal.predicate + "'");
            }
            for (std::size_t i = 1; i < expression.list.size(); ++i) {
                if (!expression.list[i].IsAtom()) {
                    ParseError(expression.list[i].location, "nested terms are unsupported in classical subset");
                }
                literal.terms.push_back(expression.list[i].atom);
            }

            if (effectContext && literal.predicate == "increase") {
                ParseError(expression.location, "numeric increase must be parsed as an action cost effect");
            }
            return literal;
        }

        std::vector<Literal> ParseLiteralConjunction(
            const SExpr& expression,
            bool effectContext
        ) {
            std::vector<Literal> result;
            if (IsHead(expression, "and")) {
                for (std::size_t i = 1; i < expression.list.size(); ++i) {
                    auto nested = ParseLiteralConjunction(expression.list[i], effectContext);
                    result.insert(result.end(), nested.begin(), nested.end());
                }
            } else {
                result.push_back(ParseLiteral(expression, effectContext));
            }
            return result;
        }

        double ParseNumber(const SExpr& expression) {
            const std::string& text = Atom(expression, "number");
            char* end = nullptr;
            errno = 0;
            const double value = std::strtod(text.c_str(), &end);
            if (end == text.c_str() || *end != '\0' || !std::isfinite(value) || errno == ERANGE) {
                ParseError(expression.location, "expected numeric literal, got '" + text + "'");
            }
            return value;
        }

        void ParseEffects(const SExpr& expression, ActionSchema& action) {
            if (IsHead(expression, "and")) {
                for (std::size_t i = 1; i < expression.list.size(); ++i) {
                    ParseEffects(expression.list[i], action);
                }
            } else if (IsHead(expression, "increase")) {
                if (expression.list.size() != 3 || !IsTotalCost(expression.list[1])) {
                    ParseError(expression.location, "only increases of total-cost are supported");
                }
                if (!action.hasExplicitActionCost) {
                    action.actionCost = 0.0;
                }
                if (expression.list[2].IsAtom()) {
                    const double cost = ParseNumber(expression.list[2]);
                    if (cost < 0.0) {
                        ParseError(expression.location, "action cost increases must be nonnegative");
                    }
                    action.actionCost += cost;
                } else {
                    Literal function = ParseLiteral(expression.list[2]);
                    if (function.negated || function.predicate == "total-cost") {
                        ParseError(expression.list[2].location, "cost requires an immutable numeric function application");
                    }
                    action.staticActionCosts.push_back(std::move(function));
                }
                action.hasExplicitActionCost = true;
            } else {
                action.effects.push_back(ParseLiteral(expression, true));
            }
        }

        void ParseAction(const SExpr& section, Domain& domain) {
            if (section.list.size() < 2) {
                ParseError(section.location, "malformed :action section");
            }

            ActionSchema action;
            action.location = section.location;
            action.name = Atom(section.list[1], "action name");
            std::unordered_set<std::string> fields;

            for (std::size_t i = 2; i < section.list.size();) {
                const std::string& keyword = Atom(section.list[i], "action keyword");
                if (!fields.insert(keyword).second) {
                    ParseError(section.list[i].location, "duplicate action field '" + keyword + "'");
                }
                if (i + 1 >= section.list.size()) {
                    ParseError(section.list[i].location, "action keyword lacks value");
                }

                const SExpr& value = section.list[i + 1];
                if (keyword == ":parameters") {
                    if (value.IsAtom()) {
                        ParseError(value.location, ":parameters must be a list");
                    }
                    action.parameters = ParseTypedList(value.list, 0);
                } else if (keyword == ":precondition") {
                    action.preconditions = ParseLiteralConjunction(value, false);
                } else if (keyword == ":effect") {
                    ParseEffects(value, action);
                } else {
                    ParseError(section.list[i].location, "unsupported action field '" + keyword + "'");
                }
                i += 2;
            }

            domain.actions.push_back(std::move(action));
        }

        Domain BuildDomain(const SExpr& root) {
            if (!IsHead(root, "define") || root.list.size() < 2 ||
                !IsHead(root.list[1], "domain")) {
                ParseError(root.location, "expected (define (domain ...))");
            }

            Domain domain;
            if (root.list[1].list.size() != 2) {
                ParseError(root.list[1].location, "domain declaration must contain one name");
            }
            domain.name = Atom(root.list[1].list[1], "domain name");
            domain.types.push_back(TypedName{"object", ""});
            std::unordered_set<std::string> sections;

            for (std::size_t i = 2; i < root.list.size(); ++i) {
                const SExpr& section = root.list[i];
                if (section.list.empty()) {
                    ParseError(section.location, "expected domain section");
                }

                const std::string& head = Atom(section.list.front(), "domain section");
                if (head != ":action" && !sections.insert(head).second) {
                    ParseError(section.location, "duplicate domain section '" + head + "'");
                }
                if (head == ":requirements") {
                    for (std::size_t j = 1; j < section.list.size(); ++j) {
                        domain.requirements.push_back(
                            RequirementFromString(Atom(section.list[j], "requirement"))
                        );
                    }
                } else if (head == ":types") {
                    auto parsed = ParseTypedList(section.list, 1);
                    domain.types.insert(domain.types.end(), parsed.begin(), parsed.end());
                } else if (head == ":constants") {
                    domain.constants = ParseTypedList(section.list, 1);
                } else if (head == ":predicates") {
                    for (std::size_t j = 1; j < section.list.size(); ++j) {
                        const SExpr& predicate = section.list[j];
                        if (predicate.list.empty()) {
                            ParseError(predicate.location, "malformed predicate declaration");
                        }
                        PredicateDecl decl;
                        decl.name = Atom(predicate.list.front(), "predicate name");
                        decl.parameters = ParseTypedList(predicate.list, 1);
                        domain.predicates.push_back(std::move(decl));
                    }
                } else if (head == ":functions") {
                    bool haveUntypedFunction = false;
                    std::unordered_set<std::string> names;
                    for (std::size_t j = 1; j < section.list.size(); ++j) {
                        const auto& function = section.list[j];
                        if (function.IsAtom()) {
                            if (function.atom != "-" || !haveUntypedFunction ||
                                j + 1 >= section.list.size() || section.list[j + 1].atom != "number") {
                                ParseError(function.location, "numeric function declarations require return type number");
                            }
                            ++j;
                            haveUntypedFunction = false;
                            continue;
                        }
                        if (function.list.empty()) {
                            ParseError(function.location, "malformed function declaration");
                        }
                        PredicateDecl declaration;
                        declaration.name = Atom(function.list.front(), "function name");
                        declaration.parameters = ParseTypedList(function.list, 1);
                        if (!names.insert(declaration.name).second) {
                            ParseError(function.location, "duplicate function declaration");
                        }
                        if (declaration.name == "total-cost") {
                            if (!declaration.parameters.empty()) {
                                ParseError(function.location, "total-cost must have zero arguments");
                            }
                        } else {
                            domain.staticFunctions.push_back(std::move(declaration));
                        }
                        haveUntypedFunction = true;
                    }
                } else if (head == ":action") {
                    ParseAction(section, domain);
                } else {
                    ParseError(section.location, "unsupported domain section '" + head + "'");
                }
            }
            return domain;
        }

        Problem BuildProblem(const SExpr& root) {
            if (!IsHead(root, "define") || root.list.size() < 3 ||
                !IsHead(root.list[1], "problem")) {
                ParseError(root.location, "expected (define (problem ...))");
            }

            Problem problem;
            if (root.list[1].list.size() != 2) {
                ParseError(root.list[1].location, "problem declaration must contain one name");
            }
            problem.name = Atom(root.list[1].list[1], "problem name");
            std::unordered_set<std::string> sections;
            bool initializedTotalCost = false;

            for (std::size_t i = 2; i < root.list.size(); ++i) {
                const SExpr& section = root.list[i];
                if (section.list.empty()) {
                    ParseError(section.location, "expected problem section");
                }
                const std::string& head = Atom(section.list.front(), "problem section");
                if (!sections.insert(head).second) {
                    ParseError(section.location, "duplicate problem section '" + head + "'");
                }

                if (head == ":domain") {
                    if (section.list.size() != 2) {
                        ParseError(section.location, "malformed :domain section");
                    }
                    problem.domainName = Atom(section.list[1], "domain name");
                } else if (head == ":objects") {
                    problem.objects = ParseTypedList(section.list, 1);
                } else if (head == ":init") {
                    for (std::size_t j = 1; j < section.list.size(); ++j) {
                        const SExpr& term = section.list[j];
                        if (IsHead(term, "=")) {
                            if (term.list.size() == 3 &&
                                IsTotalCost(term.list[1])) {
                                if (initializedTotalCost) {
                                    ParseError(term.location, "duplicate total-cost initialization");
                                }
                                initializedTotalCost = true;
                                problem.initialTotalCost = ParseNumber(term.list[2]);
                                continue;
                            }
                            if (term.list.size() == 3 && !term.list[1].IsAtom()) {
                                auto function = ParseLiteral(term.list[1]);
                                if (function.negated || function.predicate == "total-cost") {
                                    ParseError(term.location, "malformed numeric initialization");
                                }
                                problem.staticFunctionValues.emplace_back(
                                    std::move(function), ParseNumber(term.list[2]));
                                continue;
                            }
                        }
                        Literal literal = ParseLiteral(term, false);
                        if (literal.negated) {
                            ParseError(term.location, "negative initial facts are unnecessary under closed-world semantics");
                        }
                        problem.initialFacts.push_back(std::move(literal));
                    }
                } else if (head == ":goal") {
                    if (section.list.size() != 2) {
                        ParseError(section.location, "malformed :goal section");
                    }
                    problem.goal = ParseLiteralConjunction(section.list[1], false);
                } else if (head == ":metric") {
                    if (section.list.size() != 3 ||
                        Atom(section.list[1], "metric direction") != "minimize" ||
                        !IsTotalCost(section.list[2])) {
                        ParseError(section.location, "only (:metric minimize (total-cost)) is supported");
                    }
                    problem.minimizeTotalCost = true;
                } else {
                    ParseError(section.location, "unsupported problem section '" + head + "'");
                }
            }

            if (!sections.contains(":domain") || !sections.contains(":init") ||
                !sections.contains(":goal")) {
                ParseError(root.location, "problem requires :domain, :init and :goal sections");
            }

            return problem;
        }

        SExpr ParseRoot(const std::vector<Token>& tokens) {
            if (tokens.empty() || tokens.back().kind != TokenKind::end) {
                ParseError({}, "token stream must end with an end token");
            }
            std::size_t index = 0;
            SExpr root = ParseSExpr(tokens, index);
            if (tokens[index].kind != TokenKind::end) {
                ParseError(tokens[index].location, "extra tokens after top-level form");
            }
            return root;
        }
    }

    std::string ToString(Requirement requirement) {
        switch (requirement) {
            case Requirement::strips: return ":strips";
            case Requirement::typing: return ":typing";
            case Requirement::negativePreconditions: return ":negative-preconditions";
            case Requirement::equality: return ":equality";
            case Requirement::actionCosts: return ":action-costs";
            case Requirement::adl: return ":adl";
            case Requirement::disjunctivePreconditions: return ":disjunctive-preconditions";
            case Requirement::existentialPreconditions: return ":existential-preconditions";
            case Requirement::universalPreconditions: return ":universal-preconditions";
            case Requirement::conditionalEffects: return ":conditional-effects";
            case Requirement::numericFluents: return ":fluents";
            case Requirement::durativeActions: return ":durative-actions";
            case Requirement::unknown: return ":unknown";
        }
        return ":unknown";
    }

    Requirement RequirementFromString(const std::string& text) {
        if (text == ":strips") return Requirement::strips;
        if (text == ":typing") return Requirement::typing;
        if (text == ":negative-preconditions") return Requirement::negativePreconditions;
        if (text == ":equality") return Requirement::equality;
        if (text == ":action-costs") return Requirement::actionCosts;
        if (text == ":adl") return Requirement::adl;
        if (text == ":disjunctive-preconditions") return Requirement::disjunctivePreconditions;
        if (text == ":existential-preconditions") return Requirement::existentialPreconditions;
        if (text == ":universal-preconditions") return Requirement::universalPreconditions;
        if (text == ":conditional-effects") return Requirement::conditionalEffects;
        if (text == ":fluents" || text == ":numeric-fluents") return Requirement::numericFluents;
        if (text == ":durative-actions") return Requirement::durativeActions;
        return Requirement::unknown;
    }

    Domain Parser::ParseDomain(const std::vector<Token>& tokens) const {
        return BuildDomain(ParseRoot(tokens));
    }

    Problem Parser::ParseProblem(const std::vector<Token>& tokens) const {
        return BuildProblem(ParseRoot(tokens));
    }

    Domain Parser::ParseDomainText(const std::string& text) const {
        return ParseDomain(Lexer{}.Tokenize(text));
    }

    Problem Parser::ParseProblemText(const std::string& text) const {
        return ParseProblem(Lexer{}.Tokenize(text));
    }
}
