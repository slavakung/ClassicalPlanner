#pragma once

#include "Lexer.h"

#include <string>
#include <vector>

namespace planning::pddl
{
    class Parser {
    public:
        [[nodiscard]] Domain ParseDomain(const std::vector<Token>& tokens) const;
        [[nodiscard]] Problem ParseProblem(const std::vector<Token>& tokens) const;

        [[nodiscard]] Domain ParseDomainText(const std::string& text) const;
        [[nodiscard]] Problem ParseProblemText(const std::string& text) const;
    };
}
