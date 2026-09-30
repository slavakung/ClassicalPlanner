#pragma once

#include "Ast.h"

#include <string>
#include <string_view>
#include <vector>

namespace planning::pddl
{
    enum class TokenKind {
        leftParen,
        rightParen,
        atom,
        end
    };

    struct Token {
        TokenKind kind = TokenKind::end;
        std::string text;
        SourceLocation location{};
    };

    class Lexer {
    public:
        [[nodiscard]] std::vector<Token> Tokenize(std::string_view input) const;
    };
}
