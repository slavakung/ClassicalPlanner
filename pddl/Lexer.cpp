#include "pddl/Lexer.h"

#include <cctype>
#include <stdexcept>

namespace planning::pddl
{
    std::vector<Token> Lexer::Tokenize(std::string_view input) const {
        std::vector<Token> tokens;
        tokens.reserve(input.size() / 6U + 8U);

        std::size_t i = 0;
        std::size_t line = 1;
        std::size_t column = 1;

        auto advance = [&](char c) {
            ++i;
            if (c == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        };

        while (i < input.size()) {
            const char c = input[i];

            if (std::isspace(static_cast<unsigned char>(c)) != 0) {
                advance(c);
                continue;
            }

            // PDDL comments begin with ';' and continue to end of line.
            if (c == ';') {
                while (i < input.size() && input[i] != '\n') {
                    advance(input[i]);
                }
                continue;
            }

            const SourceLocation location{line, column};

            if (c == '(') {
                tokens.push_back(Token{TokenKind::leftParen, "(", location});
                advance(c);
                continue;
            }

            if (c == ')') {
                tokens.push_back(Token{TokenKind::rightParen, ")", location});
                advance(c);
                continue;
            }

            std::string atom;
            while (i < input.size()) {
                const char current = input[i];
                if (std::isspace(static_cast<unsigned char>(current)) != 0 ||
                    current == '(' || current == ')' || current == ';') {
                    break;
                }

                // PDDL symbols are case-insensitive. Normalizing once here means
                // every later symbol table and equality comparison is cheaper and
                // deterministic.
                atom.push_back(static_cast<char>(
                    std::tolower(static_cast<unsigned char>(current))
                ));
                advance(current);
            }

            if (atom.empty()) {
                throw std::runtime_error(
                    "PDDL lexer could not consume input at line " +
                    std::to_string(line) + ", column " +
                    std::to_string(column)
                );
            }

            tokens.push_back(Token{TokenKind::atom, std::move(atom), location});
        }

        tokens.push_back(Token{
            TokenKind::end,
            {},
            SourceLocation{line, column}
        });

        return tokens;
    }
}
