#pragma once

#include "Ast.h"

#include <filesystem>
#include <string>

namespace planning::pddl
{
    class Writer {
    public:
        [[nodiscard]] std::string DomainText(const Domain& domain) const;
        [[nodiscard]] std::string ProblemText(const Problem& problem) const;

        void WriteDomain(
            const Domain& domain,
            const std::filesystem::path& path
        ) const;

        void WriteProblem(
            const Problem& problem,
            const std::filesystem::path& path
        ) const;
    };
}
