#pragma once

#include "PDDLPlanningProblem.h"
#include "Parser.h"
#include "SemanticAnalyzer.h"

#include <filesystem>
#include <string>

namespace planning::pddl
{
    struct ParsedFiles {
        Domain domain;
        Problem problem;
    };

    [[nodiscard]] std::string ReadTextFile(const std::filesystem::path& path);

    [[nodiscard]] ParsedFiles ParseFiles(
        const std::filesystem::path& domainPath,
        const std::filesystem::path& problemPath
    );

    [[nodiscard]] PDDLPlanningProblem LoadGroundedProblem(
        const std::filesystem::path& domainPath,
        const std::filesystem::path& problemPath,
        ExecutionMode mode = ExecutionMode::groundAll
    );

    [[nodiscard]] PDDLPlanningProblem LoadGroundedProblemFromText(
        const std::string& domainText,
        const std::string& problemText,
        ExecutionMode mode = ExecutionMode::groundAll
    );
}
