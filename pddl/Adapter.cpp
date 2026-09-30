#include "pddl/Adapter.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace planning::pddl
{
    std::string ReadTextFile(const std::filesystem::path& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(
                "unable to open PDDL file: " + path.string()
            );
        }

        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }

    ParsedFiles ParseFiles(
        const std::filesystem::path& domainPath,
        const std::filesystem::path& problemPath
    ) {
        Parser parser;
        return ParsedFiles{
            parser.ParseDomainText(ReadTextFile(domainPath)),
            parser.ParseProblemText(ReadTextFile(problemPath))
        };
    }

    PDDLPlanningProblem LoadGroundedProblem(
        const std::filesystem::path& domainPath,
        const std::filesystem::path& problemPath,
        ExecutionMode mode
    ) {
        const ParsedFiles parsed = ParseFiles(domainPath, problemPath);
        const ir::Problem irProblem =
            SemanticAnalyzer{}.Analyze(parsed.domain, parsed.problem);
        return PDDLPlanningProblem(Grounder{}.Ground(irProblem, mode));
    }

    PDDLPlanningProblem LoadGroundedProblemFromText(
        const std::string& domainText,
        const std::string& problemText,
        ExecutionMode mode
    ) {
        Parser parser;
        const Domain domain = parser.ParseDomainText(domainText);
        const Problem problem = parser.ParseProblemText(problemText);
        const ir::Problem irProblem = SemanticAnalyzer{}.Analyze(domain, problem);
        return PDDLPlanningProblem(Grounder{}.Ground(irProblem, mode));
    }
}
