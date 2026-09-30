#pragma once


#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {
inline std::size_t checks = 0;

inline void Require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error("CHECK FAILED: " + message);
    }
}

template<typename F>
void Throws(F&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    Require(threw, message);
}

// Invoke the planner directly; this adapter creates no workers or packed engine.
template<class Factory, class Problem, class Consumer>
void Run(Factory factory, const Problem& problem, Consumer consume) {
    auto planner = factory();
    planner.Search(problem, consume);
}
} // namespace test
