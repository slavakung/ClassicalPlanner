cmake_minimum_required(VERSION 3.24)
project(Planning_Public_serial VERSION 0.7.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(PLANNING_BUILD_TESTS "Build serial algorithm and PDDL regression tests" ON)
option(PLANNING_ENABLE_SANITIZERS "ASan/UBSan (GCC/Clang)" OFF)

function(planning_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(PLANNING_ENABLE_SANITIZERS)
            target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PUBLIC -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

add_library(classical_planner_core STATIC
    Planner.cpp pddl/Lexer.cpp pddl/Parser.cpp pddl/SemanticAnalyzer.cpp
    pddl/Grounder.cpp pddl/PDDLPlanningProblem.cpp pddl/Adapter.cpp
    pddl/Writer.cpp pddl/GridPDDL.cpp symbolic/ClassicalPlanners.cpp)
target_include_directories(classical_planner_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
planning_warnings(classical_planner_core)
foreach(name pddl_planner_demo benchmark_planner)
    if(name STREQUAL "benchmark_planner")
        add_executable(${name} benchmarks/benchmark_planner.cpp)
    else()
        add_executable(${name} pddl_demo.cpp)
    endif()
    target_link_libraries(${name} PRIVATE classical_planner_core)
    planning_warnings(${name})
endforeach()

if(PLANNING_BUILD_TESTS)
    enable_testing()
    add_executable(classical_planner_tests tests.cpp)
    target_link_libraries(classical_planner_tests PRIVATE classical_planner_core)
    planning_warnings(classical_planner_tests)
    if(NOT MSVC)
        target_compile_options(classical_planner_tests PRIVATE -UNDEBUG)
    endif()
    add_test(NAME legacy_regression COMMAND classical_planner_tests)
    set_tests_properties(legacy_regression PROPERTIES TIMEOUT 120)
    include(cmake/AlgorithmTests.cmake)
    add_executable(benchmark_verify_pairs benchmarks/verify_pairs.cpp)
    target_link_libraries(benchmark_verify_pairs PRIVATE classical_planner_core)
    planning_warnings(benchmark_verify_pairs)
    add_test(NAME benchmark_hanoi_encoding_equivalence COMMAND benchmark_verify_pairs
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/hanoi/domain.pddl"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/hanoi/disks-06.pddl"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/hanoi/propositional-06-domain.pddl"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/hanoi/propositional-06.pddl" -1)
    add_test(NAME benchmark_sliding_encoding_equivalence COMMAND benchmark_verify_pairs
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/sliding_tile/domain.pddl"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/sliding_tile/distance-31.pddl"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/sliding_tile/propositional-domain.pddl"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/instances/generated/sliding_tile/propositional-31.pddl" 10)
    set_tests_properties(benchmark_hanoi_encoding_equivalence benchmark_sliding_encoding_equivalence PROPERTIES TIMEOUT 180)
endif()
message(STATUS "Native serial-only build: no worker, transport, or accelerator dependencies")
