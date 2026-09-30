foreach(name search_regressions bfs_duplicate_tests pddl_regressions informed_tests partial_order_tests iterative_tests symbolic_regressions finite_state_tests local_search_tests approximate_dp_tests serial_optimization_tests)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE classical_planner_core)
    planning_warnings(${name})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "serial;regression" TIMEOUT 180
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
foreach(algorithm bfs dfs recursive-dfs bbdfs ucs astar greedy weighted-astar beam
                  iddfs idastar rbfs hill ehc backward-bfs bidirectional forward-vi backward-vi
                  rollout adp adp-astar graphplan satplan csp regression partial-order)
    add_test(NAME cli_${algorithm} COMMAND benchmark_planner
        --domain "${CMAKE_CURRENT_SOURCE_DIR}/examples/pddl/blocks/domain.pddl"
        --problem "${CMAKE_CURRENT_SOURCE_DIR}/examples/pddl/blocks/problem.pddl"
        --algorithm ${algorithm} --max-depth 8 --rollout-depth 4 --adp-trials 4 --trial-depth 8)
    set_tests_properties(cli_${algorithm} PROPERTIES LABELS "serial;cli" TIMEOUT 120)
endforeach()
add_test(NAME cli_reject_inadmissible_astar COMMAND benchmark_planner
    --domain "${CMAKE_CURRENT_SOURCE_DIR}/examples/pddl/blocks/domain.pddl"
    --problem "${CMAKE_CURRENT_SOURCE_DIR}/examples/pddl/blocks/problem.pddl"
    --algorithm astar --heuristic hadd)
set_tests_properties(cli_reject_inadmissible_astar PROPERTIES WILL_FAIL TRUE TIMEOUT 10)
