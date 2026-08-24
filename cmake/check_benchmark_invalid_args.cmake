if(NOT DEFINED PLAPOINT_BENCHMARK_EXE)
    message(FATAL_ERROR "PLAPOINT_BENCHMARK_EXE is required")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --points not-a-number
    RESULT_VARIABLE invalid_result
    OUTPUT_VARIABLE invalid_output
    ERROR_VARIABLE invalid_error
)

if(invalid_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks accepted an invalid --points value\n"
        "stdout:\n${invalid_output}\n"
        "stderr:\n${invalid_error}")
endif()

if(NOT invalid_error MATCHES "Invalid value for --points")
    message(FATAL_ERROR
        "plapoint_benchmarks did not explain the invalid --points value\n"
        "stdout:\n${invalid_output}\n"
        "stderr:\n${invalid_error}")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --points
    RESULT_VARIABLE missing_result
    OUTPUT_VARIABLE missing_output
    ERROR_VARIABLE missing_error
)

if(missing_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks accepted --points without a value\n"
        "stdout:\n${missing_output}\n"
        "stderr:\n${missing_error}")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --points 0
    RESULT_VARIABLE zero_result
    OUTPUT_VARIABLE zero_output
    ERROR_VARIABLE zero_error
)

if(zero_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks accepted --points 0\n"
        "stdout:\n${zero_output}\n"
        "stderr:\n${zero_error}")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --definitely-unknown
    RESULT_VARIABLE unknown_result
    OUTPUT_VARIABLE unknown_output
    ERROR_VARIABLE unknown_error
)

if(unknown_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks accepted an unknown option\n"
        "stdout:\n${unknown_output}\n"
        "stderr:\n${unknown_error}")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --mesh-only
        --search-features-only
        --poisson-points 8192
        --poisson-depth 8
    RESULT_VARIABLE sparse_poisson_result
    OUTPUT_VARIABLE sparse_poisson_output
    ERROR_VARIABLE sparse_poisson_error
)

if(sparse_poisson_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks accepted an undersampled Poisson fixture\n"
        "stdout:\n${sparse_poisson_output}\n"
        "stderr:\n${sparse_poisson_error}")
endif()

if(NOT sparse_poisson_output STREQUAL "")
    message(FATAL_ERROR
        "plapoint_benchmarks emitted output before rejecting the undersampled Poisson fixture\n"
        "stdout:\n${sparse_poisson_output}\n"
        "stderr:\n${sparse_poisson_error}")
endif()

if(NOT sparse_poisson_error MATCHES
   "Poisson benchmark sampling is too sparse.*use at least 65536 points or reduce the depth")
    message(FATAL_ERROR
        "plapoint_benchmarks did not explain the undersampled Poisson fixture\n"
        "stdout:\n${sparse_poisson_output}\n"
        "stderr:\n${sparse_poisson_error}")
endif()
