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
