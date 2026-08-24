if(NOT DEFINED PLAPOINT_BENCHMARK_EXE)
    message(FATAL_ERROR "PLAPOINT_BENCHMARK_EXE is required")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --points 128
        --iterations 1
        --search-features-only
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_output
    ERROR_VARIABLE benchmark_error
)

if(NOT benchmark_result EQUAL 0)
    message(FATAL_ERROR
        "PlaPoint CPU benchmark smoke failed with exit code ${benchmark_result}\n"
        "stdout:\n${benchmark_output}\n"
        "stderr:\n${benchmark_error}")
endif()

string(REPLACE "\r\n" "\n" normalized_output "${benchmark_output}")
set(expected_header "benchmark,points,iterations,best_ms,median_ms,p95_ms,stddev_ms,cv")
if(NOT normalized_output MATCHES "^${expected_header}\n")
    message(FATAL_ERROR
        "PlaPoint benchmark CSV header does not match the v2 schema\n"
        "stdout:\n${benchmark_output}")
endif()

string(REGEX MATCH "(^|\n)cpu_knn_batch_k8,[^\n]*" row_output "${normalized_output}")
string(REGEX REPLACE "^\n" "" row_output "${row_output}")
if(NOT row_output)
    message(FATAL_ERROR
        "PlaPoint CPU benchmark smoke did not emit cpu_knn_batch_k8\n"
        "stdout:\n${benchmark_output}")
endif()

string(REPLACE "," ";" row_fields "${row_output}")
list(LENGTH row_fields field_count)
if(NOT field_count EQUAL 8)
    message(FATAL_ERROR "Invalid PlaPoint CPU benchmark row: ${row_output}")
endif()

foreach(field_index RANGE 3 7)
    list(GET row_fields ${field_index} timing_value)
    if(NOT timing_value MATCHES "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$")
        message(FATAL_ERROR "Invalid PlaPoint CPU benchmark statistic: ${row_output}")
    endif()
endforeach()

list(GET row_fields 3 best_ms)
list(GET row_fields 4 median_ms)
list(GET row_fields 5 p95_ms)
if(best_ms GREATER median_ms OR median_ms GREATER p95_ms)
    message(FATAL_ERROR
        "PlaPoint CPU benchmark statistics are not ordered best <= median <= p95: ${row_output}")
endif()
