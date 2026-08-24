execute_process(
    COMMAND "${PLAPOINT_BENCHMARK_EXE}"
            --points 128
            --poisson-points 1024
            --poisson-depth 4
            --iterations 1
            --mesh-only
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_output
    ERROR_VARIABLE benchmark_error
)

if(NOT benchmark_result EQUAL 0)
    message(FATAL_ERROR "PlaPoint GPU mesh benchmark failed: ${benchmark_error}")
endif()

set(required_rows
    marching_cubes_field
    height_grid_fill
    poisson_solve
    poisson_end_to_end
)

foreach(row IN LISTS required_rows)
    string(REGEX MATCH "${row},[^\r\n]*" row_output "${benchmark_output}")
    if(NOT row_output)
        message(FATAL_ERROR "Missing PlaPoint GPU mesh benchmark row: ${row}\n${benchmark_output}")
    endif()
    string(REPLACE "," ";" row_fields "${row_output}")
    list(LENGTH row_fields field_count)
    if(NOT field_count EQUAL 8)
        message(FATAL_ERROR "Invalid PlaPoint GPU mesh benchmark row: ${row_output}")
    endif()

    foreach(field_index RANGE 3 7)
        list(GET row_fields ${field_index} timing_value)
        if(NOT timing_value MATCHES "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$")
            message(FATAL_ERROR "Invalid PlaPoint GPU mesh timing statistic: ${row_output}")
        endif()
    endforeach()

    list(GET row_fields 3 best_ms)
    list(GET row_fields 4 median_ms)
    list(GET row_fields 5 p95_ms)
    if(NOT best_ms GREATER 0 OR NOT median_ms GREATER 0 OR NOT p95_ms GREATER 0)
        message(FATAL_ERROR "Non-positive PlaPoint GPU mesh timing: ${row_output}")
    endif()
    if(best_ms GREATER median_ms OR median_ms GREATER p95_ms)
        message(FATAL_ERROR "Inconsistent PlaPoint GPU mesh timing statistics: ${row_output}")
    endif()
endforeach()
