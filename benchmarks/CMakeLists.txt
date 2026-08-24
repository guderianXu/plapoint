add_executable(plapoint_benchmarks
    plapoint_benchmarks.cpp
)

target_link_libraries(plapoint_benchmarks PRIVATE plapoint::plapoint)

set_target_properties(plapoint_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
)

add_test(
    NAME plapoint.benchmarks.invalid_args_rejected
    COMMAND
        ${CMAKE_COMMAND}
        -DPLAPOINT_BENCHMARK_EXE=$<TARGET_FILE:plapoint_benchmarks>
        -P ${PROJECT_SOURCE_DIR}/cmake/check_benchmark_invalid_args.cmake
)

add_test(
    NAME plapoint.benchmarks.statistics_self_test
    COMMAND plapoint_benchmarks --self-test-benchmark-statistics
)

add_test(
    NAME plapoint.benchmarks.cpu_csv_schema_smoke
    COMMAND
        ${CMAKE_COMMAND}
        -DPLAPOINT_BENCHMARK_EXE=$<TARGET_FILE:plapoint_benchmarks>
        -P ${PROJECT_SOURCE_DIR}/cmake/check_cpu_benchmark_csv.cmake
)

if(PLAPOINT_WITH_CUDA)
    add_test(
        NAME plapoint.benchmarks.gpu_search_feature_rows_registered
        COMMAND
            ${CMAKE_COMMAND}
            -DPLAPOINT_BENCHMARK_EXE=$<TARGET_FILE:plapoint_benchmarks>
            -P ${PROJECT_SOURCE_DIR}/cmake/check_gpu_search_benchmark_rows.cmake
    )

    add_test(
        NAME plapoint.benchmarks.gpu_mesh_rows_registered
        COMMAND ${CMAKE_COMMAND}
            -DPLAPOINT_BENCHMARK_EXE=$<TARGET_FILE:plapoint_benchmarks>
            -P ${PROJECT_SOURCE_DIR}/cmake/check_gpu_mesh_benchmark_rows.cmake
    )

    add_test(
        NAME plapoint.benchmarks.gpu_icp_rows_registered
        COMMAND
            ${CMAKE_COMMAND}
            -DPLAPOINT_BENCHMARK_EXE=$<TARGET_FILE:plapoint_benchmarks>
            -P ${PROJECT_SOURCE_DIR}/cmake/check_gpu_icp_benchmark_rows.cmake
    )
endif()
