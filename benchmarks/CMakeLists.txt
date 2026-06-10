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

if(PLAPOINT_WITH_CUDA)
    add_test(
        NAME plapoint.benchmarks.gpu_icp_rows_registered
        COMMAND
            ${CMAKE_COMMAND}
            -DPLAPOINT_BENCHMARK_EXE=$<TARGET_FILE:plapoint_benchmarks>
            -P ${PROJECT_SOURCE_DIR}/cmake/check_gpu_icp_benchmark_rows.cmake
    )
endif()
