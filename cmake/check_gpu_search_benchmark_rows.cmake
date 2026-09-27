execute_process(
    COMMAND "${PLAPOINT_BENCHMARK_EXE}"
            --points 128
            --iterations 1
            --search-features-only
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_output
    ERROR_VARIABLE benchmark_error
)

if(NOT benchmark_result EQUAL 0)
    message(FATAL_ERROR "PlaPoint search/feature benchmark failed: ${benchmark_error}")
endif()

set(required_rows
    gpu_spatial_index_build_adaptive
    gpu_knn_brute_force_k8
    gpu_knn_indexed_k8
    gpu_radius_count
    gpu_normal_estimation_k8
    gpu_normal_smoothing_k8
    gpu_statistical_outlier_removal_k8
    gpu_radius_outlier_removal
)

foreach(row IN LISTS required_rows)
    string(FIND "${benchmark_output}" "${row}," row_position)
    if(row_position EQUAL -1)
        message(FATAL_ERROR "Missing PlaPoint search/feature benchmark row: ${row}")
    endif()
endforeach()
