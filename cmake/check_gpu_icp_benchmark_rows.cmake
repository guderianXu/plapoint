if(NOT DEFINED PLAPOINT_BENCHMARK_EXE)
    message(FATAL_ERROR "PLAPOINT_BENCHMARK_EXE is required")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --points 1000
        --iterations 1
        --icp-points 1000
        --icp-max-iterations 1
        --skip-cpu-icp
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_output
    ERROR_VARIABLE benchmark_error
)

if(NOT benchmark_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks failed with exit code ${benchmark_result}\n"
        "stdout:\n${benchmark_output}\n"
        "stderr:\n${benchmark_error}")
endif()

set(expected_gpu_icp_rows
    gpu_icp_identity
    gpu_icp_identity_same_buffer_reuse_output
    gpu_icp_finite_radius
    gpu_icp_finite_radius_identity_reuse_output
    gpu_icp_finite_radius_identity_exact_probe_reuse_output
    gpu_icp_finite_radius_translation
    gpu_icp_finite_radius_translation_reuse
    gpu_icp_finite_radius_translation_reuse_shrinking
    gpu_icp_finite_radius_translation_reuse_output
    gpu_icp_finite_radius_translation_reuse_output_one_iteration
    gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics
    gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics_one_iteration
    gpu_icp_finite_radius_translation_ordered_output
    gpu_icp_finite_radius_translation_ordered_output_one_iteration
    gpu_icp_finite_radius_translation_ordered_transform_only
    gpu_icp_finite_radius_translation_ordered_transform_only_one_iteration
    gpu_icp_finite_radius_translation_ordered_output_skip_final_metrics
    gpu_icp_finite_radius_translation_ordered_output_skip_final_metrics_one_iteration
    gpu_icp_finite_radius_ordered_low_residual_output_one_iteration
    gpu_icp_ordered_infinite_radius_output_one_iteration
    gpu_icp_finite_radius_translation_ordered_reuse_target_output
    gpu_icp_finite_radius_nonrigid_transform_only_two_iterations
    gpu_icp_finite_radius_nonrigid_transform_only_prewarmed_workspace_two_iterations
    gpu_icp_finite_radius_nonrigid_output_transform_only_two_iterations
    gpu_icp_finite_radius_nonrigid_output_transform_only_fresh_target_two_iterations
    gpu_icp_finite_radius_nonrigid_target_alias_transform_only_two_iterations
    gpu_icp_finite_radius_nonrigid_final_metrics_two_iterations
    gpu_icp_finite_radius_nonrigid_output_final_metrics_two_iterations
    gpu_icp_finite_radius_nonrigid_target_alias_final_metrics_two_iterations
    gpu_icp_finite_radius_nonrigid_transform_only_preflight_two_iterations
    gpu_icp_finite_radius_nonrigid_transform_only_cache_reuse_two_iterations
    gpu_icp_finite_radius_nonrigid_verified_ordered_transform_only_two_iterations
    gpu_icp_finite_radius_nonrigid_verified_ordered_final_metrics_two_iterations
    gpu_icp_finite_radius_nonrigid_ordered_transform_only_two_iterations
    gpu_icp_finite_radius_translation_transform_only_skip_final_metrics
    gpu_icp_finite_radius_translation_transform_only_skip_final_metrics_one_iteration
    gpu_icp_finite_radius_binary_translation_transform_only_two_iterations
    gpu_icp_finite_radius_binary_translation_transform_only_preflight_two_iterations
    gpu_icp_finite_radius_binary_translation_reuse_output_one_iteration
    gpu_icp_finite_radius_binary_translation_reuse_output_two_iterations
    gpu_icp_finite_radius_binary_translation_reuse_output_preflight_two_iterations
    gpu_icp_finite_radius_translation_reuse_target_output
    gpu_icp_finite_radius_translation_reuse_target_output_skip_final_metrics
    gpu_icp_stats_step_finite_radius_translation_new_workspace
    gpu_icp_stats_step_finite_radius_translation_cached_grid
    gpu_icp_stats_step_finite_radius_translation_ordered
    gpu_icp_alignment_step_finite_radius_translation_new_workspace
    gpu_icp_alignment_step_finite_radius_translation_new_workspace_one_source
    gpu_icp_alignment_step_finite_radius_translation_rebuild_reserved_grid
    gpu_icp_alignment_step_finite_radius_translation_cached_grid
    gpu_icp_alignment_step_finite_radius_translation_cached_grid_reserved_workspace
    gpu_icp_alignment_step_finite_radius_translation_async_launch_cached_grid
    gpu_icp_alignment_step_transformed_accumulated_async_launch_cached_grid
    gpu_icp_alignment_step_two_step_async_launch_separate_workspaces
    gpu_icp_alignment_step_exact_pointwise_same_buffer
    gpu_icp_alignment_step_exact_pointwise_same_buffer_reserved_workspace
    gpu_icp_alignment_step_ordered_same_buffer_finite_radius
    gpu_icp_alignment_step_transformed_exact_pointwise_cached_grid
    gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit
    gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit_preflight
    gpu_icp_stats_finite_radius_translation_cached_grid
    gpu_icp_residual_stats_finite_radius_translation_new_workspace
    gpu_icp_residual_stats_finite_radius_translation_cached_grid
    gpu_icp_residual_stats_finite_radius_translation_cached_grid_reserved_workspace
    gpu_icp_residual_stats_finite_radius_translation_ordered
    gpu_icp_transform_residual_stats_finite_radius_translation_new_workspace
    gpu_icp_transform_residual_stats_finite_radius_translation_cached_grid
    gpu_icp_transform_residual_stats_transformed_exact_pointwise_new_workspace
    gpu_icp_stats_fallback_tile_bounds_new_workspace
    gpu_icp_stats_fallback_tile_bounds_cached_bounds
    gpu_icp_stats_step_fallback_tile_bounds_new_workspace
    gpu_icp_stats_step_fallback_tile_bounds_cached_bounds
    gpu_icp_alignment_step_fallback_tile_bounds_new_workspace
    gpu_icp_alignment_step_fallback_tile_bounds_cached_bounds
    gpu_icp_alignment_step_small_finite_radius_target_4
    gpu_icp_alignment_step_small_finite_radius_target_16
    gpu_icp_alignment_step_small_finite_radius_target_64
    gpu_icp_alignment_step_small_finite_radius_target_below_grid_threshold
    gpu_icp_alignment_step_small_finite_radius_target_async_launch_below_grid_threshold
    gpu_icp_alignment_step_small_finite_radius_target_at_grid_threshold
    gpu_icp_alignment_step_small_finite_radius_target_256
    gpu_icp_stats_step_small_finite_radius_target_below_grid_threshold
    gpu_icp_residual_stats_small_finite_radius_target_below_grid_threshold
    gpu_icp_transform_residual_stats_small_finite_radius_target_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_transform_only_two_iterations_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_output_transform_only_two_iterations_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_target_alias_transform_only_two_iterations_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_final_metrics_one_iteration_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_final_metrics_two_iterations_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_reuse_final_metrics_two_iterations_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_output_final_metrics_two_iterations_below_grid_threshold
    gpu_icp_small_finite_radius_nonrigid_target_alias_final_metrics_two_iterations_below_grid_threshold
    gpu_icp_small_finite_radius_transformed_accumulated_async_launch_below_grid_threshold
    gpu_icp_small_finite_radius_two_step_transform_only_async_launch_below_grid_threshold
    gpu_icp_small_finite_radius_terminal_async_launch_below_grid_threshold
    gpu_icp_small_finite_radius_two_step_terminal_async_launch_below_grid_threshold
)

foreach(row_name IN LISTS expected_gpu_icp_rows)
    if(NOT benchmark_output MATCHES "(^|\n)${row_name},")
        message(FATAL_ERROR
            "Missing GPU ICP benchmark row '${row_name}'\n"
            "stdout:\n${benchmark_output}")
    endif()
endforeach()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --points 1000
        --iterations 1
        --icp-points 1000
        --icp-max-iterations 1
        --skip-cpu-icp
        --skip-icp-identity
    RESULT_VARIABLE disabled_result
    OUTPUT_VARIABLE disabled_output
    ERROR_VARIABLE disabled_error
)

if(NOT disabled_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks --skip-icp-identity failed with exit code ${disabled_result}\n"
        "stdout:\n${disabled_output}\n"
        "stderr:\n${disabled_error}")
endif()

if(NOT disabled_output MATCHES
        "(^|\n)gpu_icp_identity_same_buffer_reuse_output,skipped,disabled,")
    message(FATAL_ERROR
        "Missing disabled same-buffer identity row when --skip-icp-identity is set\n"
        "stdout:\n${disabled_output}")
endif()

execute_process(
    COMMAND
        "${PLAPOINT_BENCHMARK_EXE}"
        --self-test-benchmark-gpu-sync
    RESULT_VARIABLE sync_self_test_result
    OUTPUT_VARIABLE sync_self_test_output
    ERROR_VARIABLE sync_self_test_error
)

if(NOT sync_self_test_result EQUAL 0)
    message(FATAL_ERROR
        "plapoint_benchmarks --self-test-benchmark-gpu-sync failed with exit code ${sync_self_test_result}\n"
        "stdout:\n${sync_self_test_output}\n"
        "stderr:\n${sync_self_test_error}")
endif()

if(NOT sync_self_test_output MATCHES "(^|\n)benchmark_gpu_sync_self_test,(passed|skipped)")
    message(FATAL_ERROR
        "Missing GPU benchmark sync self-test marker\n"
        "stdout:\n${sync_self_test_output}")
endif()

if(NOT sync_self_test_output MATCHES "(^|\n)benchmark_gpu_sync_scope_self_test,passed")
    message(FATAL_ERROR
        "Missing GPU benchmark sync scope self-test pass marker\n"
        "stdout:\n${sync_self_test_output}")
endif()
