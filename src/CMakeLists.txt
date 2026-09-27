# CPU sources (always compiled)
target_sources(plapoint PRIVATE
    plapoint.cpp
    height_grid_dispatch.cpp
)

if(PLAPOINT_WITH_OPENCL)
    target_sources(plapoint PRIVATE
        opencl_runtime.cpp
        height_grid_opencl.cpp
        knn_opencl.cpp
        normal_estimation_opencl.cpp
        outlier_removal_opencl.cpp
        voxel_grid_opencl.cpp
    )
else()
    target_sources(plapoint PRIVATE
        opencl_algorithm_stubs.cpp
        opencl_runtime_stub.cpp
    )
endif()

# GPU / CUDA sources
if(PLAPOINT_WITH_CUDA)
    target_sources(plapoint PRIVATE
        knn_gpu.cu
        batch_knn_gpu.cu
        filter_compaction_gpu.cu
        filter_indices_gpu.cu
        voxel_grid_gpu.cu
        voxel_cluster_gpu.cu
        height_grid_gpu.cu
        icp_gpu.cu
        marching_cubes_gpu.cu
        mesh_normals_gpu.cu
        normal_estimation_gpu.cu
        normal_refinement_gpu.cu
        spatial_index_gpu.cu
        spatial_index_query_gpu.cu
        spatial_index_knn_gpu.cu
    )
endif()
