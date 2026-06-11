# CPU sources (always compiled)
target_sources(plapoint PRIVATE
    plapoint.cpp
)

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
        mesh_normals_gpu.cu
    )
endif()
