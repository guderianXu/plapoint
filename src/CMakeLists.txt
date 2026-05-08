# CPU sources (always compiled)
target_sources(plapoint PRIVATE
    plapoint.cpp
)

# GPU / CUDA sources
if(PLAPOINT_WITH_CUDA)
    target_sources(plapoint PRIVATE
        knn_gpu.cu
        voxel_grid_gpu.cu
    )
endif()
