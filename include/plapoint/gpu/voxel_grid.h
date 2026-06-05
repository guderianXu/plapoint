#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

namespace plapoint
{
namespace gpu
{

/// Downsample PlaMatrix column-major Nx3 device points into sorted voxel centroids.
/// d_out_points must have capacity for N x 3 Scalars. Returns the centroid count after stream synchronization.
int voxelGridDownsampleColumnMajor(const float* d_points, int N,
                                   float leaf_x, float leaf_y, float leaf_z,
                                   float* d_out_points,
                                   cudaStream_t stream = 0);

int voxelGridDownsampleColumnMajor(const double* d_points, int N,
                                   double leaf_x, double leaf_y, double leaf_z,
                                   double* d_out_points,
                                   cudaStream_t stream = 0);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
