#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>
#include <plamatrix/dense/dense_matrix.h>

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

int voxelGridDownsampleColumnMajor(
    const plamatrix::DenseMatrix<float, plamatrix::Device::GPU>& points,
    float leaf_x, float leaf_y, float leaf_z,
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU>& out_points,
    cudaStream_t stream = 0);

int voxelGridDownsampleColumnMajor(
    const plamatrix::DenseMatrix<double, plamatrix::Device::GPU>& points,
    double leaf_x, double leaf_y, double leaf_z,
    plamatrix::DenseMatrix<double, plamatrix::Device::GPU>& out_points,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
