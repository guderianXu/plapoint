#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <plapoint/core/point_cloud.h>
#include <plamatrix/internal/core/device.h>

namespace plapoint
{
namespace mesh
{

/// Recompute mesh vertex normals on CUDA from column-major GPU points and Fx3 GPU faces.
/// Face normal accumulation and per-vertex normalization run on the GPU.
plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> recomputeVertexNormals(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Recompute mesh vertex normals on CUDA from column-major GPU points and Fx3 GPU faces.
/// Face normal accumulation and per-vertex normalization run on the GPU.
plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> recomputeVertexNormals(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Orient existing GPU normals outward from the mesh centroid.
/// The centroid and orientation vote are reduced on the GPU; if flipped, face winding is swapped on GPU.
plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> orientNormalsOutwardFromCentroid(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Orient existing GPU normals outward from the mesh centroid.
/// The centroid and orientation vote are reduced on the GPU; if flipped, face winding is swapped on GPU.
plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> orientNormalsOutwardFromCentroid(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Apply Taubin smoothing to GPU mesh points.
/// First version boundary: face adjacency and optional boundary flags are built on the host from the face matrix,
/// while each Laplacian update iteration runs on CUDA over the CSR adjacency.
plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> taubinSmooth(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& mesh,
    int iterations,
    float lambda = 0.5f,
    float mu = -0.53f,
    bool fix_boundary = false,
    cudaStream_t stream = 0);

/// Apply Taubin smoothing to GPU mesh points.
/// First version boundary: face adjacency and optional boundary flags are built on the host from the face matrix,
/// while each Laplacian update iteration runs on CUDA over the CSR adjacency.
plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> taubinSmooth(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& mesh,
    int iterations,
    double lambda = 0.5,
    double mu = -0.53,
    bool fix_boundary = false,
    cudaStream_t stream = 0);

} // namespace mesh
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
