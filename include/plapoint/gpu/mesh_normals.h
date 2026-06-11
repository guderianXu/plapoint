#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <plapoint/core/point_cloud.h>

namespace plapoint
{
namespace mesh
{

/// Recompute mesh vertex normals on CUDA from column-major GPU points and Fx3 GPU faces.
/// Face normal accumulation and per-vertex normalization run on the GPU.
PointCloud<float, plamatrix::Device::GPU> recomputeVertexNormals(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Recompute mesh vertex normals on CUDA from column-major GPU points and Fx3 GPU faces.
/// Face normal accumulation and per-vertex normalization run on the GPU.
PointCloud<double, plamatrix::Device::GPU> recomputeVertexNormals(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Orient existing GPU normals outward from the mesh centroid.
/// The centroid and orientation vote are reduced on the GPU; if flipped, face winding is swapped on GPU.
PointCloud<float, plamatrix::Device::GPU> orientNormalsOutwardFromCentroid(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Orient existing GPU normals outward from the mesh centroid.
/// The centroid and orientation vote are reduced on the GPU; if flipped, face winding is swapped on GPU.
PointCloud<double, plamatrix::Device::GPU> orientNormalsOutwardFromCentroid(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream = 0);

/// Apply Taubin smoothing to GPU mesh points.
/// First version boundary: face adjacency and optional boundary flags are built on the host from the face matrix,
/// while each Laplacian update iteration runs on CUDA over the CSR adjacency.
PointCloud<float, plamatrix::Device::GPU> taubinSmooth(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    int iterations,
    float lambda = 0.5f,
    float mu = -0.53f,
    bool fix_boundary = false,
    cudaStream_t stream = 0);

/// Apply Taubin smoothing to GPU mesh points.
/// First version boundary: face adjacency and optional boundary flags are built on the host from the face matrix,
/// while each Laplacian update iteration runs on CUDA over the CSR adjacency.
PointCloud<double, plamatrix::Device::GPU> taubinSmooth(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    int iterations,
    double lambda = 0.5,
    double mu = -0.53,
    bool fix_boundary = false,
    cudaStream_t stream = 0);

} // namespace mesh
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
