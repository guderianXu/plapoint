#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <plapoint/core/point_cloud.h>

namespace plapoint::mesh
{

/// Simplify a GPU mesh by voxel-clustering vertices and averaging each occupied voxel.
/// Faces are remapped to clustered vertices; degenerate faces are removed.
/// Throws std::invalid_argument when cluster_size is not finite and positive.
PointCloud<float, plamatrix::Device::GPU> voxelClusterSimplify(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    float cluster_size);

/// Double-precision overload of GPU voxel-cluster mesh simplification.
PointCloud<double, plamatrix::Device::GPU> voxelClusterSimplify(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    double cluster_size);

} // namespace plapoint::mesh

#endif // PLAPOINT_WITH_CUDA
