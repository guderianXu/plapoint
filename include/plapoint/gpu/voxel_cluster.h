#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <plapoint/core/point_cloud.h>
#include <plamatrix/internal/core/device.h>

namespace plapoint::mesh
{

/// Simplify a GPU mesh by voxel-clustering vertices and averaging each occupied voxel.
/// Faces are remapped to clustered vertices; degenerate faces are removed.
/// Throws std::invalid_argument when cluster_size is not finite and positive.
plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> voxelClusterSimplify(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& mesh,
    float cluster_size);

/// Double-precision overload of GPU voxel-cluster mesh simplification.
plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> voxelClusterSimplify(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& mesh,
    double cluster_size);

} // namespace plapoint::mesh

#endif // PLAPOINT_WITH_CUDA
