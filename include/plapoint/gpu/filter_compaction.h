#pragma once

#include <vector>

#include <plamatrix/core/types.h>

#include <plapoint/core/point_cloud.h>

namespace plapoint {
namespace gpu {

/// Gather a GPU point cloud by host-selected point indices, preserving point-aligned attributes.
PointCloud<float, plamatrix::Device::GPU> gatherPointCloudByIndices(
    const PointCloud<float, plamatrix::Device::GPU>& input,
    const std::vector<int>& indices);

/// Gather a GPU point cloud by host-selected point indices, preserving point-aligned attributes.
PointCloud<double, plamatrix::Device::GPU> gatherPointCloudByIndices(
    const PointCloud<double, plamatrix::Device::GPU>& input,
    const std::vector<int>& indices);

} // namespace gpu
} // namespace plapoint
