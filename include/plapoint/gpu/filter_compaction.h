#pragma once

#include <cstdint>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>
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

#ifdef PLAPOINT_WITH_CUDA
/// Stably compact a GPU cloud by a device keep mask without staging point data on the host.
PointCloud<float, plamatrix::Device::GPU> compactPointCloudByKeepMask(
    const PointCloud<float, plamatrix::Device::GPU>& input,
    const plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>& keep_mask,
    cudaStream_t stream = nullptr);

/// Stably compact a GPU cloud by a device keep mask without staging point data on the host.
PointCloud<double, plamatrix::Device::GPU> compactPointCloudByKeepMask(
    const PointCloud<double, plamatrix::Device::GPU>& input,
    const plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>& keep_mask,
    cudaStream_t stream = nullptr);
#endif

} // namespace gpu
} // namespace plapoint
