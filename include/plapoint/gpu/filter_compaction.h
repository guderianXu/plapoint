#pragma once

#include <cstdint>
#include <vector>

#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/core/device.h>

#include <plapoint/core/point_cloud.h>

namespace plapoint {
namespace gpu {

/// Gather a GPU point cloud by host-selected point indices, preserving point-aligned attributes.
plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> gatherPointCloudByIndices(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& input,
    const std::vector<int>& indices);

/// Gather a GPU point cloud by host-selected point indices, preserving point-aligned attributes.
plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> gatherPointCloudByIndices(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& input,
    const std::vector<int>& indices);

#ifdef PLAPOINT_WITH_CUDA
/// Stably compact a GPU cloud by a device keep mask without staging point data on the host.
plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> compactPointCloudByKeepMask(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& input,
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask,
    cudaStream_t stream = nullptr);

/// Stably compact a GPU cloud by a device keep mask without staging point data on the host.
plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> compactPointCloudByKeepMask(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& input,
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask,
    cudaStream_t stream = nullptr);
#endif

} // namespace gpu
} // namespace plapoint
