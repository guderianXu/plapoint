#pragma once

#include <plamatrix/internal/core/device.h>
#include <plapoint/filters/detail/preprocessing_common.h>
#include <plapoint/filters/detail/geometry_bridge.h>

namespace plapoint::detail
{

/// Downsample an internal device cloud using voxel centroids.
template <typename Scalar, plamatrix::internal::Device Dev>
plapoint::internal::DeviceCloud<Scalar, Dev> voxelDownsample(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& input,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z)
{
    return detail::voxelDownsampleOnInputDevice(input, leaf_x, leaf_y, leaf_z);
}

/// Downsample a point cloud on its current device using cubic voxels.
template <typename Scalar, plamatrix::internal::Device Dev>
plapoint::internal::DeviceCloud<Scalar, Dev> voxelDownsample(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& input,
    Scalar leaf_size)
{
    return voxelDownsample(input, leaf_size, leaf_size, leaf_size);
}

/// Remove statistical outliers on the point cloud's current device.
template <typename Scalar, plamatrix::internal::Device Dev>
plapoint::internal::DeviceCloud<Scalar, Dev> statisticalOutlierRemoval(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& input,
    int mean_k,
    Scalar stddev_mul,
    std::vector<int>* removed_indices = nullptr)
{
    return detail::statisticalOutlierRemovalOnInputDevice(input, mean_k, stddev_mul, removed_indices);
}

/// Remove radius outliers on the point cloud's current device.
template <typename Scalar, plamatrix::internal::Device Dev>
plapoint::internal::DeviceCloud<Scalar, Dev> radiusOutlierRemoval(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& input,
    Scalar radius,
    int min_neighbors,
    std::vector<int>* removed_indices = nullptr)
{
    return detail::radiusOutlierRemovalOnInputDevice(input, radius, min_neighbors, removed_indices);
}

} // namespace plapoint::detail

#include <plapoint/filters/detail/preprocessing_voxel_dispatch.h>
#include <plapoint/filters/detail/preprocessing_outlier_dispatch.h>

namespace plapoint
{

/// Downsample an owning geometry cloud while preserving its optional attributes.
template <typename Scalar>
GeometryCloud<Scalar> voxelDownsample(const GeometryCloud<Scalar>& input,
                                     Scalar leaf_x, Scalar leaf_y, Scalar leaf_z,
                                     ProcessingDevice device = ProcessingDevice::Auto,
                                     ProcessingReport* report = nullptr)
{
    auto device_input = detail::toDeviceCloud(input);
    auto device_output = detail::voxelDownsample(device_input, leaf_x, leaf_y, leaf_z, device, report);
    return detail::fromDeviceCloud(device_output);
}

/// Downsample using a cubic voxel size.
template <typename Scalar>
GeometryCloud<Scalar> voxelDownsample(const GeometryCloud<Scalar>& input,
                                     Scalar leaf_size,
                                     ProcessingDevice device = ProcessingDevice::Auto,
                                     ProcessingReport* report = nullptr)
{
    return voxelDownsample(input, leaf_size, leaf_size, leaf_size, device, report);
}

/// Remove statistical outliers, optionally reporting removed point indices and device selection.
template <typename Scalar>
GeometryCloud<Scalar> statisticalOutlierRemoval(const GeometryCloud<Scalar>& input,
                                                int mean_k, Scalar stddev_mul,
                                                ProcessingDevice device = ProcessingDevice::Auto,
                                                std::vector<int>* removed_indices = nullptr,
                                                ProcessingReport* report = nullptr)
{
    auto device_input = detail::toDeviceCloud(input);
    auto device_output = detail::statisticalOutlierRemoval(
        device_input, mean_k, stddev_mul, device, removed_indices, report);
    return detail::fromDeviceCloud(device_output);
}

/// Remove radius outliers, optionally reporting removed point indices and device selection.
template <typename Scalar>
GeometryCloud<Scalar> radiusOutlierRemoval(const GeometryCloud<Scalar>& input,
                                          Scalar radius, int min_neighbors,
                                          ProcessingDevice device = ProcessingDevice::Auto,
                                          std::vector<int>* removed_indices = nullptr,
                                          ProcessingReport* report = nullptr)
{
    auto device_input = detail::toDeviceCloud(input);
    auto device_output = detail::radiusOutlierRemoval(
        device_input, radius, min_neighbors, device, removed_indices, report);
    return detail::fromDeviceCloud(device_output);
}

} // namespace plapoint
