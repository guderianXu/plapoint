#pragma once

#include <plapoint/filters/detail/preprocessing_common.h>

namespace plapoint
{

/// Downsample a point cloud on its current device using voxel centroids.
template <typename Scalar, plamatrix::Device Dev>
PointCloud<Scalar, Dev> voxelDownsample(
    const PointCloud<Scalar, Dev>& input,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z)
{
    return detail::voxelDownsampleOnInputDevice(input, leaf_x, leaf_y, leaf_z);
}

/// Downsample a point cloud on its current device using cubic voxels.
template <typename Scalar, plamatrix::Device Dev>
PointCloud<Scalar, Dev> voxelDownsample(
    const PointCloud<Scalar, Dev>& input,
    Scalar leaf_size)
{
    return voxelDownsample(input, leaf_size, leaf_size, leaf_size);
}

/// Remove statistical outliers on the point cloud's current device.
template <typename Scalar, plamatrix::Device Dev>
PointCloud<Scalar, Dev> statisticalOutlierRemoval(
    const PointCloud<Scalar, Dev>& input,
    int mean_k,
    Scalar stddev_mul,
    std::vector<int>* removed_indices = nullptr)
{
    return detail::statisticalOutlierRemovalOnInputDevice(input, mean_k, stddev_mul, removed_indices);
}

/// Remove radius outliers on the point cloud's current device.
template <typename Scalar, plamatrix::Device Dev>
PointCloud<Scalar, Dev> radiusOutlierRemoval(
    const PointCloud<Scalar, Dev>& input,
    Scalar radius,
    int min_neighbors,
    std::vector<int>* removed_indices = nullptr)
{
    return detail::radiusOutlierRemovalOnInputDevice(input, radius, min_neighbors, removed_indices);
}

} // namespace plapoint

#include <plapoint/filters/detail/preprocessing_voxel_dispatch.h>
#include <plapoint/filters/detail/preprocessing_outlier_dispatch.h>
