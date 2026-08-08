#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <plapoint/core/point_cloud.h>

namespace plapoint
{
namespace opencl
{

struct OpenClKnnResult
{
    int k = 0;
    std::vector<int> indices;
    std::vector<std::uint8_t> finiteQueries;
};

/// Exact KNN through the deterministic OpenCL uniform-grid backend (row-major indices).
OpenClKnnResult knnSearch(
    const PointCloud<float, plamatrix::Device::CPU>& input,
    int k);

/// Exact KNN through the deterministic OpenCL uniform-grid backend (row-major indices).
OpenClKnnResult knnSearch(
    const PointCloud<double, plamatrix::Device::CPU>& input,
    int k);

/// Downsample CPU-owned float points through private OpenCL buffers.
PointCloud<float, plamatrix::Device::CPU> voxelDownsample(
    const PointCloud<float, plamatrix::Device::CPU>& input,
    float leaf_x,
    float leaf_y,
    float leaf_z);

/// Downsample CPU-owned double points through private OpenCL buffers.
PointCloud<double, plamatrix::Device::CPU> voxelDownsample(
    const PointCloud<double, plamatrix::Device::CPU>& input,
    double leaf_x,
    double leaf_y,
    double leaf_z);

/// Compute an OpenCL uniform-grid statistical outlier keep mask for float points.
std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const PointCloud<float, plamatrix::Device::CPU>& input,
    int mean_k,
    float stddev_mul);

/// Compute an OpenCL uniform-grid statistical outlier keep mask for double points.
std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const PointCloud<double, plamatrix::Device::CPU>& input,
    int mean_k,
    double stddev_mul);

/// Compute an OpenCL uniform-grid radius outlier keep mask for float points.
std::vector<std::uint8_t> radiusOutlierKeepMask(
    const PointCloud<float, plamatrix::Device::CPU>& input,
    float radius,
    int min_neighbors);

/// Compute an OpenCL uniform-grid radius outlier keep mask for double points.
std::vector<std::uint8_t> radiusOutlierKeepMask(
    const PointCloud<double, plamatrix::Device::CPU>& input,
    double radius,
    int min_neighbors);

} // namespace opencl
} // namespace plapoint
