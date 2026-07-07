#pragma once

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/search/kdtree.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif

namespace plapoint
{

/// Execution device preference for CPU-owned preprocessing convenience APIs.
enum class ProcessingDevice
{
    CPU,
    GPU,
    Auto
};

/// Reports which preprocessing device was requested and which one actually ran.
struct ProcessingReport
{
    ProcessingDevice requestedDevice = ProcessingDevice::CPU;
    ProcessingDevice usedDevice = ProcessingDevice::CPU;
    bool usedFallback = false;
    std::string fallbackReason;
};

namespace detail
{

template <typename Scalar, plamatrix::Device Dev>
std::shared_ptr<const PointCloud<Scalar, Dev>> nonOwningCloudPtr(
    const PointCloud<Scalar, Dev>& cloud)
{
    return std::shared_ptr<const PointCloud<Scalar, Dev>>(&cloud, [](const PointCloud<Scalar, Dev>*) {});
}

inline bool shouldTryGpu(ProcessingDevice device)
{
    return device == ProcessingDevice::GPU || device == ProcessingDevice::Auto;
}

inline bool gpuIsAvailable()
{
#ifdef PLAPOINT_WITH_CUDA
    return gpu::hasUsableCudaDevice();
#else
    return false;
#endif
}

template <typename Scalar, plamatrix::Device Dev>
bool hasPointAttributes(const PointCloud<Scalar, Dev>& input)
{
    return input.hasNormals() || input.hasColors() || input.hasIntensities() || input.hasScalarFields();
}

inline void setReport(ProcessingReport* report,
                      ProcessingDevice requested,
                      ProcessingDevice used,
                      bool fallback,
                      std::string reason = {})
{
    if (!report)
    {
        return;
    }
    report->requestedDevice = requested;
    report->usedDevice = used;
    report->usedFallback = fallback;
    report->fallbackReason = std::move(reason);
}

template <typename Scalar, plamatrix::Device Dev>
PointCloud<Scalar, Dev> voxelDownsampleOnInputDevice(
    const PointCloud<Scalar, Dev>& input,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z)
{
    VoxelGrid<Scalar, Dev> filter;
    filter.setInputCloud(nonOwningCloudPtr(input));
    filter.setLeafSize(leaf_x, leaf_y, leaf_z);

    PointCloud<Scalar, Dev> output;
    filter.filter(output);
    return output;
}

template <typename Scalar, plamatrix::Device Dev>
PointCloud<Scalar, Dev> statisticalOutlierRemovalOnInputDevice(
    const PointCloud<Scalar, Dev>& input,
    int mean_k,
    Scalar stddev_mul,
    std::vector<int>* removed_indices)
{
    auto input_ptr = nonOwningCloudPtr(input);
    auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
    tree->setInputCloud(input_ptr);
    tree->build();

    StatisticalOutlierRemoval<Scalar, Dev> filter;
    filter.setInputCloud(input_ptr);
    filter.setSearchMethod(tree);
    filter.setMeanK(mean_k);
    filter.setStddevMulThresh(stddev_mul);

    PointCloud<Scalar, Dev> output;
    if (removed_indices)
    {
        filter.filter(output, *removed_indices);
    }
    else
    {
        filter.filter(output);
    }
    return output;
}

template <typename Scalar, plamatrix::Device Dev>
PointCloud<Scalar, Dev> radiusOutlierRemovalOnInputDevice(
    const PointCloud<Scalar, Dev>& input,
    Scalar radius,
    int min_neighbors,
    std::vector<int>* removed_indices)
{
    RadiusOutlierRemoval<Scalar, Dev> filter;
    filter.setInputCloud(nonOwningCloudPtr(input));
    filter.setRadius(radius);
    filter.setMinNeighbors(min_neighbors);

    PointCloud<Scalar, Dev> output;
    if (removed_indices)
    {
        filter.filter(output, *removed_indices);
    }
    else
    {
        filter.filter(output);
    }
    return output;
}

} // namespace detail

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

/// Downsample a CPU-owned point cloud using the requested device, returning CPU-owned output.
template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> voxelDownsample(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z,
    ProcessingDevice device,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    if (detail::shouldTryGpu(device))
    {
        if (detail::hasPointAttributes(input))
        {
            fallback_reason = "Voxel downsampling of attributed CPU clouds uses the CPU path to preserve attributes";
        }
        else
        {
#ifdef PLAPOINT_WITH_CUDA
            if (detail::gpuIsAvailable())
            {
                try
                {
                    const auto gpu_input = input.toGpu();
                    const auto gpu_output = voxelDownsample(gpu_input, leaf_x, leaf_y, leaf_z);
                    detail::setReport(report, device, ProcessingDevice::GPU, false);
                    return gpu_output.toCpu();
                }
                catch (const std::exception& ex)
                {
                    fallback_reason = ex.what();
                }
            }
            else
            {
                fallback_reason = "CUDA device is not available";
            }
#else
            fallback_reason = "PlaPoint was built without CUDA support";
#endif
        }
    }

    auto output = voxelDownsample(input, leaf_x, leaf_y, leaf_z);
    detail::setReport(report, device, ProcessingDevice::CPU,
                      detail::shouldTryGpu(device) && !fallback_reason.empty(),
                      fallback_reason);
    return output;
}

/// Downsample a CPU-owned point cloud using cubic voxels and the requested device.
template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> voxelDownsample(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    Scalar leaf_size,
    ProcessingDevice device,
    ProcessingReport* report = nullptr)
{
    return voxelDownsample(input, leaf_size, leaf_size, leaf_size, device, report);
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

/// Remove statistical outliers from a CPU-owned point cloud using the requested device.
template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> statisticalOutlierRemoval(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    int mean_k,
    Scalar stddev_mul,
    ProcessingDevice device,
    std::vector<int>* removed_indices = nullptr,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    if (detail::shouldTryGpu(device))
    {
#ifdef PLAPOINT_WITH_CUDA
        if (detail::gpuIsAvailable())
        {
            try
            {
                std::vector<int> gpu_removed;
                const auto gpu_input = input.toGpu();
                const auto gpu_output = statisticalOutlierRemoval(
                    gpu_input, mean_k, stddev_mul, removed_indices ? &gpu_removed : nullptr);
                if (removed_indices)
                {
                    *removed_indices = std::move(gpu_removed);
                }
                detail::setReport(report, device, ProcessingDevice::GPU, false);
                return gpu_output.toCpu();
            }
            catch (const std::exception& ex)
            {
                fallback_reason = ex.what();
            }
        }
        else
        {
            fallback_reason = "CUDA device is not available";
        }
#else
        fallback_reason = "PlaPoint was built without CUDA support";
#endif
    }

    auto output = statisticalOutlierRemoval(input, mean_k, stddev_mul, removed_indices);
    detail::setReport(report, device, ProcessingDevice::CPU,
                      detail::shouldTryGpu(device) && !fallback_reason.empty(),
                      fallback_reason);
    return output;
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

/// Remove radius outliers from a CPU-owned point cloud using the requested device.
template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> radiusOutlierRemoval(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    Scalar radius,
    int min_neighbors,
    ProcessingDevice device,
    std::vector<int>* removed_indices = nullptr,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    if (detail::shouldTryGpu(device))
    {
#ifdef PLAPOINT_WITH_CUDA
        if (detail::gpuIsAvailable())
        {
            try
            {
                std::vector<int> gpu_removed;
                const auto gpu_input = input.toGpu();
                const auto gpu_output = radiusOutlierRemoval(
                    gpu_input, radius, min_neighbors, removed_indices ? &gpu_removed : nullptr);
                if (removed_indices)
                {
                    *removed_indices = std::move(gpu_removed);
                }
                detail::setReport(report, device, ProcessingDevice::GPU, false);
                return gpu_output.toCpu();
            }
            catch (const std::exception& ex)
            {
                fallback_reason = ex.what();
            }
        }
        else
        {
            fallback_reason = "CUDA device is not available";
        }
#else
        fallback_reason = "PlaPoint was built without CUDA support";
#endif
    }

    auto output = radiusOutlierRemoval(input, radius, min_neighbors, removed_indices);
    detail::setReport(report, device, ProcessingDevice::CPU,
                      detail::shouldTryGpu(device) && !fallback_reason.empty(),
                      fallback_reason);
    return output;
}

} // namespace plapoint
