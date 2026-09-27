#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <plapoint/core/point_cloud.h>
#include <plapoint/core/processing_policy.h>
#include <plapoint/filters/filter.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/internal/core/device.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif
#ifdef PLAPOINT_WITH_OPENCL
#include <plapoint/opencl/opencl_runtime.h>
#include <plapoint/opencl/preprocessing.h>
#endif

namespace plapoint
{
namespace detail
{

template <typename Scalar, plamatrix::internal::Device Dev>
std::shared_ptr<const plapoint::internal::DeviceCloud<Scalar, Dev>> nonOwningCloudPtr(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& cloud)
{
    return std::shared_ptr<const plapoint::internal::DeviceCloud<Scalar, Dev>>(&cloud, [](const plapoint::internal::DeviceCloud<Scalar, Dev>*) {});
}

inline bool gpuIsAvailable()
{
#ifdef PLAPOINT_WITH_CUDA
    return gpu::hasUsableCudaDevice();
#else
    return false;
#endif
}

inline void appendFallbackReason(std::string& reasons,
                                 const char* backend,
                                 const std::string& reason)
{
    if (!reasons.empty())
    {
        reasons += "; ";
    }
    reasons += backend;
    reasons += ": ";
    reasons += reason;
}

inline void setReport(ProcessingReport* report,
                      ProcessingDevice requested,
                      ProcessingDevice used,
                      bool fallback,
                      std::string reason = {},
                      ProcessingNeighborBackend backend = ProcessingNeighborBackend::None)
{
    if (!report)
    {
        return;
    }
    report->requestedDevice = requested;
    report->actualDevice = used;
    report->usedDevice = used;
    report->neighborBackend = backend;
    report->usedFallback = fallback;
    report->fallbackReason = std::move(reason);
    report->selectionReason.clear();
}

inline void setAutoCpuReport(ProcessingReport* report,
                             ProcessingNeighborBackend backend,
                             std::string reason)
{
    setReport(report, ProcessingDevice::Auto, ProcessingDevice::CPU, false, {}, backend);
    if (report)
    {
        report->selectionReason = std::move(reason);
    }
}

template <typename Scalar>
class CpuPointSelection final : public Filter<Scalar, plamatrix::internal::Device::CPU>
{
public:
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    Cloud select(const Cloud& input,
                 const std::vector<std::uint8_t>& keep_mask,
                 std::vector<int>* removed_indices)
    {
        if (keep_mask.size() != input.size())
        {
            throw std::invalid_argument("OpenCL filter keep-mask size does not match point count");
        }
        this->setInputCloud(nonOwningCloudPtr(input));
        std::vector<int> kept;
        kept.reserve(keep_mask.size());
        for (std::size_t index = 0; index < keep_mask.size(); ++index)
        {
            if (keep_mask[index] != 0)
            {
                kept.push_back(static_cast<int>(index));
            }
        }
        Cloud output;
        this->copyPointsAndAttributesForIndices(kept, output);
        if (removed_indices)
        {
            *removed_indices = this->removedIndicesFromKept(kept);
        }
        return output;
    }

protected:
    void applyFilter(Cloud&) override {}
};

template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> selectByKeepMask(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& input,
    const std::vector<std::uint8_t>& keep_mask,
    std::vector<int>* removed_indices)
{
    CpuPointSelection<Scalar> selection;
    return selection.select(input, keep_mask, removed_indices);
}

template <typename Scalar, plamatrix::internal::Device Dev>
plapoint::internal::DeviceCloud<Scalar, Dev> voxelDownsampleOnInputDevice(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& input,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z)
{
    VoxelGrid<Scalar, Dev> filter;
    filter.setInputCloud(nonOwningCloudPtr(input));
    filter.setLeafSize(leaf_x, leaf_y, leaf_z);
    plapoint::internal::DeviceCloud<Scalar, Dev> output;
    filter.filter(output);
    return output;
}

template <typename Scalar, plamatrix::internal::Device Dev>
plapoint::internal::DeviceCloud<Scalar, Dev> statisticalOutlierRemovalOnInputDevice(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& input,
    int mean_k,
    Scalar stddev_mul,
    std::vector<int>* removed_indices)
{
    auto input_ptr = nonOwningCloudPtr(input);
    StatisticalOutlierRemoval<Scalar, Dev> filter;
    filter.setInputCloud(input_ptr);
    filter.setMeanK(mean_k);
    filter.setStddevMulThresh(stddev_mul);
    const auto indexed_k = input.size() == 0
        ? std::size_t(0)
        : std::min<std::size_t>(static_cast<std::size_t>(mean_k) + 1, input.size());
    if constexpr (Dev == plamatrix::internal::Device::CPU)
    {
        auto tree = std::make_shared<search::internal::DeviceKdTree<Scalar, Dev>>();
        tree->setInputCloud(input_ptr);
        tree->build();
        filter.setSearchMethod(std::move(tree));
    }
    else if (indexed_k > 32)
    {
        auto tree = std::make_shared<search::internal::DeviceKdTree<Scalar, Dev>>();
        tree->setInputCloud(input_ptr);
        tree->build();
        filter.setSearchMethod(std::move(tree));
    }
    plapoint::internal::DeviceCloud<Scalar, Dev> output;
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

template <typename Scalar, plamatrix::internal::Device Dev>
plapoint::internal::DeviceCloud<Scalar, Dev> radiusOutlierRemovalOnInputDevice(
    const plapoint::internal::DeviceCloud<Scalar, Dev>& input,
    Scalar radius,
    int min_neighbors,
    std::vector<int>* removed_indices)
{
    RadiusOutlierRemoval<Scalar, Dev> filter;
    filter.setInputCloud(nonOwningCloudPtr(input));
    filter.setRadius(radius);
    filter.setMinNeighbors(min_neighbors);
    plapoint::internal::DeviceCloud<Scalar, Dev> output;
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
} // namespace plapoint
