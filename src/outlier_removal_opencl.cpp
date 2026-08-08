#include <plapoint/opencl/preprocessing.h>

#include "opencl_preprocessing_detail.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace plapoint
{
namespace opencl
{
namespace
{

template <typename Scalar>
std::vector<std::uint8_t> statisticalOutlierKeepMaskImpl(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    int mean_k,
    Scalar stddev_mul)
{
    if (mean_k <= 0 || mean_k > std::numeric_limits<int>::max() - 1)
    {
        throw std::invalid_argument("OpenCL statistical outlier removal: mean k must be positive");
    }
    if (!std::isfinite(stddev_mul) || stddev_mul < Scalar(0))
    {
        throw std::invalid_argument(
            "OpenCL statistical outlier removal: stddev multiplier must be finite and non-negative");
    }
    if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("OpenCL statistical outlier removal: point count exceeds int range");
    }
    if (input.size() == 0) return {};

    const auto points = detail::rowMajorPoints(input);
    const Scalar cell_size = detail::adaptiveCellSize(points, input.size());
    const detail::HostGrid grid = detail::buildGrid(points, input.size(), cell_size);
    detail::validateKnnGridWork(grid);
    const int finite_count = static_cast<int>(grid.sortedIndices.size());
    const int k = std::min(mean_k + 1, finite_count);
    if (k > detail::maximumKnn)
    {
        throw std::invalid_argument(
            "OpenCL statistical outlier removal requires meanK + self <= 32");
    }
    if (finite_count == 0) return std::vector<std::uint8_t>(input.size(), 0);
    if (std::max({grid.spanX, grid.spanY, grid.spanZ}) > detail::maximumEnumeratedShell)
    {
        throw std::runtime_error("OpenCL uniform-grid KNN span exceeds the bounded shell-search limit");
    }

    auto& runtime = detail::OpenClRuntime::instance();
    detail::requireFp64<Scalar>(runtime);
    detail::CommandQueue queue(runtime.createQueue());
    auto buffers = detail::makeGridBuffers(runtime, points, grid);
    std::vector<Scalar> mean_distances(input.size(), Scalar(0));
    std::vector<std::uint8_t> valid_distances(input.size(), 0);
    detail::DeviceBuffer mean_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY, detail::byteSize<Scalar>(mean_distances.size()));
    detail::DeviceBuffer valid_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY,
        detail::byteSize<std::uint8_t>(valid_distances.size()));
    detail::DeviceBuffer dummy_neighbor(runtime.context(), CL_MEM_WRITE_ONLY, sizeof(int));
    const cl_program program = runtime.program(
        detail::programKey<Scalar>("preprocessing_neighbors"),
        detail::neighborKernelSource,
        detail::realBuildOptions<Scalar>());
    detail::CompiledKernel kernel(program, "knnMeanDistances");
    const int point_count = static_cast<int>(input.size());
    detail::setGridKernelArgs(kernel, buffers, grid, point_count);
    detail::kernelArg(kernel, 12, cell_size);
    detail::kernelArg(kernel, 13, k);
    detail::kernelBufferArg(kernel, 14, mean_buffer);
    detail::kernelBufferArg(kernel, 15, valid_buffer);
    const int write_neighbors = 0;
    detail::kernelArg(kernel, 16, write_neighbors);
    detail::kernelBufferArg(kernel, 17, dummy_neighbor);
    const std::size_t global_size = input.size();
    detail::checkOpenCl(
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(knnMeanDistances)");
    detail::readVector(queue, mean_buffer, mean_distances);
    detail::readVector(queue, valid_buffer, valid_distances);

    long double scale = 0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (!grid.finiteMask[index]) continue;
        if (!valid_distances[index] || !std::isfinite(mean_distances[index]))
        {
            throw std::runtime_error(
                "OpenCL uniform-grid KNN failed to produce a finite neighborhood distance");
        }
        scale = std::max(scale, static_cast<long double>(mean_distances[index]));
    }
    long double normalized_mean = 0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (grid.finiteMask[index])
        {
            normalized_mean += scale == 0
                ? 0
                : static_cast<long double>(mean_distances[index]) / scale;
        }
    }
    normalized_mean /= static_cast<long double>(finite_count);
    long double variance = 0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (grid.finiteMask[index])
        {
            const long double normalized = scale == 0
                ? 0
                : static_cast<long double>(mean_distances[index]) / scale;
            const long double difference = normalized - normalized_mean;
            variance += difference * difference;
        }
    }
    variance /= static_cast<long double>(finite_count);
    const long double threshold = normalized_mean
        + static_cast<long double>(stddev_mul) * std::sqrt(variance);
    if (!std::isfinite(threshold)) return std::vector<std::uint8_t>(input.size(), 0);

    std::vector<std::uint8_t> keep_mask(input.size(), 0);
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (grid.finiteMask[index])
        {
            const long double normalized = scale == 0
                ? 0
                : static_cast<long double>(mean_distances[index]) / scale;
            keep_mask[index] = normalized <= threshold ? 1 : 0;
        }
    }
    return keep_mask;
}

template <typename Scalar>
std::vector<std::uint8_t> radiusOutlierKeepMaskImpl(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    Scalar radius,
    int min_neighbors)
{
    if (!std::isfinite(radius) || radius < Scalar(0))
    {
        throw std::invalid_argument(
            "OpenCL radius outlier removal: radius must be finite and non-negative");
    }
    if (min_neighbors <= 0)
    {
        throw std::invalid_argument("OpenCL radius outlier removal: minimum neighbors must be positive");
    }
    if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("OpenCL radius outlier removal: point count exceeds int range");
    }
    if (input.size() == 0) return {};

    const auto points = detail::rowMajorPoints(input);
    const Scalar cell_size = radius > Scalar(0)
        ? radius
        : detail::adaptiveCellSize(points, input.size());
    const detail::HostGrid grid = detail::buildGrid(points, input.size(), cell_size);
    if (grid.sortedIndices.empty()) return std::vector<std::uint8_t>(input.size(), 0);
    if (min_neighbors > static_cast<int>(grid.sortedIndices.size()))
    {
        return std::vector<std::uint8_t>(input.size(), 0);
    }
    detail::validateRadiusGridWork(grid);

    auto& runtime = detail::OpenClRuntime::instance();
    detail::requireFp64<Scalar>(runtime);
    detail::CommandQueue queue(runtime.createQueue());
    auto buffers = detail::makeGridBuffers(runtime, points, grid);
    std::vector<std::uint8_t> keep_mask(input.size(), 0);
    detail::DeviceBuffer keep_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY, detail::byteSize<std::uint8_t>(keep_mask.size()));
    const cl_program program = runtime.program(
        detail::programKey<Scalar>("preprocessing_neighbors"),
        detail::neighborKernelSource,
        detail::realBuildOptions<Scalar>());
    detail::CompiledKernel kernel(program, "radiusKeepMask");
    const int point_count = static_cast<int>(input.size());
    detail::setGridKernelArgs(kernel, buffers, grid, point_count);
    detail::kernelArg(kernel, 12, radius);
    detail::kernelArg(kernel, 13, min_neighbors);
    detail::kernelBufferArg(kernel, 14, keep_buffer);
    const std::size_t global_size = input.size();
    detail::checkOpenCl(
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(radiusKeepMask)");
    detail::readVector(queue, keep_buffer, keep_mask);
    return keep_mask;
}

} // namespace

std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const PointCloud<float, plamatrix::Device::CPU>& input, int mean_k, float stddev_mul)
{
    return statisticalOutlierKeepMaskImpl(input, mean_k, stddev_mul);
}

std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const PointCloud<double, plamatrix::Device::CPU>& input, int mean_k, double stddev_mul)
{
    return statisticalOutlierKeepMaskImpl(input, mean_k, stddev_mul);
}

std::vector<std::uint8_t> radiusOutlierKeepMask(
    const PointCloud<float, plamatrix::Device::CPU>& input, float radius, int min_neighbors)
{
    return radiusOutlierKeepMaskImpl(input, radius, min_neighbors);
}

std::vector<std::uint8_t> radiusOutlierKeepMask(
    const PointCloud<double, plamatrix::Device::CPU>& input, double radius, int min_neighbors)
{
    return radiusOutlierKeepMaskImpl(input, radius, min_neighbors);
}

} // namespace opencl
} // namespace plapoint
