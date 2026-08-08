#include <plapoint/opencl/preprocessing.h>

#include "opencl_preprocessing_detail.h"

#include <algorithm>
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
OpenClKnnResult knnSearchImpl(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    int requested_k)
{
    if (requested_k <= 0 || requested_k > detail::maximumKnn)
    {
        throw std::invalid_argument("OpenCL KNN requires 1 <= k <= 32");
    }
    if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("OpenCL KNN point count exceeds int range");
    }
    OpenClKnnResult result;
    result.finiteQueries.assign(input.size(), 0);
    if (input.size() == 0) return result;

    const auto points = detail::rowMajorPoints(input);
    const Scalar cell_size = detail::adaptiveCellSize(points, input.size());
    const detail::HostGrid grid = detail::buildGrid(points, input.size(), cell_size);
    detail::validateKnnGridWork(grid);
    result.finiteQueries = grid.finiteMask;
    const int finite_count = static_cast<int>(grid.sortedIndices.size());
    if (finite_count == 0) return result;
    result.k = std::min(requested_k, finite_count);
    result.indices.assign(input.size() * static_cast<std::size_t>(result.k), -1);
    if (std::max({grid.spanX, grid.spanY, grid.spanZ}) > detail::maximumEnumeratedShell)
    {
        throw std::runtime_error("OpenCL uniform-grid KNN span exceeds the bounded shell-search limit");
    }

    auto& runtime = detail::OpenClRuntime::instance();
    detail::requireFp64<Scalar>(runtime);
    detail::CommandQueue queue(runtime.createQueue());
    auto buffers = detail::makeGridBuffers(runtime, points, grid);
    std::vector<Scalar> ignored_means(input.size(), Scalar(0));
    std::vector<std::uint8_t> valid_distances(input.size(), 0);
    detail::DeviceBuffer mean_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY, detail::byteSize<Scalar>(ignored_means.size()));
    detail::DeviceBuffer valid_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY,
        detail::byteSize<std::uint8_t>(valid_distances.size()));
    detail::DeviceBuffer neighbor_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY, detail::byteSize<int>(result.indices.size()));
    const cl_program program = runtime.program(
        detail::programKey<Scalar>("preprocessing_neighbors"),
        detail::neighborKernelSource,
        detail::realBuildOptions<Scalar>());
    detail::CompiledKernel kernel(program, "knnMeanDistances");
    const int point_count = static_cast<int>(input.size());
    detail::setGridKernelArgs(kernel, buffers, grid, point_count);
    detail::kernelArg(kernel, 12, cell_size);
    detail::kernelArg(kernel, 13, result.k);
    detail::kernelBufferArg(kernel, 14, mean_buffer);
    detail::kernelBufferArg(kernel, 15, valid_buffer);
    const int write_neighbors = 1;
    detail::kernelArg(kernel, 16, write_neighbors);
    detail::kernelBufferArg(kernel, 17, neighbor_buffer);
    const std::size_t global_size = input.size();
    detail::checkOpenCl(
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(knnSearch)");
    detail::readVector(queue, neighbor_buffer, result.indices);
    detail::readVector(queue, valid_buffer, valid_distances);
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (grid.finiteMask[index] && !valid_distances[index])
        {
            throw std::runtime_error("OpenCL uniform-grid KNN did not complete a finite query");
        }
    }
    return result;
}

} // namespace

OpenClKnnResult knnSearch(
    const PointCloud<float, plamatrix::Device::CPU>& input,
    int k)
{
    return knnSearchImpl(input, k);
}

OpenClKnnResult knnSearch(
    const PointCloud<double, plamatrix::Device::CPU>& input,
    int k)
{
    return knnSearchImpl(input, k);
}

} // namespace opencl
} // namespace plapoint
