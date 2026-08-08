#pragma once

#include "opencl_execution.h"
#include "opencl_preprocessing_kernels.h"
#include "opencl_uniform_grid.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <plapoint/core/point_cloud.h>

namespace plapoint
{
namespace opencl
{
namespace detail
{

template <typename Scalar>
std::vector<Scalar> rowMajorPoints(const PointCloud<Scalar, plamatrix::Device::CPU>& input)
{
    std::vector<Scalar> points(input.size() * 3u);
    const auto& matrix = input.points();
    for (std::size_t row = 0; row < input.size(); ++row)
    {
        points[row * 3u] = matrix(static_cast<plamatrix::Index>(row), 0);
        points[row * 3u + 1u] = matrix(static_cast<plamatrix::Index>(row), 1);
        points[row * 3u + 2u] = matrix(static_cast<plamatrix::Index>(row), 2);
    }
    return points;
}

template <typename Scalar>
std::string programKey(const char* prefix)
{
    return std::string(prefix) + (std::is_same_v<Scalar, double> ? "_f64" : "_f32");
}

template <typename Scalar>
struct GridBuffers
{
    DeviceBuffer points;
    DeviceBuffer finiteMask;
    DeviceBuffer queryCells;
    DeviceBuffer cells;
    DeviceBuffer sortedIndices;
    DeviceBuffer offsets;
    DeviceBuffer counts;
};

template <typename Scalar>
GridBuffers<Scalar> makeGridBuffers(
    OpenClRuntime& runtime,
    const std::vector<Scalar>& points,
    const HostGrid& grid)
{
    return {
        inputVector(runtime, points),
        inputVector(runtime, grid.finiteMask),
        inputVector(runtime, grid.queryCells),
        inputVector(runtime, grid.cellCoords),
        inputVector(runtime, grid.sortedIndices),
        inputVector(runtime, grid.offsets),
        inputVector(runtime, grid.counts)};
}

template <typename Scalar>
void setGridKernelArgs(
    cl_kernel kernel,
    const GridBuffers<Scalar>& buffers,
    const HostGrid& grid,
    int point_count)
{
    kernelBufferArg(kernel, 0, buffers.points);
    kernelBufferArg(kernel, 1, buffers.finiteMask);
    kernelBufferArg(kernel, 2, buffers.queryCells);
    kernelBufferArg(kernel, 3, buffers.cells);
    kernelBufferArg(kernel, 4, buffers.sortedIndices);
    kernelBufferArg(kernel, 5, buffers.offsets);
    kernelBufferArg(kernel, 6, buffers.counts);
    kernelArg(kernel, 7, point_count);
    const int cell_count = static_cast<int>(grid.counts.size());
    kernelArg(kernel, 8, cell_count);
    kernelArg(kernel, 9, grid.spanX);
    kernelArg(kernel, 10, grid.spanY);
    kernelArg(kernel, 11, grid.spanZ);
}

} // namespace detail
} // namespace opencl
} // namespace plapoint
