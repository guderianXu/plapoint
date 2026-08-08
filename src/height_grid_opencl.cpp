#include <plapoint/opencl/height_grid.h>

#include "opencl_execution.h"
#include "opencl_height_grid_kernel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace plapoint
{
namespace opencl
{
namespace
{

std::atomic<std::uint64_t> height_grid_execution_count{0};

template <typename Scalar>
struct Contribution
{
    int cell = 0;
    int point = 0;
    Scalar weight = Scalar(0);
};

template <typename Scalar>
void appendContribution(
    std::vector<Contribution<Scalar>>& contributions,
    int cell,
    int point,
    Scalar weight)
{
    if (weight > Scalar(0))
    {
        contributions.push_back({cell, point, weight});
    }
}

template <typename Scalar>
std::vector<Contribution<Scalar>> buildContributions(
    const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    const mesh::HeightGrid<Scalar>& grid)
{
    std::vector<Contribution<Scalar>> result;
    result.reserve(cloud.size() * (options.useBilinearSplat ? 4u : 1u));
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        const Scalar x = cloud.points().getValue(row, 0);
        const Scalar y = cloud.points().getValue(row, 1);
        const Scalar z = cloud.points().getValue(row, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            continue;
        }
        const Scalar gx = (x - grid.minX) / grid.stepX;
        const Scalar gy = (y - grid.minY) / grid.stepY;
        if (gx < Scalar(0) || gx > Scalar(grid.width - 1)
            || gy < Scalar(0) || gy > Scalar(grid.height - 1))
        {
            continue;
        }
        if (!options.useBilinearSplat)
        {
            const int ix = std::clamp(static_cast<int>(std::round(gx)), 0, grid.width - 1);
            const int iy = std::clamp(static_cast<int>(std::round(gy)), 0, grid.height - 1);
            appendContribution(result, iy * grid.width + ix, static_cast<int>(index), Scalar(1));
            continue;
        }

        const int ix = std::clamp(static_cast<int>(std::floor(gx)), 0, grid.width - 2);
        const int iy = std::clamp(static_cast<int>(std::floor(gy)), 0, grid.height - 2);
        const Scalar tx = std::clamp(gx - Scalar(ix), Scalar(0), Scalar(1));
        const Scalar ty = std::clamp(gy - Scalar(iy), Scalar(0), Scalar(1));
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const Scalar wx = dx != 0 ? tx : Scalar(1) - tx;
                const Scalar wy = dy != 0 ? ty : Scalar(1) - ty;
                appendContribution(
                    result, (iy + dy) * grid.width + ix + dx,
                    static_cast<int>(index), wx * wy);
            }
        }
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs)
    {
        return lhs.cell < rhs.cell;
    });
    return result;
}

int aggregationCode(mesh::ElevationAggregation aggregation)
{
    switch (aggregation)
    {
    case mesh::ElevationAggregation::Mean: return 0;
    case mesh::ElevationAggregation::Min: return 1;
    case mesh::ElevationAggregation::Max: return 2;
    }
    throw std::invalid_argument("OpenCL height grid: unsupported elevation aggregation");
}

template <typename Scalar>
mesh::HeightGrid<Scalar> initializeGridGeometry(
    const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options)
{
    mesh::HeightGrid<Scalar> grid;
    if (cloud.size() == 0) return grid;
    if (!std::isfinite(options.padding) || options.padding < Scalar(0))
    {
        throw std::invalid_argument("OpenCL height grid: padding must be finite and non-negative");
    }
    Scalar min_x = Scalar(0);
    Scalar max_x = Scalar(0);
    Scalar min_y = Scalar(0);
    Scalar max_y = Scalar(0);
    bool have_bounds = false;
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        const Scalar x = cloud.points().getValue(row, 0);
        const Scalar y = cloud.points().getValue(row, 1);
        const Scalar z = cloud.points().getValue(row, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            if (options.skipNonFinite) continue;
            throw std::invalid_argument("OpenCL height grid: points must be finite");
        }
        if (!have_bounds)
        {
            min_x = max_x = x;
            min_y = max_y = y;
            have_bounds = true;
        }
        else
        {
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
    }
    if (!have_bounds) return grid;
    if (options.useExplicitBounds)
    {
        if (!std::isfinite(options.minX) || !std::isfinite(options.maxX)
            || !std::isfinite(options.minY) || !std::isfinite(options.maxY)
            || options.maxX <= options.minX || options.maxY <= options.minY)
        {
            throw std::invalid_argument("OpenCL height grid: explicit bounds must be finite and non-empty");
        }
        min_x = options.minX;
        max_x = options.maxX;
        min_y = options.minY;
        max_y = options.maxY;
    }
    Scalar span_x = std::max(max_x - min_x, Scalar(1.0e-6));
    Scalar span_y = std::max(max_y - min_y, Scalar(1.0e-6));
    if (!options.useExplicitBounds)
    {
        min_x -= span_x * options.padding;
        max_x += span_x * options.padding;
        min_y -= span_y * options.padding;
        max_y += span_y * options.padding;
    }
    span_x = std::max(max_x - min_x, Scalar(1.0e-6));
    span_y = std::max(max_y - min_y, Scalar(1.0e-6));
    grid.width = options.width > 0 ? options.width : options.resolution;
    grid.height = options.height > 0 ? options.height : options.resolution;
    if (grid.width < 2 || grid.height < 2)
    {
        throw std::invalid_argument("OpenCL height grid: grid dimensions must be at least 2");
    }
    const auto width = static_cast<std::size_t>(grid.width);
    const auto height = static_cast<std::size_t>(grid.height);
    if (width > std::numeric_limits<std::size_t>::max() / height)
    {
        throw std::overflow_error("OpenCL height grid: cell count overflow");
    }
    const std::size_t cell_count = width * height;
    if (cell_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("OpenCL height grid: cell count exceeds int range");
    }
    grid.minX = min_x;
    grid.minY = min_y;
    grid.stepX = span_x / Scalar(grid.width - 1);
    grid.stepY = span_y / Scalar(grid.height - 1);
    grid.heights.assign(cell_count, Scalar(0));
    grid.weights.assign(cell_count, Scalar(0));
    grid.valid.assign(cell_count, 0);
    grid.fillPass.assign(cell_count, 0);
    if (cloud.hasColors()) grid.colors.assign(cell_count * 3u, 0);
    return grid;
}

template <typename Scalar>
void populateExactColors(
    const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
    const std::vector<Contribution<Scalar>>& contributions,
    mesh::HeightGrid<Scalar>& grid)
{
    if (!cloud.hasColors()) return;
    for (std::size_t begin = 0; begin < contributions.size();)
    {
        std::size_t end = begin + 1u;
        while (end < contributions.size() && contributions[end].cell == contributions[begin].cell) ++end;
        std::array<long double, 3> sums{0, 0, 0};
        long double total_weight = 0;
        for (std::size_t cursor = begin; cursor < end; ++cursor)
        {
            const long double weight = static_cast<long double>(contributions[cursor].weight);
            total_weight += weight;
            for (int channel = 0; channel < 3; ++channel)
            {
                sums[channel] += static_cast<long double>(cloud.colors()->getValue(
                    contributions[cursor].point, channel)) * weight;
            }
        }
        const std::size_t output = static_cast<std::size_t>(contributions[begin].cell) * 3u;
        if (!grid.valid[static_cast<std::size_t>(contributions[begin].cell)])
        {
            begin = end;
            continue;
        }
        for (int channel = 0; channel < 3; ++channel)
        {
            grid.colors[output + static_cast<std::size_t>(channel)] =
                mesh::detail::colorByteFromWeightedSum(sums[channel], total_weight);
        }
        begin = end;
    }
}

template <typename Scalar>
mesh::HeightGrid<Scalar> buildHeightGridImpl(
    const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options)
{
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("OpenCL height grid: point count exceeds int range");
    }
    mesh::HeightGrid<Scalar> grid = initializeGridGeometry(cloud, options);
    if (grid.width == 0 || grid.height == 0)
    {
        return grid;
    }
    const auto contributions = buildContributions(cloud, options, grid);
    if (contributions.empty())
    {
        return grid;
    }
    if (contributions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("OpenCL height grid: contribution count exceeds int range");
    }

    std::vector<Scalar> point_z(cloud.size());
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        point_z[index] = cloud.points().getValue(static_cast<plamatrix::Index>(index), 2);
    }

    std::vector<int> source_indices;
    std::vector<Scalar> contribution_weights;
    std::vector<int> cell_ids;
    std::vector<int> offsets;
    std::vector<int> counts;
    source_indices.reserve(contributions.size());
    contribution_weights.reserve(contributions.size());
    for (std::size_t begin = 0; begin < contributions.size();)
    {
        std::size_t end = begin + 1;
        while (end < contributions.size() && contributions[end].cell == contributions[begin].cell)
        {
            ++end;
        }
        if (end - begin > static_cast<std::size_t>(detail::maximumSequentialReductionItems))
        {
            throw std::runtime_error(
                "OpenCL height grid rejected a cell that exceeds the bounded serial reduction size");
        }
        cell_ids.push_back(contributions[begin].cell);
        offsets.push_back(static_cast<int>(begin));
        counts.push_back(static_cast<int>(end - begin));
        for (std::size_t cursor = begin; cursor < end; ++cursor)
        {
            source_indices.push_back(contributions[cursor].point);
            contribution_weights.push_back(contributions[cursor].weight);
        }
        begin = end;
    }

    std::fill(grid.heights.begin(), grid.heights.end(), Scalar(0));
    std::fill(grid.weights.begin(), grid.weights.end(), Scalar(0));
    std::fill(grid.valid.begin(), grid.valid.end(), std::uint8_t(0));
    std::fill(grid.colors.begin(), grid.colors.end(), std::uint8_t(0));
    auto& runtime = detail::OpenClRuntime::instance();
    detail::requireFp64<Scalar>(runtime);
    detail::CommandQueue queue(runtime.createQueue());
    auto z_buffer = detail::inputVector(runtime, point_z);
    std::vector<std::uint8_t> color_placeholder(1, 0);
    auto color_buffer = detail::inputVector(runtime, color_placeholder);
    auto source_buffer = detail::inputVector(runtime, source_indices);
    auto contribution_buffer = detail::inputVector(runtime, contribution_weights);
    auto cell_buffer = detail::inputVector(runtime, cell_ids);
    auto offset_buffer = detail::inputVector(runtime, offsets);
    auto count_buffer = detail::inputVector(runtime, counts);
    auto height_buffer = detail::inOutVector(runtime, grid.heights);
    auto weight_buffer = detail::inOutVector(runtime, grid.weights);
    auto valid_buffer = detail::inOutVector(runtime, grid.valid);
    std::vector<std::uint8_t> output_color_placeholder(1, 0);
    auto output_color_buffer = detail::inOutVector(runtime, output_color_placeholder);

    const std::string key = std::string("height_grid_")
        + (std::is_same_v<Scalar, double> ? "f64" : "f32");
    const cl_program program = runtime.program(
        key, detail::heightGridKernelSource, detail::realBuildOptions<Scalar>());
    detail::CompiledKernel kernel(program, "aggregateHeightCells");
    detail::kernelBufferArg(kernel, 0, z_buffer);
    detail::kernelBufferArg(kernel, 1, color_buffer);
    detail::kernelBufferArg(kernel, 2, source_buffer);
    detail::kernelBufferArg(kernel, 3, contribution_buffer);
    detail::kernelBufferArg(kernel, 4, cell_buffer);
    detail::kernelBufferArg(kernel, 5, offset_buffer);
    detail::kernelBufferArg(kernel, 6, count_buffer);
    const int occupied_count = static_cast<int>(cell_ids.size());
    detail::kernelArg(kernel, 7, occupied_count);
    const int aggregation = aggregationCode(options.elevationAggregation);
    detail::kernelArg(kernel, 8, aggregation);
    const int has_colors = 0;
    detail::kernelArg(kernel, 9, has_colors);
    detail::kernelBufferArg(kernel, 10, height_buffer);
    detail::kernelBufferArg(kernel, 11, weight_buffer);
    detail::kernelBufferArg(kernel, 12, valid_buffer);
    detail::kernelBufferArg(kernel, 13, output_color_buffer);
    const std::size_t global_size = cell_ids.size();
    detail::checkOpenCl(
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(aggregateHeightCells)");
    detail::readVector(queue, height_buffer, grid.heights);
    detail::readVector(queue, weight_buffer, grid.weights);
    detail::readVector(queue, valid_buffer, grid.valid);
    height_grid_execution_count.fetch_add(1, std::memory_order_relaxed);
    if (cloud.hasColors())
    {
        populateExactColors(cloud, contributions, grid);
    }
    return grid;
}

} // namespace

std::uint64_t heightGridOpenClExecutionCount() noexcept
{
    return height_grid_execution_count.load(std::memory_order_relaxed);
}

mesh::HeightGrid<float> buildHeightGrid(
    const PointCloud<float, plamatrix::Device::CPU>& cloud,
    const mesh::HeightGridOptions<float>& options)
{
    return buildHeightGridImpl(cloud, options);
}

mesh::HeightGrid<double> buildHeightGrid(
    const PointCloud<double, plamatrix::Device::CPU>& cloud,
    const mesh::HeightGridOptions<double>& options)
{
    return buildHeightGridImpl(cloud, options);
}

} // namespace opencl
} // namespace plapoint
