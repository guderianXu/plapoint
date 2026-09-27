#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <plamatrix/internal/ops/statistics.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/ops/reduction.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/height_grid.h>
#include <plapoint/filters/detail/geometry_bridge.h>

namespace plapoint
{
namespace gpu
{

struct HeightGridGpuWorkspaceAccess
{
    template <typename Scalar>
    static void prepare(
        HeightGridGpuWorkspace<Scalar>& workspace,
        std::size_t cell_count,
        cudaStream_t stream)
    {
        workspace.bindStream(stream);
        workspace.reserve(cell_count);
    }

    template <typename Scalar>
    static Scalar* colorSums(HeightGridGpuWorkspace<Scalar>& workspace)
    {
        return workspace._colorSums.get();
    }

    template <typename Scalar>
    static Scalar* colorWeights(HeightGridGpuWorkspace<Scalar>& workspace)
    {
        return workspace._colorWeights.get();
    }

    template <typename Scalar>
    static Scalar* nextHeights(HeightGridGpuWorkspace<Scalar>& workspace)
    {
        return workspace._nextHeights.get();
    }

    template <typename Scalar>
    static std::uint8_t* nextValid(HeightGridGpuWorkspace<Scalar>& workspace)
    {
        return workspace._nextValid.get();
    }

    template <typename Scalar>
    static std::uint8_t* nextColors(HeightGridGpuWorkspace<Scalar>& workspace)
    {
        return workspace._nextColors.get();
    }

    template <typename Scalar>
    static std::uint16_t* nextFillPass(HeightGridGpuWorkspace<Scalar>& workspace)
    {
        return workspace._nextFillPass.get();
    }
};

struct GpuHeightGridAccess
{
    template <typename Scalar>
    static void checkStream(
        const GpuHeightGrid<Scalar>& grid,
        const char* operation,
        cudaStream_t stream)
    {
        if (grid._hasPendingWork && grid._pendingStream != stream)
        {
            throw std::logic_error(
                std::string(operation) +
                " must use the stream with pending GpuHeightGrid work");
        }
    }

    template <typename Scalar>
    static void markPending(GpuHeightGrid<Scalar>& grid, cudaStream_t stream) noexcept
    {
        grid._pendingStream = stream;
        grid._hasPendingWork = true;
    }

    template <typename Scalar>
    static int* status(GpuHeightGrid<Scalar>& grid) noexcept
    {
        return grid._status ? grid._status->data() : nullptr;
    }

    template <typename Scalar>
    static const int* status(const GpuHeightGrid<Scalar>& grid) noexcept
    {
        return grid._status ? grid._status->data() : nullptr;
    }

    template <typename Scalar>
    static void allocateStatus(GpuHeightGrid<Scalar>& grid)
    {
        grid._status.emplace(1, 1, grid.heights->contextOwner());
    }
};

namespace
{

#include "height_grid_gpu_kernels.cuh"

constexpr int kBlockSize = 256;

template <typename Scalar>
struct HeightGridBounds
{
    Scalar minX;
    Scalar maxX;
    Scalar minY;
    Scalar maxY;
    int finiteCount;
};

int checkedBlockCount(int count)
{
    return count <= 0 ? 0 : (count - 1) / kBlockSize + 1;
}

template <typename Scalar>
int aggregationCode(mesh::ElevationAggregation aggregation)
{
    static_cast<void>(sizeof(Scalar));
    switch (aggregation)
    {
    case mesh::ElevationAggregation::Mean:
        return 0;
    case mesh::ElevationAggregation::Min:
        return 1;
    case mesh::ElevationAggregation::Max:
        return 2;
    }
    throw std::invalid_argument("buildHeightGrid GPU: unsupported elevation aggregation");
}

std::size_t checkedCellCount(int width, int height)
{
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (h != 0 && w > std::numeric_limits<std::size_t>::max() / h)
    {
        throw std::overflow_error("buildHeightGrid GPU: grid cell count overflow");
    }
    const std::size_t count = w * h;
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("buildHeightGrid GPU supports at most INT_MAX grid cells");
    }
    return count;
}

template <typename Scalar>
HeightGridBounds<Scalar> computeFiniteBounds(
    const plamatrix::internal::ResidentMatrix<Scalar>& points,
    cudaStream_t stream)
{
    plamatrix::internal::ResidentMatrix<Scalar> minimum(1, points.cols(), points.context());
    plamatrix::internal::ResidentMatrix<Scalar> maximum(1, points.cols(), points.context());
    plamatrix::internal::ResidentMatrix<plamatrix::Index> valid_count(1, 1, points.context());
    plamatrix::internal::ReductionWorkspace workspace;
    plamatrix::internal::finiteColumnBoundsAsync(
        points.template view<plamatrix::internal::Device::GPU>(),
        minimum.template view<plamatrix::internal::Device::GPU>(),
        maximum.template view<plamatrix::internal::Device::GPU>(),
        valid_count.template view<plamatrix::internal::Device::GPU>(), workspace, stream);

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace.closeAsyncAllocation();
    const auto host_minimum = minimum.toHostMatrix();
    const auto host_maximum = maximum.toHostMatrix();
    const auto host_valid_count = valid_count.toHostMatrix();

    const auto finite_count = host_valid_count(0, 0);
    if (finite_count < 0 || finite_count > points.rows())
    {
        throw std::runtime_error(
            "buildHeightGrid GPU: PlaMatrix returned an invalid finite point count");
    }
    return {
        host_minimum(0, 0),
        host_maximum(0, 0),
        host_minimum(0, 1),
        host_maximum(0, 1),
        static_cast<int>(finite_count)};
}

template <typename Scalar>
void validateDeviceGridShape(const GpuHeightGrid<Scalar>& grid, const char* operation)
{
    if (grid.width == 0 && grid.height == 0)
    {
        if (grid.heights || grid.weights || grid.valid || grid.colors || grid.fillPass ||
            GpuHeightGridAccess::status(grid) != nullptr)
        {
            throw std::invalid_argument(std::string(operation) +
                                        ": empty metadata has non-empty device storage");
        }
        return;
    }
    if (grid.width <= 0 || grid.height <= 0)
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": width and height must both be positive");
    }
    const std::size_t count = checkedCellCount(grid.width, grid.height);
    const auto rows = static_cast<plamatrix::Index>(count);
    const auto is_column = [rows](const auto& matrix) {
        return matrix && matrix->rows() == rows && matrix->cols() == 1;
    };
    if (!is_column(grid.heights) || !is_column(grid.weights) ||
        !is_column(grid.valid) || !is_column(grid.fillPass))
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": device fields must contain one value per grid cell");
    }
    if (GpuHeightGridAccess::status(grid) == nullptr)
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": device status storage is missing");
    }
    if (grid.colors && !grid.hasColors())
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": colors must have cell_count rows and three columns");
    }
}

template <typename Value>
void copyDeviceToHostAsync(
    Value* host,
    const Value* device,
    std::size_t count,
    cudaStream_t stream)
{
    if (count != 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            host, device, count * sizeof(Value), cudaMemcpyDeviceToHost, stream));
    }
}

template <typename Scalar>
GpuHeightGrid<Scalar> buildHeightGridDeviceImpl(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    GpuHeightGrid<Scalar> grid;
    if (cloud.size() == 0)
    {
        return grid;
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("buildHeightGrid GPU supports at most INT_MAX points");
    }
    if (!std::isfinite(options.padding) || options.padding < Scalar(0))
    {
        throw std::invalid_argument("buildHeightGrid GPU: padding must be finite and non-negative");
    }
    if (!options.useExplicitBounds)
    {
        throw std::invalid_argument(
            "buildHeightGridDeviceAsync requires explicit bounds; use buildHeightGrid "
            "for synchronous automatic bounds");
    }
    if (!std::isfinite(options.minX) || !std::isfinite(options.maxX) ||
        !std::isfinite(options.minY) || !std::isfinite(options.maxY) ||
        options.maxX <= options.minX || options.maxY <= options.minY)
    {
        throw std::invalid_argument(
            "buildHeightGrid GPU: explicit bounds must be finite and non-empty");
    }

    const int point_count = static_cast<int>(cloud.size());
    const Scalar min_x = options.minX;
    const Scalar min_y = options.minY;
    const Scalar span_x = std::max(options.maxX - options.minX, Scalar(1.0e-6));
    const Scalar span_y = std::max(options.maxY - options.minY, Scalar(1.0e-6));

    const int width = options.width > 0 ? options.width : options.resolution;
    const int height = options.height > 0 ? options.height : options.resolution;
    if (width < 2 || height < 2)
    {
        throw std::invalid_argument("buildHeightGrid GPU: grid dimensions must be at least 2");
    }
    const std::size_t cell_count = checkedCellCount(width, height);
    const auto cell_rows = static_cast<plamatrix::Index>(cell_count);
    const bool has_colors = cloud.hasColors();
    const int aggregation = aggregationCode<Scalar>(options.elevationAggregation);

    HeightGridGpuWorkspaceAccess::prepare(workspace, cell_count, stream);
    grid.width = width;
    grid.height = height;
    grid.minX = min_x;
    grid.minY = min_y;
    grid.stepX = span_x / Scalar(width - 1);
    grid.stepY = span_y / Scalar(height - 1);
    grid.heights.emplace(cell_rows, 1, cloud.executionContext());
    grid.weights.emplace(cell_rows, 1, cloud.executionContext());
    grid.valid.emplace(cell_rows, 1, cloud.executionContext());
    grid.fillPass.emplace(cell_rows, 1, cloud.executionContext());
    GpuHeightGridAccess::allocateStatus(grid);
    if (has_colors)
    {
        grid.colors.emplace(cell_rows, 3, cloud.executionContext());
    }

    const int cell_count_int = static_cast<int>(cell_count);
    const int cell_blocks = checkedBlockCount(cell_count_int);
    const int point_blocks = checkedBlockCount(point_count);
    PLAPOINT_CHECK_CUDA(cudaMemsetAsync(
        GpuHeightGridAccess::status(grid), 0, sizeof(int), stream));
    GpuHeightGridAccess::markPending(grid, stream);
    if (!options.skipNonFinite)
    {
        validateHeightGridPointsKernel<<<point_blocks, kBlockSize, 0, stream>>>(
            cloud.points().data(), point_count, GpuHeightGridAccess::status(grid));
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }
    const Scalar extreme = std::numeric_limits<Scalar>::max();
    initializeHeightGridKernel<<<cell_blocks, kBlockSize, 0, stream>>>(
        cell_count_int,
        aggregation,
        has_colors,
        extreme,
        grid.heights->data(),
        grid.weights->data(),
        grid.valid->data(),
        grid.fillPass->data(),
        HeightGridGpuWorkspaceAccess::colorSums(workspace),
        HeightGridGpuWorkspaceAccess::colorWeights(workspace),
        has_colors ? grid.colors->data() : nullptr);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    splatHeightGridKernel<<<point_blocks, kBlockSize, 0, stream>>>(
        cloud.points().data(),
        has_colors ? cloud.colors()->data() : nullptr,
        point_count,
        width,
        height,
        has_colors,
        options.skipNonFinite,
        options.useBilinearSplat,
        aggregation,
        grid.minX,
        grid.minY,
        grid.stepX,
        grid.stepY,
        grid.heights->data(),
        grid.weights->data(),
        HeightGridGpuWorkspaceAccess::colorSums(workspace),
        HeightGridGpuWorkspaceAccess::colorWeights(workspace));
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    normalizeHeightGridKernel<<<cell_blocks, kBlockSize, 0, stream>>>(
        cell_count_int,
        aggregation,
        has_colors,
        grid.heights->data(),
        grid.weights->data(),
        HeightGridGpuWorkspaceAccess::colorSums(workspace),
        HeightGridGpuWorkspaceAccess::colorWeights(workspace),
        grid.valid->data(),
        has_colors ? grid.colors->data() : nullptr);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    GpuHeightGridAccess::markPending(grid, stream);
    return grid;
}

template <typename Scalar>
void fillHolesDeviceImpl(
    GpuHeightGrid<Scalar>& grid,
    int max_passes,
    int min_neighbors,
    int search_radius,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    validateDeviceGridShape(grid, "fillHolesAsync");
    if (grid.width == 0 || max_passes <= 0)
    {
        return;
    }
    GpuHeightGridAccess::checkStream(grid, "fillHolesAsync", stream);
    if (max_passes > static_cast<int>(std::numeric_limits<std::uint16_t>::max()))
    {
        throw std::overflow_error("fillHolesAsync: max_passes exceeds fillPass range");
    }
    min_neighbors = std::max(1, min_neighbors);
    search_radius = std::max(1, search_radius);
    search_radius = std::min(search_radius, std::max(grid.width, grid.height));

    const std::size_t cell_count = grid.cellCount();
    const int cell_count_int = static_cast<int>(cell_count);
    const int blocks = checkedBlockCount(cell_count_int);
    const bool has_colors = grid.hasColors();
    HeightGridGpuWorkspaceAccess::prepare(workspace, cell_count, stream);

    const Scalar* input_heights = grid.heights->data();
    const std::uint8_t* input_valid = grid.valid->data();
    const std::uint8_t* input_colors = has_colors ? grid.colors->data() : nullptr;
    const std::uint16_t* input_fill_pass = grid.fillPass->data();
    for (int pass = 0; pass < max_passes; ++pass)
    {
        const bool write_workspace = (pass % 2) == 0;
        Scalar* output_heights = write_workspace
            ? HeightGridGpuWorkspaceAccess::nextHeights(workspace)
            : grid.heights->data();
        std::uint8_t* output_valid = write_workspace
            ? HeightGridGpuWorkspaceAccess::nextValid(workspace)
            : grid.valid->data();
        std::uint8_t* output_colors = has_colors
            ? (write_workspace ? HeightGridGpuWorkspaceAccess::nextColors(workspace)
                               : grid.colors->data())
            : nullptr;
        std::uint16_t* output_fill_pass = write_workspace
            ? HeightGridGpuWorkspaceAccess::nextFillPass(workspace)
            : grid.fillPass->data();
        fillHeightGridHolesKernel<<<blocks, kBlockSize, 0, stream>>>(
            grid.width,
            grid.height,
            pass,
            min_neighbors,
            search_radius,
            has_colors,
            input_heights,
            input_valid,
            input_colors,
            input_fill_pass,
            output_heights,
            output_valid,
            output_colors,
            output_fill_pass);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
        input_heights = output_heights;
        input_valid = output_valid;
        input_colors = output_colors;
        input_fill_pass = output_fill_pass;
    }

    if ((max_passes % 2) != 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            grid.heights->data(), HeightGridGpuWorkspaceAccess::nextHeights(workspace),
            cell_count * sizeof(Scalar), cudaMemcpyDeviceToDevice, stream));
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            grid.valid->data(), HeightGridGpuWorkspaceAccess::nextValid(workspace),
            cell_count * sizeof(std::uint8_t), cudaMemcpyDeviceToDevice, stream));
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            grid.fillPass->data(), HeightGridGpuWorkspaceAccess::nextFillPass(workspace),
            cell_count * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice, stream));
        if (has_colors)
        {
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                grid.colors->data(), HeightGridGpuWorkspaceAccess::nextColors(workspace),
                cell_count * 3 * sizeof(std::uint8_t), cudaMemcpyDeviceToDevice, stream));
        }
    }
    GpuHeightGridAccess::markPending(grid, stream);
}

template <typename Scalar>
mesh::HeightGrid<Scalar> downloadHeightGridImpl(
    const GpuHeightGrid<Scalar>& grid,
    cudaStream_t stream)
{
    validateDeviceGridShape(grid, "downloadHeightGrid");
    mesh::HeightGrid<Scalar> host;
    if (grid.width == 0)
    {
        return host;
    }
    GpuHeightGridAccess::checkStream(grid, "downloadHeightGrid", stream);
    const std::size_t cell_count = grid.cellCount();
    host.width = grid.width;
    host.height = grid.height;
    host.minX = grid.minX;
    host.minY = grid.minY;
    host.stepX = grid.stepX;
    host.stepY = grid.stepY;
    host.heights.resize(cell_count);
    host.weights.resize(cell_count);
    host.valid.resize(cell_count);
    host.fillPass.resize(cell_count);
    copyDeviceToHostAsync(host.heights.data(), grid.heights->data(), cell_count, stream);
    copyDeviceToHostAsync(host.weights.data(), grid.weights->data(), cell_count, stream);
    copyDeviceToHostAsync(host.valid.data(), grid.valid->data(), cell_count, stream);
    copyDeviceToHostAsync(host.fillPass.data(), grid.fillPass->data(), cell_count, stream);
    int input_status = 0;
    copyDeviceToHostAsync(
        &input_status, GpuHeightGridAccess::status(grid), std::size_t{1}, stream);

    std::vector<std::uint8_t> planar_colors;
    if (grid.hasColors())
    {
        planar_colors.resize(cell_count * 3);
        copyDeviceToHostAsync(
            planar_colors.data(), grid.colors->data(), planar_colors.size(), stream);
    }
    grid.synchronize(stream);
    if (input_status != 0)
    {
        throw std::invalid_argument("buildHeightGrid GPU: points must be finite");
    }
    if (!planar_colors.empty())
    {
        host.colors.resize(cell_count * 3);
        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            for (std::size_t channel = 0; channel < 3; ++channel)
            {
                host.colors[cell * 3 + channel] =
                    planar_colors[cell + channel * cell_count];
            }
        }
    }
    return host;
}

} // namespace

template <typename Scalar>
GpuHeightGrid<Scalar> buildHeightGridDeviceAsync(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    cloud.validate();
    return buildHeightGridDeviceImpl(cloud, options, workspace, stream);
}

template <typename Scalar>
void fillHolesAsync(
    GpuHeightGrid<Scalar>& grid,
    int max_passes,
    int min_neighbors,
    int search_radius,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    fillHolesDeviceImpl(
        grid, max_passes, min_neighbors, search_radius, workspace, stream);
}

template <typename Scalar>
mesh::HeightGrid<Scalar> downloadHeightGrid(
    const GpuHeightGrid<Scalar>& grid,
    cudaStream_t stream)
{
    return downloadHeightGridImpl(grid, stream);
}

template <typename Scalar>
GpuHeightGrid<Scalar> buildHeightGridDeviceResolved(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    cloud.validate();
    if (cloud.size() == 0)
    {
        return {};
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("buildHeightGrid GPU supports at most INT_MAX points");
    }
    if (!std::isfinite(options.padding) || options.padding < Scalar(0))
    {
        throw std::invalid_argument("buildHeightGrid GPU: padding must be finite and non-negative");
    }
    const int point_count = static_cast<int>(cloud.size());
    const auto bounds = computeFiniteBounds(cloud.points(), stream);
    if (!options.skipNonFinite && bounds.finiteCount != point_count)
    {
        throw std::invalid_argument("buildHeightGrid GPU: points must be finite");
    }
    if (bounds.finiteCount == 0)
    {
        return {};
    }

    auto resolved = options;
    if (!resolved.useExplicitBounds)
    {
        Scalar span_x = std::max(bounds.maxX - bounds.minX, Scalar(1.0e-6));
        Scalar span_y = std::max(bounds.maxY - bounds.minY, Scalar(1.0e-6));
        resolved.minX = bounds.minX - span_x * options.padding;
        resolved.maxX = bounds.maxX + span_x * options.padding;
        resolved.minY = bounds.minY - span_y * options.padding;
        resolved.maxY = bounds.maxY + span_y * options.padding;
        resolved.useExplicitBounds = true;
        resolved.padding = Scalar(0);
    }
    return buildHeightGridDeviceImpl(cloud, resolved, workspace, stream);
}

template <typename Scalar>
mesh::HeightGrid<Scalar> buildHeightGridCompat(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    cudaStream_t stream)
{
    HeightGridGpuWorkspace<Scalar> workspace;
    const auto device_grid = buildHeightGridDeviceResolved(cloud, options, workspace, stream);
    return downloadHeightGrid(device_grid, stream);
}

mesh::HeightGrid<float> buildHeightGrid(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& cloud,
    const mesh::HeightGridOptions<float>& options,
    cudaStream_t stream)
{
    return buildHeightGridCompat(cloud, options, stream);
}

mesh::HeightGrid<double> buildHeightGrid(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& cloud,
    const mesh::HeightGridOptions<double>& options,
    cudaStream_t stream)
{
    return buildHeightGridCompat(cloud, options, stream);
}

void fillHoles(mesh::HeightGrid<float>& grid, int max_passes)
{
    mesh::fillHoles(grid, max_passes);
}

void fillHoles(mesh::HeightGrid<double>& grid, int max_passes)
{
    mesh::fillHoles(grid, max_passes);
}

template <typename Scalar>
GeometryCloud<Scalar> heightGridToMeshImpl(
    const mesh::HeightGrid<Scalar>& grid,
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<Scalar>& options)
{
    return mesh::heightGridToMesh(grid, plapoint::detail::fromDeviceCloud(source_cloud.toCpu()), options);
}

GeometryCloud<float> heightGridToMesh(
    const mesh::HeightGrid<float>& grid,
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options)
{
    return heightGridToMeshImpl(grid, source_cloud, options);
}

GeometryCloud<double> heightGridToMesh(
    const mesh::HeightGrid<double>& grid,
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options)
{
    return heightGridToMeshImpl(grid, source_cloud, options);
}

template <typename Scalar>
GeometryCloud<Scalar> heightGridToMeshPipeline(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    int fill_passes,
    cudaStream_t stream)
{
    HeightGridGpuWorkspace<Scalar> workspace;
    auto device_grid = buildHeightGridDeviceResolved(source_cloud, options, workspace, stream);
    fillHolesAsync(
        device_grid,
        fill_passes,
        options.holeFillMinNeighbors,
        options.holeFillSearchRadius,
        workspace,
        stream);
    const auto grid = downloadHeightGrid(device_grid, stream);
    return heightGridToMeshImpl(grid, source_cloud, options);
}

GeometryCloud<float> heightGridToMesh(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options,
    int fill_passes,
    cudaStream_t stream)
{
    return heightGridToMeshPipeline(source_cloud, options, fill_passes, stream);
}

GeometryCloud<double> heightGridToMesh(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options,
    int fill_passes,
    cudaStream_t stream)
{
    return heightGridToMeshPipeline(source_cloud, options, fill_passes, stream);
}

template GpuHeightGrid<float> buildHeightGridDeviceAsync<float>(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>&,
    const mesh::HeightGridOptions<float>&,
    HeightGridGpuWorkspace<float>&,
    cudaStream_t);
template GpuHeightGrid<double> buildHeightGridDeviceAsync<double>(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>&,
    const mesh::HeightGridOptions<double>&,
    HeightGridGpuWorkspace<double>&,
    cudaStream_t);
template void fillHolesAsync<float>(
    GpuHeightGrid<float>&, int, int, int, HeightGridGpuWorkspace<float>&, cudaStream_t);
template void fillHolesAsync<double>(
    GpuHeightGrid<double>&, int, int, int, HeightGridGpuWorkspace<double>&, cudaStream_t);
template mesh::HeightGrid<float> downloadHeightGrid<float>(
    const GpuHeightGrid<float>&, cudaStream_t);
template mesh::HeightGrid<double> downloadHeightGrid<double>(
    const GpuHeightGrid<double>&, cudaStream_t);

} // namespace gpu
} // namespace plapoint
