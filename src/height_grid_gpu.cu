#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/height_grid.h>

namespace plapoint
{
namespace gpu
{
namespace
{

template <typename Scalar>
__device__ Scalar clamp01Device(Scalar value)
{
    if (value < Scalar(0))
    {
        return Scalar(0);
    }
    if (value > Scalar(1))
    {
        return Scalar(1);
    }
    return value;
}

__device__ int clampGridIndexDevice(int value, int upper_inclusive)
{
    if (value < 0)
    {
        return 0;
    }
    if (value > upper_inclusive)
    {
        return upper_inclusive;
    }
    return value;
}

__device__ void atomicAddScalar(float* address, float value)
{
    atomicAdd(address, value);
}

__device__ void atomicAddScalar(double* address, double value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#else
    atomicAdd(address, value);
#endif
}

template <typename Scalar>
__global__ void validateHeightGridInputKernel(const Scalar* points, int point_count, int* error_code)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    const double x = static_cast<double>(points[idx]);
    const double y = static_cast<double>(points[point_count + idx]);
    const double z = static_cast<double>(points[2 * point_count + idx]);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z))
    {
        atomicCAS(error_code, 0, 1);
    }
}

template <typename Scalar>
void copyFromDevice(Scalar* h_values, const Scalar* d_values, std::size_t count, cudaStream_t stream)
{
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        h_values, d_values, count * sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
}

template <typename Scalar>
void copyScalarFromDevice(Scalar& h_value, const Scalar* d_value, cudaStream_t stream)
{
    copyFromDevice(&h_value, d_value, std::size_t{1}, stream);
}

template <typename Scalar>
void validateHeightGridInputColumnMajor(const Scalar* d_points, int point_count, cudaStream_t stream)
{
    if (!d_points)
    {
        throw std::invalid_argument("HeightGrid GPU: device points must not be null");
    }

    DeviceBuffer<int> d_error(1);
    int host_error = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        d_error.get(), &host_error, sizeof(host_error), cudaMemcpyHostToDevice, stream));

    constexpr int block_size = 256;
    const int grid_size = (point_count + block_size - 1) / block_size;
    validateHeightGridInputKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_points, point_count, d_error.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    copyScalarFromDevice(host_error, d_error.get(), stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

    if (host_error != 0)
    {
        throw std::invalid_argument("buildHeightGrid GPU: points must be finite");
    }
}

template <typename Scalar>
void computeBoundsColumnMajor(
    const Scalar* d_points, int point_count, cudaStream_t stream,
    Scalar& min_x, Scalar& max_x, Scalar& min_y, Scalar& max_y)
{
    auto policy = thrust::cuda::par.on(stream);
    auto d_x = thrust::device_pointer_cast(d_points);
    auto d_y = d_x + point_count;

    const auto minmax_x = thrust::minmax_element(policy, d_x, d_x + point_count);
    const auto minmax_y = thrust::minmax_element(policy, d_y, d_y + point_count);

    copyScalarFromDevice(min_x, thrust::raw_pointer_cast(minmax_x.first), stream);
    copyScalarFromDevice(max_x, thrust::raw_pointer_cast(minmax_x.second), stream);
    copyScalarFromDevice(min_y, thrust::raw_pointer_cast(minmax_y.first), stream);
    copyScalarFromDevice(max_y, thrust::raw_pointer_cast(minmax_y.second), stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
__global__ void splatHeightGridKernel(
    const Scalar* points,
    const std::uint8_t* colors,
    int point_count,
    int width,
    int height,
    bool has_colors,
    bool skip_non_finite,
    bool use_bilinear_splat,
    Scalar min_x,
    Scalar min_y,
    Scalar step_x,
    Scalar step_y,
    Scalar* heights,
    Scalar* weights,
    Scalar* color_sums,
    Scalar* color_weights)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    const Scalar x = points[idx];
    const Scalar y = points[point_count + idx];
    const Scalar z = points[2 * point_count + idx];
    if (!isfinite(static_cast<double>(x)) ||
        !isfinite(static_cast<double>(y)) ||
        !isfinite(static_cast<double>(z)))
    {
        if (skip_non_finite)
        {
            return;
        }
    }

    const Scalar gx = (x - min_x) / step_x;
    const Scalar gy = (y - min_y) / step_y;
    if (gx < Scalar(0) || gx > Scalar(width - 1) ||
        gy < Scalar(0) || gy > Scalar(height - 1))
    {
        return;
    }

    const auto accumulate_cell = [&](int cell, Scalar weight) {
        if (weight <= Scalar(0))
        {
            return;
        }
        atomicAddScalar(heights + cell, z * weight);
        atomicAddScalar(weights + cell, weight);
        if (has_colors)
        {
            atomicAddScalar(color_sums + cell * 3, static_cast<Scalar>(colors[idx]) * weight);
            atomicAddScalar(color_sums + cell * 3 + 1, static_cast<Scalar>(colors[point_count + idx]) * weight);
            atomicAddScalar(color_sums + cell * 3 + 2, static_cast<Scalar>(colors[2 * point_count + idx]) * weight);
            atomicAddScalar(color_weights + cell, weight);
        }
    };

    if (!use_bilinear_splat)
    {
        const int ix = clampGridIndexDevice(static_cast<int>(floor(static_cast<double>(gx) + 0.5)), width - 1);
        const int iy = clampGridIndexDevice(static_cast<int>(floor(static_cast<double>(gy) + 0.5)), height - 1);
        accumulate_cell(iy * width + ix, Scalar(1));
        return;
    }

    const int ix = clampGridIndexDevice(static_cast<int>(floor(static_cast<double>(gx))), width - 2);
    const int iy = clampGridIndexDevice(static_cast<int>(floor(static_cast<double>(gy))), height - 2);
    const Scalar tx = clamp01Device(gx - Scalar(ix));
    const Scalar ty = clamp01Device(gy - Scalar(iy));

    for (int dy = 0; dy <= 1; ++dy)
    {
        for (int dx = 0; dx <= 1; ++dx)
        {
            const Scalar wx = dx != 0 ? tx : Scalar(1) - tx;
            const Scalar wy = dy != 0 ? ty : Scalar(1) - ty;
            const Scalar w = wx * wy;
            const int cell = (iy + dy) * width + (ix + dx);
            accumulate_cell(cell, w);
        }
    }
}

template <typename Scalar>
__device__ std::uint8_t colorByteFromWeightedSumDevice(Scalar weighted_sum, Scalar weight)
{
    if (weight <= Scalar(0))
    {
        return 0;
    }
    Scalar value = weighted_sum / weight;
    if (value <= Scalar(0))
    {
        return 0;
    }
    if (value >= Scalar(255))
    {
        return 255;
    }
    return static_cast<std::uint8_t>(floor(static_cast<double>(value) + 0.5));
}

template <typename Scalar>
__global__ void normalizeHeightGridKernel(
    int cell_count,
    bool has_colors,
    Scalar* heights,
    const Scalar* weights,
    const Scalar* color_sums,
    const Scalar* color_weights,
    std::uint8_t* valid,
    std::uint8_t* grid_colors)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cell_count)
    {
        return;
    }

    if (weights[cell] > Scalar(1.0e-9))
    {
        heights[cell] /= weights[cell];
        valid[cell] = 1;
        if (has_colors && color_weights[cell] > Scalar(0))
        {
            grid_colors[cell * 3] =
                colorByteFromWeightedSumDevice(color_sums[cell * 3], color_weights[cell]);
            grid_colors[cell * 3 + 1] =
                colorByteFromWeightedSumDevice(color_sums[cell * 3 + 1], color_weights[cell]);
            grid_colors[cell * 3 + 2] =
                colorByteFromWeightedSumDevice(color_sums[cell * 3 + 2], color_weights[cell]);
        }
    }
    else
    {
        valid[cell] = 0;
    }
}

std::size_t checkedCellCount(int width, int height)
{
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (h != 0 && w > std::numeric_limits<std::size_t>::max() / h)
    {
        throw std::overflow_error("buildHeightGrid GPU: grid cell count overflow");
    }

    const std::size_t cell_count = w * h;
    if (cell_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("buildHeightGrid GPU: first GPU path supports at most INT_MAX grid cells");
    }
    return cell_count;
}

template <typename Scalar>
mesh::HeightGrid<Scalar> buildHeightGridImpl(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    cudaStream_t stream)
{
    if (options.elevationAggregation != mesh::ElevationAggregation::Mean ||
        (options.skipNonFinite && !options.useExplicitBounds))
    {
        return mesh::buildHeightGrid(cloud.toCpu(), options);
    }

    mesh::HeightGrid<Scalar> grid;
    if (cloud.size() == 0)
    {
        return grid;
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("buildHeightGrid GPU: first GPU path supports at most INT_MAX points");
    }
    if (!std::isfinite(options.padding) || options.padding < Scalar(0))
    {
        throw std::invalid_argument("buildHeightGrid GPU: padding must be finite and non-negative");
    }

    const auto point_count = static_cast<int>(cloud.size());
    const Scalar* d_points = cloud.points().data();
    if (!options.skipNonFinite)
    {
        validateHeightGridInputColumnMajor(d_points, point_count, stream);
    }

    Scalar min_x = Scalar(0);
    Scalar max_x = Scalar(0);
    Scalar min_y = Scalar(0);
    Scalar max_y = Scalar(0);
    if (options.useExplicitBounds)
    {
        if (!std::isfinite(options.minX) || !std::isfinite(options.maxX) ||
            !std::isfinite(options.minY) || !std::isfinite(options.maxY) ||
            options.maxX <= options.minX || options.maxY <= options.minY)
        {
            throw std::invalid_argument("buildHeightGrid GPU: explicit bounds must be finite and non-empty");
        }
        min_x = options.minX;
        max_x = options.maxX;
        min_y = options.minY;
        max_y = options.maxY;
    }
    else
    {
        computeBoundsColumnMajor(d_points, point_count, stream, min_x, max_x, min_y, max_y);
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

    const int requested_width = options.width > 0 ? options.width : options.resolution;
    const int requested_height = options.height > 0 ? options.height : options.resolution;
    if (requested_width < 2 || requested_height < 2)
    {
        throw std::invalid_argument("buildHeightGrid GPU: grid dimensions must be at least 2");
    }

    grid.width = requested_width;
    grid.height = requested_height;
    grid.minX = min_x;
    grid.minY = min_y;
    grid.stepX = span_x / Scalar(grid.width - 1);
    grid.stepY = span_y / Scalar(grid.height - 1);

    const std::size_t cell_count = checkedCellCount(grid.width, grid.height);
    grid.heights.assign(cell_count, Scalar(0));
    grid.weights.assign(cell_count, Scalar(0));
    grid.valid.assign(cell_count, 0);
    grid.fillPass.assign(cell_count, 0);
    const bool source_has_colors = cloud.hasColors();
    if (source_has_colors)
    {
        grid.colors.assign(cell_count * 3, 0);
    }

    DeviceBuffer<Scalar> d_heights(cell_count);
    DeviceBuffer<Scalar> d_weights(cell_count);
    DeviceBuffer<std::uint8_t> d_valid(cell_count);
    DeviceBuffer<Scalar> d_color_sums(source_has_colors ? cell_count * 3 : std::size_t{1});
    DeviceBuffer<Scalar> d_color_weights(source_has_colors ? cell_count : std::size_t{1});
    DeviceBuffer<std::uint8_t> d_grid_colors(source_has_colors ? cell_count * 3 : std::size_t{1});
    const auto value_bytes = cell_count * sizeof(Scalar);
    PLAPOINT_CHECK_CUDA(cudaMemsetAsync(d_heights.get(), 0, value_bytes, stream));
    PLAPOINT_CHECK_CUDA(cudaMemsetAsync(d_weights.get(), 0, value_bytes, stream));
    if (source_has_colors)
    {
        PLAPOINT_CHECK_CUDA(cudaMemsetAsync(d_color_sums.get(), 0, cell_count * 3 * sizeof(Scalar), stream));
        PLAPOINT_CHECK_CUDA(cudaMemsetAsync(d_color_weights.get(), 0, value_bytes, stream));
        PLAPOINT_CHECK_CUDA(cudaMemsetAsync(d_grid_colors.get(), 0, cell_count * 3 * sizeof(std::uint8_t), stream));
    }

    constexpr int block_size = 256;
    const int point_grid_size = (point_count + block_size - 1) / block_size;
    const std::uint8_t* d_colors = source_has_colors ? cloud.colors()->data() : nullptr;
    splatHeightGridKernel<Scalar><<<point_grid_size, block_size, 0, stream>>>(
        d_points, d_colors, point_count, grid.width, grid.height,
        source_has_colors, options.skipNonFinite, options.useBilinearSplat,
        grid.minX, grid.minY, grid.stepX, grid.stepY,
        d_heights.get(), d_weights.get(), d_color_sums.get(), d_color_weights.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    const int cell_count_int = static_cast<int>(cell_count);
    const int cell_grid_size = (cell_count_int + block_size - 1) / block_size;
    normalizeHeightGridKernel<Scalar><<<cell_grid_size, block_size, 0, stream>>>(
        cell_count_int, source_has_colors, d_heights.get(), d_weights.get(),
        d_color_sums.get(), d_color_weights.get(), d_valid.get(), d_grid_colors.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    copyFromDevice(grid.heights.data(), d_heights.get(), cell_count, stream);
    copyFromDevice(grid.weights.data(), d_weights.get(), cell_count, stream);
    copyFromDevice(grid.valid.data(), d_valid.get(), cell_count, stream);
    if (source_has_colors)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            grid.colors.data(), d_grid_colors.get(), cell_count * 3 * sizeof(std::uint8_t),
            cudaMemcpyDeviceToHost, stream));
    }
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return grid;
}

} // namespace

mesh::HeightGrid<float> buildHeightGrid(
    const PointCloud<float, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<float>& options,
    cudaStream_t stream)
{
    return buildHeightGridImpl<float>(cloud, options, stream);
}

mesh::HeightGrid<double> buildHeightGrid(
    const PointCloud<double, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<double>& options,
    cudaStream_t stream)
{
    return buildHeightGridImpl<double>(cloud, options, stream);
}

void fillHoles(mesh::HeightGrid<float>& grid, int max_passes)
{
    mesh::fillHoles(grid, max_passes);
}

void fillHoles(mesh::HeightGrid<double>& grid, int max_passes)
{
    mesh::fillHoles(grid, max_passes);
}

PointCloud<float, plamatrix::Device::CPU> heightGridToMesh(
    const mesh::HeightGrid<float>& grid,
    const PointCloud<float, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options)
{
    const auto cpu_cloud = source_cloud.toCpu();
    return mesh::heightGridToMesh(grid, cpu_cloud, options);
}

PointCloud<double, plamatrix::Device::CPU> heightGridToMesh(
    const mesh::HeightGrid<double>& grid,
    const PointCloud<double, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options)
{
    const auto cpu_cloud = source_cloud.toCpu();
    return mesh::heightGridToMesh(grid, cpu_cloud, options);
}

PointCloud<float, plamatrix::Device::CPU> heightGridToMesh(
    const PointCloud<float, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options,
    int fill_passes,
    cudaStream_t stream)
{
    auto grid = buildHeightGrid(source_cloud, options, stream);
    fillHoles(grid, fill_passes);
    return heightGridToMesh(grid, source_cloud, options);
}

PointCloud<double, plamatrix::Device::CPU> heightGridToMesh(
    const PointCloud<double, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options,
    int fill_passes,
    cudaStream_t stream)
{
    auto grid = buildHeightGrid(source_cloud, options, stream);
    fillHoles(grid, fill_passes);
    return heightGridToMesh(grid, source_cloud, options);
}

} // namespace gpu
} // namespace plapoint
