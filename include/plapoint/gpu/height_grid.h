#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/height_grid.h>

namespace plapoint
{
namespace gpu
{

struct HeightGridGpuWorkspaceAccess;
struct GpuHeightGridAccess;

template <typename Scalar>
struct GpuHeightGrid
{
    GpuHeightGrid() = default;
    GpuHeightGrid(const GpuHeightGrid&) = delete;
    GpuHeightGrid& operator=(const GpuHeightGrid&) = delete;

    GpuHeightGrid(GpuHeightGrid&& other) noexcept
    {
        *this = std::move(other);
    }

    GpuHeightGrid& operator=(GpuHeightGrid&& other) noexcept
    {
        if (this != &other)
        {
            width = other.width;
            height = other.height;
            minX = other.minX;
            minY = other.minY;
            stepX = other.stepX;
            stepY = other.stepY;
            heights = std::move(other.heights);
            weights = std::move(other.weights);
            valid = std::move(other.valid);
            colors = std::move(other.colors);
            fillPass = std::move(other.fillPass);
            _status = std::move(other._status);
            _pendingStream = other._pendingStream;
            _hasPendingWork = other._hasPendingWork;
            other.width = 0;
            other.height = 0;
            other._pendingStream = nullptr;
            other._hasPendingWork = false;
        }
        return *this;
    }

    int width = 0;
    int height = 0;
    Scalar minX = Scalar(0);
    Scalar minY = Scalar(0);
    Scalar stepX = Scalar(1);
    Scalar stepY = Scalar(1);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> heights;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> weights;
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU> valid;
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU> colors;
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::GPU> fillPass;

    std::size_t cellCount() const noexcept
    {
        return width > 0 && height > 0
            ? static_cast<std::size_t>(width) * static_cast<std::size_t>(height)
            : std::size_t{0};
    }

    bool hasColors() const noexcept
    {
        return cellCount() != 0 &&
            colors.rows() == static_cast<plamatrix::Index>(cellCount()) && colors.cols() == 3;
    }

    void synchronize(cudaStream_t stream = nullptr) const
    {
        if (_hasPendingWork && _pendingStream != stream)
        {
            throw std::logic_error(
                "GpuHeightGrid::synchronize must use the stream that produced the grid");
        }
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
        _pendingStream = nullptr;
        _hasPendingWork = false;
    }

private:
    friend struct GpuHeightGridAccess;
    plamatrix::DenseMatrix<int, plamatrix::Device::GPU> _status;
    mutable cudaStream_t _pendingStream = nullptr;
    mutable bool _hasPendingWork = false;
};

template <typename Scalar>
class HeightGridGpuWorkspace
{
public:
    HeightGridGpuWorkspace() = default;
    HeightGridGpuWorkspace(const HeightGridGpuWorkspace&) = delete;
    HeightGridGpuWorkspace& operator=(const HeightGridGpuWorkspace&) = delete;
    HeightGridGpuWorkspace(HeightGridGpuWorkspace&& other) noexcept
    {
        *this = std::move(other);
    }

    HeightGridGpuWorkspace& operator=(HeightGridGpuWorkspace&& other) noexcept
    {
        if (this != &other)
        {
            _cellCapacity = other._cellCapacity;
            _stream = other._stream;
            _hasStream = other._hasStream;
            _colorSums = std::move(other._colorSums);
            _colorWeights = std::move(other._colorWeights);
            _nextHeights = std::move(other._nextHeights);
            _nextValid = std::move(other._nextValid);
            _nextColors = std::move(other._nextColors);
            _nextFillPass = std::move(other._nextFillPass);
            other._cellCapacity = 0;
            other._stream = nullptr;
            other._hasStream = false;
        }
        return *this;
    }

    std::size_t cellCapacity() const noexcept { return _cellCapacity; }

    void resetStream(cudaStream_t stream = nullptr)
    {
        if (_hasStream)
        {
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(_stream));
        }
        _stream = stream;
        _hasStream = true;
    }

private:
    friend struct HeightGridGpuWorkspaceAccess;
    template <typename OtherScalar>
    friend GpuHeightGrid<OtherScalar> buildHeightGridDeviceAsync(
        const PointCloud<OtherScalar, plamatrix::Device::GPU>&,
        const mesh::HeightGridOptions<OtherScalar>&,
        HeightGridGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    template <typename OtherScalar>
    friend void fillHolesAsync(
        GpuHeightGrid<OtherScalar>&,
        int,
        int,
        int,
        HeightGridGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    void bindStream(cudaStream_t stream)
    {
        if (_hasStream && _stream != stream)
        {
            throw std::logic_error(
                "HeightGridGpuWorkspace cannot be reused on another stream before resetStream()");
        }
        _stream = stream;
        _hasStream = true;
    }

    void reserve(std::size_t cell_count)
    {
        if (cell_count <= _cellCapacity)
        {
            return;
        }

        DeviceBuffer<Scalar> color_sums(cell_count * 3);
        DeviceBuffer<Scalar> color_weights(cell_count);
        DeviceBuffer<Scalar> next_heights(cell_count);
        DeviceBuffer<std::uint8_t> next_valid(cell_count);
        DeviceBuffer<std::uint8_t> next_colors(cell_count * 3);
        DeviceBuffer<std::uint16_t> next_fill_pass(cell_count);

        _colorSums = std::move(color_sums);
        _colorWeights = std::move(color_weights);
        _nextHeights = std::move(next_heights);
        _nextValid = std::move(next_valid);
        _nextColors = std::move(next_colors);
        _nextFillPass = std::move(next_fill_pass);
        _cellCapacity = cell_count;
    }

    std::size_t _cellCapacity = 0;
    cudaStream_t _stream = nullptr;
    bool _hasStream = false;
    DeviceBuffer<Scalar> _colorSums;
    DeviceBuffer<Scalar> _colorWeights;
    DeviceBuffer<Scalar> _nextHeights;
    DeviceBuffer<std::uint8_t> _nextValid;
    DeviceBuffer<std::uint8_t> _nextColors;
    DeviceBuffer<std::uint16_t> _nextFillPass;
};

/// Enqueue device aggregation using explicit bounds. Strict non-finite input errors are reported
/// by downloadHeightGrid(). Use buildHeightGrid() when bounds must be computed automatically.
template <typename Scalar>
GpuHeightGrid<Scalar> buildHeightGridDeviceAsync(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void fillHolesAsync(
    GpuHeightGrid<Scalar>& grid,
    int max_passes,
    int min_neighbors,
    int search_radius,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
mesh::HeightGrid<Scalar> downloadHeightGrid(
    const GpuHeightGrid<Scalar>& grid,
    cudaStream_t stream = nullptr);

/// Build a CPU-readable height grid from a GPU-resident point cloud.
/// Automatic bounds are resolved synchronously before device aggregation.
mesh::HeightGrid<float> buildHeightGrid(
    const PointCloud<float, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<float>& options = mesh::HeightGridOptions<float>{},
    cudaStream_t stream = 0);

/// Build a CPU-readable height grid from a GPU-resident point cloud.
/// Automatic bounds are resolved synchronously before device aggregation.
mesh::HeightGrid<double> buildHeightGrid(
    const PointCloud<double, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<double>& options = mesh::HeightGridOptions<double>{},
    cudaStream_t stream = 0);

/// Fill a CPU-owned height grid using the CPU implementation.
void fillHoles(mesh::HeightGrid<float>& grid, int max_passes = 8);

/// Fill a CPU-owned height grid using the CPU implementation.
void fillHoles(mesh::HeightGrid<double>& grid, int max_passes = 8);

/// Convert a height grid to a CPU mesh while preserving CPU-supported source attributes.
/// The current mesh implementation preserves colors by transferring the GPU source cloud back to CPU.
PointCloud<float, plamatrix::Device::CPU> heightGridToMesh(
    const mesh::HeightGrid<float>& grid,
    const PointCloud<float, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options = mesh::HeightGridOptions<float>{});

/// Convert a height grid to a CPU mesh while preserving CPU-supported source attributes.
/// The current mesh implementation preserves colors by transferring the GPU source cloud back to CPU.
PointCloud<double, plamatrix::Device::CPU> heightGridToMesh(
    const mesh::HeightGrid<double>& grid,
    const PointCloud<double, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options = mesh::HeightGridOptions<double>{});

/// Convenience path: build and fill a device grid, then download once and emit a CPU mesh.
PointCloud<float, plamatrix::Device::CPU> heightGridToMesh(
    const PointCloud<float, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options = mesh::HeightGridOptions<float>{},
    int fill_passes = 8,
    cudaStream_t stream = 0);

/// Convenience path: build and fill a device grid, then download once and emit a CPU mesh.
PointCloud<double, plamatrix::Device::CPU> heightGridToMesh(
    const PointCloud<double, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options = mesh::HeightGridOptions<double>{},
    int fill_passes = 8,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
