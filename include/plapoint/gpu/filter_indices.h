#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <plamatrix/internal/device/device_matrix.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plamatrix/internal/ops/reduction.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/spatial_index.h>
#endif

namespace plapoint {
namespace gpu {

#ifdef PLAPOINT_WITH_CUDA
enum class GpuOutlierRemovalBackend
{
    None,
    UniformGrid,
    CpuCompatibility
};

template <typename Scalar>
class OutlierRemovalGpuWorkspace
{
public:
    OutlierRemovalGpuWorkspace() = default;
    OutlierRemovalGpuWorkspace(const OutlierRemovalGpuWorkspace&) = delete;
    OutlierRemovalGpuWorkspace& operator=(const OutlierRemovalGpuWorkspace&) = delete;
    OutlierRemovalGpuWorkspace(OutlierRemovalGpuWorkspace&&) noexcept = default;
    OutlierRemovalGpuWorkspace& operator=(OutlierRemovalGpuWorkspace&&) noexcept = default;

    GpuOutlierRemovalBackend lastBackend() const noexcept { return _lastBackend; }
    std::size_t indexBuildCount() const noexcept { return _indexBuildCount; }

private:
    template <typename OtherScalar>
    friend plamatrix::internal::ResidentMatrix<std::uint8_t>
    radiusOutlierRemovalKeepMaskDevice(
        const plapoint::internal::DeviceCloud<OtherScalar, plamatrix::internal::Device::GPU>&,
        OtherScalar,
        int,
        OutlierRemovalGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    template <typename OtherScalar>
    friend plamatrix::internal::ResidentMatrix<std::uint8_t>
    statisticalOutlierRemovalKeepMaskDevice(
        const plapoint::internal::DeviceCloud<OtherScalar, plamatrix::internal::Device::GPU>&,
        int,
        OtherScalar,
        OutlierRemovalGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    void ensureBuffers(plamatrix::Index rows,
                       const std::shared_ptr<plamatrix::internal::ExecutionContext>& context)
    {
        if (_context != context)
        {
            _meanDistances.reset();
            _finiteWeights.reset();
            _invalidDistances.reset();
            _squaredDeviations.reset();
            _context = context;
        }
        if (!_meanDistances || _meanDistances->rows() != rows)
        {
            _meanDistances.emplace(rows, 1, context);
            _finiteWeights.emplace(rows, 1, context);
            _invalidDistances.emplace(rows, 1, context);
            _squaredDeviations.emplace(rows, 1, context);
        }
    }

    GpuSpatialIndex<Scalar> _index;
    GpuSpatialQueryWorkspace<Scalar> _queryWorkspace;
    std::shared_ptr<plamatrix::internal::ExecutionContext> _context;
    std::optional<plamatrix::internal::ResidentMatrix<double>> _meanDistances;
    std::optional<plamatrix::internal::ResidentMatrix<double>> _finiteWeights;
    std::optional<plamatrix::internal::ResidentMatrix<double>> _invalidDistances;
    std::optional<plamatrix::internal::ResidentMatrix<double>> _squaredDeviations;
    GpuOutlierRemovalBackend _lastBackend = GpuOutlierRemovalBackend::None;
    std::size_t _indexBuildCount = 0;
};
#endif

/// Convert a byte keep-mask to source indices that should be preserved.
std::vector<int> keptIndicesFromKeepMask(const std::vector<std::uint8_t>& keep_mask);

/// Convert a byte keep-mask to source indices that should be reported as removed.
std::vector<int> removedIndicesFromKeepMask(const std::vector<std::uint8_t>& keep_mask);

/// Download the byte keep mask for compatibility diagnostics only.
std::vector<std::uint8_t> keepMaskToHost(
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask);

#ifdef PLAPOINT_WITH_CUDA
/// Stably compact removed source indices on the device and download only those indices.
std::vector<int> removedIndicesFromKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask,
    cudaStream_t stream = nullptr);

/// Indexed RadiusOutlierRemoval mask. The returned ordinary allocation is independent of workspace.
template <typename Scalar>
plamatrix::internal::ResidentMatrix<std::uint8_t>
radiusOutlierRemovalKeepMaskDevice(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    Scalar radius,
    int min_neighbors,
    OutlierRemovalGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);

/// Indexed StatisticalOutlierRemoval mask. The returned ordinary allocation is independent of workspace.
template <typename Scalar>
plamatrix::internal::ResidentMatrix<std::uint8_t>
statisticalOutlierRemovalKeepMaskDevice(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    int mean_k,
    Scalar stddev_mul,
    OutlierRemovalGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);
#endif

/// Compute the RadiusOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> radiusOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, float radius, int min_neighbors);

/// Compute the RadiusOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> radiusOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, double radius, int min_neighbors);

/// Compute the RadiusOutlierRemoval keep-mask into PlaMatrix GPU storage.
/// The input must own a shared CUDA execution context; the result retains it.
plamatrix::internal::ResidentMatrix<std::uint8_t> radiusOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<float>& points,
    float radius,
    int min_neighbors);

/// Compute the RadiusOutlierRemoval keep-mask into PlaMatrix GPU storage.
/// The input must own a shared CUDA execution context; the result retains it.
plamatrix::internal::ResidentMatrix<std::uint8_t> radiusOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<double>& points,
    double radius,
    int min_neighbors);

/// Compute the StatisticalOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, int mean_k, float stddev_mul);

/// Compute the StatisticalOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, int mean_k, double stddev_mul);

/// Compute the StatisticalOutlierRemoval keep-mask into PlaMatrix GPU storage.
/// The input must own a shared CUDA execution context; the result retains it.
plamatrix::internal::ResidentMatrix<std::uint8_t> statisticalOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<float>& points,
    int mean_k,
    float stddev_mul);

/// Compute the StatisticalOutlierRemoval keep-mask into PlaMatrix GPU storage.
/// The input must own a shared CUDA execution context; the result retains it.
plamatrix::internal::ResidentMatrix<std::uint8_t> statisticalOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<double>& points,
    int mean_k,
    double stddev_mul);

} // namespace gpu
} // namespace plapoint
