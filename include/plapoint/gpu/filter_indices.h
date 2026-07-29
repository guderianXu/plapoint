#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plamatrix/ops/reduction.h>

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
    friend plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>
    radiusOutlierRemovalKeepMaskDevice(
        const PointCloud<OtherScalar, plamatrix::Device::GPU>&,
        OtherScalar,
        int,
        OutlierRemovalGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    template <typename OtherScalar>
    friend plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>
    statisticalOutlierRemovalKeepMaskDevice(
        const PointCloud<OtherScalar, plamatrix::Device::GPU>&,
        int,
        OtherScalar,
        OutlierRemovalGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    void ensureBuffers(plamatrix::Index rows)
    {
        if (_meanDistances.rows() == rows)
        {
            return;
        }
        _meanDistances = plamatrix::DenseMatrix<double, plamatrix::Device::GPU>
            ::uninitialized(rows, 1);
        _finiteWeights = plamatrix::DenseMatrix<double, plamatrix::Device::GPU>
            ::uninitialized(rows, 1);
        _invalidDistances = plamatrix::DenseMatrix<double, plamatrix::Device::GPU>
            ::uninitialized(rows, 1);
        _squaredDeviations = plamatrix::DenseMatrix<double, plamatrix::Device::GPU>
            ::uninitialized(rows, 1);
    }

    GpuSpatialIndex<Scalar> _index;
    GpuSpatialQueryWorkspace<Scalar> _queryWorkspace;
    plamatrix::DenseMatrix<double, plamatrix::Device::GPU> _meanDistances;
    plamatrix::DenseMatrix<double, plamatrix::Device::GPU> _finiteWeights;
    plamatrix::DenseMatrix<double, plamatrix::Device::GPU> _invalidDistances;
    plamatrix::DenseMatrix<double, plamatrix::Device::GPU> _squaredDeviations;
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
    const plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>& keep_mask);

#ifdef PLAPOINT_WITH_CUDA
/// Stably compact removed source indices on the device and download only those indices.
std::vector<int> removedIndicesFromKeepMaskDevice(
    const plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>& keep_mask,
    cudaStream_t stream = nullptr);

/// Indexed RadiusOutlierRemoval mask. The returned ordinary allocation is independent of workspace.
template <typename Scalar>
plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>
radiusOutlierRemovalKeepMaskDevice(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    Scalar radius,
    int min_neighbors,
    OutlierRemovalGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);

/// Indexed StatisticalOutlierRemoval mask. The returned ordinary allocation is independent of workspace.
template <typename Scalar>
plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>
statisticalOutlierRemovalKeepMaskDevice(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
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
plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU> radiusOutlierRemovalKeepMaskDevice(
    const plamatrix::DenseMatrix<float, plamatrix::Device::GPU>& points,
    float radius,
    int min_neighbors);

/// Compute the RadiusOutlierRemoval keep-mask into PlaMatrix GPU storage.
plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU> radiusOutlierRemovalKeepMaskDevice(
    const plamatrix::DenseMatrix<double, plamatrix::Device::GPU>& points,
    double radius,
    int min_neighbors);

/// Compute the StatisticalOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, int mean_k, float stddev_mul);

/// Compute the StatisticalOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, int mean_k, double stddev_mul);

/// Compute the StatisticalOutlierRemoval keep-mask into PlaMatrix GPU storage.
plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU> statisticalOutlierRemovalKeepMaskDevice(
    const plamatrix::DenseMatrix<float, plamatrix::Device::GPU>& points,
    int mean_k,
    float stddev_mul);

/// Compute the StatisticalOutlierRemoval keep-mask into PlaMatrix GPU storage.
plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU> statisticalOutlierRemovalKeepMaskDevice(
    const plamatrix::DenseMatrix<double, plamatrix::Device::GPU>& points,
    int mean_k,
    double stddev_mul);

} // namespace gpu
} // namespace plapoint
