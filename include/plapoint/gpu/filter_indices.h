#pragma once

#include <cstdint>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>

namespace plapoint {
namespace gpu {

/// Convert a byte keep-mask to source indices that should be preserved.
std::vector<int> keptIndicesFromKeepMask(const std::vector<std::uint8_t>& keep_mask);

/// Convert a byte keep-mask to source indices that should be reported as removed.
std::vector<int> removedIndicesFromKeepMask(const std::vector<std::uint8_t>& keep_mask);

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
