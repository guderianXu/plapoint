#pragma once

#include <cstdint>
#include <vector>

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

/// Compute the StatisticalOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, int mean_k, float stddev_mul);

/// Compute the StatisticalOutlierRemoval keep-mask for column-major GPU point storage.
std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, int mean_k, double stddev_mul);

} // namespace gpu
} // namespace plapoint
