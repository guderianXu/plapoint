#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

namespace plapoint
{
namespace gpu
{

/// Small host-side summary of GPU ICP correspondences and centered moment sums.
template <typename Scalar>
struct IcpCorrespondenceStats
{
    int active_count = 0;
    int invalid_source_count = 0;
    double src_centroid[3]{};
    double tgt_centroid[3]{};
    double cross_covariance[9]{};
    double src_covariance[9]{};
    double tgt_covariance[9]{};
    double residual_sq_sum = 0.0;
};

/// Compute nearest-neighbor ICP correspondences for PlaMatrix column-major Nx3 device point arrays.
/// The returned stats are copied to host after synchronizing the supplied stream.
IcpCorrespondenceStats<float> computeIcpCorrespondenceStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream = 0);

IcpCorrespondenceStats<double> computeIcpCorrespondenceStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
