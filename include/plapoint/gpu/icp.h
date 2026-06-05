#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <plapoint/gpu/cuda_check.h>

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

/// Reusable device storage for ICP correspondence stats reductions.
/// Reserve it once for a source size and pass it to repeated stats calls to avoid repeated allocations.
class IcpCorrespondenceStatsWorkspace
{
public:
    IcpCorrespondenceStatsWorkspace() = default;

    /// Reserve enough storage for the supplied source point count. Throws for negative counts.
    void reserve(int source_count);

    /// Return the currently reserved partial reduction capacity, in blocks.
    int partialCapacity() const { return _partial_capacity; }

    /// Return the reusable partial reduction storage pointer, or null before reserve().
    unsigned char* partialStorage() { return _partial_storage.get(); }

    /// Return the reusable final stats storage pointer, or null before reserve().
    unsigned char* statsStorage() { return _stats_storage.get(); }

private:
    DeviceBuffer<unsigned char> _partial_storage;
    DeviceBuffer<unsigned char> _stats_storage;
    int _partial_capacity = 0;
};

/// Compute nearest-neighbor ICP correspondences for PlaMatrix column-major Nx3 device point arrays.
/// The returned stats are copied to host after synchronizing the supplied stream. If
/// d_correspondence_indices is null, per-source correspondence indices are not written.
IcpCorrespondenceStats<float> computeIcpCorrespondenceStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream = 0);

/// Compute nearest-neighbor ICP stats using caller-owned reusable device workspace.
IcpCorrespondenceStats<float> computeIcpCorrespondenceStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    int* d_correspondence_indices,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

IcpCorrespondenceStats<double> computeIcpCorrespondenceStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream = 0);

/// Compute nearest-neighbor ICP stats using caller-owned reusable device workspace.
IcpCorrespondenceStats<double> computeIcpCorrespondenceStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    int* d_correspondence_indices,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Multiply two 4x4 column-major device transforms and write C = A * B.
/// Throws if any device pointer is null or the CUDA launch fails.
void multiplyTransform4x4(
    const float* d_A,
    const float* d_B,
    float* d_C,
    cudaStream_t stream = 0);

/// Multiply two 4x4 column-major device transforms and write C = A * B.
/// Throws if any device pointer is null or the CUDA launch fails.
void multiplyTransform4x4(
    const double* d_A,
    const double* d_B,
    double* d_C,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
