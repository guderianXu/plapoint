#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/icp.h>

namespace plapoint
{
namespace gpu
{

namespace
{

struct RawIcpStats
{
    int active_count;
    int invalid_source_count;
    double src_sum[3];
    double tgt_sum[3];
    double cross_sum[9];
    double src_outer_sum[9];
    double tgt_outer_sum[9];
    double residual_sq_sum;
};

__device__ double atomicAddDouble(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
#endif
}

template <typename Scalar>
__device__ bool loadFiniteColumnMajorPoint(const Scalar* points, int point_count, int idx,
                                           double& x, double& y, double& z)
{
    x = static_cast<double>(points[idx]);
    y = static_cast<double>(points[point_count + idx]);
    z = static_cast<double>(points[2 * point_count + idx]);
    return isfinite(x) && isfinite(y) && isfinite(z);
}

template <typename Scalar>
__global__ void collectCorrespondenceStatsKernel(
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    int* correspondence_indices,
    RawIcpStats* stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_idx >= source_count)
    {
        return;
    }

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    if (!loadFiniteColumnMajorPoint(source_points, source_count, source_idx, sx, sy, sz))
    {
        correspondence_indices[source_idx] = -1;
        atomicAdd(&stats->invalid_source_count, 1);
        return;
    }

    int best_idx = -1;
    double best_dist_sq = INFINITY;
    for (int target_idx = 0; target_idx < target_count; ++target_idx)
    {
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        if (!loadFiniteColumnMajorPoint(target_points, target_count, target_idx, tx, ty, tz))
        {
            continue;
        }

        const double dx = sx - tx;
        const double dy = sy - ty;
        const double dz = sz - tz;
        const double dist_sq = dx * dx + dy * dy + dz * dz;
        if (isfinite(dist_sq) && dist_sq < best_dist_sq)
        {
            best_dist_sq = dist_sq;
            best_idx = target_idx;
        }
    }

    bool accepted = best_idx >= 0;
    if (accepted && isfinite(static_cast<double>(max_correspondence_distance)))
    {
        const double max_dist = static_cast<double>(max_correspondence_distance);
        accepted = best_dist_sq <= max_dist * max_dist;
    }

    if (!accepted)
    {
        correspondence_indices[source_idx] = -1;
        return;
    }

    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    if (!loadFiniteColumnMajorPoint(target_points, target_count, best_idx, tx, ty, tz))
    {
        correspondence_indices[source_idx] = -1;
        return;
    }

    correspondence_indices[source_idx] = best_idx;
    atomicAdd(&stats->active_count, 1);

    const double source_values[3]{sx, sy, sz};
    const double target_values[3]{tx, ty, tz};
    for (int r = 0; r < 3; ++r)
    {
        atomicAddDouble(&stats->src_sum[r], source_values[r]);
        atomicAddDouble(&stats->tgt_sum[r], target_values[r]);
        for (int c = 0; c < 3; ++c)
        {
            atomicAddDouble(&stats->cross_sum[r * 3 + c], source_values[r] * target_values[c]);
            atomicAddDouble(&stats->src_outer_sum[r * 3 + c], source_values[r] * source_values[c]);
            atomicAddDouble(&stats->tgt_outer_sum[r * 3 + c], target_values[r] * target_values[c]);
        }
    }
    atomicAddDouble(&stats->residual_sq_sum, best_dist_sq);
}

template <typename Scalar>
__global__ void multiplyTransform4x4Kernel(const Scalar* A, const Scalar* B, Scalar* C)
{
    const int idx = threadIdx.x;
    if (idx >= 16)
    {
        return;
    }

    const int row = idx & 3;
    const int col = idx >> 2;
    double sum = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        sum += static_cast<double>(A[row + k * 4]) * static_cast<double>(B[k + col * 4]);
    }
    C[row + col * 4] = static_cast<Scalar>(sum);
}

template <typename Scalar>
IcpCorrespondenceStats<Scalar> makeHostStats(const RawIcpStats& raw)
{
    IcpCorrespondenceStats<Scalar> stats;
    stats.active_count = raw.active_count;
    stats.invalid_source_count = raw.invalid_source_count;
    stats.residual_sq_sum = raw.residual_sq_sum;

    if (raw.active_count <= 0)
    {
        return stats;
    }

    const double inv_count = 1.0 / static_cast<double>(raw.active_count);
    for (int c = 0; c < 3; ++c)
    {
        stats.src_centroid[c] = raw.src_sum[c] * inv_count;
        stats.tgt_centroid[c] = raw.tgt_sum[c] * inv_count;
    }

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            const int idx = r * 3 + c;
            stats.cross_covariance[idx] = raw.cross_sum[idx] -
                raw.src_sum[r] * raw.tgt_sum[c] * inv_count;
            stats.src_covariance[idx] = raw.src_outer_sum[idx] -
                raw.src_sum[r] * raw.src_sum[c] * inv_count;
            stats.tgt_covariance[idx] = raw.tgt_outer_sum[idx] -
                raw.tgt_sum[r] * raw.tgt_sum[c] * inv_count;
        }
    }
    return stats;
}

template <typename Scalar>
IcpCorrespondenceStats<Scalar> computeIcpCorrespondenceStatsColumnMajorImpl(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream)
{
    if (source_count <= 0 || target_count <= 0)
    {
        return {};
    }
    if (!d_source_points || !d_target_points || !d_correspondence_indices)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }

    DeviceBuffer<RawIcpStats> d_stats(1);
    PLAPOINT_CHECK_CUDA(cudaMemsetAsync(d_stats.get(), 0, sizeof(RawIcpStats), stream));

    constexpr int block_size = 128;
    const int grid_size = (source_count + block_size - 1) / block_size;
    collectCorrespondenceStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices,
        d_stats.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    RawIcpStats raw{};
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats.get(), sizeof(RawIcpStats),
                                        cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostStats<Scalar>(raw);
}

template <typename Scalar>
void multiplyTransform4x4Impl(
    const Scalar* d_A,
    const Scalar* d_B,
    Scalar* d_C,
    cudaStream_t stream)
{
    if (!d_A || !d_B || !d_C)
    {
        throw std::invalid_argument("ICP GPU: transform pointers must not be null");
    }

    multiplyTransform4x4Kernel<Scalar><<<1, 16, 0, stream>>>(d_A, d_B, d_C);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

} // namespace

IcpCorrespondenceStats<float> computeIcpCorrespondenceStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream)
{
    return computeIcpCorrespondenceStatsColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices,
        stream);
}

IcpCorrespondenceStats<double> computeIcpCorrespondenceStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream)
{
    return computeIcpCorrespondenceStatsColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices,
        stream);
}

void multiplyTransform4x4(
    const float* d_A,
    const float* d_B,
    float* d_C,
    cudaStream_t stream)
{
    multiplyTransform4x4Impl(d_A, d_B, d_C, stream);
}

void multiplyTransform4x4(
    const double* d_A,
    const double* d_B,
    double* d_C,
    cudaStream_t stream)
{
    multiplyTransform4x4Impl(d_A, d_B, d_C, stream);
}

} // namespace gpu
} // namespace plapoint
