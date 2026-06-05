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

constexpr int kIcpStatsBlockSize = 128;

__device__ void addRawIcpStats(RawIcpStats& dst, const RawIcpStats& src)
{
    dst.active_count += src.active_count;
    dst.invalid_source_count += src.invalid_source_count;
    for (int c = 0; c < 3; ++c)
    {
        dst.src_sum[c] += src.src_sum[c];
        dst.tgt_sum[c] += src.tgt_sum[c];
    }
    for (int idx = 0; idx < 9; ++idx)
    {
        dst.cross_sum[idx] += src.cross_sum[idx];
        dst.src_outer_sum[idx] += src.src_outer_sum[idx];
        dst.tgt_outer_sum[idx] += src.tgt_outer_sum[idx];
    }
    dst.residual_sq_sum += src.residual_sq_sum;
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
    RawIcpStats* partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;

    if (source_idx < source_count)
    {
        if (!loadFiniteColumnMajorPoint(source_points, source_count, source_idx, sx, sy, sz))
        {
            if (correspondence_indices)
            {
                correspondence_indices[source_idx] = -1;
            }
            local.invalid_source_count = 1;
        }
        else
        {
            source_valid = true;
        }
    }

    int best_idx = -1;
    double best_dist_sq = INFINITY;
    double best_tx = 0.0;
    double best_ty = 0.0;
    double best_tz = 0.0;
    __shared__ double target_tile_x[kIcpStatsBlockSize];
    __shared__ double target_tile_y[kIcpStatsBlockSize];
    __shared__ double target_tile_z[kIcpStatsBlockSize];
    __shared__ int target_tile_valid[kIcpStatsBlockSize];

    for (int tile_start = 0; tile_start < target_count; tile_start += kIcpStatsBlockSize)
    {
        const int target_idx = tile_start + local_idx;
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        const bool target_valid = target_idx < target_count &&
            loadFiniteColumnMajorPoint(target_points, target_count, target_idx, tx, ty, tz);
        target_tile_x[local_idx] = tx;
        target_tile_y[local_idx] = ty;
        target_tile_z[local_idx] = tz;
        target_tile_valid[local_idx] = target_valid ? 1 : 0;
        __syncthreads();

        if (source_valid)
        {
            const int tile_count = min(kIcpStatsBlockSize, target_count - tile_start);
            for (int tile_offset = 0; tile_offset < tile_count; ++tile_offset)
            {
                if (!target_tile_valid[tile_offset])
                {
                    continue;
                }

                const double dx = sx - target_tile_x[tile_offset];
                const double dy = sy - target_tile_y[tile_offset];
                const double dz = sz - target_tile_z[tile_offset];
                const double dist_sq = dx * dx + dy * dy + dz * dz;
                if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                {
                    best_dist_sq = dist_sq;
                    best_idx = tile_start + tile_offset;
                    best_tx = target_tile_x[tile_offset];
                    best_ty = target_tile_y[tile_offset];
                    best_tz = target_tile_z[tile_offset];
                }
            }
        }
        __syncthreads();
    }

    if (source_valid)
    {
        bool accepted = best_idx >= 0;
        if (accepted && isfinite(static_cast<double>(max_correspondence_distance)))
        {
            const double max_dist = static_cast<double>(max_correspondence_distance);
            accepted = best_dist_sq <= max_dist * max_dist;
        }

        if (!accepted)
        {
            if (correspondence_indices)
            {
                correspondence_indices[source_idx] = -1;
            }
        }
        else
        {
            if (correspondence_indices)
            {
                correspondence_indices[source_idx] = best_idx;
            }
            local.active_count = 1;
            local.residual_sq_sum = best_dist_sq;

            const double source_values[3]{sx, sy, sz};
            const double target_values[3]{best_tx, best_ty, best_tz};
            for (int r = 0; r < 3; ++r)
            {
                local.src_sum[r] = source_values[r];
                local.tgt_sum[r] = target_values[r];
                for (int c = 0; c < 3; ++c)
                {
                    local.cross_sum[r * 3 + c] = source_values[r] * target_values[c];
                    local.src_outer_sum[r * 3 + c] = source_values[r] * source_values[c];
                    local.tgt_outer_sum[r * 3 + c] = target_values[r] * target_values[c];
                }
            }
        }
    }

    __shared__ RawIcpStats shared_stats[kIcpStatsBlockSize];
    shared_stats[local_idx] = local;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            addRawIcpStats(shared_stats[local_idx], shared_stats[local_idx + stride]);
        }
        __syncthreads();
    }

    if (local_idx == 0)
    {
        partial_stats[blockIdx.x] = shared_stats[0];
    }
}

__global__ void reduceRawIcpStatsKernel(
    const RawIcpStats* partial_stats,
    int partial_count,
    RawIcpStats* stats)
{
    const int local_idx = threadIdx.x;
    RawIcpStats local{};
    for (int idx = local_idx; idx < partial_count; idx += blockDim.x)
    {
        addRawIcpStats(local, partial_stats[idx]);
    }

    __shared__ RawIcpStats shared_stats[kIcpStatsBlockSize];
    shared_stats[local_idx] = local;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            addRawIcpStats(shared_stats[local_idx], shared_stats[local_idx + stride]);
        }
        __syncthreads();
    }

    if (local_idx == 0)
    {
        *stats = shared_stats[0];
    }
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
    if (!d_source_points || !d_target_points)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = (source_count + block_size - 1) / block_size;
    DeviceBuffer<RawIcpStats> d_partials(static_cast<std::size_t>(grid_size));
    DeviceBuffer<RawIcpStats> d_stats(1);

    collectCorrespondenceStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices,
        d_partials.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials.get(),
        grid_size,
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
