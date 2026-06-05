#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

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

struct IcpStepTransformInput
{
    double src_centroid[3];
    double tgt_centroid[3];
    double cross_covariance[9];
};

struct IcpStepTransformRawResult
{
    double delta;
    int valid;
};

constexpr int kIcpStatsBlockSize = 128;

#ifdef PLAPOINT_ENABLE_TESTING
std::atomic<int> g_icp_correspondence_stats_call_count{0};
std::atomic<std::uintptr_t> g_icp_first_stats_source_pointer{0};
std::atomic<int> g_icp_step_transform_input_copy_count{0};
std::atomic<int> g_icp_host_synchronization_count{0};
__device__ unsigned long long g_icp_full_distance_evaluation_count;
__device__ unsigned long long g_icp_target_candidate_visit_count;
#endif

int icpStatsPartialCount(int source_count)
{
    return (source_count + kIcpStatsBlockSize - 1) / kIcpStatsBlockSize;
}

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
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const bool can_prune_by_radius = isfinite(max_dist) && max_dist >= 0.0;
    __shared__ double target_tile_x[kIcpStatsBlockSize];
    __shared__ double target_tile_y[kIcpStatsBlockSize];
    __shared__ double target_tile_z[kIcpStatsBlockSize];
    __shared__ double target_tile_min_x[kIcpStatsBlockSize];
    __shared__ double target_tile_min_y[kIcpStatsBlockSize];
    __shared__ double target_tile_min_z[kIcpStatsBlockSize];
    __shared__ double target_tile_max_x[kIcpStatsBlockSize];
    __shared__ double target_tile_max_y[kIcpStatsBlockSize];
    __shared__ double target_tile_max_z[kIcpStatsBlockSize];
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
        if (can_prune_by_radius)
        {
            target_tile_min_x[local_idx] = target_valid ? tx : INFINITY;
            target_tile_min_y[local_idx] = target_valid ? ty : INFINITY;
            target_tile_min_z[local_idx] = target_valid ? tz : INFINITY;
            target_tile_max_x[local_idx] = target_valid ? tx : -INFINITY;
            target_tile_max_y[local_idx] = target_valid ? ty : -INFINITY;
            target_tile_max_z[local_idx] = target_valid ? tz : -INFINITY;
        }
        __syncthreads();

        if (can_prune_by_radius)
        {
            for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
            {
                if (local_idx < stride)
                {
                    target_tile_min_x[local_idx] =
                        fmin(target_tile_min_x[local_idx], target_tile_min_x[local_idx + stride]);
                    target_tile_min_y[local_idx] =
                        fmin(target_tile_min_y[local_idx], target_tile_min_y[local_idx + stride]);
                    target_tile_min_z[local_idx] =
                        fmin(target_tile_min_z[local_idx], target_tile_min_z[local_idx + stride]);
                    target_tile_max_x[local_idx] =
                        fmax(target_tile_max_x[local_idx], target_tile_max_x[local_idx + stride]);
                    target_tile_max_y[local_idx] =
                        fmax(target_tile_max_y[local_idx], target_tile_max_y[local_idx + stride]);
                    target_tile_max_z[local_idx] =
                        fmax(target_tile_max_z[local_idx], target_tile_max_z[local_idx + stride]);
                }
                __syncthreads();
            }
        }

        bool tile_relevant = true;
        if (source_valid && can_prune_by_radius)
        {
            tile_relevant =
                isfinite(target_tile_min_x[0]) &&
                sx >= target_tile_min_x[0] - max_dist && sx <= target_tile_max_x[0] + max_dist &&
                sy >= target_tile_min_y[0] - max_dist && sy <= target_tile_max_y[0] + max_dist &&
                sz >= target_tile_min_z[0] - max_dist && sz <= target_tile_max_z[0] + max_dist;
        }

        if (source_valid && tile_relevant)
        {
            const int tile_count = min(kIcpStatsBlockSize, target_count - tile_start);
            for (int tile_offset = 0; tile_offset < tile_count; ++tile_offset)
            {
#ifdef PLAPOINT_ENABLE_TESTING
                atomicAdd(&g_icp_target_candidate_visit_count, 1ull);
#endif
                if (!target_tile_valid[tile_offset])
                {
                    continue;
                }

                const double dx = sx - target_tile_x[tile_offset];
                if (can_prune_by_radius && fabs(dx) > max_dist)
                {
                    continue;
                }
                const double dy = sy - target_tile_y[tile_offset];
                if (can_prune_by_radius && fabs(dy) > max_dist)
                {
                    continue;
                }
                const double dz = sz - target_tile_z[tile_offset];
                if (can_prune_by_radius && fabs(dz) > max_dist)
                {
                    continue;
                }
#ifdef PLAPOINT_ENABLE_TESTING
                atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
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
        if (accepted && isfinite(max_dist))
        {
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
__global__ void setIdentityTransform4x4Kernel(Scalar* transform)
{
    const int idx = threadIdx.x;
    if (idx >= 16)
    {
        return;
    }

    const int row = idx & 3;
    const int col = idx >> 2;
    transform[idx] = row == col ? Scalar(1) : Scalar(0);
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
__global__ void transformPointsColumnMajorKernel(
    const Scalar* transform,
    const Scalar* points,
    int point_count,
    Scalar* output_points)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    const Scalar px = points[idx];
    const Scalar py = points[point_count + idx];
    const Scalar pz = points[2 * point_count + idx];

    output_points[idx] = transform[0] * px + transform[4] * py + transform[8] * pz + transform[12];
    output_points[point_count + idx] = transform[1] * px + transform[5] * py + transform[9] * pz + transform[13];
    output_points[2 * point_count + idx] = transform[2] * px + transform[6] * py + transform[10] * pz + transform[14];
}

__device__ void jacobiRotate4x4(double A[16], double V[16], int p, int q)
{
    const double apq = A[p * 4 + q];
    const double app = A[p * 4 + p];
    const double aqq = A[q * 4 + q];
    if (fabs(apq) <= 1.0e-15 * (fabs(app) + fabs(aqq) + 1.0))
    {
        return;
    }

    const double tau = (aqq - app) / (2.0 * apq);
    const double tau_sign = tau >= 0.0 ? 1.0 : -1.0;
    const double t = tau_sign / (fabs(tau) + sqrt(1.0 + tau * tau));
    const double c = 1.0 / sqrt(1.0 + t * t);
    const double s = t * c;

    for (int k = 0; k < 4; ++k)
    {
        if (k == p || k == q)
        {
            continue;
        }
        const double akp = A[k * 4 + p];
        const double akq = A[k * 4 + q];
        const double new_akp = c * akp - s * akq;
        const double new_akq = s * akp + c * akq;
        A[k * 4 + p] = new_akp;
        A[p * 4 + k] = new_akp;
        A[k * 4 + q] = new_akq;
        A[q * 4 + k] = new_akq;
    }

    A[p * 4 + p] = app - t * apq;
    A[q * 4 + q] = aqq + t * apq;
    A[p * 4 + q] = 0.0;
    A[q * 4 + p] = 0.0;

    for (int k = 0; k < 4; ++k)
    {
        const double vkp = V[k * 4 + p];
        const double vkq = V[k * 4 + q];
        V[k * 4 + p] = c * vkp - s * vkq;
        V[k * 4 + q] = s * vkp + c * vkq;
    }
}

__device__ void largestEigenvectorSymmetric4x4(const double A_in[16], double eigenvector[4])
{
    double A[16];
    double V[16];
    for (int idx = 0; idx < 16; ++idx)
    {
        A[idx] = A_in[idx];
        V[idx] = 0.0;
    }
    for (int i = 0; i < 4; ++i)
    {
        V[i * 4 + i] = 1.0;
    }

    for (int sweep = 0; sweep < 32; ++sweep)
    {
        jacobiRotate4x4(A, V, 0, 1);
        jacobiRotate4x4(A, V, 0, 2);
        jacobiRotate4x4(A, V, 0, 3);
        jacobiRotate4x4(A, V, 1, 2);
        jacobiRotate4x4(A, V, 1, 3);
        jacobiRotate4x4(A, V, 2, 3);
    }

    int best = 0;
    for (int i = 1; i < 4; ++i)
    {
        if (A[i * 4 + i] > A[best * 4 + best])
        {
            best = i;
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        eigenvector[i] = V[i * 4 + best];
    }
}

template <typename Scalar>
__device__ bool scalarRepresentable(double value)
{
    if (!isfinite(value))
    {
        return false;
    }
    if constexpr (std::is_same_v<Scalar, float>)
    {
        return fabs(value) <= static_cast<double>(FLT_MAX);
    }
    else
    {
        return true;
    }
}

template <typename Scalar>
__device__ Scalar checkedDeviceScalar(double value, int& valid)
{
    if (!scalarRepresentable<Scalar>(value))
    {
        valid = 0;
    }
    return static_cast<Scalar>(value);
}

template <typename Scalar>
__device__ void computeStepTransformFromInput(
    const IcpStepTransformInput& input,
    Scalar* step_transform,
    IcpStepTransformRawResult* result)
{
    const double* h = input.cross_covariance;
    double N[16];
    N[0] = h[0] + h[4] + h[8];
    N[1] = h[5] - h[7];
    N[2] = h[6] - h[2];
    N[3] = h[1] - h[3];

    N[4] = h[5] - h[7];
    N[5] = h[0] - h[4] - h[8];
    N[6] = h[1] + h[3];
    N[7] = h[6] + h[2];

    N[8] = h[6] - h[2];
    N[9] = h[1] + h[3];
    N[10] = -h[0] + h[4] - h[8];
    N[11] = h[5] + h[7];

    N[12] = h[1] - h[3];
    N[13] = h[6] + h[2];
    N[14] = h[5] + h[7];
    N[15] = -h[0] - h[4] + h[8];

    double q[4];
    largestEigenvectorSymmetric4x4(N, q);
    if (q[0] < 0.0)
    {
        for (double& value : q)
        {
            value = -value;
        }
    }

    const double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    int valid = norm > 0.0 && isfinite(norm) ? 1 : 0;
    const double inv_norm = valid ? 1.0 / norm : 1.0;
    const double w = q[0] * inv_norm;
    const double x = q[1] * inv_norm;
    const double y = q[2] * inv_norm;
    const double z = q[3] * inv_norm;

    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - w * z);
    const double r02 = 2.0 * (x * z + w * y);
    const double r10 = 2.0 * (x * y + w * z);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - w * x);
    const double r20 = 2.0 * (x * z - w * y);
    const double r21 = 2.0 * (y * z + w * x);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);

    const double tx = input.tgt_centroid[0] -
        (r00 * input.src_centroid[0] + r01 * input.src_centroid[1] + r02 * input.src_centroid[2]);
    const double ty = input.tgt_centroid[1] -
        (r10 * input.src_centroid[0] + r11 * input.src_centroid[1] + r12 * input.src_centroid[2]);
    const double tz = input.tgt_centroid[2] -
        (r20 * input.src_centroid[0] + r21 * input.src_centroid[1] + r22 * input.src_centroid[2]);

    step_transform[0] = checkedDeviceScalar<Scalar>(r00, valid);
    step_transform[1] = checkedDeviceScalar<Scalar>(r10, valid);
    step_transform[2] = checkedDeviceScalar<Scalar>(r20, valid);
    step_transform[3] = Scalar(0);
    step_transform[4] = checkedDeviceScalar<Scalar>(r01, valid);
    step_transform[5] = checkedDeviceScalar<Scalar>(r11, valid);
    step_transform[6] = checkedDeviceScalar<Scalar>(r21, valid);
    step_transform[7] = Scalar(0);
    step_transform[8] = checkedDeviceScalar<Scalar>(r02, valid);
    step_transform[9] = checkedDeviceScalar<Scalar>(r12, valid);
    step_transform[10] = checkedDeviceScalar<Scalar>(r22, valid);
    step_transform[11] = Scalar(0);
    step_transform[12] = checkedDeviceScalar<Scalar>(tx, valid);
    step_transform[13] = checkedDeviceScalar<Scalar>(ty, valid);
    step_transform[14] = checkedDeviceScalar<Scalar>(tz, valid);
    step_transform[15] = Scalar(1);

    result->delta = fabs(r00 - 1.0) + fabs(r11 - 1.0) + fabs(r22 - 1.0)
                  + fabs(r01) + fabs(r02) + fabs(r10)
                  + fabs(r12) + fabs(r20) + fabs(r21)
                  + fabs(tx) + fabs(ty) + fabs(tz);
    result->valid = valid;
}

template <typename Scalar>
__global__ void computeStepTransformFromStatsKernel(
    const IcpStepTransformInput* input,
    Scalar* step_transform,
    IcpStepTransformRawResult* result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    computeStepTransformFromInput<Scalar>(*input, step_transform, result);
}

template <typename Scalar>
__global__ void computeStepTransformFromRawStatsKernel(
    const RawIcpStats* raw_stats,
    Scalar* step_transform,
    IcpStepTransformRawResult* result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    if (raw_stats->active_count <= 0)
    {
        result->delta = 0.0;
        result->valid = 0;
        return;
    }

    IcpStepTransformInput input{};
    const RawIcpStats raw = *raw_stats;
    const double inv_count = 1.0 / static_cast<double>(raw.active_count);
    for (int c = 0; c < 3; ++c)
    {
        input.src_centroid[c] = raw.src_sum[c] * inv_count;
        input.tgt_centroid[c] = raw.tgt_sum[c] * inv_count;
    }
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            const int idx = r * 3 + c;
            input.cross_covariance[idx] = raw.cross_sum[idx] -
                raw.src_sum[r] * raw.tgt_sum[c] * inv_count;
        }
    }
    computeStepTransformFromInput<Scalar>(input, step_transform, result);
}

bool covarianceHasNonCollinearGeometry(const double covariance[9]);

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
    stats.src_has_non_collinear_geometry = covarianceHasNonCollinearGeometry(stats.src_covariance);
    stats.tgt_has_non_collinear_geometry = covarianceHasNonCollinearGeometry(stats.tgt_covariance);
    return stats;
}

bool covarianceHasNonCollinearGeometry(const double covariance[9])
{
    const double c00 = covariance[0];
    const double c01 = covariance[1];
    const double c02 = covariance[2];
    const double c11 = covariance[4];
    const double c12 = covariance[5];
    const double c22 = covariance[8];
    const double trace = c00 + c11 + c22;
    if (!std::isfinite(trace) || trace <= 0.0)
    {
        return false;
    }

    const double principal_minor_sum = c00 * c11 + c00 * c22 + c11 * c22 -
        (c01 * c01 + c02 * c02 + c12 * c12);
    if (!std::isfinite(principal_minor_sum))
    {
        return false;
    }

    const double threshold = std::max(trace * trace * 1.0e-12, trace * 1.0e-30);
    return principal_minor_sum > threshold;
}

template <typename Scalar>
IcpCorrespondenceStats<Scalar> computeIcpCorrespondenceStatsColumnMajorImpl(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    int* d_correspondence_indices,
    IcpCorrespondenceStatsWorkspace* workspace,
    cudaStream_t stream)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_correspondence_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    std::uintptr_t empty = 0;
    g_icp_first_stats_source_pointer.compare_exchange_strong(
        empty,
        reinterpret_cast<std::uintptr_t>(d_source_points),
        std::memory_order_relaxed,
        std::memory_order_relaxed);
#endif

    if (source_count <= 0 || target_count <= 0)
    {
        return {};
    }
    if (!d_source_points || !d_target_points)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }

    IcpCorrespondenceStatsWorkspace local_workspace;
    IcpCorrespondenceStatsWorkspace& active_workspace = workspace ? *workspace : local_workspace;
    active_workspace.reserve(source_count);

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpStats*>(active_workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpStats*>(active_workspace.statsStorage());

    collectCorrespondenceStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        grid_size,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    RawIcpStats raw{};
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats, sizeof(RawIcpStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostStats<Scalar>(raw);
}

template <typename Scalar>
void setIdentityTransform4x4Impl(Scalar* d_transform, cudaStream_t stream)
{
    if (!d_transform)
    {
        throw std::invalid_argument("ICP GPU: transform pointer must not be null");
    }

    setIdentityTransform4x4Kernel<Scalar><<<1, 16, 0, stream>>>(d_transform);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
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
}

template <typename Scalar>
void transformPointsColumnMajorImpl(
    const Scalar* d_transform,
    const Scalar* d_points,
    int point_count,
    Scalar* d_output_points,
    cudaStream_t stream)
{
    if (point_count < 0)
    {
        throw std::invalid_argument("ICP GPU: point count must not be negative");
    }
    if (point_count == 0)
    {
        return;
    }
    if (!d_transform || !d_points || !d_output_points)
    {
        throw std::invalid_argument("ICP GPU: transform point pointers must not be null");
    }

    constexpr int block_size = 256;
    const int grid_size = (point_count + block_size - 1) / block_size;
    transformPointsColumnMajorKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_transform,
        d_points,
        point_count,
        d_output_points);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void validateStepTransformStatsInput(const IcpCorrespondenceStats<Scalar>& stats)
{
    if (stats.active_count <= 0)
    {
        throw std::invalid_argument("ICP GPU: step stats must contain correspondences");
    }

    const double max_scalar = static_cast<double>(std::numeric_limits<Scalar>::max());
    for (int c = 0; c < 3; ++c)
    {
        if (!std::isfinite(stats.src_centroid[c]) || std::abs(stats.src_centroid[c]) > max_scalar ||
            !std::isfinite(stats.tgt_centroid[c]) || std::abs(stats.tgt_centroid[c]) > max_scalar)
        {
            throw std::runtime_error("ICP: correspondence centroid is not representable");
        }
    }
    for (int idx = 0; idx < 9; ++idx)
    {
        if (!std::isfinite(stats.cross_covariance[idx]) ||
            std::abs(stats.cross_covariance[idx]) > max_scalar)
        {
            throw std::runtime_error("ICP: cross-covariance is not representable");
        }
    }
}

template <typename Scalar>
IcpStepTransformResult<Scalar> computeIcpStepTransformFromStatsImpl(
    const IcpCorrespondenceStats<Scalar>& stats,
    Scalar* d_step_transform,
    IcpStepTransformWorkspace* workspace,
    cudaStream_t stream)
{
    if (!d_step_transform)
    {
        throw std::invalid_argument("ICP GPU: step transform pointer must not be null");
    }
    validateStepTransformStatsInput(stats);

    IcpStepTransformWorkspace local_workspace;
    IcpStepTransformWorkspace& active_workspace = workspace ? *workspace : local_workspace;
    active_workspace.reserve();

    IcpStepTransformInput input{};
    for (int c = 0; c < 3; ++c)
    {
        input.src_centroid[c] = stats.src_centroid[c];
        input.tgt_centroid[c] = stats.tgt_centroid[c];
    }
    for (int idx = 0; idx < 9; ++idx)
    {
        input.cross_covariance[idx] = stats.cross_covariance[idx];
    }

    auto* d_input = reinterpret_cast<IcpStepTransformInput*>(active_workspace.inputStorage());
    auto* d_result = reinterpret_cast<IcpStepTransformRawResult*>(active_workspace.resultStorage());
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_step_transform_input_copy_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(d_input, &input, sizeof(IcpStepTransformInput),
                                        cudaMemcpyHostToDevice, stream));
    computeStepTransformFromStatsKernel<Scalar><<<1, 1, 0, stream>>>(
        d_input,
        d_step_transform,
        d_result);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    IcpStepTransformRawResult raw_result{};
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw_result, d_result, sizeof(IcpStepTransformRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (!raw_result.valid)
    {
        throw std::runtime_error("ICP: transform step is not representable");
    }

    IcpStepTransformResult<Scalar> result;
    result.delta = static_cast<Scalar>(raw_result.delta);
    return result;
}

template <typename Scalar>
IcpStepTransformResult<Scalar> computeIcpStepTransformFromDeviceStatsImpl(
    const IcpCorrespondenceStats<Scalar>& stats,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    Scalar* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream)
{
    if (!d_step_transform)
    {
        throw std::invalid_argument("ICP GPU: step transform pointer must not be null");
    }
    if (!stats_workspace.statsStorage())
    {
        throw std::invalid_argument("ICP GPU: stats workspace must contain reduced stats");
    }
    validateStepTransformStatsInput(stats);

    step_workspace.reserve();

    const auto* d_stats = reinterpret_cast<const RawIcpStats*>(stats_workspace.statsStorage());
    auto* d_result = reinterpret_cast<IcpStepTransformRawResult*>(step_workspace.resultStorage());
    computeStepTransformFromRawStatsKernel<Scalar><<<1, 1, 0, stream>>>(
        d_stats,
        d_step_transform,
        d_result);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    IcpStepTransformRawResult raw_result{};
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw_result, d_result, sizeof(IcpStepTransformRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (!raw_result.valid)
    {
        throw std::runtime_error("ICP: transform step is not representable");
    }

    IcpStepTransformResult<Scalar> result;
    result.delta = static_cast<Scalar>(raw_result.delta);
    return result;
}

template <typename Scalar>
IcpStatsAndStepTransformResult<Scalar> computeIcpStatsAndStepTransformColumnMajorImpl(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    Scalar* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_correspondence_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    std::uintptr_t empty = 0;
    g_icp_first_stats_source_pointer.compare_exchange_strong(
        empty,
        reinterpret_cast<std::uintptr_t>(d_source_points),
        std::memory_order_relaxed,
        std::memory_order_relaxed);
#endif

    if (source_count <= 0 || target_count <= 0)
    {
        return {};
    }
    if (!d_source_points || !d_target_points || !d_step_transform)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }

    stats_workspace.reserve(source_count);
    step_workspace.reserve();

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpStats*>(stats_workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpStats*>(stats_workspace.statsStorage());
    auto* d_result = reinterpret_cast<IcpStepTransformRawResult*>(step_workspace.resultStorage());

    collectCorrespondenceStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        nullptr,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        grid_size,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    computeStepTransformFromRawStatsKernel<Scalar><<<1, 1, 0, stream>>>(
        d_stats,
        d_step_transform,
        d_result);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    RawIcpStats raw{};
    IcpStepTransformRawResult raw_result{};
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats, sizeof(RawIcpStats),
                                        cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw_result, d_result, sizeof(IcpStepTransformRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

    IcpStatsAndStepTransformResult<Scalar> result;
    result.stats = makeHostStats<Scalar>(raw);
    result.step.delta = static_cast<Scalar>(raw_result.delta);
    result.step_valid = raw_result.valid != 0;
    return result;
}

} // namespace

#ifdef PLAPOINT_ENABLE_TESTING
void resetIcpCorrespondenceStatsCallCountForTesting()
{
    g_icp_correspondence_stats_call_count.store(0, std::memory_order_relaxed);
}

int icpCorrespondenceStatsCallCountForTesting()
{
    return g_icp_correspondence_stats_call_count.load(std::memory_order_relaxed);
}

void resetIcpFirstStatsSourcePointerForTesting()
{
    g_icp_first_stats_source_pointer.store(0, std::memory_order_relaxed);
}

const void* icpFirstStatsSourcePointerForTesting()
{
    return reinterpret_cast<const void*>(
        g_icp_first_stats_source_pointer.load(std::memory_order_relaxed));
}

void resetIcpFullDistanceEvaluationCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_full_distance_evaluation_count, &zero, sizeof(zero)));
}

unsigned long long icpFullDistanceEvaluationCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_full_distance_evaluation_count, sizeof(count)));
    return count;
}

void resetIcpTargetCandidateVisitCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_target_candidate_visit_count, &zero, sizeof(zero)));
}

unsigned long long icpTargetCandidateVisitCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_target_candidate_visit_count, sizeof(count)));
    return count;
}

void resetIcpStepTransformInputCopyCountForTesting()
{
    g_icp_step_transform_input_copy_count.store(0, std::memory_order_relaxed);
}

int icpStepTransformInputCopyCountForTesting()
{
    return g_icp_step_transform_input_copy_count.load(std::memory_order_relaxed);
}

void resetIcpHostSynchronizationCountForTesting()
{
    g_icp_host_synchronization_count.store(0, std::memory_order_relaxed);
}

int icpHostSynchronizationCountForTesting()
{
    return g_icp_host_synchronization_count.load(std::memory_order_relaxed);
}
#endif

void IcpCorrespondenceStatsWorkspace::reserve(int source_count)
{
    if (source_count < 0)
    {
        throw std::invalid_argument("ICP GPU: source point count must not be negative");
    }
    if (source_count == 0)
    {
        return;
    }

    const int required_partials = icpStatsPartialCount(source_count);
    if (partialCapacity() < required_partials)
    {
        _partial_storage.allocate(static_cast<std::size_t>(required_partials) * sizeof(RawIcpStats));
        _partial_capacity = required_partials;
    }
    if (_stats_storage.size() < sizeof(RawIcpStats))
    {
        _stats_storage.allocate(sizeof(RawIcpStats));
    }
}

void IcpStepTransformWorkspace::reserve()
{
    if (_input_storage.size() < sizeof(IcpStepTransformInput))
    {
        _input_storage.allocate(sizeof(IcpStepTransformInput));
    }
    if (_result_storage.size() < sizeof(IcpStepTransformRawResult))
    {
        _result_storage.allocate(sizeof(IcpStepTransformRawResult));
    }
}

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
        nullptr,
        stream);
}

IcpCorrespondenceStats<float> computeIcpCorrespondenceStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    int* d_correspondence_indices,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    return computeIcpCorrespondenceStatsColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices,
        &workspace,
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
        nullptr,
        stream);
}

IcpCorrespondenceStats<double> computeIcpCorrespondenceStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    int* d_correspondence_indices,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    return computeIcpCorrespondenceStatsColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices,
        &workspace,
        stream);
}

IcpStepTransformResult<float> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<float>& stats,
    float* d_step_transform,
    cudaStream_t stream)
{
    return computeIcpStepTransformFromStatsImpl(stats, d_step_transform, nullptr, stream);
}

IcpStepTransformResult<float> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<float>& stats,
    float* d_step_transform,
    IcpStepTransformWorkspace& workspace,
    cudaStream_t stream)
{
    return computeIcpStepTransformFromStatsImpl(stats, d_step_transform, &workspace, stream);
}

IcpStepTransformResult<double> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<double>& stats,
    double* d_step_transform,
    cudaStream_t stream)
{
    return computeIcpStepTransformFromStatsImpl(stats, d_step_transform, nullptr, stream);
}

IcpStepTransformResult<double> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<double>& stats,
    double* d_step_transform,
    IcpStepTransformWorkspace& workspace,
    cudaStream_t stream)
{
    return computeIcpStepTransformFromStatsImpl(stats, d_step_transform, &workspace, stream);
}

IcpStepTransformResult<float> computeIcpStepTransformFromDeviceStats(
    const IcpCorrespondenceStats<float>& stats,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream)
{
    return computeIcpStepTransformFromDeviceStatsImpl(
        stats,
        stats_workspace,
        d_step_transform,
        step_workspace,
        stream);
}

IcpStepTransformResult<double> computeIcpStepTransformFromDeviceStats(
    const IcpCorrespondenceStats<double>& stats,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream)
{
    return computeIcpStepTransformFromDeviceStatsImpl(
        stats,
        stats_workspace,
        d_step_transform,
        step_workspace,
        stream);
}

IcpStatsAndStepTransformResult<float> computeIcpStatsAndStepTransformColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream)
{
    return computeIcpStatsAndStepTransformColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        step_workspace,
        stream);
}

IcpStatsAndStepTransformResult<double> computeIcpStatsAndStepTransformColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream)
{
    return computeIcpStatsAndStepTransformColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        step_workspace,
        stream);
}

void multiplyTransform4x4(
    const float* d_A,
    const float* d_B,
    float* d_C,
    cudaStream_t stream)
{
    multiplyTransform4x4Impl(d_A, d_B, d_C, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void multiplyTransform4x4(
    const double* d_A,
    const double* d_B,
    double* d_C,
    cudaStream_t stream)
{
    multiplyTransform4x4Impl(d_A, d_B, d_C, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void multiplyTransform4x4Async(
    const float* d_A,
    const float* d_B,
    float* d_C,
    cudaStream_t stream)
{
    multiplyTransform4x4Impl(d_A, d_B, d_C, stream);
}

void multiplyTransform4x4Async(
    const double* d_A,
    const double* d_B,
    double* d_C,
    cudaStream_t stream)
{
    multiplyTransform4x4Impl(d_A, d_B, d_C, stream);
}

void setIdentityTransform4x4(float* d_transform, cudaStream_t stream)
{
    setIdentityTransform4x4Impl(d_transform, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void setIdentityTransform4x4(double* d_transform, cudaStream_t stream)
{
    setIdentityTransform4x4Impl(d_transform, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void setIdentityTransform4x4Async(float* d_transform, cudaStream_t stream)
{
    setIdentityTransform4x4Impl(d_transform, stream);
}

void setIdentityTransform4x4Async(double* d_transform, cudaStream_t stream)
{
    setIdentityTransform4x4Impl(d_transform, stream);
}

void transformPointsColumnMajor(
    const float* d_transform,
    const float* d_points,
    int point_count,
    float* d_output_points,
    cudaStream_t stream)
{
    transformPointsColumnMajorImpl(d_transform, d_points, point_count, d_output_points, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void transformPointsColumnMajor(
    const double* d_transform,
    const double* d_points,
    int point_count,
    double* d_output_points,
    cudaStream_t stream)
{
    transformPointsColumnMajorImpl(d_transform, d_points, point_count, d_output_points, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void transformPointsColumnMajorAsync(
    const float* d_transform,
    const float* d_points,
    int point_count,
    float* d_output_points,
    cudaStream_t stream)
{
    transformPointsColumnMajorImpl(d_transform, d_points, point_count, d_output_points, stream);
}

void transformPointsColumnMajorAsync(
    const double* d_transform,
    const double* d_points,
    int point_count,
    double* d_output_points,
    cudaStream_t stream)
{
    transformPointsColumnMajorImpl(d_transform, d_points, point_count, d_output_points, stream);
}

} // namespace gpu
} // namespace plapoint
