#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

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

struct RawIcpResidualStats
{
    int active_count;
    int invalid_source_count;
    double residual_sq_sum;
};

struct IcpTargetTileBounds
{
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;
    int has_valid_point;
};

struct IcpGridCellKey
{
    int x;
    int y;
    int z;
};

struct IcpGridCellKeyLess
{
    __host__ __device__ bool operator()(const IcpGridCellKey& lhs, const IcpGridCellKey& rhs) const
    {
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        return lhs.z < rhs.z;
    }
};

struct IcpGridCellKeyEqual
{
    __host__ __device__ bool operator()(const IcpGridCellKey& lhs, const IcpGridCellKey& rhs) const
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

struct IcpTargetSpatialGrid
{
    const IcpGridCellKey* cell_keys = nullptr;
    const int* sorted_target_indices = nullptr;
    const int* cell_starts = nullptr;
    const int* cell_counts = nullptr;
    int cell_count = 0;
    double cell_size = 0.0;
    bool active = false;
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

struct IcpAlignmentStepRawResult
{
    int active_count;
    int invalid_source_count;
    int src_has_non_collinear_geometry;
    int tgt_has_non_collinear_geometry;
    int step_valid;
    double residual_sq_sum;
    double delta;
};

constexpr int kIcpStatsBlockSize = 128;

template <typename Scalar>
__device__ void computeStepTransformFromInput(
    const IcpStepTransformInput& input,
    Scalar* step_transform,
    IcpStepTransformRawResult* result);

template <typename Scalar>
__device__ void computeStepTransformFromRawStatsValue(
    const RawIcpStats& raw,
    Scalar* step_transform,
    IcpStepTransformRawResult* result);

template <typename Scalar>
__device__ void writeAlignmentStepRawResultFromRawStats(
    const RawIcpStats& raw,
    Scalar* step_transform,
    IcpAlignmentStepRawResult* result,
    bool exact_identity_step);

#ifdef PLAPOINT_ENABLE_TESTING
std::atomic<int> g_icp_correspondence_stats_call_count{0};
std::atomic<int> g_icp_residual_stats_call_count{0};
std::atomic<std::uintptr_t> g_icp_first_stats_source_pointer{0};
std::atomic<int> g_icp_step_transform_input_copy_count{0};
std::atomic<int> g_icp_exact_pointwise_step_call_count{0};
std::atomic<int> g_icp_raw_stats_step_kernel_launch_count{0};
std::atomic<int> g_icp_alignment_step_call_count{0};
std::atomic<int> g_icp_host_synchronization_count{0};
std::atomic<int> g_icp_target_spatial_grid_build_count{0};
std::atomic<std::uintptr_t> g_icp_last_transform_output_pointer{0};
std::atomic<int> g_icp_transform_points_call_count{0};
std::atomic<int> g_icp_transform_multiply_call_count{0};
__device__ unsigned long long g_icp_full_distance_evaluation_count;
__device__ unsigned long long g_icp_target_candidate_visit_count;
__device__ unsigned long long g_icp_target_tile_bound_computation_count;
__device__ unsigned long long g_icp_grid_cell_lookup_count;
#endif

int icpStatsPartialCount(int source_count)
{
    return (source_count + kIcpStatsBlockSize - 1) / kIcpStatsBlockSize;
}

int icpTargetTileCount(int target_count)
{
    return (target_count + kIcpStatsBlockSize - 1) / kIcpStatsBlockSize;
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

__device__ void addRawIcpResidualStats(RawIcpResidualStats& dst, const RawIcpResidualStats& src)
{
    dst.active_count += src.active_count;
    dst.invalid_source_count += src.invalid_source_count;
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
__device__ void loadColumnMajorPoint(const Scalar* points, int point_count, int idx,
                                     double& x, double& y, double& z)
{
    x = static_cast<double>(points[idx]);
    y = static_cast<double>(points[point_count + idx]);
    z = static_cast<double>(points[2 * point_count + idx]);
}

__host__ __device__ int icpGridCellCoordinate(double value, double cell_size)
{
    const double coordinate = floor(value / cell_size);
    if (coordinate <= static_cast<double>(INT_MIN))
    {
        return INT_MIN;
    }
    if (coordinate >= static_cast<double>(INT_MAX))
    {
        return INT_MAX;
    }
    return static_cast<int>(coordinate);
}

__device__ bool offsetGridCellCoordinate(int base, int offset, int& result)
{
    if ((offset < 0 && base == INT_MIN) || (offset > 0 && base == INT_MAX))
    {
        return false;
    }
    result = base + offset;
    return true;
}

template <typename Scalar>
struct ComputeIcpTargetGridCellKey
{
    const Scalar* points;
    int point_count;
    double cell_size;

    __host__ __device__ IcpGridCellKey operator()(int idx) const
    {
        const double x = static_cast<double>(points[idx]);
        const double y = static_cast<double>(points[point_count + idx]);
        const double z = static_cast<double>(points[2 * point_count + idx]);
        if (!isfinite(x) || !isfinite(y) || !isfinite(z))
        {
            return {INT_MAX, INT_MAX, INT_MAX};
        }
        return {
            icpGridCellCoordinate(x, cell_size),
            icpGridCellCoordinate(y, cell_size),
            icpGridCellCoordinate(z, cell_size)
        };
    }
};

__device__ int lowerBoundIcpGridCell(const IcpGridCellKey* cell_keys, int cell_count, const IcpGridCellKey& query)
{
#ifdef PLAPOINT_ENABLE_TESTING
    atomicAdd(&g_icp_grid_cell_lookup_count, 1ull);
#endif
    int first = 0;
    int last = cell_count;
    const IcpGridCellKeyLess less{};
    while (first < last)
    {
        const int mid = first + (last - first) / 2;
        if (less(cell_keys[mid], query))
        {
            first = mid + 1;
        }
        else
        {
            last = mid;
        }
    }
    return first;
}

__device__ double distanceOutsideIcpGridCellAxis(double value, int cell_coordinate, double cell_size)
{
    const double cell_min = static_cast<double>(cell_coordinate) * cell_size;
    const double cell_max = cell_min + cell_size;
    if (!isfinite(cell_min) || !isfinite(cell_max))
    {
        return 0.0;
    }
    if (value < cell_min)
    {
        return cell_min - value;
    }
    if (value > cell_max)
    {
        return value - cell_max;
    }
    return 0.0;
}

__device__ double minDistanceSqToIcpGridCell(
    double x,
    double y,
    double z,
    const IcpGridCellKey& cell_key,
    double cell_size)
{
    const double dx = distanceOutsideIcpGridCellAxis(x, cell_key.x, cell_size);
    const double dy = distanceOutsideIcpGridCellAxis(y, cell_key.y, cell_size);
    const double dz = distanceOutsideIcpGridCellAxis(z, cell_key.z, cell_size);
    return dx * dx + dy * dy + dz * dz;
}

__device__ double minDistanceSqToIcpGridCellXY(
    double x,
    double y,
    int cell_x,
    int cell_y,
    double cell_size)
{
    const double dx = distanceOutsideIcpGridCellAxis(x, cell_x, cell_size);
    const double dy = distanceOutsideIcpGridCellAxis(y, cell_y, cell_size);
    return dx * dx + dy * dy;
}

__device__ void recordAcceptedCorrespondence(
    RawIcpStats& local,
    double sx,
    double sy,
    double sz,
    double tx,
    double ty,
    double tz,
    double residual_sq)
{
    local.active_count = 1;
    local.residual_sq_sum = residual_sq;

    const double source_values[3]{sx, sy, sz};
    const double target_values[3]{tx, ty, tz};
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

template <typename Scalar>
__global__ void computeTargetTileBoundsKernel(
    const Scalar* target_points,
    int target_count,
    IcpTargetTileBounds* target_tile_bounds)
{
    const int local_idx = threadIdx.x;
    const int tile_start = static_cast<int>(blockIdx.x) * kIcpStatsBlockSize;
    const int target_idx = tile_start + local_idx;

    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    const bool target_valid = target_idx < target_count &&
        loadFiniteColumnMajorPoint(target_points, target_count, target_idx, tx, ty, tz);

    __shared__ double min_x[kIcpStatsBlockSize];
    __shared__ double min_y[kIcpStatsBlockSize];
    __shared__ double min_z[kIcpStatsBlockSize];
    __shared__ double max_x[kIcpStatsBlockSize];
    __shared__ double max_y[kIcpStatsBlockSize];
    __shared__ double max_z[kIcpStatsBlockSize];

    min_x[local_idx] = target_valid ? tx : INFINITY;
    min_y[local_idx] = target_valid ? ty : INFINITY;
    min_z[local_idx] = target_valid ? tz : INFINITY;
    max_x[local_idx] = target_valid ? tx : -INFINITY;
    max_y[local_idx] = target_valid ? ty : -INFINITY;
    max_z[local_idx] = target_valid ? tz : -INFINITY;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            min_x[local_idx] = fmin(min_x[local_idx], min_x[local_idx + stride]);
            min_y[local_idx] = fmin(min_y[local_idx], min_y[local_idx + stride]);
            min_z[local_idx] = fmin(min_z[local_idx], min_z[local_idx + stride]);
            max_x[local_idx] = fmax(max_x[local_idx], max_x[local_idx + stride]);
            max_y[local_idx] = fmax(max_y[local_idx], max_y[local_idx + stride]);
            max_z[local_idx] = fmax(max_z[local_idx], max_z[local_idx + stride]);
        }
        __syncthreads();
    }

    if (local_idx == 0)
    {
        IcpTargetTileBounds bounds{};
        bounds.min_x = min_x[0];
        bounds.min_y = min_y[0];
        bounds.min_z = min_z[0];
        bounds.max_x = max_x[0];
        bounds.max_y = max_y[0];
        bounds.max_z = max_z[0];
        bounds.has_valid_point = isfinite(min_x[0]) ? 1 : 0;
        target_tile_bounds[blockIdx.x] = bounds;
#ifdef PLAPOINT_ENABLE_TESTING
        atomicAdd(&g_icp_target_tile_bound_computation_count, 1ull);
#endif
    }
}

template <typename Scalar>
__global__ void collectCorrespondenceStatsKernel(
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    int* correspondence_indices,
    const IcpTargetTileBounds* target_tile_bounds,
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
    __shared__ int target_tile_valid[kIcpStatsBlockSize];

    for (int tile_start = 0; tile_start < target_count; tile_start += kIcpStatsBlockSize)
    {
        const int tile_idx = tile_start / kIcpStatsBlockSize;
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

        bool tile_relevant = true;
        if (source_valid && can_prune_by_radius && target_tile_bounds)
        {
            const IcpTargetTileBounds bounds = target_tile_bounds[tile_idx];
            tile_relevant =
                bounds.has_valid_point &&
                sx >= bounds.min_x - max_dist && sx <= bounds.max_x + max_dist &&
                sy >= bounds.min_y - max_dist && sy <= bounds.max_y + max_dist &&
                sz >= bounds.min_z - max_dist && sz <= bounds.max_z + max_dist;
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
            recordAcceptedCorrespondence(local, sx, sy, sz, best_tx, best_ty, best_tz, best_dist_sq);
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

template <typename Scalar>
__global__ void collectCorrespondenceStatsSpatialGridKernel(
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    int* correspondence_indices,
    IcpTargetSpatialGrid target_grid,
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
    const double max_dist_sq = max_dist * max_dist;

    if (source_valid && target_grid.active)
    {
        const IcpGridCellKey source_key{
            icpGridCellCoordinate(sx, target_grid.cell_size),
            icpGridCellCoordinate(sy, target_grid.cell_size),
            icpGridCellCoordinate(sz, target_grid.cell_size)
        };

        int min_z = source_key.z;
        int max_z = source_key.z;
        if (!offsetGridCellCoordinate(source_key.z, -1, min_z))
        {
            min_z = source_key.z;
        }
        if (!offsetGridCellCoordinate(source_key.z, 1, max_z))
        {
            max_z = source_key.z;
        }

        const int offset_order[3]{0, -1, 1};
        const bool can_stop_after_exact_match = correspondence_indices == nullptr;
        bool stop_cell_scan = false;
        for (int dx_offset_idx = 0; dx_offset_idx < 3 && !stop_cell_scan; ++dx_offset_idx)
        {
            int query_x = 0;
            const int dx_cell = offset_order[dx_offset_idx];
            if (!offsetGridCellCoordinate(source_key.x, dx_cell, query_x))
            {
                continue;
            }
            for (int dy_offset_idx = 0; dy_offset_idx < 3 && !stop_cell_scan; ++dy_offset_idx)
            {
                int query_y = 0;
                const int dy_cell = offset_order[dy_offset_idx];
                if (!offsetGridCellCoordinate(source_key.y, dy_cell, query_y))
                {
                    continue;
                }

                const double min_xy_dist_sq = minDistanceSqToIcpGridCellXY(
                    sx,
                    sy,
                    query_x,
                    query_y,
                    target_grid.cell_size);
                if (min_xy_dist_sq > max_dist_sq || min_xy_dist_sq > best_dist_sq)
                {
                    continue;
                }

                IcpGridCellKey query_key{query_x, query_y, min_z};
                int cell_idx = lowerBoundIcpGridCell(target_grid.cell_keys, target_grid.cell_count, query_key);
                while (cell_idx < target_grid.cell_count && !stop_cell_scan)
                {
                    const IcpGridCellKey cell_key = target_grid.cell_keys[cell_idx];
                    if (cell_key.x != query_x || cell_key.y != query_y || cell_key.z > max_z)
                    {
                        break;
                    }

                    const double min_cell_dist_sq = minDistanceSqToIcpGridCell(
                        sx,
                        sy,
                        sz,
                        cell_key,
                        target_grid.cell_size);
                    if (min_cell_dist_sq > max_dist_sq || min_cell_dist_sq > best_dist_sq)
                    {
                        ++cell_idx;
                        continue;
                    }

                    const int start = target_grid.cell_starts[cell_idx];
                    const int count = target_grid.cell_counts[cell_idx];
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int target_idx = target_grid.sorted_target_indices[start + offset];
#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_target_candidate_visit_count, 1ull);
#endif
                        double tx = 0.0;
                        double ty = 0.0;
                        double tz = 0.0;
                        loadColumnMajorPoint(target_points, target_count, target_idx, tx, ty, tz);

                        const double dx = sx - tx;
                        if (fabs(dx) > max_dist)
                        {
                            continue;
                        }
                        const double dy = sy - ty;
                        if (fabs(dy) > max_dist)
                        {
                            continue;
                        }
                        const double dz = sz - tz;
                        if (fabs(dz) > max_dist)
                        {
                            continue;
                        }
#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
                        const double dist_sq = dx * dx + dy * dy + dz * dz;
                        if (isfinite(dist_sq) &&
                            (dist_sq < best_dist_sq ||
                             (dist_sq == best_dist_sq && (best_idx < 0 || target_idx < best_idx))))
                        {
                            best_dist_sq = dist_sq;
                            best_idx = target_idx;
                            best_tx = tx;
                            best_ty = ty;
                            best_tz = tz;
                            if (can_stop_after_exact_match && dist_sq <= 0.0)
                            {
                                stop_cell_scan = true;
                                break;
                            }
                        }
                    }
                    ++cell_idx;
                }
            }
        }
    }

    if (source_valid)
    {
        bool accepted = best_idx >= 0;
        if (accepted)
        {
            accepted = best_dist_sq <= max_dist_sq;
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
            recordAcceptedCorrespondence(local, sx, sy, sz, best_tx, best_ty, best_tz, best_dist_sq);
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

template <typename Scalar>
__global__ void collectExactPointwiseCorrespondenceStatsKernel(
    const Scalar* source_points,
    const Scalar* target_points,
    int point_count,
    RawIcpStats* partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};

    if (source_idx < point_count)
    {
        const Scalar raw_sx = source_points[source_idx];
        const Scalar raw_sy = source_points[point_count + source_idx];
        const Scalar raw_sz = source_points[2 * point_count + source_idx];
        const Scalar raw_tx = target_points[source_idx];
        const Scalar raw_ty = target_points[point_count + source_idx];
        const Scalar raw_tz = target_points[2 * point_count + source_idx];

        if (raw_sx != raw_tx || raw_sy != raw_ty || raw_sz != raw_tz)
        {
            local.residual_sq_sum = INFINITY;
        }
        else
        {
            const double sx = static_cast<double>(raw_sx);
            const double sy = static_cast<double>(raw_sy);
            const double sz = static_cast<double>(raw_sz);
            if (!isfinite(sx) || !isfinite(sy) || !isfinite(sz))
            {
                local.invalid_source_count = 1;
            }
            else
            {
                recordAcceptedCorrespondence(local, sx, sy, sz, sx, sy, sz, 0.0);
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

template <typename Scalar>
__global__ void collectResidualStatsKernel(
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const IcpTargetTileBounds* target_tile_bounds,
    RawIcpResidualStats* partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;

    if (source_idx < source_count)
    {
        if (!loadFiniteColumnMajorPoint(source_points, source_count, source_idx, sx, sy, sz))
        {
            local.invalid_source_count = 1;
        }
        else
        {
            source_valid = true;
        }
    }

    double best_dist_sq = INFINITY;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const bool can_prune_by_radius = isfinite(max_dist) && max_dist >= 0.0;
    __shared__ double target_tile_x[kIcpStatsBlockSize];
    __shared__ double target_tile_y[kIcpStatsBlockSize];
    __shared__ double target_tile_z[kIcpStatsBlockSize];
    __shared__ int target_tile_valid[kIcpStatsBlockSize];

    for (int tile_start = 0; tile_start < target_count; tile_start += kIcpStatsBlockSize)
    {
        const int tile_idx = tile_start / kIcpStatsBlockSize;
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

        bool tile_relevant = true;
        if (source_valid && can_prune_by_radius && target_tile_bounds)
        {
            const IcpTargetTileBounds bounds = target_tile_bounds[tile_idx];
            tile_relevant =
                bounds.has_valid_point &&
                sx >= bounds.min_x - max_dist && sx <= bounds.max_x + max_dist &&
                sy >= bounds.min_y - max_dist && sy <= bounds.max_y + max_dist &&
                sz >= bounds.min_z - max_dist && sz <= bounds.max_z + max_dist;
        }

        if (source_valid && tile_relevant)
        {
            const int tile_count = min(kIcpStatsBlockSize, target_count - tile_start);
            for (int tile_offset = 0; tile_offset < tile_count; ++tile_offset)
            {
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

                const double dist_sq = dx * dx + dy * dy + dz * dz;
                if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                {
                    best_dist_sq = dist_sq;
                }
            }
        }
        __syncthreads();
    }

    if (source_valid)
    {
        bool accepted = isfinite(best_dist_sq);
        if (accepted && isfinite(max_dist))
        {
            accepted = best_dist_sq <= max_dist * max_dist;
        }
        if (accepted)
        {
            local.active_count = 1;
            local.residual_sq_sum = best_dist_sq;
        }
    }

    __shared__ RawIcpResidualStats shared_stats[kIcpStatsBlockSize];
    shared_stats[local_idx] = local;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            addRawIcpResidualStats(shared_stats[local_idx], shared_stats[local_idx + stride]);
        }
        __syncthreads();
    }

    if (local_idx == 0)
    {
        partial_stats[blockIdx.x] = shared_stats[0];
    }
}

template <typename Scalar>
__global__ void collectResidualStatsSpatialGridKernel(
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    IcpTargetSpatialGrid target_grid,
    RawIcpResidualStats* partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;

    if (source_idx < source_count)
    {
        if (!loadFiniteColumnMajorPoint(source_points, source_count, source_idx, sx, sy, sz))
        {
            local.invalid_source_count = 1;
        }
        else
        {
            source_valid = true;
        }
    }

    double best_dist_sq = INFINITY;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const double max_dist_sq = max_dist * max_dist;

    if (source_valid && target_grid.active)
    {
        const IcpGridCellKey source_key{
            icpGridCellCoordinate(sx, target_grid.cell_size),
            icpGridCellCoordinate(sy, target_grid.cell_size),
            icpGridCellCoordinate(sz, target_grid.cell_size)
        };

        int min_z = source_key.z;
        int max_z = source_key.z;
        if (!offsetGridCellCoordinate(source_key.z, -1, min_z))
        {
            min_z = source_key.z;
        }
        if (!offsetGridCellCoordinate(source_key.z, 1, max_z))
        {
            max_z = source_key.z;
        }

        const int offset_order[3]{0, -1, 1};
        bool stop_cell_scan = false;
        for (int dx_offset_idx = 0; dx_offset_idx < 3 && !stop_cell_scan; ++dx_offset_idx)
        {
            int query_x = 0;
            const int dx_cell = offset_order[dx_offset_idx];
            if (!offsetGridCellCoordinate(source_key.x, dx_cell, query_x))
            {
                continue;
            }
            for (int dy_offset_idx = 0; dy_offset_idx < 3 && !stop_cell_scan; ++dy_offset_idx)
            {
                int query_y = 0;
                const int dy_cell = offset_order[dy_offset_idx];
                if (!offsetGridCellCoordinate(source_key.y, dy_cell, query_y))
                {
                    continue;
                }

                const double min_xy_dist_sq = minDistanceSqToIcpGridCellXY(
                    sx,
                    sy,
                    query_x,
                    query_y,
                    target_grid.cell_size);
                if (min_xy_dist_sq > max_dist_sq || min_xy_dist_sq >= best_dist_sq)
                {
                    continue;
                }

                IcpGridCellKey query_key{query_x, query_y, min_z};
                int cell_idx = lowerBoundIcpGridCell(target_grid.cell_keys, target_grid.cell_count, query_key);
                while (cell_idx < target_grid.cell_count && !stop_cell_scan)
                {
                    const IcpGridCellKey cell_key = target_grid.cell_keys[cell_idx];
                    if (cell_key.x != query_x || cell_key.y != query_y || cell_key.z > max_z)
                    {
                        break;
                    }

                    const double min_cell_dist_sq = minDistanceSqToIcpGridCell(
                        sx,
                        sy,
                        sz,
                        cell_key,
                        target_grid.cell_size);
                    if (min_cell_dist_sq > max_dist_sq || min_cell_dist_sq >= best_dist_sq)
                    {
                        ++cell_idx;
                        continue;
                    }

                    const int start = target_grid.cell_starts[cell_idx];
                    const int count = target_grid.cell_counts[cell_idx];
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int target_idx = target_grid.sorted_target_indices[start + offset];
                        double tx = 0.0;
                        double ty = 0.0;
                        double tz = 0.0;
                        loadColumnMajorPoint(target_points, target_count, target_idx, tx, ty, tz);

                        const double dx = sx - tx;
                        if (fabs(dx) > max_dist)
                        {
                            continue;
                        }
                        const double dy = sy - ty;
                        if (fabs(dy) > max_dist)
                        {
                            continue;
                        }
                        const double dz = sz - tz;
                        if (fabs(dz) > max_dist)
                        {
                            continue;
                        }

                        const double dist_sq = dx * dx + dy * dy + dz * dz;
                        if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                        {
                            best_dist_sq = dist_sq;
                            if (dist_sq <= 0.0)
                            {
                                stop_cell_scan = true;
                                break;
                            }
                        }
                    }
                    ++cell_idx;
                }
            }
        }
    }

    if (source_valid && isfinite(best_dist_sq) && best_dist_sq <= max_dist_sq)
    {
        local.active_count = 1;
        local.residual_sq_sum = best_dist_sq;
    }

    __shared__ RawIcpResidualStats shared_stats[kIcpStatsBlockSize];
    shared_stats[local_idx] = local;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            addRawIcpResidualStats(shared_stats[local_idx], shared_stats[local_idx + stride]);
        }
        __syncthreads();
    }

    if (local_idx == 0)
    {
        partial_stats[blockIdx.x] = shared_stats[0];
    }
}

template <typename Scalar>
__global__ void transformAndCollectResidualStatsKernel(
    const Scalar* transform,
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    Scalar* output_points,
    const IcpTargetTileBounds* target_tile_bounds,
    RawIcpResidualStats* partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;

    if (source_idx < source_count)
    {
        const Scalar px = source_points[source_idx];
        const Scalar py = source_points[source_count + source_idx];
        const Scalar pz = source_points[2 * source_count + source_idx];

        const Scalar ox = transform[0] * px + transform[4] * py + transform[8] * pz + transform[12];
        const Scalar oy = transform[1] * px + transform[5] * py + transform[9] * pz + transform[13];
        const Scalar oz = transform[2] * px + transform[6] * py + transform[10] * pz + transform[14];
        output_points[source_idx] = ox;
        output_points[source_count + source_idx] = oy;
        output_points[2 * source_count + source_idx] = oz;

        sx = static_cast<double>(ox);
        sy = static_cast<double>(oy);
        sz = static_cast<double>(oz);
        if (!isfinite(sx) || !isfinite(sy) || !isfinite(sz))
        {
            local.invalid_source_count = 1;
        }
        else
        {
            source_valid = true;
        }
    }

    double best_dist_sq = INFINITY;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const bool can_prune_by_radius = isfinite(max_dist) && max_dist >= 0.0;
    __shared__ double target_tile_x[kIcpStatsBlockSize];
    __shared__ double target_tile_y[kIcpStatsBlockSize];
    __shared__ double target_tile_z[kIcpStatsBlockSize];
    __shared__ int target_tile_valid[kIcpStatsBlockSize];

    for (int tile_start = 0; tile_start < target_count; tile_start += kIcpStatsBlockSize)
    {
        const int tile_idx = tile_start / kIcpStatsBlockSize;
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

        bool tile_relevant = true;
        if (source_valid && can_prune_by_radius && target_tile_bounds)
        {
            const IcpTargetTileBounds bounds = target_tile_bounds[tile_idx];
            tile_relevant =
                bounds.has_valid_point &&
                sx >= bounds.min_x - max_dist && sx <= bounds.max_x + max_dist &&
                sy >= bounds.min_y - max_dist && sy <= bounds.max_y + max_dist &&
                sz >= bounds.min_z - max_dist && sz <= bounds.max_z + max_dist;
        }

        if (source_valid && tile_relevant)
        {
            const int tile_count = min(kIcpStatsBlockSize, target_count - tile_start);
            for (int tile_offset = 0; tile_offset < tile_count; ++tile_offset)
            {
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

                const double dist_sq = dx * dx + dy * dy + dz * dz;
                if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                {
                    best_dist_sq = dist_sq;
                }
            }
        }
        __syncthreads();
    }

    if (source_valid)
    {
        bool accepted = isfinite(best_dist_sq);
        if (accepted && isfinite(max_dist))
        {
            accepted = best_dist_sq <= max_dist * max_dist;
        }
        if (accepted)
        {
            local.active_count = 1;
            local.residual_sq_sum = best_dist_sq;
        }
    }

    __shared__ RawIcpResidualStats shared_stats[kIcpStatsBlockSize];
    shared_stats[local_idx] = local;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            addRawIcpResidualStats(shared_stats[local_idx], shared_stats[local_idx + stride]);
        }
        __syncthreads();
    }

    if (local_idx == 0)
    {
        partial_stats[blockIdx.x] = shared_stats[0];
    }
}

template <typename Scalar>
__global__ void transformAndCollectResidualStatsSpatialGridKernel(
    const Scalar* transform,
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    Scalar* output_points,
    IcpTargetSpatialGrid target_grid,
    RawIcpResidualStats* partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;

    if (source_idx < source_count)
    {
        const Scalar px = source_points[source_idx];
        const Scalar py = source_points[source_count + source_idx];
        const Scalar pz = source_points[2 * source_count + source_idx];

        const Scalar ox = transform[0] * px + transform[4] * py + transform[8] * pz + transform[12];
        const Scalar oy = transform[1] * px + transform[5] * py + transform[9] * pz + transform[13];
        const Scalar oz = transform[2] * px + transform[6] * py + transform[10] * pz + transform[14];
        output_points[source_idx] = ox;
        output_points[source_count + source_idx] = oy;
        output_points[2 * source_count + source_idx] = oz;

        sx = static_cast<double>(ox);
        sy = static_cast<double>(oy);
        sz = static_cast<double>(oz);
        if (!isfinite(sx) || !isfinite(sy) || !isfinite(sz))
        {
            local.invalid_source_count = 1;
        }
        else
        {
            source_valid = true;
        }
    }

    double best_dist_sq = INFINITY;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const double max_dist_sq = max_dist * max_dist;

    if (source_valid && target_grid.active)
    {
        const IcpGridCellKey source_key{
            icpGridCellCoordinate(sx, target_grid.cell_size),
            icpGridCellCoordinate(sy, target_grid.cell_size),
            icpGridCellCoordinate(sz, target_grid.cell_size)
        };

        int min_z = source_key.z;
        int max_z = source_key.z;
        if (!offsetGridCellCoordinate(source_key.z, -1, min_z))
        {
            min_z = source_key.z;
        }
        if (!offsetGridCellCoordinate(source_key.z, 1, max_z))
        {
            max_z = source_key.z;
        }

        const int offset_order[3]{0, -1, 1};
        bool stop_cell_scan = false;
        for (int dx_offset_idx = 0; dx_offset_idx < 3 && !stop_cell_scan; ++dx_offset_idx)
        {
            int query_x = 0;
            const int dx_cell = offset_order[dx_offset_idx];
            if (!offsetGridCellCoordinate(source_key.x, dx_cell, query_x))
            {
                continue;
            }
            for (int dy_offset_idx = 0; dy_offset_idx < 3 && !stop_cell_scan; ++dy_offset_idx)
            {
                int query_y = 0;
                const int dy_cell = offset_order[dy_offset_idx];
                if (!offsetGridCellCoordinate(source_key.y, dy_cell, query_y))
                {
                    continue;
                }

                const double min_xy_dist_sq = minDistanceSqToIcpGridCellXY(
                    sx,
                    sy,
                    query_x,
                    query_y,
                    target_grid.cell_size);
                if (min_xy_dist_sq > max_dist_sq || min_xy_dist_sq >= best_dist_sq)
                {
                    continue;
                }

                IcpGridCellKey query_key{query_x, query_y, min_z};
                int cell_idx = lowerBoundIcpGridCell(target_grid.cell_keys, target_grid.cell_count, query_key);
                while (cell_idx < target_grid.cell_count && !stop_cell_scan)
                {
                    const IcpGridCellKey cell_key = target_grid.cell_keys[cell_idx];
                    if (cell_key.x != query_x || cell_key.y != query_y || cell_key.z > max_z)
                    {
                        break;
                    }

                    const double min_cell_dist_sq = minDistanceSqToIcpGridCell(
                        sx,
                        sy,
                        sz,
                        cell_key,
                        target_grid.cell_size);
                    if (min_cell_dist_sq > max_dist_sq || min_cell_dist_sq >= best_dist_sq)
                    {
                        ++cell_idx;
                        continue;
                    }

                    const int start = target_grid.cell_starts[cell_idx];
                    const int count = target_grid.cell_counts[cell_idx];
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int target_idx = target_grid.sorted_target_indices[start + offset];
                        double tx = 0.0;
                        double ty = 0.0;
                        double tz = 0.0;
                        loadColumnMajorPoint(target_points, target_count, target_idx, tx, ty, tz);

                        const double dx = sx - tx;
                        if (fabs(dx) > max_dist)
                        {
                            continue;
                        }
                        const double dy = sy - ty;
                        if (fabs(dy) > max_dist)
                        {
                            continue;
                        }
                        const double dz = sz - tz;
                        if (fabs(dz) > max_dist)
                        {
                            continue;
                        }

                        const double dist_sq = dx * dx + dy * dy + dz * dz;
                        if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                        {
                            best_dist_sq = dist_sq;
                            if (dist_sq <= 0.0)
                            {
                                stop_cell_scan = true;
                                break;
                            }
                        }
                    }
                    ++cell_idx;
                }
            }
        }
    }

    if (source_valid && isfinite(best_dist_sq) && best_dist_sq <= max_dist_sq)
    {
        local.active_count = 1;
        local.residual_sq_sum = best_dist_sq;
    }

    __shared__ RawIcpResidualStats shared_stats[kIcpStatsBlockSize];
    shared_stats[local_idx] = local;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            addRawIcpResidualStats(shared_stats[local_idx], shared_stats[local_idx + stride]);
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
__global__ void reduceRawIcpStatsAndSetExactPointwiseIdentityStepKernel(
    const RawIcpStats* partial_stats,
    int partial_count,
    RawIcpStats* stats,
    Scalar* step_transform,
    IcpStepTransformRawResult* result)
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

    if (local_idx < 16)
    {
        const int row = local_idx & 3;
        const int col = local_idx >> 2;
        step_transform[local_idx] = row == col ? Scalar(1) : Scalar(0);
    }

    if (local_idx == 0)
    {
        const RawIcpStats reduced = shared_stats[0];
        *stats = reduced;
        result->delta = 0.0;
        result->valid = reduced.active_count > 0 && isfinite(reduced.residual_sq_sum) ? 1 : 0;
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndSetExactPointwiseIdentityAlignmentStepKernel(
    const RawIcpStats* partial_stats,
    int partial_count,
    Scalar* step_transform,
    IcpAlignmentStepRawResult* result)
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
        writeAlignmentStepRawResultFromRawStats<Scalar>(shared_stats[0], step_transform, result, true);
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndComputeStepTransformKernel(
    const RawIcpStats* partial_stats,
    int partial_count,
    RawIcpStats* stats,
    Scalar* step_transform,
    IcpStepTransformRawResult* result)
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
        const RawIcpStats raw = shared_stats[0];
        *stats = raw;
        computeStepTransformFromRawStatsValue<Scalar>(raw, step_transform, result);
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndComputeAlignmentStepKernel(
    const RawIcpStats* partial_stats,
    int partial_count,
    Scalar* step_transform,
    IcpAlignmentStepRawResult* result)
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
        writeAlignmentStepRawResultFromRawStats<Scalar>(shared_stats[0], step_transform, result, false);
    }
}

__global__ void reduceRawIcpResidualStatsKernel(
    const RawIcpResidualStats* partial_stats,
    int partial_count,
    RawIcpResidualStats* stats)
{
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    for (int idx = local_idx; idx < partial_count; idx += blockDim.x)
    {
        addRawIcpResidualStats(local, partial_stats[idx]);
    }

    __shared__ RawIcpResidualStats shared_stats[kIcpStatsBlockSize];
    shared_stats[local_idx] = local;
    __syncthreads();

    for (int stride = kIcpStatsBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (local_idx < stride)
        {
            addRawIcpResidualStats(shared_stats[local_idx], shared_stats[local_idx + stride]);
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

    computeStepTransformFromRawStatsValue<Scalar>(*raw_stats, step_transform, result);
}

template <typename Scalar>
__device__ void computeStepTransformFromRawStatsValue(
    const RawIcpStats& raw,
    Scalar* step_transform,
    IcpStepTransformRawResult* result)
{
    if (raw.active_count <= 0)
    {
        result->delta = 0.0;
        result->valid = 0;
        return;
    }

    IcpStepTransformInput input{};
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

__device__ bool rawStatsCovarianceHasNonCollinearGeometry(
    const double sum[3],
    const double outer_sum[9],
    int active_count)
{
    if (active_count <= 0)
    {
        return false;
    }

    const double inv_count = 1.0 / static_cast<double>(active_count);
    const double c00 = outer_sum[0] - sum[0] * sum[0] * inv_count;
    const double c01 = outer_sum[1] - sum[0] * sum[1] * inv_count;
    const double c02 = outer_sum[2] - sum[0] * sum[2] * inv_count;
    const double c11 = outer_sum[4] - sum[1] * sum[1] * inv_count;
    const double c12 = outer_sum[5] - sum[1] * sum[2] * inv_count;
    const double c22 = outer_sum[8] - sum[2] * sum[2] * inv_count;
    const double trace = c00 + c11 + c22;
    if (!isfinite(trace) || trace <= 0.0)
    {
        return false;
    }

    const double principal_minor_sum = c00 * c11 + c00 * c22 + c11 * c22 -
        (c01 * c01 + c02 * c02 + c12 * c12);
    if (!isfinite(principal_minor_sum))
    {
        return false;
    }

    const double threshold = fmax(trace * trace * 1.0e-12, trace * 1.0e-30);
    return principal_minor_sum > threshold;
}

template <typename Scalar>
__device__ void writeAlignmentStepRawResultFromRawStats(
    const RawIcpStats& raw,
    Scalar* step_transform,
    IcpAlignmentStepRawResult* result,
    bool exact_identity_step)
{
    IcpStepTransformRawResult step_result{};
    if (exact_identity_step)
    {
        for (int idx = 0; idx < 16; ++idx)
        {
            const int row = idx & 3;
            const int col = idx >> 2;
            step_transform[idx] = row == col ? Scalar(1) : Scalar(0);
        }
        step_result.delta = 0.0;
        step_result.valid = raw.active_count > 0 && isfinite(raw.residual_sq_sum) ? 1 : 0;
    }
    else
    {
        computeStepTransformFromRawStatsValue<Scalar>(raw, step_transform, &step_result);
    }

    result->active_count = raw.active_count;
    result->invalid_source_count = raw.invalid_source_count;
    result->src_has_non_collinear_geometry =
        rawStatsCovarianceHasNonCollinearGeometry(raw.src_sum, raw.src_outer_sum, raw.active_count) ? 1 : 0;
    result->tgt_has_non_collinear_geometry =
        rawStatsCovarianceHasNonCollinearGeometry(raw.tgt_sum, raw.tgt_outer_sum, raw.active_count) ? 1 : 0;
    result->step_valid = step_result.valid;
    result->residual_sq_sum = raw.residual_sq_sum;
    result->delta = step_result.delta;
}

bool covarianceHasNonCollinearGeometry(const double covariance[9]);

template <typename Scalar>
bool shouldPrecomputeTargetTileBounds(Scalar max_correspondence_distance)
{
    const double max_dist = static_cast<double>(max_correspondence_distance);
    return std::isfinite(max_dist) && max_dist >= 0.0;
}

template <typename Scalar>
bool shouldUseTargetSpatialGrid(Scalar max_correspondence_distance)
{
    const double max_dist = static_cast<double>(max_correspondence_distance);
    return std::isfinite(max_dist) && max_dist > 0.0;
}

template <typename Scalar>
bool shouldTryExactPointwiseStats(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const int* d_correspondence_indices)
{
    if (d_correspondence_indices || source_count != target_count)
    {
        return false;
    }
    if (d_source_points == d_target_points)
    {
        return true;
    }
    const double max_dist = static_cast<double>(max_correspondence_distance);
    return !std::isfinite(max_dist);
}

template <typename Scalar>
bool launchExactPointwiseStats(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const int* d_correspondence_indices,
    RawIcpStats* d_partials,
    int partial_count,
    RawIcpStats* d_stats,
    cudaStream_t stream)
{
    if (!shouldTryExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_correspondence_indices))
    {
        return false;
    }

    constexpr int block_size = kIcpStatsBlockSize;
    collectExactPointwiseCorrespondenceStatsKernel<Scalar><<<partial_count, block_size, 0, stream>>>(
        d_source_points,
        d_target_points,
        source_count,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        partial_count,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    return true;
}

template <typename Scalar>
bool launchExactPointwiseStatsAndIdentityStep(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const int* d_correspondence_indices,
    RawIcpStats* d_partials,
    int partial_count,
    RawIcpStats* d_stats,
    Scalar* d_step_transform,
    IcpStepTransformRawResult* d_result,
    cudaStream_t stream)
{
    if (!shouldTryExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_correspondence_indices))
    {
        return false;
    }

    constexpr int block_size = kIcpStatsBlockSize;
    collectExactPointwiseCorrespondenceStatsKernel<Scalar><<<partial_count, block_size, 0, stream>>>(
        d_source_points,
        d_target_points,
        source_count,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_exact_pointwise_step_call_count.fetch_add(1, std::memory_order_relaxed);
#endif
    reduceRawIcpStatsAndSetExactPointwiseIdentityStepKernel<Scalar><<<1, block_size, 0, stream>>>(
        d_partials,
        partial_count,
        d_stats,
        d_step_transform,
        d_result);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    return true;
}

template <typename Scalar>
bool launchExactPointwiseAlignmentStep(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    RawIcpStats* d_partials,
    int partial_count,
    Scalar* d_step_transform,
    IcpAlignmentStepRawResult* d_result,
    cudaStream_t stream)
{
    if (!shouldTryExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            nullptr))
    {
        return false;
    }

    constexpr int block_size = kIcpStatsBlockSize;
    collectExactPointwiseCorrespondenceStatsKernel<Scalar><<<partial_count, block_size, 0, stream>>>(
        d_source_points,
        d_target_points,
        source_count,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_exact_pointwise_step_call_count.fetch_add(1, std::memory_order_relaxed);
#endif
    reduceRawIcpStatsAndSetExactPointwiseIdentityAlignmentStepKernel<Scalar><<<1, block_size, 0, stream>>>(
        d_partials,
        partial_count,
        d_step_transform,
        d_result);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    return true;
}

template <typename Scalar>
bool tryComputeExactPointwiseStats(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const int* d_correspondence_indices,
    RawIcpStats* d_partials,
    int partial_count,
    RawIcpStats* d_stats,
    RawIcpStats& raw,
    cudaStream_t stream)
{
    if (!launchExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_correspondence_indices,
            d_partials,
            partial_count,
            d_stats,
            stream))
    {
        return false;
    }

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats, sizeof(RawIcpStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return std::isfinite(raw.residual_sq_sum);
}

template <typename Scalar>
const IcpTargetTileBounds* prepareTargetTileBounds(
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    if (!shouldPrecomputeTargetTileBounds(max_correspondence_distance))
    {
        return nullptr;
    }

    workspace.reserveTargetTileBounds(target_count);
    auto* d_bounds = reinterpret_cast<IcpTargetTileBounds*>(workspace.targetTileBoundsStorage());
    const int tile_count = icpTargetTileCount(target_count);
    computeTargetTileBoundsKernel<Scalar><<<tile_count, kIcpStatsBlockSize, 0, stream>>>(
        d_target_points,
        target_count,
        d_bounds);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    return d_bounds;
}

template <typename Scalar>
IcpTargetSpatialGrid prepareTargetSpatialGrid(
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    IcpTargetSpatialGrid grid{};
    if (!shouldUseTargetSpatialGrid(max_correspondence_distance))
    {
        return grid;
    }

    const double cell_size = static_cast<double>(max_correspondence_distance);
    workspace.reserveTargetSpatialGrid(target_count);
    auto* d_keys = reinterpret_cast<IcpGridCellKey*>(workspace.targetSpatialGridKeysStorage());
    auto* d_unique_keys = reinterpret_cast<IcpGridCellKey*>(workspace.targetSpatialGridUniqueKeysStorage());
    auto* d_indices = reinterpret_cast<int*>(workspace.targetSpatialGridIndicesStorage());
    auto* d_cell_starts = reinterpret_cast<int*>(workspace.targetSpatialGridCellStartsStorage());
    auto* d_cell_counts = reinterpret_cast<int*>(workspace.targetSpatialGridCellCountsStorage());

    grid.cell_keys = d_unique_keys;
    grid.sorted_target_indices = d_indices;
    grid.cell_starts = d_cell_starts;
    grid.cell_counts = d_cell_counts;
    grid.cell_size = cell_size;

    if (workspace.targetSpatialGridCacheMatches(d_target_points, target_count, cell_size))
    {
        grid.cell_count = workspace.targetSpatialGridCellCount();
        grid.active = true;
        return grid;
    }

    auto policy = thrust::cuda::par.on(stream);
    auto keys = thrust::device_pointer_cast(d_keys);
    auto unique_keys = thrust::device_pointer_cast(d_unique_keys);
    auto indices = thrust::device_pointer_cast(d_indices);
    auto cell_starts = thrust::device_pointer_cast(d_cell_starts);
    auto cell_counts = thrust::device_pointer_cast(d_cell_counts);

    thrust::sequence(policy, indices, indices + target_count, 0);
    thrust::transform(
        policy,
        indices,
        indices + target_count,
        keys,
        ComputeIcpTargetGridCellKey<Scalar>{
            d_target_points,
            target_count,
            cell_size
        });
    thrust::sort_by_key(policy, keys, keys + target_count, indices, IcpGridCellKeyLess{});
    const auto reduced = thrust::reduce_by_key(
        policy,
        keys,
        keys + target_count,
        thrust::make_constant_iterator(1),
        unique_keys,
        cell_counts,
        IcpGridCellKeyEqual{},
        thrust::plus<int>{});
    const int cell_count = static_cast<int>(reduced.first - unique_keys);
    thrust::exclusive_scan(policy, cell_counts, cell_counts + cell_count, cell_starts);

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_target_spatial_grid_build_count.fetch_add(1, std::memory_order_relaxed);
#endif

    grid.cell_count = cell_count;
    grid.active = true;
    workspace.markTargetSpatialGridCache(d_target_points, target_count, cell_size, cell_count);
    return grid;
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
    stats.src_has_non_collinear_geometry = covarianceHasNonCollinearGeometry(stats.src_covariance);
    stats.tgt_has_non_collinear_geometry = covarianceHasNonCollinearGeometry(stats.tgt_covariance);
    return stats;
}

template <typename Scalar>
IcpResidualStats<Scalar> makeHostResidualStats(const RawIcpResidualStats& raw)
{
    IcpResidualStats<Scalar> stats;
    stats.active_count = raw.active_count;
    stats.invalid_source_count = raw.invalid_source_count;
    stats.residual_sq_sum = raw.residual_sq_sum;
    return stats;
}

template <typename Scalar>
IcpAlignmentStepResult<Scalar> makeHostAlignmentStepResult(const IcpAlignmentStepRawResult& raw)
{
    IcpAlignmentStepResult<Scalar> result;
    result.active_count = raw.active_count;
    result.invalid_source_count = raw.invalid_source_count;
    result.residual_sq_sum = raw.residual_sq_sum;
    result.src_has_non_collinear_geometry = raw.src_has_non_collinear_geometry != 0;
    result.tgt_has_non_collinear_geometry = raw.tgt_has_non_collinear_geometry != 0;
    result.step.delta = static_cast<Scalar>(raw.delta);
    result.step_valid = raw.step_valid != 0;
    return result;
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
    RawIcpStats raw{};
    if (tryComputeExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_correspondence_indices,
            d_partials,
            grid_size,
            d_stats,
            raw,
            stream))
    {
        return makeHostStats<Scalar>(raw);
    }

    const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
        d_target_points,
        target_count,
        max_correspondence_distance,
        active_workspace,
        stream);

    if (target_grid.active)
    {
        collectCorrespondenceStatsSpatialGridKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_correspondence_indices,
            target_grid,
            d_partials);
    }
    else
    {
        const IcpTargetTileBounds* d_target_tile_bounds = prepareTargetTileBounds(
            d_target_points,
            target_count,
            max_correspondence_distance,
            active_workspace,
            stream);

        collectCorrespondenceStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_correspondence_indices,
            d_target_tile_bounds,
            d_partials);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        grid_size,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats, sizeof(RawIcpStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostStats<Scalar>(raw);
}

template <typename Scalar>
IcpResidualStats<Scalar> computeIcpResidualStatsColumnMajorImpl(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_correspondence_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_residual_stats_call_count.fetch_add(1, std::memory_order_relaxed);
#endif

    if (source_count <= 0 || target_count <= 0)
    {
        return {};
    }
    if (!d_source_points || !d_target_points)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }

    workspace.reserve(source_count);

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpResidualStats*>(workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.statsStorage());
    const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
        d_target_points,
        target_count,
        max_correspondence_distance,
        workspace,
        stream);

    if (target_grid.active)
    {
        collectResidualStatsSpatialGridKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            target_grid,
            d_partials);
    }
    else
    {
        const IcpTargetTileBounds* d_target_tile_bounds = prepareTargetTileBounds(
            d_target_points,
            target_count,
            max_correspondence_distance,
            workspace,
            stream);

        collectResidualStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_target_tile_bounds,
            d_partials);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpResidualStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        grid_size,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    RawIcpResidualStats raw{};
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostResidualStats<Scalar>(raw);
}

template <typename Scalar>
IcpResidualStats<Scalar> transformPointsAndComputeIcpResidualStatsColumnMajorImpl(
    const Scalar* d_transform,
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    Scalar* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_correspondence_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_residual_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_last_transform_output_pointer.store(
        reinterpret_cast<std::uintptr_t>(d_output_points),
        std::memory_order_relaxed);
#endif

    if (source_count <= 0 || target_count <= 0)
    {
        return {};
    }
    if (!d_transform || !d_source_points || !d_target_points || !d_output_points)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }

    workspace.reserve(source_count);

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpResidualStats*>(workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.statsStorage());
    const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
        d_target_points,
        target_count,
        max_correspondence_distance,
        workspace,
        stream);

    if (target_grid.active)
    {
        transformAndCollectResidualStatsSpatialGridKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
            d_transform,
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_output_points,
            target_grid,
            d_partials);
    }
    else
    {
        const IcpTargetTileBounds* d_target_tile_bounds = prepareTargetTileBounds(
            d_target_points,
            target_count,
            max_correspondence_distance,
            workspace,
            stream);

        transformAndCollectResidualStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
            d_transform,
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_output_points,
            d_target_tile_bounds,
            d_partials);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpResidualStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        grid_size,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    RawIcpResidualStats raw{};
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostResidualStats<Scalar>(raw);
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

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_transform_multiply_call_count.fetch_add(1, std::memory_order_relaxed);
#endif

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

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_transform_points_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_last_transform_output_pointer.store(
        reinterpret_cast<std::uintptr_t>(d_output_points),
        std::memory_order_relaxed);
#endif

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
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_raw_stats_step_kernel_launch_count.fetch_add(1, std::memory_order_relaxed);
#endif
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
    RawIcpStats raw{};
    IcpStepTransformRawResult raw_result{};
    const bool exact_pointwise_stats = launchExactPointwiseStatsAndIdentityStep(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        nullptr,
        d_partials,
        grid_size,
        d_stats,
        d_step_transform,
        d_result,
        stream);

    if (exact_pointwise_stats)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw, d_stats, sizeof(RawIcpStats),
                                            cudaMemcpyDeviceToHost, stream));
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw_result, d_result, sizeof(IcpStepTransformRawResult),
                                            cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
        g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
        if (std::isfinite(raw.residual_sq_sum))
        {
            IcpStatsAndStepTransformResult<Scalar> result;
            result.stats = makeHostStats<Scalar>(raw);
            result.step.delta = static_cast<Scalar>(raw_result.delta);
            result.step_valid = raw_result.valid != 0;
            return result;
        }
    }

    {
        const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
            d_target_points,
            target_count,
            max_correspondence_distance,
            stats_workspace,
            stream);

        if (target_grid.active)
        {
            collectCorrespondenceStatsSpatialGridKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
                d_source_points,
                source_count,
                d_target_points,
                target_count,
                max_correspondence_distance,
                nullptr,
                target_grid,
                d_partials);
        }
        else
        {
            const IcpTargetTileBounds* d_target_tile_bounds = prepareTargetTileBounds(
                d_target_points,
                target_count,
                max_correspondence_distance,
                stats_workspace,
                stream);

            collectCorrespondenceStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
                d_source_points,
                source_count,
                d_target_points,
                target_count,
                max_correspondence_distance,
                nullptr,
                d_target_tile_bounds,
                d_partials);
        }
        PLAPOINT_CHECK_CUDA(cudaGetLastError());

        reduceRawIcpStatsAndComputeStepTransformKernel<Scalar><<<1, block_size, 0, stream>>>(
            d_partials,
            grid_size,
            d_stats,
            d_step_transform,
            d_result);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }

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

template <typename Scalar>
IcpAlignmentStepResult<Scalar> computeIcpAlignmentStepColumnMajorImpl(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    Scalar* d_step_transform,
    cudaStream_t stream)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_correspondence_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_alignment_step_call_count.fetch_add(1, std::memory_order_relaxed);
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

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpStats*>(stats_workspace.partialStorage());
    auto* d_result = reinterpret_cast<IcpAlignmentStepRawResult*>(stats_workspace.statsStorage());
    IcpAlignmentStepRawResult raw_result{};

    const bool exact_pointwise_stats = launchExactPointwiseAlignmentStep(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_partials,
        grid_size,
        d_step_transform,
        d_result,
        stream);

    if (exact_pointwise_stats)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw_result, d_result, sizeof(IcpAlignmentStepRawResult),
                                            cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
        g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
        if (std::isfinite(raw_result.residual_sq_sum))
        {
            return makeHostAlignmentStepResult<Scalar>(raw_result);
        }
    }

    {
        const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
            d_target_points,
            target_count,
            max_correspondence_distance,
            stats_workspace,
            stream);

        if (target_grid.active)
        {
            collectCorrespondenceStatsSpatialGridKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
                d_source_points,
                source_count,
                d_target_points,
                target_count,
                max_correspondence_distance,
                nullptr,
                target_grid,
                d_partials);
        }
        else
        {
            const IcpTargetTileBounds* d_target_tile_bounds = prepareTargetTileBounds(
                d_target_points,
                target_count,
                max_correspondence_distance,
                stats_workspace,
                stream);

            collectCorrespondenceStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
                d_source_points,
                source_count,
                d_target_points,
                target_count,
                max_correspondence_distance,
                nullptr,
                d_target_tile_bounds,
                d_partials);
        }
        PLAPOINT_CHECK_CUDA(cudaGetLastError());

        reduceRawIcpStatsAndComputeAlignmentStepKernel<Scalar><<<1, block_size, 0, stream>>>(
            d_partials,
            grid_size,
            d_step_transform,
            d_result);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&raw_result, d_result, sizeof(IcpAlignmentStepRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostAlignmentStepResult<Scalar>(raw_result);
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

void resetIcpResidualStatsCallCountForTesting()
{
    g_icp_residual_stats_call_count.store(0, std::memory_order_relaxed);
}

int icpResidualStatsCallCountForTesting()
{
    return g_icp_residual_stats_call_count.load(std::memory_order_relaxed);
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

void resetIcpExactPointwiseStepCallCountForTesting()
{
    g_icp_exact_pointwise_step_call_count.store(0, std::memory_order_relaxed);
}

int icpExactPointwiseStepCallCountForTesting()
{
    return g_icp_exact_pointwise_step_call_count.load(std::memory_order_relaxed);
}

void resetIcpRawStatsStepKernelLaunchCountForTesting()
{
    g_icp_raw_stats_step_kernel_launch_count.store(0, std::memory_order_relaxed);
}

int icpRawStatsStepKernelLaunchCountForTesting()
{
    return g_icp_raw_stats_step_kernel_launch_count.load(std::memory_order_relaxed);
}

void resetIcpAlignmentStepCallCountForTesting()
{
    g_icp_alignment_step_call_count.store(0, std::memory_order_relaxed);
}

int icpAlignmentStepCallCountForTesting()
{
    return g_icp_alignment_step_call_count.load(std::memory_order_relaxed);
}

void resetIcpHostSynchronizationCountForTesting()
{
    g_icp_host_synchronization_count.store(0, std::memory_order_relaxed);
}

int icpHostSynchronizationCountForTesting()
{
    return g_icp_host_synchronization_count.load(std::memory_order_relaxed);
}

void resetIcpTargetTileBoundComputationCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_target_tile_bound_computation_count, &zero, sizeof(zero)));
}

unsigned long long icpTargetTileBoundComputationCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_target_tile_bound_computation_count, sizeof(count)));
    return count;
}

void resetIcpTargetSpatialGridBuildCountForTesting()
{
    g_icp_target_spatial_grid_build_count.store(0, std::memory_order_relaxed);
}

int icpTargetSpatialGridBuildCountForTesting()
{
    return g_icp_target_spatial_grid_build_count.load(std::memory_order_relaxed);
}

void resetIcpGridCellLookupCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_grid_cell_lookup_count, &zero, sizeof(zero)));
}

unsigned long long icpGridCellLookupCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_grid_cell_lookup_count, sizeof(count)));
    return count;
}

void resetIcpLastTransformOutputPointerForTesting()
{
    g_icp_last_transform_output_pointer.store(0, std::memory_order_relaxed);
}

const void* icpLastTransformOutputPointerForTesting()
{
    return reinterpret_cast<const void*>(
        g_icp_last_transform_output_pointer.load(std::memory_order_relaxed));
}

void resetIcpTransformPointsCallCountForTesting()
{
    g_icp_transform_points_call_count.store(0, std::memory_order_relaxed);
}

int icpTransformPointsCallCountForTesting()
{
    return g_icp_transform_points_call_count.load(std::memory_order_relaxed);
}

void resetIcpTransformMultiplyCallCountForTesting()
{
    g_icp_transform_multiply_call_count.store(0, std::memory_order_relaxed);
}

int icpTransformMultiplyCallCountForTesting()
{
    return g_icp_transform_multiply_call_count.load(std::memory_order_relaxed);
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

void IcpCorrespondenceStatsWorkspace::reserveTargetTileBounds(int target_count)
{
    if (target_count < 0)
    {
        throw std::invalid_argument("ICP GPU: target point count must not be negative");
    }
    if (target_count == 0)
    {
        return;
    }

    const int required_tiles = icpTargetTileCount(target_count);
    if (targetTileBoundCapacity() < required_tiles)
    {
        _target_tile_bounds_storage.allocate(
            static_cast<std::size_t>(required_tiles) * sizeof(IcpTargetTileBounds));
        _target_tile_bound_capacity = required_tiles;
    }
}

void IcpCorrespondenceStatsWorkspace::reserveTargetSpatialGrid(int target_count)
{
    if (target_count < 0)
    {
        throw std::invalid_argument("ICP GPU: target point count must not be negative");
    }
    if (target_count == 0)
    {
        return;
    }

    if (targetSpatialGridCapacity() < target_count)
    {
        invalidateTargetSpatialGridCache();
        _target_spatial_grid_keys_storage.allocate(
            static_cast<std::size_t>(target_count) * sizeof(IcpGridCellKey));
        _target_spatial_grid_unique_keys_storage.allocate(
            static_cast<std::size_t>(target_count) * sizeof(IcpGridCellKey));
        _target_spatial_grid_indices_storage.allocate(
            static_cast<std::size_t>(target_count) * sizeof(int));
        _target_spatial_grid_cell_starts_storage.allocate(
            static_cast<std::size_t>(target_count) * sizeof(int));
        _target_spatial_grid_cell_counts_storage.allocate(
            static_cast<std::size_t>(target_count) * sizeof(int));
        _target_spatial_grid_capacity = target_count;
    }
}

void IcpCorrespondenceStatsWorkspace::invalidateTargetSpatialGridCache()
{
    _target_spatial_grid_cache_valid = false;
    _target_spatial_grid_points = nullptr;
    _target_spatial_grid_point_count = 0;
    _target_spatial_grid_cell_size = 0.0;
    _target_spatial_grid_cell_count = 0;
}

bool IcpCorrespondenceStatsWorkspace::targetSpatialGridCacheMatches(
    const void* target_points,
    int target_count,
    double cell_size) const
{
    return _target_spatial_grid_cache_valid &&
        _target_spatial_grid_points == target_points &&
        _target_spatial_grid_point_count == target_count &&
        _target_spatial_grid_cell_size == cell_size;
}

void IcpCorrespondenceStatsWorkspace::markTargetSpatialGridCache(
    const void* target_points,
    int target_count,
    double cell_size,
    int cell_count)
{
    _target_spatial_grid_cache_valid = true;
    _target_spatial_grid_points = target_points;
    _target_spatial_grid_point_count = target_count;
    _target_spatial_grid_cell_size = cell_size;
    _target_spatial_grid_cell_count = cell_count;
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

IcpResidualStats<float> computeIcpResidualStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    return computeIcpResidualStatsColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        workspace,
        stream);
}

IcpResidualStats<double> computeIcpResidualStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    return computeIcpResidualStatsColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        workspace,
        stream);
}

IcpResidualStats<float> transformPointsAndComputeIcpResidualStatsColumnMajor(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    return transformPointsAndComputeIcpResidualStatsColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
        stream);
}

IcpResidualStats<double> transformPointsAndComputeIcpResidualStatsColumnMajor(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream)
{
    return transformPointsAndComputeIcpResidualStatsColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
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

IcpAlignmentStepResult<float> computeIcpAlignmentStepColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream)
{
    return computeIcpAlignmentStepColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        stream);
}

IcpAlignmentStepResult<double> computeIcpAlignmentStepColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream)
{
    return computeIcpAlignmentStepColumnMajorImpl(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
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
