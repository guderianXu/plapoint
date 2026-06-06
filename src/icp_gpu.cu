#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
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
    __host__ __device__ __forceinline__ bool operator()(const IcpGridCellKey& lhs, const IcpGridCellKey& rhs) const
    {
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        return lhs.z < rhs.z;
    }
};

struct IcpGridCellKeyEqual
{
    __host__ __device__ __forceinline__ bool operator()(const IcpGridCellKey& lhs, const IcpGridCellKey& rhs) const
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

struct IcpTargetSpatialGrid
{
    const void* target_points = nullptr;
    int target_count = 0;
    const IcpGridCellKey* __restrict__ cell_keys = nullptr;
    const int* __restrict__ sorted_target_indices = nullptr;
    const void* __restrict__ sorted_target_x = nullptr;
    const void* __restrict__ sorted_target_y = nullptr;
    const void* __restrict__ sorted_target_z = nullptr;
    const int* __restrict__ cell_starts = nullptr;
    const int* __restrict__ cell_counts = nullptr;
    int cell_count = 0;
    double cell_size = 0.0;
    bool finite_cell_bounds = false;
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

constexpr unsigned int kIcpAlignmentStepSrcNonCollinearFlag = 1u << 0;
constexpr unsigned int kIcpAlignmentStepTgtNonCollinearFlag = 1u << 1;
constexpr unsigned int kIcpAlignmentStepValidFlag = 1u << 2;

template <typename Scalar>
struct IcpAlignmentStepRawResult
{
    double residual_sq_sum;
    Scalar delta;
    int active_count;
    int invalid_source_count;
    unsigned int flags;
};

struct IcpStatsAndStepRawResult
{
    RawIcpStats stats;
    IcpStepTransformRawResult step;
};

static_assert(sizeof(RawIcpStats) >= sizeof(RawIcpResidualStats),
              "ICP alignment-step partial workspace must cover residual-stats partials");
static_assert(sizeof(IcpAlignmentStepRawResult<float>) >= sizeof(RawIcpResidualStats),
              "ICP alignment-step result workspace must cover residual-stats results");
static_assert(sizeof(IcpAlignmentStepRawResult<float>) <= 24,
              "Float ICP alignment-step host result should use a float-sized delta");
static_assert(sizeof(IcpAlignmentStepRawResult<double>) <= 32,
              "ICP alignment-step host result should stay compact");
static_assert(offsetof(IcpStatsAndStepRawResult, stats) == 0,
              "ICP stats-step result must expose RawIcpStats at the start of the storage");

constexpr int kIcpStatsBlockSize = 128;
constexpr int kIcpTransform3x4ValueCount = 12;

template <typename Scalar>
__device__ __forceinline__ void computeStepTransformFromInput(
    const IcpStepTransformInput& input,
    Scalar* __restrict__ step_transform,
    IcpStepTransformRawResult* __restrict__ result);

template <typename Scalar>
__device__ __forceinline__ void computeStepTransformFromRawStatsValue(
    const RawIcpStats& raw,
    Scalar* __restrict__ step_transform,
    IcpStepTransformRawResult* __restrict__ result);

template <typename Scalar>
__device__ __forceinline__ void writeAlignmentStepRawResultFromRawStats(
    const RawIcpStats& raw,
    Scalar* __restrict__ step_transform,
    IcpAlignmentStepRawResult<Scalar>* __restrict__ result);

template <typename Scalar>
__device__ __forceinline__ void writeExactPointwiseStatsAndStepRawResult(
    const RawIcpStats& raw,
    IcpStatsAndStepRawResult* __restrict__ result);

template <typename Scalar>
__device__ __forceinline__ void multiplyTransform4x4SingleThread(
    const Scalar* __restrict__ A,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ C);

__device__ __forceinline__ bool rawStatsCovarianceHasNonCollinearGeometry(
    const double sum[3],
    const double outer_sum[9],
    int active_count);

#ifdef PLAPOINT_ENABLE_TESTING
std::atomic<int> g_icp_correspondence_stats_call_count{0};
std::atomic<int> g_icp_residual_stats_call_count{0};
std::atomic<std::uintptr_t> g_icp_first_stats_source_pointer{0};
std::atomic<int> g_icp_step_transform_input_copy_count{0};
std::atomic<int> g_icp_exact_pointwise_step_call_count{0};
std::atomic<int> g_icp_transformed_exact_pointwise_alignment_step_call_count{0};
std::atomic<int> g_icp_raw_stats_step_kernel_launch_count{0};
std::atomic<int> g_icp_stats_step_host_result_copy_count{0};
std::atomic<int> g_icp_alignment_step_call_count{0};
std::atomic<int> g_icp_transformed_alignment_step_call_count{0};
std::atomic<int> g_icp_accumulated_alignment_step_call_count{0};
std::atomic<int> g_icp_transformed_exact_pointwise_residual_call_count{0};
std::atomic<int> g_icp_alignment_step_reserve_count{0};
std::atomic<int> g_icp_alignment_step_reserve_check_count{0};
std::atomic<int> g_icp_residual_stats_reserve_check_count{0};
std::atomic<int> g_icp_host_synchronization_count{0};
std::atomic<int> g_icp_target_spatial_grid_build_count{0};
std::atomic<int> g_icp_fallback_tile_bound_kernel_launch_count{0};
std::atomic<int> g_icp_fallback_unbounded_kernel_launch_count{0};
std::atomic<std::uintptr_t> g_icp_last_transform_output_pointer{0};
std::atomic<int> g_icp_transform_points_call_count{0};
std::atomic<int> g_icp_transform_multiply_call_count{0};
std::atomic<int> g_icp_identity_transform_write_count{0};
std::atomic<int> g_icp_target_spatial_grid_prepare_count{0};
std::atomic<int> g_icp_host_result_storage_allocation_count{0};
__device__ unsigned long long g_icp_full_distance_evaluation_count;
__device__ unsigned long long g_icp_target_candidate_visit_count;
__device__ unsigned long long g_icp_target_index_load_count;
__device__ unsigned long long g_icp_sorted_target_coordinate_load_count;
__device__ unsigned long long g_icp_target_tile_bound_computation_count;
__device__ unsigned long long g_icp_target_tile_load_count;
__device__ unsigned long long g_icp_exact_pointwise_target_load_count;
__device__ unsigned long long g_icp_grid_cell_lookup_count;
#endif

#ifdef PLAPOINT_ENABLE_TESTING
void recordFallbackKernelLaunchForTesting(bool uses_target_tile_bounds)
{
    if (uses_target_tile_bounds)
    {
        g_icp_fallback_tile_bound_kernel_launch_count.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        g_icp_fallback_unbounded_kernel_launch_count.fetch_add(1, std::memory_order_relaxed);
    }
}

void recordTransformedExactPointwiseResidualCallForTesting(bool enabled)
{
    if (enabled)
    {
        g_icp_transformed_exact_pointwise_residual_call_count.fetch_add(1, std::memory_order_relaxed);
    }
}
#endif

void synchronizeIcpStreamWithHost(cudaStream_t stream)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
}

int icpStatsPartialCount(int source_count)
{
    return (source_count + kIcpStatsBlockSize - 1) / kIcpStatsBlockSize;
}

int icpTargetTileCount(int target_count)
{
    return (target_count + kIcpStatsBlockSize - 1) / kIcpStatsBlockSize;
}

bool icpGridCellBoundsAreFinite(double cell_size)
{
    constexpr double max_abs_grid_coordinate = static_cast<double>(INT_MAX) + 2.0;
    return std::isfinite(cell_size) &&
        cell_size > 0.0 &&
        cell_size <= std::numeric_limits<double>::max() / max_abs_grid_coordinate;
}

__device__ __forceinline__ void addRawIcpStats(RawIcpStats& dst, const RawIcpStats& src)
{
    dst.active_count += src.active_count;
    dst.invalid_source_count += src.invalid_source_count;
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        dst.src_sum[c] += src.src_sum[c];
        dst.tgt_sum[c] += src.tgt_sum[c];
    }
#pragma unroll
    for (int idx = 0; idx < 9; ++idx)
    {
        dst.cross_sum[idx] += src.cross_sum[idx];
        dst.src_outer_sum[idx] += src.src_outer_sum[idx];
        dst.tgt_outer_sum[idx] += src.tgt_outer_sum[idx];
    }
    dst.residual_sq_sum += src.residual_sq_sum;
}

__device__ __forceinline__ void addRawIcpResidualStats(RawIcpResidualStats& dst, const RawIcpResidualStats& src)
{
    dst.active_count += src.active_count;
    dst.invalid_source_count += src.invalid_source_count;
    dst.residual_sq_sum += src.residual_sq_sum;
}

template <typename Scalar>
__device__ __forceinline__ void writeAlignmentStepRawResultFields(
    const RawIcpStats& raw,
    IcpAlignmentStepRawResult<Scalar>* __restrict__ result,
    int step_valid,
    double delta)
{
    result->residual_sq_sum = raw.residual_sq_sum;
    result->delta = static_cast<Scalar>(delta);
    result->active_count = raw.active_count;
    result->invalid_source_count = raw.invalid_source_count;
    unsigned int flags = 0;
    if (rawStatsCovarianceHasNonCollinearGeometry(raw.src_sum, raw.src_outer_sum, raw.active_count))
    {
        flags |= kIcpAlignmentStepSrcNonCollinearFlag;
    }
    if (rawStatsCovarianceHasNonCollinearGeometry(raw.tgt_sum, raw.tgt_outer_sum, raw.active_count))
    {
        flags |= kIcpAlignmentStepTgtNonCollinearFlag;
    }
    if (step_valid != 0)
    {
        flags |= kIcpAlignmentStepValidFlag;
    }
    result->flags = flags;
}

template <typename Value>
__host__ __device__ __forceinline__ Value loadReadOnlyIcpValue(const Value* __restrict__ value)
{
#if defined(__CUDA_ARCH__)
    return __ldg(value);
#else
    return *value;
#endif
}

__device__ __forceinline__ IcpTargetTileBounds loadIcpTargetTileBounds(
    const IcpTargetTileBounds* __restrict__ target_tile_bounds,
    int tile_idx)
{
    const IcpTargetTileBounds* __restrict__ bounds = target_tile_bounds + tile_idx;
    return IcpTargetTileBounds{
        loadReadOnlyIcpValue(&bounds->min_x),
        loadReadOnlyIcpValue(&bounds->min_y),
        loadReadOnlyIcpValue(&bounds->min_z),
        loadReadOnlyIcpValue(&bounds->max_x),
        loadReadOnlyIcpValue(&bounds->max_y),
        loadReadOnlyIcpValue(&bounds->max_z),
        loadReadOnlyIcpValue(&bounds->has_valid_point)
    };
}

template <typename Scalar>
__device__ __forceinline__ void transformColumnMajorPoint3x4(
    const Scalar* transform_values,
    Scalar px,
    Scalar py,
    Scalar pz,
    Scalar& ox,
    Scalar& oy,
    Scalar& oz)
{
    const Scalar t00 = transform_values[0];
    const Scalar t10 = transform_values[1];
    const Scalar t20 = transform_values[2];
    const Scalar t01 = transform_values[3];
    const Scalar t11 = transform_values[4];
    const Scalar t21 = transform_values[5];
    const Scalar t02 = transform_values[6];
    const Scalar t12 = transform_values[7];
    const Scalar t22 = transform_values[8];
    const Scalar t03 = transform_values[9];
    const Scalar t13 = transform_values[10];
    const Scalar t23 = transform_values[11];

    ox = t00 * px + t01 * py + t02 * pz + t03;
    oy = t10 * px + t11 * py + t12 * pz + t13;
    oz = t20 * px + t21 * py + t22 * pz + t23;
}

__device__ __forceinline__ int columnMajorTransform3x4Offset(int packed_idx)
{
    return packed_idx + packed_idx / 3;
}

template <typename Scalar>
__device__ __forceinline__ void loadColumnMajorTransform3x4Block(
    const Scalar* __restrict__ transform,
    Scalar* transform_values,
    int local_idx)
{
    if (local_idx < kIcpTransform3x4ValueCount)
    {
        transform_values[local_idx] = loadReadOnlyIcpValue(
            transform + columnMajorTransform3x4Offset(local_idx));
    }
    __syncthreads();
}

template <typename Scalar>
__device__ __forceinline__ bool loadFiniteColumnMajorPoint(
    const Scalar* __restrict__ points,
    int point_count,
    int idx,
    double& x,
    double& y,
    double& z)
{
    x = static_cast<double>(loadReadOnlyIcpValue(points + idx));
    y = static_cast<double>(loadReadOnlyIcpValue(points + point_count + idx));
    z = static_cast<double>(loadReadOnlyIcpValue(points + 2 * point_count + idx));
    return isfinite(x) && isfinite(y) && isfinite(z);
}

template <typename Scalar>
__device__ __forceinline__ bool loadFiniteTransformedColumnMajorPoint(
    const Scalar* __restrict__ points,
    int point_count,
    int idx,
    const Scalar* __restrict__ transform_values,
    double& x,
    double& y,
    double& z)
{
    const Scalar px = loadReadOnlyIcpValue(points + idx);
    const Scalar py = loadReadOnlyIcpValue(points + point_count + idx);
    const Scalar pz = loadReadOnlyIcpValue(points + 2 * point_count + idx);

    Scalar ox = Scalar(0);
    Scalar oy = Scalar(0);
    Scalar oz = Scalar(0);
    transformColumnMajorPoint3x4(transform_values, px, py, pz, ox, oy, oz);

    x = static_cast<double>(ox);
    y = static_cast<double>(oy);
    z = static_cast<double>(oz);
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__host__ __device__ __forceinline__ int icpGridCellCoordinate(double value, double cell_size)
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

__device__ __forceinline__ bool offsetGridCellCoordinate(int base, int offset, int& result)
{
    if ((offset < 0 && base == INT_MIN) || (offset > 0 && base == INT_MAX))
    {
        return false;
    }
    result = base + offset;
    return true;
}

__device__ __forceinline__ int icpNeighborCellOffset(int offset_index)
{
    return offset_index == 0 ? 0 : (offset_index == 1 ? -1 : 1);
}

__device__ __forceinline__ IcpGridCellKey loadIcpGridCellKey(
    const IcpGridCellKey* __restrict__ cell_keys,
    int cell_idx)
{
    const IcpGridCellKey* __restrict__ key = cell_keys + cell_idx;
    return {
        loadReadOnlyIcpValue(&key->x),
        loadReadOnlyIcpValue(&key->y),
        loadReadOnlyIcpValue(&key->z)
    };
}

template <typename Scalar>
struct ComputeIcpTargetGridCellKey
{
    const Scalar* __restrict__ points;
    int point_count;
    double cell_size;

    __host__ __device__ __forceinline__ IcpGridCellKey operator()(int idx) const
    {
        const double x = static_cast<double>(loadReadOnlyIcpValue(points + idx));
        const double y = static_cast<double>(loadReadOnlyIcpValue(points + point_count + idx));
        const double z = static_cast<double>(loadReadOnlyIcpValue(points + 2 * point_count + idx));
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

template <typename Scalar>
__global__ void gatherSortedIcpTargetPointsKernel(
    const Scalar* __restrict__ target_points,
    int target_count,
    const int* __restrict__ sorted_target_indices,
    Scalar* __restrict__ sorted_x,
    Scalar* __restrict__ sorted_y,
    Scalar* __restrict__ sorted_z)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= target_count)
    {
        return;
    }

    const int target_idx = loadReadOnlyIcpValue(sorted_target_indices + idx);
    sorted_x[idx] = loadReadOnlyIcpValue(target_points + target_idx);
    sorted_y[idx] = loadReadOnlyIcpValue(target_points + target_count + target_idx);
    sorted_z[idx] = loadReadOnlyIcpValue(target_points + 2 * target_count + target_idx);
}

__device__ __forceinline__ int lowerBoundIcpGridCell(
    const IcpGridCellKey* __restrict__ cell_keys,
    int cell_count,
    const IcpGridCellKey& query)
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
        if (less(loadIcpGridCellKey(cell_keys, mid), query))
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

__device__ __forceinline__ int loadSortedIcpTargetIndex(
    const IcpTargetSpatialGrid& target_grid,
    int sorted_offset)
{
#ifdef PLAPOINT_ENABLE_TESTING
    atomicAdd(&g_icp_target_index_load_count, 1ull);
#endif
    return loadReadOnlyIcpValue(target_grid.sorted_target_indices + sorted_offset);
}

__device__ __forceinline__ int loadIcpGridCellStart(const IcpTargetSpatialGrid& target_grid, int cell_idx)
{
    return loadReadOnlyIcpValue(target_grid.cell_starts + cell_idx);
}

__device__ __forceinline__ int loadIcpGridCellCount(const IcpTargetSpatialGrid& target_grid, int cell_idx)
{
    return loadReadOnlyIcpValue(target_grid.cell_counts + cell_idx);
}

template <typename Scalar>
__device__ __forceinline__ double loadSortedIcpTargetX(const IcpTargetSpatialGrid& target_grid, int sorted_offset)
{
#ifdef PLAPOINT_ENABLE_TESTING
    atomicAdd(&g_icp_sorted_target_coordinate_load_count, 1ull);
#endif
    const auto* sorted_x = reinterpret_cast<const Scalar*>(target_grid.sorted_target_x);
    return static_cast<double>(loadReadOnlyIcpValue(sorted_x + sorted_offset));
}

template <typename Scalar>
__device__ __forceinline__ double loadSortedIcpTargetY(const IcpTargetSpatialGrid& target_grid, int sorted_offset)
{
#ifdef PLAPOINT_ENABLE_TESTING
    atomicAdd(&g_icp_sorted_target_coordinate_load_count, 1ull);
#endif
    const auto* sorted_y = reinterpret_cast<const Scalar*>(target_grid.sorted_target_y);
    return static_cast<double>(loadReadOnlyIcpValue(sorted_y + sorted_offset));
}

template <typename Scalar>
__device__ __forceinline__ double loadSortedIcpTargetZ(const IcpTargetSpatialGrid& target_grid, int sorted_offset)
{
#ifdef PLAPOINT_ENABLE_TESTING
    atomicAdd(&g_icp_sorted_target_coordinate_load_count, 1ull);
#endif
    const auto* sorted_z = reinterpret_cast<const Scalar*>(target_grid.sorted_target_z);
    return static_cast<double>(loadReadOnlyIcpValue(sorted_z + sorted_offset));
}

__device__ __forceinline__ double distanceOutsideIcpGridCellAxis(double value, int cell_coordinate, double cell_size)
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

__device__ __forceinline__ double distanceOutsideFiniteIcpGridCellAxis(
    double value,
    int cell_coordinate,
    double cell_size)
{
    const double cell_min = static_cast<double>(cell_coordinate) * cell_size;
    const double cell_max = cell_min + cell_size;
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

__device__ __forceinline__ double minDistanceSqToIcpGridCellXY(
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

__device__ __forceinline__ double minDistanceSqToFiniteIcpGridCellXY(
    double x,
    double y,
    int cell_x,
    int cell_y,
    double cell_size)
{
    const double dx = distanceOutsideFiniteIcpGridCellAxis(x, cell_x, cell_size);
    const double dy = distanceOutsideFiniteIcpGridCellAxis(y, cell_y, cell_size);
    return dx * dx + dy * dy;
}

template <bool FiniteCellBounds>
__device__ __forceinline__ double minDistanceSqToIcpGridCellXY(
    double x,
    double y,
    int cell_x,
    int cell_y,
    double cell_size)
{
    if constexpr (FiniteCellBounds)
    {
        return minDistanceSqToFiniteIcpGridCellXY(x, y, cell_x, cell_y, cell_size);
    }
    return minDistanceSqToIcpGridCellXY(x, y, cell_x, cell_y, cell_size);
}

__device__ __forceinline__ double minDistanceSqToIcpGridCellZ(
    double z,
    int cell_z,
    double cell_size)
{
    const double dz = distanceOutsideIcpGridCellAxis(z, cell_z, cell_size);
    return dz * dz;
}

__device__ __forceinline__ double minDistanceSqToFiniteIcpGridCellZ(
    double z,
    int cell_z,
    double cell_size)
{
    const double dz = distanceOutsideFiniteIcpGridCellAxis(z, cell_z, cell_size);
    return dz * dz;
}

template <bool FiniteCellBounds>
__device__ __forceinline__ double minDistanceSqToIcpGridCellZ(
    double z,
    int cell_z,
    double cell_size)
{
    if constexpr (FiniteCellBounds)
    {
        return minDistanceSqToFiniteIcpGridCellZ(z, cell_z, cell_size);
    }
    return minDistanceSqToIcpGridCellZ(z, cell_z, cell_size);
}

__device__ __forceinline__ void recordAcceptedCorrespondence(
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
#pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        local.src_sum[r] = source_values[r];
        local.tgt_sum[r] = target_values[r];
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            local.cross_sum[r * 3 + c] = source_values[r] * target_values[c];
            local.src_outer_sum[r * 3 + c] = source_values[r] * source_values[c];
            local.tgt_outer_sum[r * 3 + c] = target_values[r] * target_values[c];
        }
    }
}

template <typename Scalar>
__device__ __forceinline__ bool tryAcceptTransformedExactPointwiseResidual(
    int source_idx,
    int source_count,
    const void* target_points_raw,
    int target_count,
    double sx,
    double sy,
    double sz,
    RawIcpResidualStats& local)
{
    if (!target_points_raw || source_count != target_count)
    {
        return false;
    }

    const auto* target_points = static_cast<const Scalar*>(target_points_raw);
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    if (!loadFiniteColumnMajorPoint(target_points, target_count, source_idx, tx, ty, tz))
    {
        return false;
    }
#ifdef PLAPOINT_ENABLE_TESTING
    atomicAdd(&g_icp_exact_pointwise_target_load_count, 3ull);
#endif

    if (sx != tx || sy != ty || sz != tz)
    {
        return false;
    }

    local.active_count = 1;
    local.residual_sq_sum = 0.0;
    return true;
}

template <typename Scalar>
__device__ __forceinline__ bool tryAcceptTransformedExactPointwiseCorrespondence(
    int source_idx,
    int source_count,
    const void* target_points_raw,
    int target_count,
    double sx,
    double sy,
    double sz,
    RawIcpStats& local)
{
    if (!target_points_raw || source_count != target_count)
    {
        return false;
    }

    const auto* target_points = static_cast<const Scalar*>(target_points_raw);
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    if (!loadFiniteColumnMajorPoint(target_points, target_count, source_idx, tx, ty, tz))
    {
        return false;
    }
#ifdef PLAPOINT_ENABLE_TESTING
    atomicAdd(&g_icp_exact_pointwise_target_load_count, 3ull);
#endif

    if (sx != tx || sy != ty || sz != tz)
    {
        return false;
    }

    recordAcceptedCorrespondence(local, sx, sy, sz, tx, ty, tz, 0.0);
    return true;
}

template <typename Scalar>
__global__ void computeTargetTileBoundsKernel(
    const Scalar* __restrict__ target_points,
    int target_count,
    IcpTargetTileBounds* __restrict__ target_tile_bounds)
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

template <
    typename Scalar,
    bool WriteCorrespondenceIndices,
    bool UseTargetTileBounds,
    bool TransformSource,
    bool TryTransformedExactPointwise>
__global__ void collectCorrespondenceStatsKernel(
    const Scalar* __restrict__ source_transform,
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    int* __restrict__ correspondence_indices,
    const IcpTargetTileBounds* __restrict__ target_tile_bounds,
    RawIcpStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    if constexpr (TransformSource)
    {
        loadColumnMajorTransform3x4Block(source_transform, block_transform, local_idx);
    }

    if (source_idx < source_count)
    {
        bool loaded_source = false;
        if constexpr (TransformSource)
        {
            loaded_source = loadFiniteTransformedColumnMajorPoint(
                source_points,
                source_count,
                source_idx,
                block_transform,
                sx,
                sy,
                sz);
        }
        else
        {
            loaded_source = loadFiniteColumnMajorPoint(source_points, source_count, source_idx, sx, sy, sz);
        }
        if (!loaded_source)
        {
            if constexpr (WriteCorrespondenceIndices)
            {
                correspondence_indices[source_idx] = -1;
            }
            local.invalid_source_count = 1;
        }
        else
        {
            source_valid = true;
            if constexpr (TransformSource && TryTransformedExactPointwise)
            {
                if (tryAcceptTransformedExactPointwiseCorrespondence<Scalar>(
                        source_idx,
                        source_count,
                        target_points,
                        target_count,
                        sx,
                        sy,
                        sz,
                        local))
                {
                    source_valid = false;
                }
            }
        }
    }

    int best_idx = -1;
    double best_dist_sq = INFINITY;
    double best_tx = 0.0;
    double best_ty = 0.0;
    double best_tz = 0.0;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const double max_dist_sq = max_dist * max_dist;
    bool stop_target_scan = false;
    __shared__ double target_tile_x[kIcpStatsBlockSize];
    __shared__ double target_tile_y[kIcpStatsBlockSize];
    __shared__ double target_tile_z[kIcpStatsBlockSize];
    __shared__ int target_tile_valid[kIcpStatsBlockSize];

    const int scan_target_count = __syncthreads_count(source_valid) > 0 ? target_count : 0;
    for (int tile_start = 0; tile_start < scan_target_count; tile_start += kIcpStatsBlockSize)
    {
        const int tile_idx = tile_start / kIcpStatsBlockSize;
        bool tile_relevant = true;
        if constexpr (UseTargetTileBounds)
        {
            tile_relevant = false;
            if (source_valid && !stop_target_scan)
            {
                const IcpTargetTileBounds bounds = loadIcpTargetTileBounds(target_tile_bounds, tile_idx);
                tile_relevant =
                    bounds.has_valid_point &&
                    sx >= bounds.min_x - max_dist && sx <= bounds.max_x + max_dist &&
                    sy >= bounds.min_y - max_dist && sy <= bounds.max_y + max_dist &&
                    sz >= bounds.min_z - max_dist && sz <= bounds.max_z + max_dist;
            }
            const int relevant_source_count = __syncthreads_count(tile_relevant);
            if (relevant_source_count == 0)
            {
                continue;
            }
        }

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
#ifdef PLAPOINT_ENABLE_TESTING
        if (local_idx == 0)
        {
            atomicAdd(&g_icp_target_tile_load_count, 1ull);
        }
#endif
        __syncthreads();

        if (source_valid && !stop_target_scan && tile_relevant)
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

                const double tx = target_tile_x[tile_offset];
                const double dx = sx - tx;
                const double dx_sq = dx * dx;
                if constexpr (UseTargetTileBounds)
                {
                    if (dx_sq > max_dist_sq)
                    {
                        continue;
                    }
                }
                const double ty = target_tile_y[tile_offset];
                const double dy = sy - ty;
                const double dy_sq = dy * dy;
                const double xy_dist_sq = dx_sq + dy_sq;
                if constexpr (UseTargetTileBounds)
                {
                    if (xy_dist_sq > max_dist_sq)
                    {
                        continue;
                    }
                }
                const double tz = target_tile_z[tile_offset];
                const double dz = sz - tz;
                const double dz_sq = dz * dz;
                const double dist_sq = xy_dist_sq + dz_sq;
                if constexpr (UseTargetTileBounds)
                {
                    if (dist_sq > max_dist_sq)
                    {
                        continue;
                    }
                }
#ifdef PLAPOINT_ENABLE_TESTING
                atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
                if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                {
                    best_dist_sq = dist_sq;
                    best_idx = tile_start + tile_offset;
                    best_tx = tx;
                    best_ty = ty;
                    best_tz = tz;
                    if constexpr (!WriteCorrespondenceIndices)
                    {
                        if (dist_sq <= 0.0)
                        {
                            stop_target_scan = true;
                            break;
                        }
                    }
                }
            }
        }
        const int unfinished_source_count = __syncthreads_count(source_valid && !stop_target_scan);
        if (unfinished_source_count == 0)
        {
            break;
        }
    }

    if (source_valid)
    {
        bool accepted = best_idx >= 0;
        if (accepted && isfinite(max_dist))
        {
            accepted = best_dist_sq <= max_dist_sq;
        }

        if (!accepted)
        {
            if constexpr (WriteCorrespondenceIndices)
            {
                correspondence_indices[source_idx] = -1;
            }
        }
        else
        {
            if constexpr (WriteCorrespondenceIndices)
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

template <
    typename Scalar,
    bool FiniteCellBounds,
    bool WriteCorrespondenceIndices,
    bool TransformSource,
    bool TryTransformedExactPointwise>
__global__ void collectCorrespondenceStatsSpatialGridKernel(
    const Scalar* __restrict__ source_transform,
    const Scalar* __restrict__ source_points,
    int source_count,
    Scalar max_correspondence_distance,
    int* __restrict__ correspondence_indices,
    IcpTargetSpatialGrid target_grid,
    RawIcpStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    if constexpr (TransformSource)
    {
        loadColumnMajorTransform3x4Block(source_transform, block_transform, local_idx);
    }

    if (source_idx < source_count)
    {
        bool loaded_source = false;
        if constexpr (TransformSource)
        {
            loaded_source = loadFiniteTransformedColumnMajorPoint(
                source_points,
                source_count,
                source_idx,
                block_transform,
                sx,
                sy,
                sz);
        }
        else
        {
            loaded_source = loadFiniteColumnMajorPoint(source_points, source_count, source_idx, sx, sy, sz);
        }
        if (!loaded_source)
        {
            if constexpr (WriteCorrespondenceIndices)
            {
                correspondence_indices[source_idx] = -1;
            }
            local.invalid_source_count = 1;
        }
        else
        {
            source_valid = true;
            if constexpr (TransformSource && TryTransformedExactPointwise)
            {
                if (tryAcceptTransformedExactPointwiseCorrespondence<Scalar>(
                        source_idx,
                        source_count,
                        target_grid.target_points,
                        target_grid.target_count,
                        sx,
                        sy,
                        sz,
                        local))
                {
                    source_valid = false;
                }
            }
        }
    }

    int best_sorted_offset = -1;
    int best_idx = -1;
    double best_dist_sq = INFINITY;
    double best_tx = 0.0;
    double best_ty = 0.0;
    double best_tz = 0.0;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const double max_dist_sq = max_dist * max_dist;

    if (source_valid)
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

        constexpr bool can_stop_after_exact_match = !WriteCorrespondenceIndices;
        bool stop_cell_scan = false;
#pragma unroll
        for (int dx_offset_idx = 0; dx_offset_idx < 3; ++dx_offset_idx)
        {
            if (stop_cell_scan)
            {
                break;
            }
            int query_x = 0;
            const int dx_cell = icpNeighborCellOffset(dx_offset_idx);
            if (!offsetGridCellCoordinate(source_key.x, dx_cell, query_x))
            {
                continue;
            }
#pragma unroll
            for (int dy_offset_idx = 0; dy_offset_idx < 3; ++dy_offset_idx)
            {
                if (stop_cell_scan)
                {
                    break;
                }
                int query_y = 0;
                const int dy_cell = icpNeighborCellOffset(dy_offset_idx);
                if (!offsetGridCellCoordinate(source_key.y, dy_cell, query_y))
                {
                    continue;
                }

                const double min_xy_dist_sq = minDistanceSqToIcpGridCellXY<FiniteCellBounds>(
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
                    const IcpGridCellKey cell_key = loadIcpGridCellKey(target_grid.cell_keys, cell_idx);
                    if (cell_key.x != query_x || cell_key.y != query_y || cell_key.z > max_z)
                    {
                        break;
                    }

                    const double min_cell_dist_sq = min_xy_dist_sq + minDistanceSqToIcpGridCellZ<FiniteCellBounds>(
                        sz,
                        cell_key.z,
                        target_grid.cell_size);
                    if (min_cell_dist_sq > max_dist_sq || min_cell_dist_sq > best_dist_sq)
                    {
                        ++cell_idx;
                        continue;
                    }

                    const int start = loadIcpGridCellStart(target_grid, cell_idx);
                    const int count = loadIcpGridCellCount(target_grid, cell_idx);
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int sorted_offset = start + offset;
#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_target_candidate_visit_count, 1ull);
#endif
                        const double tx = loadSortedIcpTargetX<Scalar>(target_grid, sorted_offset);
                        const double dx = sx - tx;
                        const double dx_sq = dx * dx;
                        if (dx_sq > max_dist_sq)
                        {
                            continue;
                        }
                        if (dx_sq > best_dist_sq)
                        {
                            continue;
                        }

                        const double ty = loadSortedIcpTargetY<Scalar>(target_grid, sorted_offset);
                        const double dy = sy - ty;
                        const double dy_sq = dy * dy;
                        const double xy_dist_sq = dx_sq + dy_sq;
                        if (xy_dist_sq > max_dist_sq)
                        {
                            continue;
                        }
                        if (xy_dist_sq > best_dist_sq)
                        {
                            continue;
                        }

                        const double tz = loadSortedIcpTargetZ<Scalar>(target_grid, sorted_offset);
                        const double dz = sz - tz;
                        const double dz_sq = dz * dz;
                        const double dist_sq = xy_dist_sq + dz_sq;
                        if (dist_sq > max_dist_sq)
                        {
                            continue;
                        }
#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
                        if (isfinite(dist_sq) && dist_sq <= best_dist_sq)
                        {
                            bool update_best = dist_sq < best_dist_sq || best_sorted_offset < 0;
                            int target_idx = -1;
                            if (!update_best)
                            {
                                target_idx = loadSortedIcpTargetIndex(target_grid, sorted_offset);
                                if (best_idx < 0)
                                {
                                    best_idx = loadSortedIcpTargetIndex(target_grid, best_sorted_offset);
                                }
                                update_best = target_idx < best_idx;
                            }
                            if (update_best)
                            {
                                best_dist_sq = dist_sq;
                                best_sorted_offset = sorted_offset;
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
                    }
                    ++cell_idx;
                }
            }
        }
    }

    if (source_valid)
    {
        bool accepted = best_sorted_offset >= 0;
        if (accepted)
        {
            accepted = best_dist_sq <= max_dist_sq;
        }

        if (!accepted)
        {
            if constexpr (WriteCorrespondenceIndices)
            {
                correspondence_indices[source_idx] = -1;
            }
        }
        else
        {
            if constexpr (WriteCorrespondenceIndices)
            {
                if (best_idx < 0)
                {
                    best_idx = loadSortedIcpTargetIndex(target_grid, best_sorted_offset);
                }
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

template <typename Scalar, bool SameBuffer>
__global__ void collectExactPointwiseCorrespondenceStatsKernel(
    const Scalar* source_points,
    const Scalar* target_points,
    int point_count,
    RawIcpStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};

    if (source_idx < point_count)
    {
        const Scalar raw_sx = loadReadOnlyIcpValue(source_points + source_idx);
        const Scalar raw_sy = loadReadOnlyIcpValue(source_points + point_count + source_idx);
        const Scalar raw_sz = loadReadOnlyIcpValue(source_points + 2 * point_count + source_idx);
        Scalar raw_tx = raw_sx;
        Scalar raw_ty = raw_sy;
        Scalar raw_tz = raw_sz;
        if constexpr (!SameBuffer)
        {
            raw_tx = loadReadOnlyIcpValue(target_points + source_idx);
            raw_ty = loadReadOnlyIcpValue(target_points + point_count + source_idx);
            raw_tz = loadReadOnlyIcpValue(target_points + 2 * point_count + source_idx);
#ifdef PLAPOINT_ENABLE_TESTING
            atomicAdd(&g_icp_exact_pointwise_target_load_count, 3ull);
#endif
        }

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

template <typename Scalar, bool SameBuffer>
__global__ void collectOrderedPointwiseCorrespondenceStatsKernel(
    const Scalar* source_points,
    const Scalar* target_points,
    int point_count,
    Scalar max_correspondence_distance,
    RawIcpStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};

    if (source_idx < point_count)
    {
        const Scalar raw_sx = loadReadOnlyIcpValue(source_points + source_idx);
        const Scalar raw_sy = loadReadOnlyIcpValue(source_points + point_count + source_idx);
        const Scalar raw_sz = loadReadOnlyIcpValue(source_points + 2 * point_count + source_idx);
        Scalar raw_tx = raw_sx;
        Scalar raw_ty = raw_sy;
        Scalar raw_tz = raw_sz;
        if constexpr (!SameBuffer)
        {
            raw_tx = loadReadOnlyIcpValue(target_points + source_idx);
            raw_ty = loadReadOnlyIcpValue(target_points + point_count + source_idx);
            raw_tz = loadReadOnlyIcpValue(target_points + 2 * point_count + source_idx);
#ifdef PLAPOINT_ENABLE_TESTING
            atomicAdd(&g_icp_exact_pointwise_target_load_count, 3ull);
#endif
        }

        const double sx = static_cast<double>(raw_sx);
        const double sy = static_cast<double>(raw_sy);
        const double sz = static_cast<double>(raw_sz);
        const double tx = static_cast<double>(raw_tx);
        const double ty = static_cast<double>(raw_ty);
        const double tz = static_cast<double>(raw_tz);
        if (!isfinite(sx) || !isfinite(sy) || !isfinite(sz))
        {
            local.invalid_source_count = 1;
        }
        else if (isfinite(tx) && isfinite(ty) && isfinite(tz))
        {
            const double dx = sx - tx;
            const double dy = sy - ty;
            const double dz = sz - tz;
            const double dist_sq = dx * dx + dy * dy + dz * dz;
            const double max_dist = static_cast<double>(max_correspondence_distance);
            if (isfinite(dist_sq) &&
                (!isfinite(max_dist) || dist_sq <= max_dist * max_dist))
            {
                recordAcceptedCorrespondence(local, sx, sy, sz, tx, ty, tz, dist_sq);
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
__global__ void collectTransformedExactPointwiseCorrespondenceStatsKernel(
    const Scalar* source_transform,
    const Scalar* source_points,
    int point_count,
    const Scalar* target_points,
    RawIcpStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    loadColumnMajorTransform3x4Block(source_transform, block_transform, local_idx);

    if (source_idx < point_count)
    {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        if (!loadFiniteTransformedColumnMajorPoint(
                source_points,
                point_count,
                source_idx,
                block_transform,
                sx,
                sy,
                sz))
        {
            local.invalid_source_count = 1;
        }
        else if (!tryAcceptTransformedExactPointwiseCorrespondence<Scalar>(
                     source_idx,
                     point_count,
                     target_points,
                     point_count,
                     sx,
                     sy,
                     sz,
                     local))
        {
            local.residual_sq_sum = INFINITY;
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
__global__ void transformAndCollectOrderedCorrespondenceStatsKernel(
    const Scalar* source_transform,
    const Scalar* source_points,
    int point_count,
    const Scalar* target_points,
    Scalar max_correspondence_distance,
    RawIcpStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpStats local{};
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    loadColumnMajorTransform3x4Block(source_transform, block_transform, local_idx);

    if (source_idx < point_count)
    {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        if (!loadFiniteTransformedColumnMajorPoint(
                source_points,
                point_count,
                source_idx,
                block_transform,
                sx,
                sy,
                sz))
        {
            local.invalid_source_count = 1;
        }
        else
        {
            double tx = 0.0;
            double ty = 0.0;
            double tz = 0.0;
            if (loadFiniteColumnMajorPoint(target_points, point_count, source_idx, tx, ty, tz))
            {
#ifdef PLAPOINT_ENABLE_TESTING
                atomicAdd(&g_icp_exact_pointwise_target_load_count, 3ull);
#endif
                const double dx = sx - tx;
                const double dy = sy - ty;
                const double dz = sz - tz;
                const double dist_sq = dx * dx + dy * dy + dz * dz;
                const double max_dist = static_cast<double>(max_correspondence_distance);
                if (isfinite(dist_sq) &&
                    (!isfinite(max_dist) || dist_sq <= max_dist * max_dist))
                {
                    recordAcceptedCorrespondence(local, sx, sy, sz, tx, ty, tz, dist_sq);
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

template <typename Scalar>
__global__ void collectTransformedExactPointwiseResidualStatsKernel(
    const Scalar* source_transform,
    const Scalar* source_points,
    int point_count,
    const Scalar* target_points,
    Scalar* output_points,
    RawIcpResidualStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    loadColumnMajorTransform3x4Block(source_transform, block_transform, local_idx);

    if (source_idx < point_count)
    {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        const bool source_valid = loadFiniteTransformedColumnMajorPoint(
            source_points,
            point_count,
            source_idx,
            block_transform,
            sx,
            sy,
            sz);
        if (output_points)
        {
            output_points[source_idx] = static_cast<Scalar>(sx);
            output_points[point_count + source_idx] = static_cast<Scalar>(sy);
            output_points[2 * point_count + source_idx] = static_cast<Scalar>(sz);
        }

        if (!source_valid)
        {
            local.invalid_source_count = 1;
        }
        else if (!tryAcceptTransformedExactPointwiseResidual<Scalar>(
                     source_idx,
                     point_count,
                     target_points,
                     point_count,
                     sx,
                     sy,
                     sz,
                     local))
        {
            local.residual_sq_sum = INFINITY;
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

template <typename Scalar, bool SameBuffer>
__global__ void collectExactPointwiseResidualStatsKernel(
    const Scalar* source_points,
    const Scalar* target_points,
    int point_count,
    RawIcpResidualStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};

    if (source_idx < point_count)
    {
        const Scalar raw_sx = loadReadOnlyIcpValue(source_points + source_idx);
        const Scalar raw_sy = loadReadOnlyIcpValue(source_points + point_count + source_idx);
        const Scalar raw_sz = loadReadOnlyIcpValue(source_points + 2 * point_count + source_idx);
        Scalar raw_tx = raw_sx;
        Scalar raw_ty = raw_sy;
        Scalar raw_tz = raw_sz;
        if constexpr (!SameBuffer)
        {
            raw_tx = loadReadOnlyIcpValue(target_points + source_idx);
            raw_ty = loadReadOnlyIcpValue(target_points + point_count + source_idx);
            raw_tz = loadReadOnlyIcpValue(target_points + 2 * point_count + source_idx);
#ifdef PLAPOINT_ENABLE_TESTING
            atomicAdd(&g_icp_exact_pointwise_target_load_count, 3ull);
#endif
        }

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
                local.active_count = 1;
            }
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

template <typename Scalar, bool UseTargetTileBounds>
__global__ void collectResidualStatsKernel(
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const IcpTargetTileBounds* __restrict__ target_tile_bounds,
    RawIcpResidualStats* __restrict__ partial_stats)
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
    bool stop_target_scan = false;
    __shared__ double target_tile_x[kIcpStatsBlockSize];
    __shared__ double target_tile_y[kIcpStatsBlockSize];
    __shared__ double target_tile_z[kIcpStatsBlockSize];
    __shared__ int target_tile_valid[kIcpStatsBlockSize];

    const int scan_target_count = __syncthreads_count(source_valid) > 0 ? target_count : 0;
    for (int tile_start = 0; tile_start < scan_target_count; tile_start += kIcpStatsBlockSize)
    {
        const int tile_idx = tile_start / kIcpStatsBlockSize;
        bool tile_relevant = true;
        if constexpr (UseTargetTileBounds)
        {
            tile_relevant = false;
            if (source_valid && !stop_target_scan)
            {
                const IcpTargetTileBounds bounds = loadIcpTargetTileBounds(target_tile_bounds, tile_idx);
                tile_relevant =
                    bounds.has_valid_point &&
                    sx >= bounds.min_x - max_dist && sx <= bounds.max_x + max_dist &&
                    sy >= bounds.min_y - max_dist && sy <= bounds.max_y + max_dist &&
                    sz >= bounds.min_z - max_dist && sz <= bounds.max_z + max_dist;
            }
            const int relevant_source_count = __syncthreads_count(tile_relevant);
            if (relevant_source_count == 0)
            {
                continue;
            }
        }

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
#ifdef PLAPOINT_ENABLE_TESTING
        if (local_idx == 0)
        {
            atomicAdd(&g_icp_target_tile_load_count, 1ull);
        }
#endif
        __syncthreads();

        if (source_valid && !stop_target_scan && tile_relevant)
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
                const double dx_sq = dx * dx;
                if constexpr (UseTargetTileBounds)
                {
                    if (dx_sq > max_dist_sq)
                    {
                        continue;
                    }
                }
                const double dy = sy - target_tile_y[tile_offset];
                const double dy_sq = dy * dy;
                const double xy_dist_sq = dx_sq + dy_sq;
                if constexpr (UseTargetTileBounds)
                {
                    if (xy_dist_sq > max_dist_sq)
                    {
                        continue;
                    }
                }
                const double dz = sz - target_tile_z[tile_offset];
                const double dz_sq = dz * dz;
                const double dist_sq = xy_dist_sq + dz_sq;
                if constexpr (UseTargetTileBounds)
                {
                    if (dist_sq > max_dist_sq)
                    {
                        continue;
                    }
                }

#ifdef PLAPOINT_ENABLE_TESTING
                atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
                if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                {
                    best_dist_sq = dist_sq;
                    if (dist_sq <= 0.0)
                    {
                        stop_target_scan = true;
                        break;
                    }
                }
            }
        }
        const int unfinished_source_count = __syncthreads_count(source_valid && !stop_target_scan);
        if (unfinished_source_count == 0)
        {
            break;
        }
    }

    if (source_valid)
    {
        bool accepted = isfinite(best_dist_sq);
        if (accepted && isfinite(max_dist))
        {
            accepted = best_dist_sq <= max_dist_sq;
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

template <typename Scalar, bool FiniteCellBounds>
__global__ void collectResidualStatsSpatialGridKernel(
    const Scalar* __restrict__ source_points,
    int source_count,
    Scalar max_correspondence_distance,
    IcpTargetSpatialGrid target_grid,
    RawIcpResidualStats* __restrict__ partial_stats)
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

    if (source_valid)
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

        bool stop_cell_scan = false;
#pragma unroll
        for (int dx_offset_idx = 0; dx_offset_idx < 3; ++dx_offset_idx)
        {
            if (stop_cell_scan)
            {
                break;
            }
            int query_x = 0;
            const int dx_cell = icpNeighborCellOffset(dx_offset_idx);
            if (!offsetGridCellCoordinate(source_key.x, dx_cell, query_x))
            {
                continue;
            }
#pragma unroll
            for (int dy_offset_idx = 0; dy_offset_idx < 3; ++dy_offset_idx)
            {
                if (stop_cell_scan)
                {
                    break;
                }
                int query_y = 0;
                const int dy_cell = icpNeighborCellOffset(dy_offset_idx);
                if (!offsetGridCellCoordinate(source_key.y, dy_cell, query_y))
                {
                    continue;
                }

                const double min_xy_dist_sq = minDistanceSqToIcpGridCellXY<FiniteCellBounds>(
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
                    const IcpGridCellKey cell_key = loadIcpGridCellKey(target_grid.cell_keys, cell_idx);
                    if (cell_key.x != query_x || cell_key.y != query_y || cell_key.z > max_z)
                    {
                        break;
                    }

                    const double min_cell_dist_sq = min_xy_dist_sq + minDistanceSqToIcpGridCellZ<FiniteCellBounds>(
                        sz,
                        cell_key.z,
                        target_grid.cell_size);
                    if (min_cell_dist_sq > max_dist_sq || min_cell_dist_sq >= best_dist_sq)
                    {
                        ++cell_idx;
                        continue;
                    }

                    const int start = loadIcpGridCellStart(target_grid, cell_idx);
                    const int count = loadIcpGridCellCount(target_grid, cell_idx);
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int sorted_offset = start + offset;
#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_target_candidate_visit_count, 1ull);
#endif
                        const double tx = loadSortedIcpTargetX<Scalar>(target_grid, sorted_offset);
                        const double dx = sx - tx;
                        const double dx_sq = dx * dx;
                        if (dx_sq > max_dist_sq)
                        {
                            continue;
                        }
                        if (dx_sq >= best_dist_sq)
                        {
                            continue;
                        }

                        const double ty = loadSortedIcpTargetY<Scalar>(target_grid, sorted_offset);
                        const double dy = sy - ty;
                        const double dy_sq = dy * dy;
                        const double xy_dist_sq = dx_sq + dy_sq;
                        if (xy_dist_sq > max_dist_sq)
                        {
                            continue;
                        }
                        if (xy_dist_sq >= best_dist_sq)
                        {
                            continue;
                        }

                        const double tz = loadSortedIcpTargetZ<Scalar>(target_grid, sorted_offset);
                        const double dz = sz - tz;
                        const double dz_sq = dz * dz;
                        const double dist_sq = xy_dist_sq + dz_sq;
                        if (dist_sq > max_dist_sq)
                        {
                            continue;
                        }

#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
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

template <typename Scalar, bool UseTargetTileBounds>
__global__ void transformAndCollectResidualStatsKernel(
    const Scalar* __restrict__ transform,
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    Scalar* output_points,
    const IcpTargetTileBounds* __restrict__ target_tile_bounds,
    RawIcpResidualStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    loadColumnMajorTransform3x4Block(transform, block_transform, local_idx);

    if (source_idx < source_count)
    {
        const Scalar px = loadReadOnlyIcpValue(source_points + source_idx);
        const Scalar py = loadReadOnlyIcpValue(source_points + source_count + source_idx);
        const Scalar pz = loadReadOnlyIcpValue(source_points + 2 * source_count + source_idx);

        Scalar ox = Scalar(0);
        Scalar oy = Scalar(0);
        Scalar oz = Scalar(0);
        transformColumnMajorPoint3x4(block_transform, px, py, pz, ox, oy, oz);
        if (output_points)
        {
            output_points[source_idx] = ox;
            output_points[source_count + source_idx] = oy;
            output_points[2 * source_count + source_idx] = oz;
        }

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
            if (tryAcceptTransformedExactPointwiseResidual<Scalar>(
                    source_idx,
                    source_count,
                    target_points,
                    target_count,
                    sx,
                    sy,
                    sz,
                    local))
            {
                source_valid = false;
            }
        }
    }

    double best_dist_sq = INFINITY;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const double max_dist_sq = max_dist * max_dist;
    bool stop_target_scan = false;
    __shared__ double target_tile_x[kIcpStatsBlockSize];
    __shared__ double target_tile_y[kIcpStatsBlockSize];
    __shared__ double target_tile_z[kIcpStatsBlockSize];
    __shared__ int target_tile_valid[kIcpStatsBlockSize];

    const int scan_target_count = __syncthreads_count(source_valid) > 0 ? target_count : 0;
    for (int tile_start = 0; tile_start < scan_target_count; tile_start += kIcpStatsBlockSize)
    {
        const int tile_idx = tile_start / kIcpStatsBlockSize;
        bool tile_relevant = true;
        if constexpr (UseTargetTileBounds)
        {
            tile_relevant = false;
            if (source_valid && !stop_target_scan)
            {
                const IcpTargetTileBounds bounds = loadIcpTargetTileBounds(target_tile_bounds, tile_idx);
                tile_relevant =
                    bounds.has_valid_point &&
                    sx >= bounds.min_x - max_dist && sx <= bounds.max_x + max_dist &&
                    sy >= bounds.min_y - max_dist && sy <= bounds.max_y + max_dist &&
                    sz >= bounds.min_z - max_dist && sz <= bounds.max_z + max_dist;
            }
            const int relevant_source_count = __syncthreads_count(tile_relevant);
            if (relevant_source_count == 0)
            {
                continue;
            }
        }

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
#ifdef PLAPOINT_ENABLE_TESTING
        if (local_idx == 0)
        {
            atomicAdd(&g_icp_target_tile_load_count, 1ull);
        }
#endif
        __syncthreads();

        if (source_valid && !stop_target_scan && tile_relevant)
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
                const double dx_sq = dx * dx;
                if constexpr (UseTargetTileBounds)
                {
                    if (dx_sq > max_dist_sq)
                    {
                        continue;
                    }
                }
                const double dy = sy - target_tile_y[tile_offset];
                const double dy_sq = dy * dy;
                const double xy_dist_sq = dx_sq + dy_sq;
                if constexpr (UseTargetTileBounds)
                {
                    if (xy_dist_sq > max_dist_sq)
                    {
                        continue;
                    }
                }
                const double dz = sz - target_tile_z[tile_offset];
                const double dz_sq = dz * dz;
                const double dist_sq = xy_dist_sq + dz_sq;
                if constexpr (UseTargetTileBounds)
                {
                    if (dist_sq > max_dist_sq)
                    {
                        continue;
                    }
                }

#ifdef PLAPOINT_ENABLE_TESTING
                atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
                if (isfinite(dist_sq) && dist_sq < best_dist_sq)
                {
                    best_dist_sq = dist_sq;
                    if (dist_sq <= 0.0)
                    {
                        stop_target_scan = true;
                        break;
                    }
                }
            }
        }
        const int unfinished_source_count = __syncthreads_count(source_valid && !stop_target_scan);
        if (unfinished_source_count == 0)
        {
            break;
        }
    }

    if (source_valid)
    {
        bool accepted = isfinite(best_dist_sq);
        if (accepted && isfinite(max_dist))
        {
            accepted = best_dist_sq <= max_dist_sq;
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
__global__ void transformAndCollectOrderedResidualStatsKernel(
    const Scalar* __restrict__ transform,
    const Scalar* source_points,
    int point_count,
    const Scalar* target_points,
    Scalar max_correspondence_distance,
    Scalar* output_points,
    RawIcpResidualStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    loadColumnMajorTransform3x4Block(transform, block_transform, local_idx);

    if (source_idx < point_count)
    {
        const Scalar px = loadReadOnlyIcpValue(source_points + source_idx);
        const Scalar py = loadReadOnlyIcpValue(source_points + point_count + source_idx);
        const Scalar pz = loadReadOnlyIcpValue(source_points + 2 * point_count + source_idx);

        Scalar ox = Scalar(0);
        Scalar oy = Scalar(0);
        Scalar oz = Scalar(0);
        transformColumnMajorPoint3x4(block_transform, px, py, pz, ox, oy, oz);
        if (output_points)
        {
            output_points[source_idx] = ox;
            output_points[point_count + source_idx] = oy;
            output_points[2 * point_count + source_idx] = oz;
        }

        const double sx = static_cast<double>(ox);
        const double sy = static_cast<double>(oy);
        const double sz = static_cast<double>(oz);
        if (!isfinite(sx) || !isfinite(sy) || !isfinite(sz))
        {
            local.invalid_source_count = 1;
        }
        else
        {
            double tx = 0.0;
            double ty = 0.0;
            double tz = 0.0;
            if (loadFiniteColumnMajorPoint(target_points, point_count, source_idx, tx, ty, tz))
            {
#ifdef PLAPOINT_ENABLE_TESTING
                atomicAdd(&g_icp_exact_pointwise_target_load_count, 3ull);
#endif
                const double dx = sx - tx;
                const double dy = sy - ty;
                const double dz = sz - tz;
                const double dist_sq = dx * dx + dy * dy + dz * dz;
                const double max_dist = static_cast<double>(max_correspondence_distance);
                bool accepted = isfinite(dist_sq);
                if (accepted && isfinite(max_dist))
                {
                    accepted = dist_sq <= max_dist * max_dist;
                }
                if (accepted)
                {
                    local.active_count = 1;
                    local.residual_sq_sum = dist_sq;
                }
            }
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

template <typename Scalar, bool FiniteCellBounds>
__global__ void transformAndCollectResidualStatsSpatialGridKernel(
    const Scalar* __restrict__ transform,
    const Scalar* source_points,
    int source_count,
    Scalar max_correspondence_distance,
    Scalar* output_points,
    IcpTargetSpatialGrid target_grid,
    RawIcpResidualStats* __restrict__ partial_stats)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    RawIcpResidualStats local{};
    bool source_valid = false;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    loadColumnMajorTransform3x4Block(transform, block_transform, local_idx);

    if (source_idx < source_count)
    {
        const Scalar px = loadReadOnlyIcpValue(source_points + source_idx);
        const Scalar py = loadReadOnlyIcpValue(source_points + source_count + source_idx);
        const Scalar pz = loadReadOnlyIcpValue(source_points + 2 * source_count + source_idx);

        Scalar ox = Scalar(0);
        Scalar oy = Scalar(0);
        Scalar oz = Scalar(0);
        transformColumnMajorPoint3x4(block_transform, px, py, pz, ox, oy, oz);
        if (output_points)
        {
            output_points[source_idx] = ox;
            output_points[source_count + source_idx] = oy;
            output_points[2 * source_count + source_idx] = oz;
        }

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
            if (tryAcceptTransformedExactPointwiseResidual<Scalar>(
                    source_idx,
                    source_count,
                    target_grid.target_points,
                    target_grid.target_count,
                    sx,
                    sy,
                    sz,
                    local))
            {
                source_valid = false;
            }
        }
    }

    double best_dist_sq = INFINITY;
    const double max_dist = static_cast<double>(max_correspondence_distance);
    const double max_dist_sq = max_dist * max_dist;

    if (source_valid)
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

        bool stop_cell_scan = false;
#pragma unroll
        for (int dx_offset_idx = 0; dx_offset_idx < 3; ++dx_offset_idx)
        {
            if (stop_cell_scan)
            {
                break;
            }
            int query_x = 0;
            const int dx_cell = icpNeighborCellOffset(dx_offset_idx);
            if (!offsetGridCellCoordinate(source_key.x, dx_cell, query_x))
            {
                continue;
            }
#pragma unroll
            for (int dy_offset_idx = 0; dy_offset_idx < 3; ++dy_offset_idx)
            {
                if (stop_cell_scan)
                {
                    break;
                }
                int query_y = 0;
                const int dy_cell = icpNeighborCellOffset(dy_offset_idx);
                if (!offsetGridCellCoordinate(source_key.y, dy_cell, query_y))
                {
                    continue;
                }

                const double min_xy_dist_sq = minDistanceSqToIcpGridCellXY<FiniteCellBounds>(
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
                    const IcpGridCellKey cell_key = loadIcpGridCellKey(target_grid.cell_keys, cell_idx);
                    if (cell_key.x != query_x || cell_key.y != query_y || cell_key.z > max_z)
                    {
                        break;
                    }

                    const double min_cell_dist_sq = min_xy_dist_sq + minDistanceSqToIcpGridCellZ<FiniteCellBounds>(
                        sz,
                        cell_key.z,
                        target_grid.cell_size);
                    if (min_cell_dist_sq > max_dist_sq || min_cell_dist_sq >= best_dist_sq)
                    {
                        ++cell_idx;
                        continue;
                    }

                    const int start = loadIcpGridCellStart(target_grid, cell_idx);
                    const int count = loadIcpGridCellCount(target_grid, cell_idx);
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int sorted_offset = start + offset;
#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_target_candidate_visit_count, 1ull);
#endif
                        const double tx = loadSortedIcpTargetX<Scalar>(target_grid, sorted_offset);
                        const double dx = sx - tx;
                        const double dx_sq = dx * dx;
                        if (dx_sq > max_dist_sq)
                        {
                            continue;
                        }
                        if (dx_sq >= best_dist_sq)
                        {
                            continue;
                        }

                        const double ty = loadSortedIcpTargetY<Scalar>(target_grid, sorted_offset);
                        const double dy = sy - ty;
                        const double dy_sq = dy * dy;
                        const double xy_dist_sq = dx_sq + dy_sq;
                        if (xy_dist_sq > max_dist_sq)
                        {
                            continue;
                        }
                        if (xy_dist_sq >= best_dist_sq)
                        {
                            continue;
                        }

                        const double tz = loadSortedIcpTargetZ<Scalar>(target_grid, sorted_offset);
                        const double dz = sz - tz;
                        const double dz_sq = dz * dz;
                        const double dist_sq = xy_dist_sq + dz_sq;
                        if (dist_sq > max_dist_sq)
                        {
                            continue;
                        }

#ifdef PLAPOINT_ENABLE_TESTING
                        atomicAdd(&g_icp_full_distance_evaluation_count, 1ull);
#endif
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
void launchCollectCorrespondenceStatsKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    int* correspondence_indices,
    const IcpTargetTileBounds* target_tile_bounds,
    RawIcpStats* partial_stats)
{
#ifdef PLAPOINT_ENABLE_TESTING
    recordFallbackKernelLaunchForTesting(target_tile_bounds != nullptr);
#endif

    if (target_tile_bounds)
    {
        if (correspondence_indices)
        {
            collectCorrespondenceStatsKernel<Scalar, true, true, false, false>
                <<<grid_size, block_size, 0, stream>>>(
                nullptr,
                source_points,
                source_count,
                target_points,
                target_count,
                max_correspondence_distance,
                correspondence_indices,
                target_tile_bounds,
                partial_stats);
            return;
        }

        collectCorrespondenceStatsKernel<Scalar, false, true, false, false>
            <<<grid_size, block_size, 0, stream>>>(
            nullptr,
            source_points,
            source_count,
            target_points,
            target_count,
            max_correspondence_distance,
            correspondence_indices,
            target_tile_bounds,
            partial_stats);
        return;
    }

    if (correspondence_indices)
    {
        collectCorrespondenceStatsKernel<Scalar, true, false, false, false>
            <<<grid_size, block_size, 0, stream>>>(
            nullptr,
            source_points,
            source_count,
            target_points,
            target_count,
            max_correspondence_distance,
            correspondence_indices,
            target_tile_bounds,
            partial_stats);
        return;
    }

    collectCorrespondenceStatsKernel<Scalar, false, false, false, false>
        <<<grid_size, block_size, 0, stream>>>(
        nullptr,
        source_points,
        source_count,
        target_points,
        target_count,
        max_correspondence_distance,
        correspondence_indices,
        target_tile_bounds,
        partial_stats);
}

template <typename Scalar>
void launchTransformAndCollectCorrespondenceStatsKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* source_transform,
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const IcpTargetTileBounds* target_tile_bounds,
    RawIcpStats* partial_stats)
{
#ifdef PLAPOINT_ENABLE_TESTING
    recordFallbackKernelLaunchForTesting(target_tile_bounds != nullptr);
#endif

    if (target_tile_bounds)
    {
        const bool try_transformed_exact_pointwise =
            detail::canProbeTransformedExactPointwiseStats(
                source_count,
                target_points,
                target_count,
                nullptr);
        if (try_transformed_exact_pointwise)
        {
            collectCorrespondenceStatsKernel<Scalar, false, true, true, true>
                <<<grid_size, block_size, 0, stream>>>(
                source_transform,
                source_points,
                source_count,
                target_points,
                target_count,
                max_correspondence_distance,
                nullptr,
                target_tile_bounds,
                partial_stats);
            return;
        }
        collectCorrespondenceStatsKernel<Scalar, false, true, true, false>
            <<<grid_size, block_size, 0, stream>>>(
            source_transform,
            source_points,
            source_count,
            target_points,
            target_count,
            max_correspondence_distance,
            nullptr,
            target_tile_bounds,
            partial_stats);
        return;
    }

    const bool try_transformed_exact_pointwise =
        detail::canProbeTransformedExactPointwiseStats(
            source_count,
            target_points,
            target_count,
            nullptr);
    if (try_transformed_exact_pointwise)
    {
        collectCorrespondenceStatsKernel<Scalar, false, false, true, true>
            <<<grid_size, block_size, 0, stream>>>(
            source_transform,
            source_points,
            source_count,
            target_points,
            target_count,
            max_correspondence_distance,
            nullptr,
            target_tile_bounds,
            partial_stats);
        return;
    }
    collectCorrespondenceStatsKernel<Scalar, false, false, true, false>
        <<<grid_size, block_size, 0, stream>>>(
        source_transform,
        source_points,
        source_count,
        target_points,
        target_count,
        max_correspondence_distance,
        nullptr,
        target_tile_bounds,
        partial_stats);
}

template <typename Scalar>
void launchCollectResidualStatsKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* source_points,
    int source_count,
    const Scalar* target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const IcpTargetTileBounds* target_tile_bounds,
    RawIcpResidualStats* partial_stats)
{
#ifdef PLAPOINT_ENABLE_TESTING
    recordFallbackKernelLaunchForTesting(target_tile_bounds != nullptr);
#endif

    if (target_tile_bounds)
    {
        collectResidualStatsKernel<Scalar, true><<<grid_size, block_size, 0, stream>>>(
            source_points,
            source_count,
            target_points,
            target_count,
            max_correspondence_distance,
            target_tile_bounds,
            partial_stats);
        return;
    }

    collectResidualStatsKernel<Scalar, false><<<grid_size, block_size, 0, stream>>>(
        source_points,
        source_count,
        target_points,
        target_count,
        max_correspondence_distance,
        target_tile_bounds,
        partial_stats);
}

template <typename Scalar>
void launchTransformAndCollectResidualStatsKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
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
#ifdef PLAPOINT_ENABLE_TESTING
    recordFallbackKernelLaunchForTesting(target_tile_bounds != nullptr);
    recordTransformedExactPointwiseResidualCallForTesting(source_count == target_count);
#endif

    if (target_tile_bounds)
    {
        transformAndCollectResidualStatsKernel<Scalar, true><<<grid_size, block_size, 0, stream>>>(
            transform,
            source_points,
            source_count,
            target_points,
            target_count,
            max_correspondence_distance,
            output_points,
            target_tile_bounds,
            partial_stats);
        return;
    }

    transformAndCollectResidualStatsKernel<Scalar, false><<<grid_size, block_size, 0, stream>>>(
        transform,
        source_points,
        source_count,
        target_points,
        target_count,
        max_correspondence_distance,
        output_points,
        target_tile_bounds,
        partial_stats);
}

template <typename Scalar>
void launchTransformAndCollectOrderedResidualStatsKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* transform,
    const Scalar* source_points,
    int point_count,
    const Scalar* target_points,
    Scalar max_correspondence_distance,
    Scalar* output_points,
    RawIcpResidualStats* partial_stats)
{
    transformAndCollectOrderedResidualStatsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        transform,
        source_points,
        point_count,
        target_points,
        max_correspondence_distance,
        output_points,
        partial_stats);
}

template <typename Scalar>
void launchCollectCorrespondenceStatsSpatialGridKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* source_points,
    int source_count,
    Scalar max_correspondence_distance,
    int* correspondence_indices,
    const IcpTargetSpatialGrid& target_grid,
    RawIcpStats* partial_stats)
{
    if (target_grid.finite_cell_bounds)
    {
        if (correspondence_indices)
        {
            collectCorrespondenceStatsSpatialGridKernel<Scalar, true, true, false, false>
                <<<grid_size, block_size, 0, stream>>>(
                nullptr,
                source_points,
                source_count,
                max_correspondence_distance,
                correspondence_indices,
                target_grid,
                partial_stats);
        }
        else
        {
            collectCorrespondenceStatsSpatialGridKernel<Scalar, true, false, false, false>
                <<<grid_size, block_size, 0, stream>>>(
                nullptr,
                source_points,
                source_count,
                max_correspondence_distance,
                correspondence_indices,
                target_grid,
                partial_stats);
        }
        return;
    }

    if (correspondence_indices)
    {
        collectCorrespondenceStatsSpatialGridKernel<Scalar, false, true, false, false>
            <<<grid_size, block_size, 0, stream>>>(
            nullptr,
            source_points,
            source_count,
            max_correspondence_distance,
            correspondence_indices,
            target_grid,
            partial_stats);
    }
    else
    {
        collectCorrespondenceStatsSpatialGridKernel<Scalar, false, false, false, false>
            <<<grid_size, block_size, 0, stream>>>(
            nullptr,
            source_points,
            source_count,
            max_correspondence_distance,
            correspondence_indices,
            target_grid,
            partial_stats);
    }
}

template <typename Scalar>
void launchTransformAndCollectCorrespondenceStatsSpatialGridKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* source_transform,
    const Scalar* source_points,
    int source_count,
    Scalar max_correspondence_distance,
    const IcpTargetSpatialGrid& target_grid,
    RawIcpStats* partial_stats)
{
    if (target_grid.finite_cell_bounds)
    {
        const bool try_transformed_exact_pointwise =
            detail::canProbeTransformedExactPointwiseStats(
                source_count,
                target_grid.target_points,
                target_grid.target_count,
                nullptr);
        if (try_transformed_exact_pointwise)
        {
            collectCorrespondenceStatsSpatialGridKernel<Scalar, true, false, true, true>
                <<<grid_size, block_size, 0, stream>>>(
                source_transform,
                source_points,
                source_count,
                max_correspondence_distance,
                nullptr,
                target_grid,
                partial_stats);
            return;
        }
        collectCorrespondenceStatsSpatialGridKernel<Scalar, true, false, true, false>
            <<<grid_size, block_size, 0, stream>>>(
            source_transform,
            source_points,
            source_count,
            max_correspondence_distance,
            nullptr,
            target_grid,
            partial_stats);
        return;
    }

    const bool try_transformed_exact_pointwise =
        detail::canProbeTransformedExactPointwiseStats(
            source_count,
            target_grid.target_points,
            target_grid.target_count,
            nullptr);
    if (try_transformed_exact_pointwise)
    {
        collectCorrespondenceStatsSpatialGridKernel<Scalar, false, false, true, true>
            <<<grid_size, block_size, 0, stream>>>(
            source_transform,
            source_points,
            source_count,
            max_correspondence_distance,
            nullptr,
            target_grid,
            partial_stats);
        return;
    }
    collectCorrespondenceStatsSpatialGridKernel<Scalar, false, false, true, false>
        <<<grid_size, block_size, 0, stream>>>(
        source_transform,
        source_points,
        source_count,
        max_correspondence_distance,
        nullptr,
        target_grid,
        partial_stats);
}

template <typename Scalar>
void launchCollectResidualStatsSpatialGridKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* source_points,
    int source_count,
    Scalar max_correspondence_distance,
    const IcpTargetSpatialGrid& target_grid,
    RawIcpResidualStats* partial_stats)
{
    if (target_grid.finite_cell_bounds)
    {
        collectResidualStatsSpatialGridKernel<Scalar, true><<<grid_size, block_size, 0, stream>>>(
            source_points,
            source_count,
            max_correspondence_distance,
            target_grid,
            partial_stats);
        return;
    }

    collectResidualStatsSpatialGridKernel<Scalar, false><<<grid_size, block_size, 0, stream>>>(
        source_points,
        source_count,
        max_correspondence_distance,
        target_grid,
        partial_stats);
}

template <typename Scalar>
void launchTransformAndCollectResidualStatsSpatialGridKernel(
    int grid_size,
    int block_size,
    cudaStream_t stream,
    const Scalar* transform,
    const Scalar* source_points,
    int source_count,
    Scalar max_correspondence_distance,
    Scalar* output_points,
    const IcpTargetSpatialGrid& target_grid,
    RawIcpResidualStats* partial_stats)
{
#ifdef PLAPOINT_ENABLE_TESTING
    recordTransformedExactPointwiseResidualCallForTesting(
        target_grid.target_points != nullptr && source_count == target_grid.target_count);
#endif

    if (target_grid.finite_cell_bounds)
    {
        transformAndCollectResidualStatsSpatialGridKernel<Scalar, true><<<grid_size, block_size, 0, stream>>>(
            transform,
            source_points,
            source_count,
            max_correspondence_distance,
            output_points,
            target_grid,
            partial_stats);
        return;
    }

    transformAndCollectResidualStatsSpatialGridKernel<Scalar, false><<<grid_size, block_size, 0, stream>>>(
        transform,
        source_points,
        source_count,
        max_correspondence_distance,
        output_points,
        target_grid,
        partial_stats);
}

__global__ void reduceRawIcpStatsKernel(
    const RawIcpStats* __restrict__ partial_stats,
    int partial_count,
    RawIcpStats* __restrict__ stats)
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
__device__ __forceinline__ void writeExactPointwiseStatsAndStepRawResult(
    const RawIcpStats& raw,
    IcpStatsAndStepRawResult* __restrict__ result)
{
    result->stats = raw;
    result->step.delta = 0.0;
    result->step.valid = raw.active_count > 0 && isfinite(raw.residual_sq_sum) ? 1 : 0;
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndSetExactPointwiseIdentityStepKernel(
    const RawIcpStats* __restrict__ partial_stats,
    int partial_count,
    Scalar* __restrict__ step_transform,
    IcpStatsAndStepRawResult* __restrict__ result)
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
        writeExactPointwiseStatsAndStepRawResult<Scalar>(reduced, result);
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndSetExactPointwiseIdentityAlignmentStepKernel(
    const RawIcpStats* __restrict__ partial_stats,
    int partial_count,
    Scalar* __restrict__ step_transform,
    IcpAlignmentStepRawResult<Scalar>* __restrict__ result)
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
        const RawIcpStats raw = shared_stats[0];
        const int step_valid = raw.active_count > 0 && isfinite(raw.residual_sq_sum) ? 1 : 0;
        writeAlignmentStepRawResultFields<Scalar>(raw, result, step_valid, 0.0);
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndSetExactPointwiseIdentityAccumulatedAlignmentStepKernel(
    const RawIcpStats* __restrict__ partial_stats,
    int partial_count,
    const Scalar* __restrict__ previous_accumulated_transform,
    Scalar* __restrict__ step_transform,
    Scalar* __restrict__ accumulated_transform,
    IcpAlignmentStepRawResult<Scalar>* __restrict__ result)
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

    const RawIcpStats raw = shared_stats[0];
    const int step_valid = raw.active_count > 0 && isfinite(raw.residual_sq_sum) ? 1 : 0;
    if (local_idx < 16)
    {
        const int row = local_idx & 3;
        const int col = local_idx >> 2;
        step_transform[local_idx] = row == col ? Scalar(1) : Scalar(0);
        if (step_valid != 0)
        {
            accumulated_transform[local_idx] = previous_accumulated_transform[local_idx];
        }
    }

    if (local_idx == 0)
    {
        writeAlignmentStepRawResultFields<Scalar>(raw, result, step_valid, 0.0);
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndComputeStepTransformKernel(
    const RawIcpStats* __restrict__ partial_stats,
    int partial_count,
    Scalar* __restrict__ step_transform,
    IcpStatsAndStepRawResult* __restrict__ result)
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
        result->stats = raw;
        computeStepTransformFromRawStatsValue<Scalar>(raw, step_transform, &result->step);
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndComputeAlignmentStepKernel(
    const RawIcpStats* __restrict__ partial_stats,
    int partial_count,
    Scalar* __restrict__ step_transform,
    IcpAlignmentStepRawResult<Scalar>* __restrict__ result)
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
        writeAlignmentStepRawResultFromRawStats<Scalar>(shared_stats[0], step_transform, result);
    }
}

template <typename Scalar>
__global__ void reduceRawIcpStatsAndComputeAlignmentStepAccumulatedTransformKernel(
    const RawIcpStats* __restrict__ partial_stats,
    int partial_count,
    const Scalar* __restrict__ previous_accumulated_transform,
    Scalar* __restrict__ step_transform,
    Scalar* __restrict__ accumulated_transform,
    IcpAlignmentStepRawResult<Scalar>* __restrict__ result)
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
        writeAlignmentStepRawResultFromRawStats<Scalar>(shared_stats[0], step_transform, result);
        if (result->flags & kIcpAlignmentStepValidFlag)
        {
            multiplyTransform4x4SingleThread(
                step_transform,
                previous_accumulated_transform,
                accumulated_transform);
        }
    }
}

__global__ void reduceRawIcpResidualStatsKernel(
    const RawIcpResidualStats* __restrict__ partial_stats,
    int partial_count,
    RawIcpResidualStats* __restrict__ stats)
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
__global__ void setIdentityTransform4x4Kernel(Scalar* __restrict__ transform)
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
__global__ void multiplyTransform4x4Kernel(
    const Scalar* __restrict__ A,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ C)
{
    const int idx = threadIdx.x;
    __shared__ Scalar shared_A[16];
    __shared__ Scalar shared_B[16];
    if (idx < 16)
    {
        shared_A[idx] = loadReadOnlyIcpValue(A + idx);
        shared_B[idx] = loadReadOnlyIcpValue(B + idx);
    }
    __syncthreads();

    if (idx >= 16)
    {
        return;
    }

    const int row = idx & 3;
    const int col = idx >> 2;
    double sum = 0.0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        sum += static_cast<double>(shared_A[row + k * 4]) *
            static_cast<double>(shared_B[k + col * 4]);
    }
    C[row + col * 4] = static_cast<Scalar>(sum);
}

template <typename Scalar>
__device__ __forceinline__ void multiplyTransform4x4SingleThread(
    const Scalar* __restrict__ A,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ C)
{
#pragma unroll
    for (int col = 0; col < 4; ++col)
    {
#pragma unroll
        for (int row = 0; row < 4; ++row)
        {
            double sum = 0.0;
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                sum += static_cast<double>(A[row + k * 4]) *
                    static_cast<double>(B[k + col * 4]);
            }
            C[row + col * 4] = static_cast<Scalar>(sum);
        }
    }
}

template <typename Scalar>
__global__ void transformPointsColumnMajorKernel(
    const Scalar* __restrict__ transform,
    const Scalar* points,
    int point_count,
    Scalar* output_points)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;
    __shared__ Scalar block_transform[kIcpTransform3x4ValueCount];
    loadColumnMajorTransform3x4Block(transform, block_transform, local_idx);
    if (idx >= point_count)
    {
        return;
    }

    const Scalar px = loadReadOnlyIcpValue(points + idx);
    const Scalar py = loadReadOnlyIcpValue(points + point_count + idx);
    const Scalar pz = loadReadOnlyIcpValue(points + 2 * point_count + idx);

    Scalar ox = Scalar(0);
    Scalar oy = Scalar(0);
    Scalar oz = Scalar(0);
    transformColumnMajorPoint3x4(block_transform, px, py, pz, ox, oy, oz);
    output_points[idx] = ox;
    output_points[point_count + idx] = oy;
    output_points[2 * point_count + idx] = oz;
}

__device__ __forceinline__ void jacobiRotate4x4(double A[16], double V[16], int p, int q)
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

#pragma unroll
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

#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const double vkp = V[k * 4 + p];
        const double vkq = V[k * 4 + q];
        V[k * 4 + p] = c * vkp - s * vkq;
        V[k * 4 + q] = s * vkp + c * vkq;
    }
}

__device__ __forceinline__ void largestEigenvectorSymmetric4x4(const double A_in[16], double eigenvector[4])
{
    double A[16];
    double V[16];
#pragma unroll
    for (int idx = 0; idx < 16; ++idx)
    {
        A[idx] = A_in[idx];
        V[idx] = 0.0;
    }
#pragma unroll
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
#pragma unroll
    for (int i = 1; i < 4; ++i)
    {
        if (A[i * 4 + i] > A[best * 4 + best])
        {
            best = i;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        eigenvector[i] = V[i * 4 + best];
    }
}

template <typename Scalar>
__device__ __forceinline__ bool scalarRepresentable(double value)
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
__device__ __forceinline__ Scalar checkedDeviceScalar(double value, int& valid)
{
    if (!scalarRepresentable<Scalar>(value))
    {
        valid = 0;
    }
    return static_cast<Scalar>(value);
}

template <typename Scalar>
__device__ __forceinline__ void computeStepTransformFromInput(
    const IcpStepTransformInput& input,
    Scalar* __restrict__ step_transform,
    IcpStepTransformRawResult* __restrict__ result)
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
    const IcpStepTransformInput* __restrict__ input,
    Scalar* __restrict__ step_transform,
    IcpStepTransformRawResult* __restrict__ result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    computeStepTransformFromInput<Scalar>(*input, step_transform, result);
}

template <typename Scalar>
__global__ void computeStepTransformFromRawStatsKernel(
    const RawIcpStats* __restrict__ raw_stats,
    Scalar* __restrict__ step_transform,
    IcpStepTransformRawResult* __restrict__ result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    computeStepTransformFromRawStatsValue<Scalar>(*raw_stats, step_transform, result);
}

template <typename Scalar>
__device__ __forceinline__ void computeStepTransformFromRawStatsValue(
    const RawIcpStats& raw,
    Scalar* __restrict__ step_transform,
    IcpStepTransformRawResult* __restrict__ result)
{
    if (raw.active_count <= 0)
    {
        result->delta = 0.0;
        result->valid = 0;
        return;
    }

    IcpStepTransformInput input{};
    const double inv_count = 1.0 / static_cast<double>(raw.active_count);
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        input.src_centroid[c] = raw.src_sum[c] * inv_count;
        input.tgt_centroid[c] = raw.tgt_sum[c] * inv_count;
    }
#pragma unroll
    for (int r = 0; r < 3; ++r)
    {
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            const int idx = r * 3 + c;
            input.cross_covariance[idx] = raw.cross_sum[idx] -
                raw.src_sum[r] * raw.tgt_sum[c] * inv_count;
        }
    }
    computeStepTransformFromInput<Scalar>(input, step_transform, result);
}

__device__ __forceinline__ bool rawStatsCovarianceHasNonCollinearGeometry(
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
__device__ __forceinline__ void writeAlignmentStepRawResultFromRawStats(
    const RawIcpStats& raw,
    Scalar* __restrict__ step_transform,
    IcpAlignmentStepRawResult<Scalar>* __restrict__ result)
{
    IcpStepTransformRawResult step_result{};
    computeStepTransformFromRawStatsValue<Scalar>(raw, step_transform, &step_result);

    writeAlignmentStepRawResultFields<Scalar>(raw, result, step_result.valid, step_result.delta);
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
    return detail::canProbeExactPointwiseStats(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_correspondence_indices);
}

template <typename Scalar>
void launchExactPointwiseCorrespondencePartials(
    const Scalar* d_source_points,
    const Scalar* d_target_points,
    int source_count,
    RawIcpStats* d_partials,
    int partial_count,
    cudaStream_t stream)
{
    constexpr int block_size = kIcpStatsBlockSize;
    if (detail::canUseSameBufferExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            source_count,
            nullptr))
    {
        collectExactPointwiseCorrespondenceStatsKernel<Scalar, true><<<partial_count, block_size, 0, stream>>>(
            d_source_points,
            d_target_points,
            source_count,
            d_partials);
    }
    else
    {
        collectExactPointwiseCorrespondenceStatsKernel<Scalar, false><<<partial_count, block_size, 0, stream>>>(
            d_source_points,
            d_target_points,
            source_count,
            d_partials);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void launchOrderedPointwiseCorrespondencePartials(
    const Scalar* d_source_points,
    const Scalar* d_target_points,
    int source_count,
    Scalar max_correspondence_distance,
    RawIcpStats* d_partials,
    int partial_count,
    cudaStream_t stream)
{
    constexpr int block_size = kIcpStatsBlockSize;
    if (detail::canUseSameBufferExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            source_count,
            nullptr))
    {
        collectOrderedPointwiseCorrespondenceStatsKernel<Scalar, true><<<partial_count, block_size, 0, stream>>>(
            d_source_points,
            d_target_points,
            source_count,
            max_correspondence_distance,
            d_partials);
    }
    else
    {
        collectOrderedPointwiseCorrespondenceStatsKernel<Scalar, false><<<partial_count, block_size, 0, stream>>>(
            d_source_points,
            d_target_points,
            source_count,
            max_correspondence_distance,
            d_partials);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void launchExactPointwiseResidualPartials(
    const Scalar* d_source_points,
    const Scalar* d_target_points,
    int source_count,
    RawIcpResidualStats* d_partials,
    int partial_count,
    cudaStream_t stream)
{
    constexpr int block_size = kIcpStatsBlockSize;
    if (detail::canUseSameBufferExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            source_count,
            nullptr))
    {
        collectExactPointwiseResidualStatsKernel<Scalar, true><<<partial_count, block_size, 0, stream>>>(
            d_source_points,
            d_target_points,
            source_count,
            d_partials);
    }
    else
    {
        collectExactPointwiseResidualStatsKernel<Scalar, false><<<partial_count, block_size, 0, stream>>>(
            d_source_points,
            d_target_points,
            source_count,
            d_partials);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
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
    launchExactPointwiseCorrespondencePartials<Scalar>(
        d_source_points,
        d_target_points,
        source_count,
        d_partials,
        partial_count,
        stream);

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
    Scalar* d_step_transform,
    IcpStatsAndStepRawResult* d_result,
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
    launchExactPointwiseCorrespondencePartials<Scalar>(
        d_source_points,
        d_target_points,
        source_count,
        d_partials,
        partial_count,
        stream);

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_exact_pointwise_step_call_count.fetch_add(1, std::memory_order_relaxed);
#endif
    reduceRawIcpStatsAndSetExactPointwiseIdentityStepKernel<Scalar><<<1, block_size, 0, stream>>>(
        d_partials,
        partial_count,
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
    IcpAlignmentStepRawResult<Scalar>* d_result,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    constexpr int block_size = kIcpStatsBlockSize;
    if (assume_ordered_correspondences)
    {
        if (source_count != target_count)
        {
            return false;
        }

        launchOrderedPointwiseCorrespondencePartials<Scalar>(
            d_source_points,
            d_target_points,
            source_count,
            max_correspondence_distance,
            d_partials,
            partial_count,
            stream);

#ifdef PLAPOINT_ENABLE_TESTING
        g_icp_exact_pointwise_step_call_count.fetch_add(1, std::memory_order_relaxed);
#endif
        reduceRawIcpStatsAndComputeAlignmentStepKernel<Scalar><<<1, block_size, 0, stream>>>(
            d_partials,
            partial_count,
            d_step_transform,
            d_result);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
        return true;
    }

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

    launchExactPointwiseCorrespondencePartials<Scalar>(
        d_source_points,
        d_target_points,
        source_count,
        d_partials,
        partial_count,
        stream);

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

template <typename Scalar, bool AccumulateTransform>
bool launchTransformedExactPointwiseAlignmentStep(
    const Scalar* d_source_transform,
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    RawIcpStats* d_partials,
    int partial_count,
    Scalar* d_step_transform,
    const Scalar* d_previous_accumulated_transform,
    Scalar* d_accumulated_transform,
    IcpAlignmentStepRawResult<Scalar>* d_result,
    cudaStream_t stream)
{
    if (!detail::canProbeTransformedExactPointwiseStats(
            source_count,
            d_target_points,
            target_count,
            nullptr))
    {
        return false;
    }

    constexpr int block_size = kIcpStatsBlockSize;
    collectTransformedExactPointwiseCorrespondenceStatsKernel<Scalar>
        <<<partial_count, block_size, 0, stream>>>(
        d_source_transform,
        d_source_points,
        source_count,
        d_target_points,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_transformed_exact_pointwise_alignment_step_call_count.fetch_add(1, std::memory_order_relaxed);
#endif
    if constexpr (AccumulateTransform)
    {
        reduceRawIcpStatsAndSetExactPointwiseIdentityAccumulatedAlignmentStepKernel<Scalar>
            <<<1, block_size, 0, stream>>>(
            d_partials,
            partial_count,
            d_previous_accumulated_transform,
            d_step_transform,
            d_accumulated_transform,
            d_result);
    }
    else
    {
        reduceRawIcpStatsAndSetExactPointwiseIdentityAlignmentStepKernel<Scalar>
            <<<1, block_size, 0, stream>>>(
            d_partials,
            partial_count,
            d_step_transform,
            d_result);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    return true;
}

template <typename Scalar, bool AccumulateTransform>
bool launchTransformedOrderedAlignmentStep(
    const Scalar* d_source_transform,
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    RawIcpStats* d_partials,
    int partial_count,
    Scalar* d_step_transform,
    const Scalar* d_previous_accumulated_transform,
    Scalar* d_accumulated_transform,
    IcpAlignmentStepRawResult<Scalar>* d_result,
    cudaStream_t stream)
{
    if (!d_target_points || source_count != target_count)
    {
        return false;
    }

    constexpr int block_size = kIcpStatsBlockSize;
    transformAndCollectOrderedCorrespondenceStatsKernel<Scalar>
        <<<partial_count, block_size, 0, stream>>>(
        d_source_transform,
        d_source_points,
        source_count,
        d_target_points,
        max_correspondence_distance,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    if constexpr (AccumulateTransform)
    {
        reduceRawIcpStatsAndComputeAlignmentStepAccumulatedTransformKernel<Scalar>
            <<<1, block_size, 0, stream>>>(
            d_partials,
            partial_count,
            d_previous_accumulated_transform,
            d_step_transform,
            d_accumulated_transform,
            d_result);
    }
    else
    {
        reduceRawIcpStatsAndComputeAlignmentStepKernel<Scalar>
            <<<1, block_size, 0, stream>>>(
            d_partials,
            partial_count,
            d_step_transform,
            d_result);
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    return true;
}

template <typename Scalar>
bool launchTransformedExactPointwiseResidualStats(
    const Scalar* d_source_transform,
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar* d_output_points,
    RawIcpResidualStats* d_partials,
    int partial_count,
    RawIcpResidualStats* d_stats,
    cudaStream_t stream)
{
    if (!detail::canProbeTransformedExactPointwiseStats(
            source_count,
            d_target_points,
            target_count,
            nullptr))
    {
        return false;
    }

    constexpr int block_size = kIcpStatsBlockSize;
    collectTransformedExactPointwiseResidualStatsKernel<Scalar>
        <<<partial_count, block_size, 0, stream>>>(
        d_source_transform,
        d_source_points,
        source_count,
        d_target_points,
        d_output_points,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_transformed_exact_pointwise_residual_call_count.fetch_add(1, std::memory_order_relaxed);
#endif
    reduceRawIcpResidualStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        partial_count,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    return true;
}

template <typename Scalar>
bool launchExactPointwiseResidualStats(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    RawIcpResidualStats* d_partials,
    int partial_count,
    RawIcpResidualStats* d_stats,
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
    launchExactPointwiseResidualPartials<Scalar>(
        d_source_points,
        d_target_points,
        source_count,
        d_partials,
        partial_count,
        stream);

    reduceRawIcpResidualStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        partial_count,
        d_stats);
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
    RawIcpStats* h_stats,
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

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return std::isfinite(h_stats->residual_sq_sum);
}

template <typename Scalar>
bool tryComputeExactPointwiseResidualStats(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    RawIcpResidualStats* d_partials,
    int partial_count,
    RawIcpResidualStats* d_stats,
    RawIcpResidualStats* h_stats,
    cudaStream_t stream)
{
    if (!launchExactPointwiseResidualStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_partials,
            partial_count,
            d_stats,
            stream))
    {
        return false;
    }

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return std::isfinite(h_stats->residual_sq_sum);
}

template <typename Scalar>
bool tryComputeTransformedExactPointwiseResidualStats(
    const Scalar* d_source_transform,
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar* d_output_points,
    RawIcpResidualStats* d_partials,
    int partial_count,
    RawIcpResidualStats* d_stats,
    RawIcpResidualStats* h_stats,
    cudaStream_t stream)
{
    if (!launchTransformedExactPointwiseResidualStats(
            d_source_transform,
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            d_output_points,
            d_partials,
            partial_count,
            d_stats,
            stream))
    {
        return false;
    }

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return std::isfinite(h_stats->residual_sq_sum);
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
    if (workspace.targetTileBoundsCacheMatches(d_target_points, target_count))
    {
        return d_bounds;
    }

    const int tile_count = icpTargetTileCount(target_count);
    computeTargetTileBoundsKernel<Scalar><<<tile_count, kIcpStatsBlockSize, 0, stream>>>(
        d_target_points,
        target_count,
        d_bounds);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    workspace.markTargetTileBoundsCache(d_target_points, target_count);
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

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_target_spatial_grid_prepare_count.fetch_add(1, std::memory_order_relaxed);
#endif

    const double cell_size = static_cast<double>(max_correspondence_distance);
    const bool finite_cell_bounds = icpGridCellBoundsAreFinite(cell_size);
    workspace.reserveTargetSpatialGridForScalar<Scalar>(target_count);
    auto* d_keys = reinterpret_cast<IcpGridCellKey*>(workspace.targetSpatialGridKeysStorage());
    auto* d_unique_keys = reinterpret_cast<IcpGridCellKey*>(workspace.targetSpatialGridUniqueKeysStorage());
    auto* d_indices = reinterpret_cast<int*>(workspace.targetSpatialGridIndicesStorage());
    auto* d_sorted_x = reinterpret_cast<Scalar*>(workspace.targetSpatialGridSortedXStorage());
    auto* d_sorted_y = reinterpret_cast<Scalar*>(workspace.targetSpatialGridSortedYStorage());
    auto* d_sorted_z = reinterpret_cast<Scalar*>(workspace.targetSpatialGridSortedZStorage());
    auto* d_cell_starts = reinterpret_cast<int*>(workspace.targetSpatialGridCellStartsStorage());
    auto* d_cell_counts = reinterpret_cast<int*>(workspace.targetSpatialGridCellCountsStorage());

    grid.target_points = d_target_points;
    grid.target_count = target_count;
    grid.cell_keys = d_unique_keys;
    grid.sorted_target_indices = d_indices;
    grid.sorted_target_x = d_sorted_x;
    grid.sorted_target_y = d_sorted_y;
    grid.sorted_target_z = d_sorted_z;
    grid.cell_starts = d_cell_starts;
    grid.cell_counts = d_cell_counts;
    grid.cell_size = cell_size;
    grid.finite_cell_bounds = finite_cell_bounds;

    if (workspace.targetSpatialGridCacheMatchesForScalar<Scalar>(d_target_points, target_count, cell_size))
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
    const int gather_block_size = kIcpStatsBlockSize;
    const int gather_grid_size = icpStatsPartialCount(target_count);
    gatherSortedIcpTargetPointsKernel<Scalar><<<gather_grid_size, gather_block_size, 0, stream>>>(
        d_target_points,
        target_count,
        d_indices,
        d_sorted_x,
        d_sorted_y,
        d_sorted_z);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

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
IcpAlignmentStepResult<Scalar> makeHostAlignmentStepResult(const IcpAlignmentStepRawResult<Scalar>& raw)
{
    IcpAlignmentStepResult<Scalar> result;
    result.active_count = raw.active_count;
    result.invalid_source_count = raw.invalid_source_count;
    result.residual_sq_sum = raw.residual_sq_sum;
    result.src_has_non_collinear_geometry = (raw.flags & kIcpAlignmentStepSrcNonCollinearFlag) != 0;
    result.tgt_has_non_collinear_geometry = (raw.flags & kIcpAlignmentStepTgtNonCollinearFlag) != 0;
    result.step.delta = static_cast<Scalar>(raw.delta);
    result.step_valid = (raw.flags & kIcpAlignmentStepValidFlag) != 0;
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
    auto* h_stats = reinterpret_cast<RawIcpStats*>(active_workspace.hostResultStorage());
    if (!h_stats)
    {
        throw std::invalid_argument("ICP GPU: stats host result workspace is not reserved");
    }
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
            h_stats,
            stream))
    {
        return makeHostStats<Scalar>(*h_stats);
    }

    const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
        d_target_points,
        target_count,
        max_correspondence_distance,
        active_workspace,
        stream);

    if (target_grid.active)
    {
        launchCollectCorrespondenceStatsSpatialGridKernel(
            grid_size,
            block_size,
            stream,
            d_source_points,
            source_count,
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

        launchCollectCorrespondenceStatsKernel(
            grid_size,
            block_size,
            stream,
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

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostStats<Scalar>(*h_stats);
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

    workspace.reserveResidualStats(source_count);

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpResidualStats*>(workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.statsStorage());
    auto* h_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.hostResultStorage());
    if (!h_stats)
    {
        throw std::invalid_argument("ICP GPU: residual-stats host result workspace is not reserved");
    }
    if (tryComputeExactPointwiseResidualStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            max_correspondence_distance,
            d_partials,
            grid_size,
            d_stats,
            h_stats,
            stream))
    {
        return makeHostResidualStats<Scalar>(*h_stats);
    }

    const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
        d_target_points,
        target_count,
        max_correspondence_distance,
        workspace,
        stream);

    if (target_grid.active)
    {
        launchCollectResidualStatsSpatialGridKernel(
            grid_size,
            block_size,
            stream,
            d_source_points,
            source_count,
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

        launchCollectResidualStatsKernel(
            grid_size,
            block_size,
            stream,
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

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostResidualStats<Scalar>(*h_stats);
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
    cudaStream_t stream,
    bool reserve_workspace)
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
    if (!d_transform || !d_source_points || !d_target_points)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }
    if (d_output_points && d_output_points == d_target_points)
    {
        throw std::invalid_argument("ICP GPU: transform residual output must not alias target points");
    }

    if (reserve_workspace)
    {
        workspace.reserveResidualStats(source_count);
    }

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpResidualStats*>(workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.statsStorage());
    auto* h_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.hostResultStorage());
    if (!h_stats)
    {
        throw std::invalid_argument("ICP GPU: residual-stats host result workspace is not reserved");
    }

    const bool use_target_spatial_grid = shouldUseTargetSpatialGrid(max_correspondence_distance);
    const double cell_size = static_cast<double>(max_correspondence_distance);
    const bool target_grid_cache_matches =
        use_target_spatial_grid &&
        workspace.targetSpatialGridCacheMatchesForScalar<Scalar>(
            d_target_points,
            target_count,
            cell_size);
    if (use_target_spatial_grid &&
        !target_grid_cache_matches &&
        tryComputeTransformedExactPointwiseResidualStats(
            d_transform,
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            d_output_points,
            d_partials,
            grid_size,
            d_stats,
            h_stats,
            stream))
    {
        return makeHostResidualStats<Scalar>(*h_stats);
    }

    const IcpTargetSpatialGrid target_grid = prepareTargetSpatialGrid(
        d_target_points,
        target_count,
        max_correspondence_distance,
        workspace,
        stream);

    if (target_grid.active)
    {
        launchTransformAndCollectResidualStatsSpatialGridKernel(
            grid_size,
            block_size,
            stream,
            d_transform,
            d_source_points,
            source_count,
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

        launchTransformAndCollectResidualStatsKernel(
            grid_size,
            block_size,
            stream,
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

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostResidualStats<Scalar>(*h_stats);
}

template <typename Scalar>
IcpResidualStats<Scalar> transformPointsAndComputeOrderedIcpResidualStatsColumnMajorImpl(
    const Scalar* d_transform,
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    Scalar* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream,
    bool reserve_workspace)
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
    if (source_count != target_count)
    {
        throw std::invalid_argument("ICP GPU: ordered residual metrics require equal source and target counts");
    }
    if (!d_transform || !d_source_points || !d_target_points)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }
    if (d_output_points && d_output_points == d_target_points)
    {
        throw std::invalid_argument("ICP GPU: ordered residual output must not alias target points");
    }

    if (reserve_workspace)
    {
        workspace.reserveResidualStats(source_count);
    }

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpResidualStats*>(workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.statsStorage());
    auto* h_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.hostResultStorage());
    if (!h_stats)
    {
        throw std::invalid_argument("ICP GPU: residual-stats host result workspace is not reserved");
    }

    launchTransformAndCollectOrderedResidualStatsKernel(
        grid_size,
        block_size,
        stream,
        d_transform,
        d_source_points,
        source_count,
        d_target_points,
        max_correspondence_distance,
        d_output_points,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpResidualStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        grid_size,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostResidualStats<Scalar>(*h_stats);
}

template <typename Scalar>
IcpResidualStats<Scalar> transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorImpl(
    const Scalar* d_transform,
    const Scalar* d_source_points,
    int source_count,
    Scalar max_correspondence_distance,
    Scalar* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream,
    bool reserve_workspace)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_correspondence_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_residual_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_last_transform_output_pointer.store(
        reinterpret_cast<std::uintptr_t>(d_output_points),
        std::memory_order_relaxed);
#endif

    if (source_count <= 0)
    {
        return {};
    }
    if (!d_transform || !d_source_points)
    {
        throw std::invalid_argument("ICP GPU: device pointers must not be null");
    }
    if (!shouldUseTargetSpatialGrid(max_correspondence_distance) ||
        target_spatial_grid_cell_count <= 0)
    {
        throw std::invalid_argument("ICP GPU: cached target spatial grid snapshot is not available");
    }

    auto* d_cell_keys = reinterpret_cast<IcpGridCellKey*>(workspace.targetSpatialGridUniqueKeysStorage());
    auto* d_indices = reinterpret_cast<int*>(workspace.targetSpatialGridIndicesStorage());
    auto* d_sorted_x = reinterpret_cast<Scalar*>(workspace.targetSpatialGridSortedXStorage());
    auto* d_sorted_y = reinterpret_cast<Scalar*>(workspace.targetSpatialGridSortedYStorage());
    auto* d_sorted_z = reinterpret_cast<Scalar*>(workspace.targetSpatialGridSortedZStorage());
    auto* d_cell_starts = reinterpret_cast<int*>(workspace.targetSpatialGridCellStartsStorage());
    auto* d_cell_counts = reinterpret_cast<int*>(workspace.targetSpatialGridCellCountsStorage());
    if (!d_cell_keys || !d_indices || !d_sorted_x || !d_sorted_y || !d_sorted_z || !d_cell_starts || !d_cell_counts)
    {
        throw std::invalid_argument("ICP GPU: cached target spatial grid snapshot is not available");
    }

    if (reserve_workspace)
    {
        workspace.reserveResidualStats(source_count);
    }

    IcpTargetSpatialGrid target_grid{};
    target_grid.active = true;
    target_grid.target_points = workspace.targetSpatialGridPoints();
    target_grid.target_count = workspace.targetSpatialGridPointCount();
    if (target_grid.target_points == d_output_points)
    {
        target_grid.target_points = nullptr;
        target_grid.target_count = 0;
    }
    target_grid.cell_keys = d_cell_keys;
    target_grid.sorted_target_indices = d_indices;
    target_grid.sorted_target_x = d_sorted_x;
    target_grid.sorted_target_y = d_sorted_y;
    target_grid.sorted_target_z = d_sorted_z;
    target_grid.cell_starts = d_cell_starts;
    target_grid.cell_counts = d_cell_counts;
    target_grid.cell_size = static_cast<double>(max_correspondence_distance);
    target_grid.finite_cell_bounds = icpGridCellBoundsAreFinite(target_grid.cell_size);
    target_grid.cell_count = target_spatial_grid_cell_count;

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpResidualStats*>(workspace.partialStorage());
    auto* d_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.statsStorage());
    auto* h_stats = reinterpret_cast<RawIcpResidualStats*>(workspace.hostResultStorage());
    if (!h_stats)
    {
        throw std::invalid_argument("ICP GPU: residual-stats host result workspace is not reserved");
    }
    launchTransformAndCollectResidualStatsSpatialGridKernel(
        grid_size,
        block_size,
        stream,
        d_transform,
        d_source_points,
        source_count,
        max_correspondence_distance,
        d_output_points,
        target_grid,
        d_partials);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    reduceRawIcpResidualStatsKernel<<<1, block_size, 0, stream>>>(
        d_partials,
        grid_size,
        d_stats);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_stats, d_stats, sizeof(RawIcpResidualStats),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostResidualStats<Scalar>(*h_stats);
}

template <typename Scalar>
void setIdentityTransform4x4Impl(Scalar* d_transform, cudaStream_t stream)
{
    if (!d_transform)
    {
        throw std::invalid_argument("ICP GPU: transform pointer must not be null");
    }

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_identity_transform_write_count.fetch_add(1, std::memory_order_relaxed);
#endif
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
    auto* h_result = reinterpret_cast<IcpStepTransformRawResult*>(active_workspace.hostResultStorage());
    if (!h_result)
    {
        throw std::invalid_argument("ICP GPU: step-transform host result workspace is not reserved");
    }
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

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(IcpStepTransformRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (!h_result->valid)
    {
        throw std::runtime_error("ICP: transform step is not representable");
    }

    IcpStepTransformResult<Scalar> result;
    result.delta = static_cast<Scalar>(h_result->delta);
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

    step_workspace.reserveResult();

    const auto* d_stats = reinterpret_cast<const RawIcpStats*>(stats_workspace.statsStorage());
    auto* d_result = reinterpret_cast<IcpStepTransformRawResult*>(step_workspace.resultStorage());
    auto* h_result = reinterpret_cast<IcpStepTransformRawResult*>(step_workspace.hostResultStorage());
    if (!h_result)
    {
        throw std::invalid_argument("ICP GPU: step-transform host result workspace is not reserved");
    }
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_raw_stats_step_kernel_launch_count.fetch_add(1, std::memory_order_relaxed);
#endif
    computeStepTransformFromRawStatsKernel<Scalar><<<1, 1, 0, stream>>>(
        d_stats,
        d_step_transform,
        d_result);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(IcpStepTransformRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (!h_result->valid)
    {
        throw std::runtime_error("ICP: transform step is not representable");
    }

    IcpStepTransformResult<Scalar> result;
    result.delta = static_cast<Scalar>(h_result->delta);
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
    cudaStream_t stream,
    bool assume_ordered_correspondences)
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

    stats_workspace.reserveStatsAndStep(source_count);
    (void)step_workspace;

    constexpr int block_size = kIcpStatsBlockSize;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpStats*>(stats_workspace.partialStorage());
    auto* d_result = reinterpret_cast<IcpStatsAndStepRawResult*>(stats_workspace.statsStorage());
    auto* h_result = reinterpret_cast<IcpStatsAndStepRawResult*>(stats_workspace.hostResultStorage());
    if (!h_result)
    {
        throw std::invalid_argument("ICP GPU: stats-step host result workspace is not reserved");
    }
    const bool exact_pointwise_stats = launchExactPointwiseStatsAndIdentityStep(
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        nullptr,
        d_partials,
        grid_size,
        d_step_transform,
        d_result,
        stream);

    if (exact_pointwise_stats)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(IcpStatsAndStepRawResult),
                                            cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
        g_icp_stats_step_host_result_copy_count.fetch_add(1, std::memory_order_relaxed);
        g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
        if (std::isfinite(h_result->stats.residual_sq_sum))
        {
            IcpStatsAndStepTransformResult<Scalar> result;
            result.stats = makeHostStats<Scalar>(h_result->stats);
            result.step.delta = static_cast<Scalar>(h_result->step.delta);
            result.step_valid = h_result->step.valid != 0;
            return result;
        }
    }

    if (assume_ordered_correspondences && source_count == target_count)
    {
        launchOrderedPointwiseCorrespondencePartials<Scalar>(
            d_source_points,
            d_target_points,
            source_count,
            max_correspondence_distance,
            d_partials,
            grid_size,
            stream);
        reduceRawIcpStatsAndComputeStepTransformKernel<Scalar><<<1, block_size, 0, stream>>>(
            d_partials,
            grid_size,
            d_step_transform,
            d_result);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());

        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(IcpStatsAndStepRawResult),
                                            cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
        g_icp_stats_step_host_result_copy_count.fetch_add(1, std::memory_order_relaxed);
        g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

        IcpStatsAndStepTransformResult<Scalar> result;
        result.stats = makeHostStats<Scalar>(h_result->stats);
        result.step.delta = static_cast<Scalar>(h_result->step.delta);
        result.step_valid = h_result->step.valid != 0;
        return result;
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
            launchCollectCorrespondenceStatsSpatialGridKernel(
                grid_size,
                block_size,
                stream,
                d_source_points,
                source_count,
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

            launchCollectCorrespondenceStatsKernel(
                grid_size,
                block_size,
                stream,
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
            d_step_transform,
            d_result);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(IcpStatsAndStepRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_stats_step_host_result_copy_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

    IcpStatsAndStepTransformResult<Scalar> result;
    result.stats = makeHostStats<Scalar>(h_result->stats);
    result.step.delta = static_cast<Scalar>(h_result->step.delta);
    result.step_valid = h_result->step.valid != 0;
    return result;
}

template <typename Scalar, bool TransformSource, bool AccumulateTransform>
IcpAlignmentStepResult<Scalar> computeIcpAlignmentStepColumnMajorImpl(
    const Scalar* d_source_transform,
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    Scalar* d_step_transform,
    const Scalar* d_previous_accumulated_transform,
    Scalar* d_accumulated_transform,
    cudaStream_t stream,
    bool reserve_workspace,
    bool assume_ordered_correspondences)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_correspondence_stats_call_count.fetch_add(1, std::memory_order_relaxed);
    g_icp_alignment_step_call_count.fetch_add(1, std::memory_order_relaxed);
    if constexpr (TransformSource)
    {
        g_icp_transformed_alignment_step_call_count.fetch_add(1, std::memory_order_relaxed);
    }
    if constexpr (AccumulateTransform)
    {
        g_icp_accumulated_alignment_step_call_count.fetch_add(1, std::memory_order_relaxed);
    }
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
    if constexpr (TransformSource)
    {
        if (!d_source_transform)
        {
            throw std::invalid_argument("ICP GPU: source transform pointer must not be null");
        }
    }
    if constexpr (AccumulateTransform)
    {
        if (!d_previous_accumulated_transform || !d_accumulated_transform)
        {
            throw std::invalid_argument("ICP GPU: accumulated transform pointers must not be null");
        }
    }

    if (reserve_workspace)
    {
        if constexpr (std::is_same_v<Scalar, float>)
        {
            stats_workspace.reserveFloatAlignmentStep(source_count);
        }
        else
        {
            stats_workspace.reserveDoubleAlignmentStep(source_count);
        }
    }

    constexpr int block_size = kIcpStatsBlockSize;
    using AlignmentStepRawResult = IcpAlignmentStepRawResult<Scalar>;
    const int grid_size = icpStatsPartialCount(source_count);
    auto* d_partials = reinterpret_cast<RawIcpStats*>(stats_workspace.partialStorage());
    auto* d_result = reinterpret_cast<AlignmentStepRawResult*>(stats_workspace.statsStorage());
    auto* h_result = reinterpret_cast<AlignmentStepRawResult*>(stats_workspace.hostResultStorage());
    if (!h_result)
    {
        throw std::invalid_argument("ICP GPU: alignment-step host result workspace is not reserved");
    }

    if constexpr (!TransformSource)
    {
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
            stream,
            assume_ordered_correspondences);

        if (exact_pointwise_stats)
        {
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(AlignmentStepRawResult),
                                                cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
            g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            if (std::isfinite(h_result->residual_sq_sum))
            {
                return makeHostAlignmentStepResult<Scalar>(*h_result);
            }
        }
    }
    else
    {
        if (assume_ordered_correspondences)
        {
            const bool ordered_pointwise_stats =
                launchTransformedOrderedAlignmentStep<Scalar, AccumulateTransform>(
                    d_source_transform,
                    d_source_points,
                    source_count,
                    d_target_points,
                    target_count,
                    max_correspondence_distance,
                    d_partials,
                    grid_size,
                    d_step_transform,
                    d_previous_accumulated_transform,
                    d_accumulated_transform,
                    d_result,
                    stream);
            if (ordered_pointwise_stats)
            {
                PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(AlignmentStepRawResult),
                                                    cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
                g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
                return makeHostAlignmentStepResult<Scalar>(*h_result);
            }
        }

        const bool use_target_spatial_grid = shouldUseTargetSpatialGrid(max_correspondence_distance);
        const double cell_size = static_cast<double>(max_correspondence_distance);
        const bool target_grid_cache_matches =
            use_target_spatial_grid &&
            stats_workspace.targetSpatialGridCacheMatchesForScalar<Scalar>(
                d_target_points,
                target_count,
                cell_size);
        if (use_target_spatial_grid && !target_grid_cache_matches)
        {
            const bool exact_pointwise_stats =
                launchTransformedExactPointwiseAlignmentStep<Scalar, AccumulateTransform>(
                    d_source_transform,
                    d_source_points,
                    source_count,
                    d_target_points,
                    target_count,
                    d_partials,
                    grid_size,
                    d_step_transform,
                    d_previous_accumulated_transform,
                    d_accumulated_transform,
                    d_result,
                    stream);

            if (exact_pointwise_stats)
            {
                PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(AlignmentStepRawResult),
                                                    cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
                g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
                if (std::isfinite(h_result->residual_sq_sum))
                {
                    return makeHostAlignmentStepResult<Scalar>(*h_result);
                }
            }
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
            if constexpr (TransformSource)
            {
                launchTransformAndCollectCorrespondenceStatsSpatialGridKernel(
                    grid_size,
                    block_size,
                    stream,
                    d_source_transform,
                    d_source_points,
                    source_count,
                    max_correspondence_distance,
                    target_grid,
                    d_partials);
            }
            else
            {
                launchCollectCorrespondenceStatsSpatialGridKernel(
                    grid_size,
                    block_size,
                    stream,
                    d_source_points,
                    source_count,
                    max_correspondence_distance,
                    nullptr,
                    target_grid,
                    d_partials);
            }
        }
        else
        {
            const IcpTargetTileBounds* d_target_tile_bounds = prepareTargetTileBounds(
                d_target_points,
                target_count,
                max_correspondence_distance,
                stats_workspace,
                stream);

            if constexpr (TransformSource)
            {
                launchTransformAndCollectCorrespondenceStatsKernel(
                    grid_size,
                    block_size,
                    stream,
                    d_source_transform,
                    d_source_points,
                    source_count,
                    d_target_points,
                    target_count,
                    max_correspondence_distance,
                    d_target_tile_bounds,
                    d_partials);
            }
            else
            {
                launchCollectCorrespondenceStatsKernel(
                    grid_size,
                    block_size,
                    stream,
                    d_source_points,
                    source_count,
                    d_target_points,
                    target_count,
                    max_correspondence_distance,
                    nullptr,
                    d_target_tile_bounds,
                    d_partials);
            }
        }
        PLAPOINT_CHECK_CUDA(cudaGetLastError());

        if constexpr (AccumulateTransform)
        {
            reduceRawIcpStatsAndComputeAlignmentStepAccumulatedTransformKernel<Scalar><<<1, block_size, 0, stream>>>(
                d_partials,
                grid_size,
                d_previous_accumulated_transform,
                d_step_transform,
                d_accumulated_transform,
                d_result);
        }
        else
        {
            reduceRawIcpStatsAndComputeAlignmentStepKernel<Scalar><<<1, block_size, 0, stream>>>(
                d_partials,
                grid_size,
                d_step_transform,
                d_result);
        }
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }

    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(AlignmentStepRawResult),
                                        cudaMemcpyDeviceToHost, stream));
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_host_synchronization_count.fetch_add(1, std::memory_order_relaxed);
#endif
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return makeHostAlignmentStepResult<Scalar>(*h_result);
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

void resetIcpTargetIndexLoadCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_target_index_load_count, &zero, sizeof(zero)));
}

unsigned long long icpTargetIndexLoadCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_target_index_load_count, sizeof(count)));
    return count;
}

void resetIcpSortedTargetCoordinateLoadCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_sorted_target_coordinate_load_count, &zero, sizeof(zero)));
}

unsigned long long icpSortedTargetCoordinateLoadCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_sorted_target_coordinate_load_count, sizeof(count)));
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

void resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting()
{
    g_icp_transformed_exact_pointwise_alignment_step_call_count.store(0, std::memory_order_relaxed);
}

int icpTransformedExactPointwiseAlignmentStepCallCountForTesting()
{
    return g_icp_transformed_exact_pointwise_alignment_step_call_count.load(std::memory_order_relaxed);
}

void resetIcpTransformedExactPointwiseResidualCallCountForTesting()
{
    g_icp_transformed_exact_pointwise_residual_call_count.store(0, std::memory_order_relaxed);
}

int icpTransformedExactPointwiseResidualCallCountForTesting()
{
    return g_icp_transformed_exact_pointwise_residual_call_count.load(std::memory_order_relaxed);
}

void resetIcpExactPointwiseTargetLoadCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_exact_pointwise_target_load_count, &zero, sizeof(zero)));
}

unsigned long long icpExactPointwiseTargetLoadCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_exact_pointwise_target_load_count, sizeof(count)));
    return count;
}

void resetIcpRawStatsStepKernelLaunchCountForTesting()
{
    g_icp_raw_stats_step_kernel_launch_count.store(0, std::memory_order_relaxed);
}

int icpRawStatsStepKernelLaunchCountForTesting()
{
    return g_icp_raw_stats_step_kernel_launch_count.load(std::memory_order_relaxed);
}

void resetIcpStatsStepHostResultCopyCountForTesting()
{
    g_icp_stats_step_host_result_copy_count.store(0, std::memory_order_relaxed);
}

int icpStatsStepHostResultCopyCountForTesting()
{
    return g_icp_stats_step_host_result_copy_count.load(std::memory_order_relaxed);
}

void resetIcpAlignmentStepCallCountForTesting()
{
    g_icp_alignment_step_call_count.store(0, std::memory_order_relaxed);
}

int icpAlignmentStepCallCountForTesting()
{
    return g_icp_alignment_step_call_count.load(std::memory_order_relaxed);
}

void resetIcpTransformedAlignmentStepCallCountForTesting()
{
    g_icp_transformed_alignment_step_call_count.store(0, std::memory_order_relaxed);
}

int icpTransformedAlignmentStepCallCountForTesting()
{
    return g_icp_transformed_alignment_step_call_count.load(std::memory_order_relaxed);
}

void resetIcpAccumulatedAlignmentStepCallCountForTesting()
{
    g_icp_accumulated_alignment_step_call_count.store(0, std::memory_order_relaxed);
}

int icpAccumulatedAlignmentStepCallCountForTesting()
{
    return g_icp_accumulated_alignment_step_call_count.load(std::memory_order_relaxed);
}

void resetIcpAlignmentStepReserveCountForTesting()
{
    g_icp_alignment_step_reserve_count.store(0, std::memory_order_relaxed);
}

int icpAlignmentStepReserveCountForTesting()
{
    return g_icp_alignment_step_reserve_count.load(std::memory_order_relaxed);
}

void resetIcpAlignmentStepReserveCheckCountForTesting()
{
    g_icp_alignment_step_reserve_check_count.store(0, std::memory_order_relaxed);
}

int icpAlignmentStepReserveCheckCountForTesting()
{
    return g_icp_alignment_step_reserve_check_count.load(std::memory_order_relaxed);
}

std::size_t icpAlignmentStepRawResultByteCountForTesting()
{
    return sizeof(IcpAlignmentStepRawResult<double>);
}

std::size_t icpFloatAlignmentStepRawResultByteCountForTesting()
{
    return sizeof(IcpAlignmentStepRawResult<float>);
}

std::size_t icpDoubleAlignmentStepRawResultByteCountForTesting()
{
    return sizeof(IcpAlignmentStepRawResult<double>);
}

void resetIcpResidualStatsReserveCheckCountForTesting()
{
    g_icp_residual_stats_reserve_check_count.store(0, std::memory_order_relaxed);
}

int icpResidualStatsReserveCheckCountForTesting()
{
    return g_icp_residual_stats_reserve_check_count.load(std::memory_order_relaxed);
}

void resetIcpHostSynchronizationCountForTesting()
{
    g_icp_host_synchronization_count.store(0, std::memory_order_relaxed);
}

int icpHostSynchronizationCountForTesting()
{
    return g_icp_host_synchronization_count.load(std::memory_order_relaxed);
}

void resetIcpHostResultStorageAllocationCountForTesting()
{
    g_icp_host_result_storage_allocation_count.store(0, std::memory_order_relaxed);
}

int icpHostResultStorageAllocationCountForTesting()
{
    return g_icp_host_result_storage_allocation_count.load(std::memory_order_relaxed);
}

void resetIcpIdentityTransformWriteCountForTesting()
{
    g_icp_identity_transform_write_count.store(0, std::memory_order_relaxed);
}

int icpIdentityTransformWriteCountForTesting()
{
    return g_icp_identity_transform_write_count.load(std::memory_order_relaxed);
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

void resetIcpTargetTileLoadCountForTesting()
{
    const unsigned long long zero = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_target_tile_load_count, &zero, sizeof(zero)));
}

unsigned long long icpTargetTileLoadCountForTesting()
{
    unsigned long long count = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(&count, g_icp_target_tile_load_count, sizeof(count)));
    return count;
}

void resetIcpFallbackTileBoundKernelLaunchCountForTesting()
{
    g_icp_fallback_tile_bound_kernel_launch_count.store(0, std::memory_order_relaxed);
}

int icpFallbackTileBoundKernelLaunchCountForTesting()
{
    return g_icp_fallback_tile_bound_kernel_launch_count.load(std::memory_order_relaxed);
}

void resetIcpFallbackUnboundedKernelLaunchCountForTesting()
{
    g_icp_fallback_unbounded_kernel_launch_count.store(0, std::memory_order_relaxed);
}

int icpFallbackUnboundedKernelLaunchCountForTesting()
{
    return g_icp_fallback_unbounded_kernel_launch_count.load(std::memory_order_relaxed);
}

void resetIcpTargetSpatialGridBuildCountForTesting()
{
    g_icp_target_spatial_grid_build_count.store(0, std::memory_order_relaxed);
}

int icpTargetSpatialGridBuildCountForTesting()
{
    return g_icp_target_spatial_grid_build_count.load(std::memory_order_relaxed);
}

void resetIcpTargetSpatialGridPrepareCountForTesting()
{
    g_icp_target_spatial_grid_prepare_count.store(0, std::memory_order_relaxed);
}

int icpTargetSpatialGridPrepareCountForTesting()
{
    return g_icp_target_spatial_grid_prepare_count.load(std::memory_order_relaxed);
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

    reservePartialStorage(source_count, sizeof(RawIcpStats));
    reserveStatsStorage(sizeof(RawIcpStats));
    reserveHostResultStorage(sizeof(RawIcpStats));
}

void IcpCorrespondenceStatsWorkspace::reserveAlignmentStepStorage(
    int source_count,
    std::size_t result_byte_count)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_alignment_step_reserve_check_count.fetch_add(1, std::memory_order_relaxed);
#endif

    if (source_count < 0)
    {
        throw std::invalid_argument("ICP GPU: source point count must not be negative");
    }
    if (source_count == 0)
    {
        return;
    }

    const int required_partials = icpStatsPartialCount(source_count);
    const std::size_t required_partial_bytes =
        static_cast<std::size_t>(required_partials) * sizeof(RawIcpStats);
    if (partialCapacity() >= required_partials &&
        _partial_storage.size() >= required_partial_bytes &&
        _stats_storage.size() >= result_byte_count &&
        _host_result_storage.size() >= result_byte_count)
    {
        return;
    }

#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_alignment_step_reserve_count.fetch_add(1, std::memory_order_relaxed);
#endif

    reservePartialStorage(source_count, sizeof(RawIcpStats));
    reserveStatsStorage(result_byte_count);
    reserveHostResultStorage(result_byte_count);
}

void IcpCorrespondenceStatsWorkspace::reserveAlignmentStep(int source_count)
{
    reserveDoubleAlignmentStep(source_count);
}

void IcpCorrespondenceStatsWorkspace::reserveFloatAlignmentStep(int source_count)
{
    reserveAlignmentStepStorage(source_count, sizeof(IcpAlignmentStepRawResult<float>));
}

void IcpCorrespondenceStatsWorkspace::reserveDoubleAlignmentStep(int source_count)
{
    reserveAlignmentStepStorage(source_count, sizeof(IcpAlignmentStepRawResult<double>));
}

void IcpCorrespondenceStatsWorkspace::reserveStatsAndStep(int source_count)
{
    if (source_count < 0)
    {
        throw std::invalid_argument("ICP GPU: source point count must not be negative");
    }
    if (source_count == 0)
    {
        return;
    }

    reservePartialStorage(source_count, sizeof(RawIcpStats));
    reserveStatsStorage(sizeof(IcpStatsAndStepRawResult));
    reserveHostResultStorage(sizeof(IcpStatsAndStepRawResult));
}

void IcpCorrespondenceStatsWorkspace::reserveResidualStats(int source_count)
{
#ifdef PLAPOINT_ENABLE_TESTING
    g_icp_residual_stats_reserve_check_count.fetch_add(1, std::memory_order_relaxed);
#endif

    if (source_count < 0)
    {
        throw std::invalid_argument("ICP GPU: source point count must not be negative");
    }
    if (source_count == 0)
    {
        return;
    }

    reservePartialStorage(source_count, sizeof(RawIcpResidualStats));
    reserveStatsStorage(sizeof(RawIcpResidualStats));
    reserveHostResultStorage(sizeof(RawIcpResidualStats));
}

void IcpCorrespondenceStatsWorkspace::reservePartialStorage(int source_count, std::size_t bytes_per_partial)
{
    const int required_partials = icpStatsPartialCount(source_count);
    const std::size_t required_bytes = static_cast<std::size_t>(required_partials) * bytes_per_partial;
    if (_partial_storage.size() < required_bytes)
    {
        _partial_storage.allocate(required_bytes);
    }
    if (partialCapacity() < required_partials)
    {
        _partial_capacity = required_partials;
    }
}

void IcpCorrespondenceStatsWorkspace::reserveStatsStorage(std::size_t byte_count)
{
    if (_stats_storage.size() < byte_count)
    {
        _stats_storage.allocate(byte_count);
    }
}

void IcpCorrespondenceStatsWorkspace::reserveHostResultStorage(std::size_t byte_count)
{
    if (_host_result_storage.size() < byte_count)
    {
        _host_result_storage.allocate(byte_count);
#ifdef PLAPOINT_ENABLE_TESTING
        g_icp_host_result_storage_allocation_count.fetch_add(1, std::memory_order_relaxed);
#endif
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
        invalidateTargetTileBoundsCache();
        _target_tile_bounds_storage.allocate(
            static_cast<std::size_t>(required_tiles) * sizeof(IcpTargetTileBounds));
        _target_tile_bound_capacity = required_tiles;
    }
}

void IcpCorrespondenceStatsWorkspace::invalidateTargetTileBoundsCache()
{
    _target_tile_bounds_cache_valid = false;
    _target_tile_bounds_points = nullptr;
    _target_tile_bounds_point_count = 0;
}

void IcpCorrespondenceStatsWorkspace::reserveTargetSpatialGrid(int target_count)
{
    reserveTargetSpatialGrid(target_count, sizeof(double));
}

void IcpCorrespondenceStatsWorkspace::reserveTargetSpatialGrid(
    int target_count,
    std::size_t coordinate_value_bytes)
{
    if (target_count < 0)
    {
        throw std::invalid_argument("ICP GPU: target point count must not be negative");
    }
    if (target_count == 0)
    {
        return;
    }

    if (detail::targetSpatialGridCoordinateStorageNeedsReserve(
            targetSpatialGridCapacity(),
            _target_spatial_grid_coordinate_value_bytes,
            target_count,
            coordinate_value_bytes))
    {
        invalidateTargetSpatialGridCache();
        const int required_capacity = std::max(targetSpatialGridCapacity(), target_count);
        _target_spatial_grid_keys_storage.allocate(
            static_cast<std::size_t>(required_capacity) * sizeof(IcpGridCellKey));
        _target_spatial_grid_unique_keys_storage.allocate(
            static_cast<std::size_t>(required_capacity) * sizeof(IcpGridCellKey));
        _target_spatial_grid_indices_storage.allocate(
            static_cast<std::size_t>(required_capacity) * sizeof(int));
        _target_spatial_grid_sorted_x_storage.allocate(
            static_cast<std::size_t>(required_capacity) * coordinate_value_bytes);
        _target_spatial_grid_sorted_y_storage.allocate(
            static_cast<std::size_t>(required_capacity) * coordinate_value_bytes);
        _target_spatial_grid_sorted_z_storage.allocate(
            static_cast<std::size_t>(required_capacity) * coordinate_value_bytes);
        _target_spatial_grid_cell_starts_storage.allocate(
            static_cast<std::size_t>(required_capacity) * sizeof(int));
        _target_spatial_grid_cell_counts_storage.allocate(
            static_cast<std::size_t>(required_capacity) * sizeof(int));
        _target_spatial_grid_capacity = required_capacity;
        _target_spatial_grid_coordinate_value_bytes = coordinate_value_bytes;
    }
}

void IcpCorrespondenceStatsWorkspace::invalidateTargetSpatialGridCache()
{
    invalidateTargetTileBoundsCache();
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
    return targetSpatialGridCacheMatches(target_points, target_count, cell_size, sizeof(double));
}

bool IcpCorrespondenceStatsWorkspace::targetSpatialGridCacheMatches(
    const void* target_points,
    int target_count,
    double cell_size,
    std::size_t coordinate_value_bytes) const
{
    return _target_spatial_grid_cache_valid &&
        _target_spatial_grid_points == target_points &&
        _target_spatial_grid_point_count == target_count &&
        _target_spatial_grid_cell_size == cell_size &&
        _target_spatial_grid_coordinate_value_bytes == coordinate_value_bytes;
}

bool IcpCorrespondenceStatsWorkspace::targetTileBoundsCacheMatches(
    const void* target_points,
    int target_count) const
{
    return _target_tile_bounds_cache_valid &&
        _target_tile_bounds_points == target_points &&
        _target_tile_bounds_point_count == target_count;
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

void IcpCorrespondenceStatsWorkspace::markTargetTileBoundsCache(
    const void* target_points,
    int target_count)
{
    _target_tile_bounds_cache_valid = true;
    _target_tile_bounds_points = target_points;
    _target_tile_bounds_point_count = target_count;
}

void IcpStepTransformWorkspace::reserve()
{
    if (_input_storage.size() < sizeof(IcpStepTransformInput))
    {
        _input_storage.allocate(sizeof(IcpStepTransformInput));
    }
    reserveResult();
}

void IcpStepTransformWorkspace::reserveResult()
{
    if (_result_storage.size() < sizeof(IcpStepTransformRawResult))
    {
        _result_storage.allocate(sizeof(IcpStepTransformRawResult));
    }
    if (_host_result_storage.size() < sizeof(IcpStepTransformRawResult))
    {
        _host_result_storage.allocate(sizeof(IcpStepTransformRawResult));
#ifdef PLAPOINT_ENABLE_TESTING
        g_icp_host_result_storage_allocation_count.fetch_add(1, std::memory_order_relaxed);
#endif
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
        stream,
        true);
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
        stream,
        true);
}

IcpResidualStats<float> transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajor(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream)
{
    return transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
        target_spatial_grid_cell_count,
        stream,
        true);
}

IcpResidualStats<double> transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajor(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream)
{
    return transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
        target_spatial_grid_cell_count,
        stream,
        true);
}

namespace detail
{

IcpResidualStats<float> transformPointsAndComputeIcpResidualStatsColumnMajorWithReservedWorkspace(
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
        stream,
        false);
}

IcpResidualStats<double> transformPointsAndComputeIcpResidualStatsColumnMajorWithReservedWorkspace(
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
        stream,
        false);
}

IcpResidualStats<float> transformPointsAndComputeOrderedIcpResidualStatsColumnMajorWithReservedWorkspace(
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
    return transformPointsAndComputeOrderedIcpResidualStatsColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
        stream,
        false);
}

IcpResidualStats<double> transformPointsAndComputeOrderedIcpResidualStatsColumnMajorWithReservedWorkspace(
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
    return transformPointsAndComputeOrderedIcpResidualStatsColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
        stream,
        false);
}

IcpResidualStats<float>
transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorWithReservedWorkspace(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream)
{
    return transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
        target_spatial_grid_cell_count,
        stream,
        false);
}

IcpResidualStats<double>
transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorWithReservedWorkspace(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream)
{
    return transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorImpl(
        d_transform,
        d_source_points,
        source_count,
        max_correspondence_distance,
        d_output_points,
        workspace,
        target_spatial_grid_cell_count,
        stream,
        false);
}

} // namespace detail

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
    cudaStream_t stream,
    bool assume_ordered_correspondences)
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
        stream,
        assume_ordered_correspondences);
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
    cudaStream_t stream,
    bool assume_ordered_correspondences)
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
        stream,
        assume_ordered_correspondences);
}

IcpAlignmentStepResult<float> computeIcpAlignmentStepColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<float, false, false>(
        nullptr,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        nullptr,
        nullptr,
        stream,
        true,
        assume_ordered_correspondences);
}

IcpAlignmentStepResult<double> computeIcpAlignmentStepColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<double, false, false>(
        nullptr,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        nullptr,
        nullptr,
        stream,
        true,
        assume_ordered_correspondences);
}

namespace detail
{

IcpAlignmentStepResult<float> computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<float, false, false>(
        nullptr,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        nullptr,
        nullptr,
        stream,
        false,
        assume_ordered_correspondences);
}

IcpAlignmentStepResult<double> computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<double, false, false>(
        nullptr,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        nullptr,
        nullptr,
        stream,
        false,
        assume_ordered_correspondences);
}

IcpAlignmentStepResult<float> computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const float* d_source_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<float, true, false>(
        d_source_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        nullptr,
        nullptr,
        stream,
        false,
        assume_ordered_correspondences);
}

IcpAlignmentStepResult<double> computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const double* d_source_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<double, true, false>(
        d_source_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        nullptr,
        nullptr,
        stream,
        false,
        assume_ordered_correspondences);
}

IcpAlignmentStepResult<float> computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
    const float* d_source_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    const float* d_previous_accumulated_transform,
    float* d_accumulated_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<float, true, true>(
        d_source_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        d_previous_accumulated_transform,
        d_accumulated_transform,
        stream,
        false,
        assume_ordered_correspondences);
}

IcpAlignmentStepResult<double> computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
    const double* d_source_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    const double* d_previous_accumulated_transform,
    double* d_accumulated_transform,
    cudaStream_t stream,
    bool assume_ordered_correspondences)
{
    return computeIcpAlignmentStepColumnMajorImpl<double, true, true>(
        d_source_transform,
        d_source_points,
        source_count,
        d_target_points,
        target_count,
        max_correspondence_distance,
        stats_workspace,
        d_step_transform,
        d_previous_accumulated_transform,
        d_accumulated_transform,
        stream,
        false,
        assume_ordered_correspondences);
}

} // namespace detail

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
    synchronizeIcpStreamWithHost(stream);
}

void transformPointsColumnMajor(
    const double* d_transform,
    const double* d_points,
    int point_count,
    double* d_output_points,
    cudaStream_t stream)
{
    transformPointsColumnMajorImpl(d_transform, d_points, point_count, d_output_points, stream);
    synchronizeIcpStreamWithHost(stream);
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
