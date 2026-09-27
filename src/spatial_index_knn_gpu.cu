#include <plapoint/gpu/spatial_index.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <plapoint/gpu/detail/distance_key.cuh>
#include <plamatrix/internal/device/device_matrix.h>

namespace plapoint
{
namespace gpu
{
namespace
{

constexpr long long kMaximumEnumeratedShell = 4096;

template <typename Scalar>
struct KnnGridView
{
    const Scalar* points;
    plamatrix::Index point_count;
    const std::uint64_t* keys;
    const plamatrix::Index* sorted_indices;
    const plamatrix::Index* offsets;
    const plamatrix::Index* counts;
    int cell_count;
    double cell_size;
    long long origin_x;
    long long origin_y;
    long long origin_z;
    long long span_x;
    long long span_y;
    long long span_z;
};

__device__ int findKnnCell(const std::uint64_t* keys, int count, std::uint64_t key)
{
    int first = 0;
    int last = count;
    while (first < last)
    {
        const int middle = first + (last - first) / 2;
        if (keys[middle] < key)
        {
            first = middle + 1;
        }
        else
        {
            last = middle;
        }
    }
    return first < count && keys[first] == key ? first : -1;
}

using detail::DistanceKey;
using detail::distanceLess;
using detail::finiteDistanceKey;

template <typename Scalar>
struct KnnCollector
{
    plamatrix::Index* indices;
    Scalar* distances;
    double* distance_keys;
    int* distance_exponents;
    int query;
    int query_count;
    int k;
    int count = 0;
    DistanceKey kth_distance{INT_MAX, DBL_MAX};

    __device__ void add(plamatrix::Index index, const DistanceKey& distance)
    {
        int position = count;
        if (position >= k)
        {
            position = k - 1;
            const std::size_t last_offset = static_cast<std::size_t>(query)
                + static_cast<std::size_t>(position) * query_count;
            const double last_distance = distance_keys[last_offset];
            const DistanceKey last_key{distance_exponents[last_offset], last_distance};
            const auto last_index = indices[last_offset];
            if (distanceLess(last_key, distance)
                || (!distanceLess(distance, last_key) && index >= last_index))
            {
                return;
            }
        }
        else
        {
            ++count;
        }

        while (position > 0)
        {
            const std::size_t previous_offset = static_cast<std::size_t>(query)
                + static_cast<std::size_t>(position - 1) * query_count;
            const double previous_distance = distance_keys[previous_offset];
            const DistanceKey previous_key{
                distance_exponents[previous_offset], previous_distance};
            const auto previous_index = indices[previous_offset];
            if (distanceLess(previous_key, distance)
                || (!distanceLess(distance, previous_key) && previous_index <= index))
            {
                break;
            }
            const std::size_t offset = static_cast<std::size_t>(query)
                + static_cast<std::size_t>(position) * query_count;
            indices[offset] = previous_index;
            distances[offset] = distances[previous_offset];
            distance_keys[offset] = previous_distance;
            distance_exponents[offset] = previous_key.exponent;
            --position;
        }
        const std::size_t offset = static_cast<std::size_t>(query)
            + static_cast<std::size_t>(position) * query_count;
        indices[offset] = index;
        distances[offset] = detail::squaredOutputDistance<Scalar>(distance);
        distance_keys[offset] = distance.mantissa;
        distance_exponents[offset] = distance.exponent;
        if (count == k)
        {
            const auto kth_offset = static_cast<std::size_t>(query)
                + static_cast<std::size_t>(k - 1) * query_count;
            kth_distance = {distance_exponents[kth_offset], distance_keys[kth_offset]};
        }
    }
};

template <typename Scalar>
__device__ void visitKnnCell(
    const KnnGridView<Scalar>& grid,
    int cell,
    double qx,
    double qy,
    double qz,
    KnnCollector<Scalar>& collector)
{
    for (plamatrix::Index offset = 0; offset < grid.counts[cell]; ++offset)
    {
        const auto index = grid.sorted_indices[grid.offsets[cell] + offset];
        const DistanceKey distance = finiteDistanceKey(
            qx, qy, qz,
            static_cast<double>(grid.points[index]),
            static_cast<double>(grid.points[grid.point_count + index]),
            static_cast<double>(grid.points[2 * grid.point_count + index]));
        collector.add(index, distance);
    }
}

template <typename Scalar>
__device__ void scanOccupiedCells(
    const KnnGridView<Scalar>& grid,
    double qx,
    double qy,
    double qz,
    KnnCollector<Scalar>& collector)
{
    for (int cell = 0; cell < grid.cell_count; ++cell)
    {
        visitKnnCell(grid, cell, qx, qy, qz, collector);
    }
}

template <typename Scalar>
__device__ void visitGridCoordinate(
    const KnnGridView<Scalar>& grid,
    long long x,
    long long y,
    long long z,
    double qx,
    double qy,
    double qz,
    KnnCollector<Scalar>& collector)
{
    if (x < 0 || x > grid.span_x
        || y < 0 || y > grid.span_y
        || z < 0 || z > grid.span_z)
    {
        return;
    }
    const std::uint64_t key = (static_cast<std::uint64_t>(x) << 42)
        | (static_cast<std::uint64_t>(y) << 21)
        | static_cast<std::uint64_t>(z);
    const int cell = findKnnCell(grid.keys, grid.cell_count, key);
    if (cell >= 0)
    {
        visitKnnCell(grid, cell, qx, qy, qz, collector);
    }
}

__device__ long long maximumShell(
    long long x, long long y, long long z,
    long long span_x, long long span_y, long long span_z)
{
    const long long x_distance = max(llabs(x), llabs(x - span_x));
    const long long y_distance = max(llabs(y), llabs(y - span_y));
    const long long z_distance = max(llabs(z), llabs(z - span_z));
    return max(x_distance, max(y_distance, z_distance));
}

template <typename Scalar>
__device__ void visitShells(
    const KnnGridView<Scalar>& grid,
    double qx,
    double qy,
    double qz,
    KnnCollector<Scalar>& collector)
{
    const double cell_x = floor(qx / grid.cell_size);
    const double cell_y = floor(qy / grid.cell_size);
    const double cell_z = floor(qz / grid.cell_size);
    const double normalized_x = cell_x - static_cast<double>(grid.origin_x);
    const double normalized_y = cell_y - static_cast<double>(grid.origin_y);
    const double normalized_z = cell_z - static_cast<double>(grid.origin_z);
    if (fabs(normalized_x) > 9007199254740991.0
        || fabs(normalized_y) > 9007199254740991.0
        || fabs(normalized_z) > 9007199254740991.0)
    {
        scanOccupiedCells(grid, qx, qy, qz, collector);
        return;
    }

    const long long center_x = static_cast<long long>(normalized_x);
    const long long center_y = static_cast<long long>(normalized_y);
    const long long center_z = static_cast<long long>(normalized_z);
    const long long last_shell = maximumShell(
        center_x, center_y, center_z, grid.span_x, grid.span_y, grid.span_z);
    if (last_shell > kMaximumEnumeratedShell)
    {
        scanOccupiedCells(grid, qx, qy, qz, collector);
        return;
    }

    const double fraction_x = qx / grid.cell_size - cell_x;
    const double fraction_y = qy / grid.cell_size - cell_y;
    const double fraction_z = qz / grid.cell_size - cell_z;
    const double boundary_fraction = min(
        min(fraction_x, 1.0 - fraction_x),
        min(min(fraction_y, 1.0 - fraction_y), min(fraction_z, 1.0 - fraction_z)));

    for (long long shell = 0; shell <= last_shell; ++shell)
    {
        const long long first_x = max(0LL, center_x - shell);
        const long long last_x = min(grid.span_x, center_x + shell);
        const long long first_y = max(0LL, center_y - shell);
        const long long last_y = min(grid.span_y, center_y + shell);
        const long long first_z = max(0LL, center_z - shell);
        const long long last_z = min(grid.span_z, center_z + shell);
        const long long negative_z = center_z - shell;
        const long long positive_z = center_z + shell;
        for (long long x = first_x; x <= last_x; ++x)
        {
            for (long long y = first_y; y <= last_y; ++y)
            {
                visitGridCoordinate(
                    grid, x, y, negative_z, qx, qy, qz, collector);
                if (positive_z != negative_z)
                {
                    visitGridCoordinate(
                        grid, x, y, positive_z, qx, qy, qz, collector);
                }
            }
        }

        const long long interior_first_z = max(first_z, center_z - shell + 1);
        const long long interior_last_z = min(last_z, center_z + shell - 1);
        const long long negative_x = center_x - shell;
        const long long positive_x = center_x + shell;
        for (long long y = first_y; y <= last_y; ++y)
        {
            for (long long z = interior_first_z; z <= interior_last_z; ++z)
            {
                visitGridCoordinate(
                    grid, negative_x, y, z, qx, qy, qz, collector);
                if (positive_x != negative_x)
                {
                    visitGridCoordinate(
                        grid, positive_x, y, z, qx, qy, qz, collector);
                }
            }
        }

        const long long interior_first_x = max(first_x, center_x - shell + 1);
        const long long interior_last_x = min(last_x, center_x + shell - 1);
        const long long negative_y = center_y - shell;
        const long long positive_y = center_y + shell;
        for (long long x = interior_first_x; x <= interior_last_x; ++x)
        {
            for (long long z = interior_first_z; z <= interior_last_z; ++z)
            {
                visitGridCoordinate(
                    grid, x, negative_y, z, qx, qy, qz, collector);
                if (positive_y != negative_y)
                {
                    visitGridCoordinate(
                        grid, x, positive_y, z, qx, qy, qz, collector);
                }
            }
        }

        const double next_distance = grid.cell_size * (static_cast<double>(shell) + boundary_fraction);
        if (collector.count == collector.k
            && isfinite(next_distance))
        {
            int next_exponent = 0;
            const double next_mantissa = frexp(next_distance, &next_exponent);
            const DistanceKey next_key{
                next_mantissa == 0.0 ? INT_MIN : next_exponent, next_mantissa};
            if (distanceLess(collector.kth_distance, next_key))
            {
                return;
            }
        }
    }
}

template <typename Scalar>
__global__ void indexedKnnKernel(
    KnnGridView<Scalar> grid,
    const Scalar* queries,
    int query_count,
    int k,
    plamatrix::Index* indices,
    Scalar* distances,
    double* distance_keys,
    int* distance_exponents)
{
    const int query = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= query_count)
    {
        return;
    }
    const Scalar sentinel = sizeof(Scalar) == sizeof(float) ? Scalar(FLT_MAX) : Scalar(DBL_MAX);
    for (int neighbor = 0; neighbor < k; ++neighbor)
    {
        const std::size_t offset = static_cast<std::size_t>(query)
            + static_cast<std::size_t>(neighbor) * query_count;
        indices[offset] = -1;
        distances[offset] = sentinel;
        distance_keys[offset] = DBL_MAX;
        distance_exponents[offset] = INT_MAX;
    }
    KnnCollector<Scalar> collector{
        indices, distances, distance_keys, distance_exponents, query, query_count, k};
    const std::size_t query_offset = static_cast<std::size_t>(query);
    const std::size_t query_stride = static_cast<std::size_t>(query_count);
    const double qx = static_cast<double>(queries[query_offset]);
    const double qy = static_cast<double>(queries[query_stride + query_offset]);
    const double qz = static_cast<double>(queries[2 * query_stride + query_offset]);
    if (!isfinite(qx) || !isfinite(qy) || !isfinite(qz))
    {
        return;
    }
    visitShells(
        grid, qx, qy, qz, collector);
}

} // namespace

template <typename Scalar>
GpuKnnSearchResult<Scalar> GpuSpatialIndex<Scalar>::knnSearchAsync(
    const plamatrix::internal::ResidentMatrix<Scalar>& queries,
    int k,
    GpuSpatialQueryWorkspace<Scalar>& workspace,
    cudaStream_t stream) const
{
    if (_cloudRevision == 0)
    {
        throw std::logic_error("GpuSpatialIndex must be built before querying");
    }
    if (queries.cols() != 3 || k <= 0 || k > 32)
    {
        throw std::invalid_argument("GpuSpatialIndex KNN requires Qx3 queries and 1 <= k <= 32");
    }
    if (queries.rows() > std::numeric_limits<int>::max())
    {
        throw std::overflow_error("GpuSpatialIndex query count exceeds int range");
    }
    queries.validateContext(*_context);
    auto& distance_keys = workspace.distanceKeys(queries.rows(), k, _context);
    auto& distance_exponents = workspace.distanceExponents(queries.rows(), k, _context);
    GpuKnnSearchResult<Scalar> result{
        plamatrix::internal::ResidentMatrix<plamatrix::Index>(queries.rows(), k, _context),
        plamatrix::internal::ResidentMatrix<Scalar>(queries.rows(), k, _context)};
    if (queries.rows() != 0)
    {
        const KnnGridView<Scalar> grid{
            _points->data(), _pointCount, uniqueCellKeysData(), sortedPointIndicesData(),
            cellOffsetsData(), cellCountsData(), _cellCount, static_cast<double>(_cellSize),
            _originX, _originY, _originZ,
            static_cast<long long>(_axisSpanX), static_cast<long long>(_axisSpanY),
            static_cast<long long>(_axisSpanZ)};
        indexedKnnKernel<<<(queries.rows() + 255) / 256, 256, 0, stream>>>(
            grid, queries.data(), static_cast<int>(queries.rows()), k,
            result.indices.data(), result.squaredDistances.data(), distance_keys.data(),
            distance_exponents.data());
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }
    return result;
}

template GpuKnnSearchResult<float> GpuSpatialIndex<float>::knnSearchAsync(
    const plamatrix::internal::ResidentMatrix<float>&,
    int, GpuSpatialQueryWorkspace<float>&, cudaStream_t) const;
template GpuKnnSearchResult<double> GpuSpatialIndex<double>::knnSearchAsync(
    const plamatrix::internal::ResidentMatrix<double>&,
    int, GpuSpatialQueryWorkspace<double>&, cudaStream_t) const;

} // namespace gpu
} // namespace plapoint

#endif
