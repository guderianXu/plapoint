#include <plapoint/gpu/spatial_index.h>
#include <plapoint/gpu/detail/spatial_index_validation.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace plapoint
{
namespace gpu
{
namespace
{

constexpr std::uint64_t kAxisLimit = std::uint64_t{1} << 21;
constexpr std::uint64_t kAxisMask = kAxisLimit - 1;

template <typename Scalar>
struct GridView
{
    const Scalar* points;
    plamatrix::Index point_count;
    const std::uint64_t* keys;
    const plamatrix::Index* sorted_indices;
    const plamatrix::Index* offsets;
    const plamatrix::Index* counts;
    int cell_count;
    double cell_size;
    std::int64_t origin_x;
    std::int64_t origin_y;
    std::int64_t origin_z;
};

struct AxisRange
{
    std::uint64_t first;
    std::uint64_t last;
    bool valid;
};

__device__ AxisRange queryAxisRange(
    double coordinate,
    double radius,
    double cell_size,
    std::int64_t origin)
{
    const double origin_cell = static_cast<double>(origin);
    double first = floor((coordinate - radius) / cell_size) - origin_cell;
    double last = floor((coordinate + radius) / cell_size) - origin_cell;
    if (last < 0.0 || first > static_cast<double>(kAxisMask))
    {
        return {0, 0, false};
    }
    first = first < 0.0 ? 0.0 : first;
    last = last > static_cast<double>(kAxisMask) ? static_cast<double>(kAxisMask) : last;
    return {
        static_cast<std::uint64_t>(first),
        static_cast<std::uint64_t>(last),
        first <= last
    };
}

__device__ int findCell(const std::uint64_t* keys, int count, std::uint64_t key)
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

__device__ std::uint64_t rangeLength(const AxisRange& range)
{
    return range.last - range.first + 1;
}

__device__ bool shouldEnumerateCells(
    const AxisRange& x,
    const AxisRange& y,
    const AxisRange& z,
    int cell_count)
{
    const std::uint64_t limit = static_cast<std::uint64_t>(cell_count) * 4 + 64;
    const std::uint64_t x_count = rangeLength(x);
    const std::uint64_t y_count = rangeLength(y);
    const std::uint64_t z_count = rangeLength(z);
    return x_count <= limit
        && y_count <= limit / x_count
        && z_count <= limit / (x_count * y_count);
}

template <typename Scalar, typename Collector>
__device__ bool visitCell(
    const GridView<Scalar>& grid,
    int cell,
    double qx,
    double qy,
    double qz,
    double radius,
    Collector& collector)
{
    const auto first = grid.offsets[cell];
    const auto count = grid.counts[cell];
    for (plamatrix::Index offset = 0; offset < count; ++offset)
    {
        const auto source_index = grid.sorted_indices[first + offset];
        const double dx = qx - static_cast<double>(grid.points[source_index]);
        if (fabs(dx) > radius)
        {
            continue;
        }
        const double dy = qy - static_cast<double>(grid.points[grid.point_count + source_index]);
        if (fabs(dy) > radius)
        {
            continue;
        }
        const double dz = qz - static_cast<double>(grid.points[2 * grid.point_count + source_index]);
        if (fabs(dz) > radius)
        {
            continue;
        }
        const double distance = norm3d(dx, dy, dz);
        if (isfinite(distance) && distance <= radius
            && collector.add(source_index, distance))
        {
            return true;
        }
    }
    return false;
}

template <typename Scalar, typename Collector>
__device__ void visitRadius(
    const GridView<Scalar>& grid,
    double qx,
    double qy,
    double qz,
    double radius,
    Collector& collector)
{
    const AxisRange x = queryAxisRange(qx, radius, grid.cell_size, grid.origin_x);
    const AxisRange y = queryAxisRange(qy, radius, grid.cell_size, grid.origin_y);
    const AxisRange z = queryAxisRange(qz, radius, grid.cell_size, grid.origin_z);
    if (!x.valid || !y.valid || !z.valid)
    {
        return;
    }

    if (shouldEnumerateCells(x, y, z, grid.cell_count))
    {
        for (std::uint64_t nx = x.first; nx <= x.last; ++nx)
        {
            for (std::uint64_t ny = y.first; ny <= y.last; ++ny)
            {
                for (std::uint64_t nz = z.first; nz <= z.last; ++nz)
                {
                    const std::uint64_t key = (nx << 42) | (ny << 21) | nz;
                    const int cell = findCell(grid.keys, grid.cell_count, key);
                    if (cell >= 0 && visitCell(grid, cell, qx, qy, qz, radius, collector))
                    {
                        return;
                    }
                }
            }
        }
        return;
    }

    for (int cell = 0; cell < grid.cell_count; ++cell)
    {
        const std::uint64_t key = grid.keys[cell];
        const std::uint64_t nx = key >> 42;
        const std::uint64_t ny = (key >> 21) & kAxisMask;
        const std::uint64_t nz = key & kAxisMask;
        if (nx >= x.first && nx <= x.last
            && ny >= y.first && ny <= y.last
            && nz >= z.first && nz <= z.last
            && visitCell(grid, cell, qx, qy, qz, radius, collector))
        {
            return;
        }
    }
}

struct CountCollector
{
    plamatrix::Index count = 0;
    int maximum;

    __device__ bool add(plamatrix::Index, double)
    {
        ++count;
        return count >= maximum;
    }
};

template <typename Scalar>
struct SearchCollector
{
    plamatrix::Index* indices;
    Scalar* distances;
    double* distance_keys;
    int query;
    int query_count;
    int maximum;
    int count = 0;

    __device__ bool add(plamatrix::Index index, double distance)
    {
        int position = count;
        if (position >= maximum)
        {
            position = maximum - 1;
            const std::size_t last_offset = static_cast<std::size_t>(query)
                + static_cast<std::size_t>(position) * query_count;
            const auto last_index = indices[last_offset];
            const double last_distance = distance_keys[last_offset];
            if (distance > last_distance || (distance == last_distance && index >= last_index))
            {
                return false;
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
            const auto previous_index = indices[previous_offset];
            if (previous_distance < distance
                || (previous_distance == distance && previous_index <= index))
            {
                break;
            }
            const std::size_t offset = static_cast<std::size_t>(query)
                + static_cast<std::size_t>(position) * query_count;
            indices[offset] = previous_index;
            distances[offset] = distances[previous_offset];
            distance_keys[offset] = previous_distance;
            --position;
        }
        const std::size_t offset = static_cast<std::size_t>(query)
            + static_cast<std::size_t>(position) * query_count;
        indices[offset] = index;
        distance_keys[offset] = distance;
        const double squared_distance = distance >= sqrt(DBL_MAX)
            ? DBL_MAX
            : distance * distance;
        const double maximum_scalar = sizeof(Scalar) == sizeof(float) ? FLT_MAX : DBL_MAX;
        distances[offset] = static_cast<Scalar>(
            squared_distance > maximum_scalar ? maximum_scalar : squared_distance);
        return false;
    }
};

template <typename Scalar>
__global__ void radiusCountKernel(
    GridView<Scalar> grid,
    const Scalar* queries,
    int query_count,
    double radius,
    int max_count,
    plamatrix::Index* output)
{
    const int query = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= query_count)
    {
        return;
    }
    const std::size_t query_offset = static_cast<std::size_t>(query);
    const std::size_t query_stride = static_cast<std::size_t>(query_count);
    const double qx = static_cast<double>(queries[query_offset]);
    const double qy = static_cast<double>(queries[query_stride + query_offset]);
    const double qz = static_cast<double>(queries[2 * query_stride + query_offset]);
    CountCollector collector{0, max_count};
    if (isfinite(qx) && isfinite(qy) && isfinite(qz))
    {
        visitRadius(grid, qx, qy, qz, radius, collector);
    }
    output[query] = collector.count;
}

template <typename Scalar>
__global__ void radiusSearchKernel(
    GridView<Scalar> grid,
    const Scalar* queries,
    int query_count,
    double radius,
    int max_neighbors,
    plamatrix::Index* indices,
    Scalar* distances,
    double* distance_keys,
    plamatrix::Index* counts)
{
    const int query = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= query_count)
    {
        return;
    }
    const Scalar sentinel = sizeof(Scalar) == sizeof(float) ? Scalar(FLT_MAX) : Scalar(DBL_MAX);
    for (int neighbor = 0; neighbor < max_neighbors; ++neighbor)
    {
        const std::size_t offset = static_cast<std::size_t>(query)
            + static_cast<std::size_t>(neighbor) * query_count;
        indices[offset] = -1;
        distances[offset] = sentinel;
        distance_keys[offset] = DBL_MAX;
    }
    const std::size_t query_offset = static_cast<std::size_t>(query);
    const std::size_t query_stride = static_cast<std::size_t>(query_count);
    const double qx = static_cast<double>(queries[query_offset]);
    const double qy = static_cast<double>(queries[query_stride + query_offset]);
    const double qz = static_cast<double>(queries[2 * query_stride + query_offset]);
    SearchCollector<Scalar> collector{
        indices, distances, distance_keys, query, query_count, max_neighbors, 0};
    if (isfinite(qx) && isfinite(qy) && isfinite(qz))
    {
        visitRadius(grid, qx, qy, qz, radius, collector);
    }
    counts[query] = collector.count;
}

} // namespace

template <typename Scalar>
plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
GpuSpatialIndex<Scalar>::radiusCountAsync(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& queries,
    Scalar radius,
    int max_count,
    GpuSpatialQueryWorkspace<Scalar>& workspace,
    cudaStream_t stream) const
{
    detail::validateRadiusArguments(queries, radius, max_count, _cloudRevision);
    static_cast<void>(workspace);
    auto result = plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
        ::uninitializedAsync(queries.rows(), 1, stream);
    if (queries.rows() != 0)
    {
        const GridView<Scalar> grid{
            _points.data(), _pointCount, _uniqueCellKeys.get(), _sortedPointIndices.data(),
            _cellOffsets.data(), _cellCounts.data(), _cellCount, static_cast<double>(_cellSize),
            _originX, _originY, _originZ};
        radiusCountKernel<<<(queries.rows() + 255) / 256, 256, 0, stream>>>(
            grid, queries.data(), static_cast<int>(queries.rows()), static_cast<double>(radius),
            max_count, result.data());
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }
    return result;
}

template <typename Scalar>
GpuRadiusSearchResult<Scalar> GpuSpatialIndex<Scalar>::radiusSearchAsync(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& queries,
    Scalar radius,
    int max_neighbors,
    GpuSpatialQueryWorkspace<Scalar>& workspace,
    cudaStream_t stream) const
{
    detail::validateRadiusArguments(queries, radius, max_neighbors, _cloudRevision);
    auto& distance_keys = workspace.distanceKeys(queries.rows(), max_neighbors, stream);
    GpuRadiusSearchResult<Scalar> result{
        plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
            ::uninitializedAsync(queries.rows(), max_neighbors, stream),
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>
            ::uninitializedAsync(queries.rows(), max_neighbors, stream),
        plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
            ::uninitializedAsync(queries.rows(), 1, stream)};
    if (queries.rows() != 0)
    {
        const GridView<Scalar> grid{
            _points.data(), _pointCount, _uniqueCellKeys.get(), _sortedPointIndices.data(),
            _cellOffsets.data(), _cellCounts.data(), _cellCount, static_cast<double>(_cellSize),
            _originX, _originY, _originZ};
        radiusSearchKernel<<<(queries.rows() + 255) / 256, 256, 0, stream>>>(
            grid, queries.data(), static_cast<int>(queries.rows()), static_cast<double>(radius),
            max_neighbors, result.indices.data(), result.squaredDistances.data(),
            distance_keys.data(), result.counts.data());
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }
    return result;
}

template plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
GpuSpatialIndex<float>::radiusCountAsync(
    const plamatrix::DenseMatrix<float, plamatrix::Device::GPU>&,
    float, int, GpuSpatialQueryWorkspace<float>&, cudaStream_t) const;
template plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
GpuSpatialIndex<double>::radiusCountAsync(
    const plamatrix::DenseMatrix<double, plamatrix::Device::GPU>&,
    double, int, GpuSpatialQueryWorkspace<double>&, cudaStream_t) const;
template GpuRadiusSearchResult<float> GpuSpatialIndex<float>::radiusSearchAsync(
    const plamatrix::DenseMatrix<float, plamatrix::Device::GPU>&,
    float, int, GpuSpatialQueryWorkspace<float>&, cudaStream_t) const;
template GpuRadiusSearchResult<double> GpuSpatialIndex<double>::radiusSearchAsync(
    const plamatrix::DenseMatrix<double, plamatrix::Device::GPU>&,
    double, int, GpuSpatialQueryWorkspace<double>&, cudaStream_t) const;

} // namespace gpu
} // namespace plapoint

#endif
