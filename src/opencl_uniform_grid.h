#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace plapoint
{
namespace opencl
{
namespace detail
{

constexpr int maximumKnn = 32;
constexpr std::int64_t maximumEnumeratedShell = 4096;
constexpr std::uint64_t maximumSameCellKnnWork = 100000000;
constexpr std::uint64_t maximumRadiusNeighborWork = 100000000;

struct CellCoord
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const CellCoord& other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator<(const CellCoord& other) const noexcept
    {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct GridEntry
{
    CellCoord cell;
    int point = -1;
};

struct HostGrid
{
    std::vector<int> sortedIndices;
    std::vector<std::int64_t> cellCoords;
    std::vector<int> offsets;
    std::vector<int> counts;
    std::vector<std::int64_t> queryCells;
    std::vector<std::uint8_t> finiteMask;
    std::int64_t spanX = 0;
    std::int64_t spanY = 0;
    std::int64_t spanZ = 0;
};

template <typename Scalar>
std::int64_t checkedRawCell(Scalar coordinate, Scalar cell_size)
{
    const long double cell = std::floor(
        static_cast<long double>(coordinate) / static_cast<long double>(cell_size));
    const long double lower = static_cast<long double>(std::numeric_limits<std::int64_t>::min());
    const long double upper = -lower;
    if (!std::isfinite(cell) || cell < lower || cell >= upper)
    {
        throw std::overflow_error("OpenCL uniform-grid coordinate exceeds the signed 64-bit cell range");
    }
    return static_cast<std::int64_t>(cell);
}

template <typename Scalar>
std::int64_t checkedVoxelCell(Scalar coordinate, Scalar leaf_size)
{
    const double cell = std::floor(
        static_cast<double>(coordinate) / static_cast<double>(leaf_size));
    if (!std::isfinite(cell)
        || cell < static_cast<double>(std::numeric_limits<int>::min())
        || cell > static_cast<double>(std::numeric_limits<int>::max()))
    {
        throw std::out_of_range("OpenCL VoxelGrid: voxel index is outside int range");
    }
    return static_cast<int>(cell);
}

inline std::int64_t checkedSpan(std::int64_t minimum, std::int64_t maximum)
{
    const auto span = static_cast<std::uint64_t>(maximum)
        - static_cast<std::uint64_t>(minimum);
    if (span > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    {
        throw std::overflow_error("OpenCL uniform-grid axis span exceeds signed 64-bit range");
    }
    return static_cast<std::int64_t>(span);
}

template <typename Scalar>
HostGrid buildGrid(
    const std::vector<Scalar>& points,
    std::size_t point_count,
    Scalar cell_size)
{
    if (!std::isfinite(cell_size) || cell_size <= Scalar(0))
    {
        throw std::invalid_argument("OpenCL uniform-grid cell size must be finite and positive");
    }
    HostGrid result;
    result.finiteMask.assign(point_count, 0);
    result.queryCells.assign(point_count * 3u, 0);
    std::vector<GridEntry> entries;
    entries.reserve(point_count);
    CellCoord minimum{
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max()};
    CellCoord maximum{
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min()};
    for (std::size_t index = 0; index < point_count; ++index)
    {
        const Scalar x = points[index * 3u];
        const Scalar y = points[index * 3u + 1u];
        const Scalar z = points[index * 3u + 2u];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
        const CellCoord cell{
            checkedRawCell(x, cell_size),
            checkedRawCell(y, cell_size),
            checkedRawCell(z, cell_size)};
        entries.push_back({cell, static_cast<int>(index)});
        result.finiteMask[index] = 1;
        minimum.x = std::min(minimum.x, cell.x);
        minimum.y = std::min(minimum.y, cell.y);
        minimum.z = std::min(minimum.z, cell.z);
        maximum.x = std::max(maximum.x, cell.x);
        maximum.y = std::max(maximum.y, cell.y);
        maximum.z = std::max(maximum.z, cell.z);
    }
    if (entries.empty()) return result;

    result.spanX = checkedSpan(minimum.x, maximum.x);
    result.spanY = checkedSpan(minimum.y, maximum.y);
    result.spanZ = checkedSpan(minimum.z, maximum.z);
    for (GridEntry& entry : entries)
    {
        entry.cell.x = static_cast<std::int64_t>(
            static_cast<std::uint64_t>(entry.cell.x) - static_cast<std::uint64_t>(minimum.x));
        entry.cell.y = static_cast<std::int64_t>(
            static_cast<std::uint64_t>(entry.cell.y) - static_cast<std::uint64_t>(minimum.y));
        entry.cell.z = static_cast<std::int64_t>(
            static_cast<std::uint64_t>(entry.cell.z) - static_cast<std::uint64_t>(minimum.z));
        const std::size_t offset = static_cast<std::size_t>(entry.point) * 3u;
        result.queryCells[offset] = entry.cell.x;
        result.queryCells[offset + 1u] = entry.cell.y;
        result.queryCells[offset + 2u] = entry.cell.z;
    }
    std::sort(entries.begin(), entries.end(), [](const GridEntry& lhs, const GridEntry& rhs)
    {
        return lhs.cell == rhs.cell ? lhs.point < rhs.point : lhs.cell < rhs.cell;
    });
    result.sortedIndices.reserve(entries.size());
    for (std::size_t begin = 0; begin < entries.size();)
    {
        std::size_t end = begin + 1u;
        while (end < entries.size() && entries[end].cell == entries[begin].cell) ++end;
        result.cellCoords.push_back(entries[begin].cell.x);
        result.cellCoords.push_back(entries[begin].cell.y);
        result.cellCoords.push_back(entries[begin].cell.z);
        result.offsets.push_back(static_cast<int>(begin));
        result.counts.push_back(static_cast<int>(end - begin));
        for (std::size_t cursor = begin; cursor < end; ++cursor)
        {
            result.sortedIndices.push_back(entries[cursor].point);
        }
        begin = end;
    }
    return result;
}

inline void validateKnnGridWork(const HostGrid& grid)
{
    std::uint64_t work = 0;
    for (const int count : grid.counts)
    {
        const auto occupancy = static_cast<std::uint64_t>(count);
        if (occupancy != 0 && occupancy > (maximumSameCellKnnWork - work) / occupancy)
        {
            throw std::runtime_error(
                "OpenCL uniform-grid KNN rejected a pathological high-occupancy grid");
        }
        work += occupancy * occupancy;
    }
}

inline int findHostCell(const HostGrid& grid, const CellCoord& coordinate)
{
    const std::size_t count = grid.cellCoords.size() / 3u;
    std::size_t first = 0;
    std::size_t last = count;
    auto at = [&](std::size_t index)
    {
        return CellCoord{
            grid.cellCoords[index * 3u], grid.cellCoords[index * 3u + 1u],
            grid.cellCoords[index * 3u + 2u]};
    };
    while (first < last)
    {
        const std::size_t middle = first + (last - first) / 2u;
        if (at(middle) < coordinate) first = middle + 1u;
        else last = middle;
    }
    return first < count && at(first) == coordinate ? static_cast<int>(first) : -1;
}

inline void validateRadiusGridWork(const HostGrid& grid)
{
    std::uint64_t work = 0;
    for (std::size_t cell = 0; cell < grid.counts.size(); ++cell)
    {
        const CellCoord center{
            grid.cellCoords[cell * 3u], grid.cellCoords[cell * 3u + 1u],
            grid.cellCoords[cell * 3u + 2u]};
        std::uint64_t candidates = 0;
        for (std::int64_t dx = -1; dx <= 1; ++dx)
        {
            if ((dx < 0 && center.x == 0) || (dx > 0 && center.x == grid.spanX)) continue;
            for (std::int64_t dy = -1; dy <= 1; ++dy)
            {
                if ((dy < 0 && center.y == 0) || (dy > 0 && center.y == grid.spanY)) continue;
                for (std::int64_t dz = -1; dz <= 1; ++dz)
                {
                    if ((dz < 0 && center.z == 0) || (dz > 0 && center.z == grid.spanZ)) continue;
                    const int neighbor = findHostCell(
                        grid, {center.x + dx, center.y + dy, center.z + dz});
                    if (neighbor >= 0) candidates += static_cast<std::uint64_t>(grid.counts[neighbor]);
                }
            }
        }
        const auto queries = static_cast<std::uint64_t>(grid.counts[cell]);
        if (candidates != 0 && queries > (maximumRadiusNeighborWork - work) / candidates)
        {
            throw std::runtime_error(
                "OpenCL radius outlier removal rejected a pathological dense-grid workload");
        }
        work += queries * candidates;
    }
}

template <typename Scalar>
Scalar adaptiveCellSize(const std::vector<Scalar>& points, std::size_t point_count)
{
    std::array<long double, 3> minimum{
        std::numeric_limits<long double>::infinity(),
        std::numeric_limits<long double>::infinity(),
        std::numeric_limits<long double>::infinity()};
    std::array<long double, 3> maximum{-minimum[0], -minimum[1], -minimum[2]};
    std::size_t finite_count = 0;
    for (std::size_t index = 0; index < point_count; ++index)
    {
        const std::array<Scalar, 3> point{
            points[index * 3u], points[index * 3u + 1u], points[index * 3u + 2u]};
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) continue;
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], static_cast<long double>(point[axis]));
            maximum[axis] = std::max(maximum[axis], static_cast<long double>(point[axis]));
        }
        ++finite_count;
    }
    if (finite_count == 0) return Scalar(1);
    long double scale = 0;
    long double extent = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        scale = std::max(scale, std::abs(minimum[axis]));
        scale = std::max(scale, std::abs(maximum[axis]));
        extent = std::max(extent, maximum[axis] - minimum[axis]);
    }
    if (scale == 0) return std::numeric_limits<Scalar>::denorm_min();
    long double estimate = extent > 0
        ? extent / std::cbrt(static_cast<long double>(finite_count))
        : static_cast<long double>(std::numeric_limits<Scalar>::epsilon()) * scale;
    const long double spacing = std::max(
        static_cast<long double>(std::numeric_limits<Scalar>::denorm_min()),
        static_cast<long double>(std::numeric_limits<Scalar>::epsilon()) * scale);
    estimate = std::max(estimate, spacing);
    if (!std::isfinite(estimate) || estimate <= 0
        || estimate > static_cast<long double>(std::numeric_limits<Scalar>::max()))
    {
        throw std::overflow_error("OpenCL adaptive uniform-grid cell size is outside scalar range");
    }
    return static_cast<Scalar>(estimate);
}

} // namespace detail
} // namespace opencl
} // namespace plapoint
