#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>

#include <plapoint/core/processing_policy.h>
#include <plapoint/gpu/spatial_index.h>

namespace plapoint
{
namespace search
{
namespace detail
{

constexpr std::size_t kIndexedKnnWorkThreshold = ProcessingPolicy::indexedKnnWorkThreshold;
constexpr int kMaximumIndexedCellOccupancy = 256;
// Wider empty-shell walks lose decisively to the tiled brute-force kernel in local baselines.
constexpr double kMaximumQueryShell = 8.0;

struct HostGridCell
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const HostGridCell& other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct HostGridCellHash
{
    std::size_t operator()(const HostGridCell& cell) const noexcept
    {
        auto mix = [](std::uint64_t value)
        {
            value ^= value >> 30;
            value *= UINT64_C(0xbf58476d1ce4e5b9);
            value ^= value >> 27;
            value *= UINT64_C(0x94d049bb133111eb);
            return value ^ (value >> 31);
        };
        const auto x = mix(static_cast<std::uint64_t>(cell.x));
        const auto y = mix(static_cast<std::uint64_t>(cell.y));
        const auto z = mix(static_cast<std::uint64_t>(cell.z));
        return static_cast<std::size_t>(x ^ (y << 1) ^ (z << 7));
    }
};

using HostGridCellSet = std::unordered_set<HostGridCell, HostGridCellHash>;

template <typename Scalar>
bool hostGridCell(Scalar x, Scalar y, Scalar z, Scalar cell_size, HostGridCell& cell)
{
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
        || !std::isfinite(cell_size) || cell_size <= Scalar(0))
    {
        return false;
    }
    const long double values[3] = {
        std::floor(static_cast<long double>(x) / static_cast<long double>(cell_size)),
        std::floor(static_cast<long double>(y) / static_cast<long double>(cell_size)),
        std::floor(static_cast<long double>(z) / static_cast<long double>(cell_size))};
    const auto lower_bound = -static_cast<long double>(UINT64_C(1) << 63);
    const auto upper_bound = static_cast<long double>(UINT64_C(1) << 63);
    for (const long double value : values)
    {
        if (!std::isfinite(value) || value < lower_bound || value >= upper_bound)
        {
            return false;
        }
    }
    cell = {
        static_cast<std::int64_t>(values[0]),
        static_cast<std::int64_t>(values[1]),
        static_cast<std::int64_t>(values[2])};
    return true;
}

template <typename Scalar>
bool buildHostGridOccupancy(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& points,
    Scalar cell_size,
    HostGridCellSet& occupied_cells)
{
    HostGridCellSet replacement;
    replacement.reserve(static_cast<std::size_t>(points.rows()));
    for (plamatrix::Index row = 0; row < points.rows(); ++row)
    {
        const Scalar x = points(row, 0);
        const Scalar y = points(row, 1);
        const Scalar z = points(row, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            continue;
        }
        HostGridCell cell;
        if (!hostGridCell(x, y, z, cell_size, cell))
        {
            return false;
        }
        replacement.insert(cell);
    }
    occupied_cells.swap(replacement);
    return !occupied_cells.empty();
}

template <typename Scalar>
Scalar estimateKnnCellSize(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& points)
{
    double minima[3] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    double maxima[3] = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    std::size_t finite_count = 0;
    for (plamatrix::Index row = 0; row < points.rows(); ++row)
    {
        const double x = static_cast<double>(points(row, 0));
        const double y = static_cast<double>(points(row, 1));
        const double z = static_cast<double>(points(row, 2));
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            continue;
        }
        const double values[3] = {x, y, z};
        for (int axis = 0; axis < 3; ++axis)
        {
            minima[axis] = std::min(minima[axis], values[axis]);
            maxima[axis] = std::max(maxima[axis], values[axis]);
        }
        ++finite_count;
    }
    if (finite_count == 0)
    {
        return Scalar(1);
    }

    const double extent = std::max({
        maxima[0] - minima[0], maxima[1] - minima[1], maxima[2] - minima[2]});
    const double scale = std::max({
        1.0, std::abs(minima[0]), std::abs(minima[1]), std::abs(minima[2]),
        std::abs(maxima[0]), std::abs(maxima[1]), std::abs(maxima[2])});
    const double spacing = std::max(
        static_cast<double>(std::numeric_limits<Scalar>::denorm_min()),
        static_cast<double>(std::numeric_limits<Scalar>::epsilon()) * scale);
    const double estimate = extent > 0.0
        ? extent / std::cbrt(static_cast<double>(finite_count))
        : spacing;
    return static_cast<Scalar>(std::max(estimate, spacing));
}

template <typename Scalar>
bool indexedGridIsPathological(const gpu::GpuSpatialIndex<Scalar>& index)
{
    if (index.cellCount() == 0
        || index.maxCellOccupancy() > kMaximumIndexedCellOccupancy)
    {
        return true;
    }
    const long double volume = (static_cast<long double>(index.axisSpanX()) + 1.0L)
        * (static_cast<long double>(index.axisSpanY()) + 1.0L)
        * (static_cast<long double>(index.axisSpanZ()) + 1.0L);
    return volume > 8192.0L
        && volume > static_cast<long double>(index.cellCount()) * 128.0L;
}

template <typename Scalar>
bool queriesFitIndexedShells(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& queries,
    Scalar cell_size,
    const HostGridCellSet& occupied_cells)
{
    for (plamatrix::Index row = 0; row < queries.rows(); ++row)
    {
        HostGridCell query_cell;
        if (!hostGridCell(
                queries(row, 0), queries(row, 1), queries(row, 2), cell_size, query_cell))
        {
            return false;
        }

        bool nearby_occupied_cell = false;
        constexpr int maximum_shell = static_cast<int>(kMaximumQueryShell);
        for (int shell = 0; shell <= maximum_shell && !nearby_occupied_cell; ++shell)
        {
            for (int dz = -shell; dz <= shell && !nearby_occupied_cell; ++dz)
            {
                for (int dy = -shell; dy <= shell && !nearby_occupied_cell; ++dy)
                {
                    for (int dx = -shell; dx <= shell; ++dx)
                    {
                        if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != shell)
                        {
                            continue;
                        }
                        if ((dx < 0 && query_cell.x < std::numeric_limits<std::int64_t>::min() - dx)
                            || (dx > 0 && query_cell.x > std::numeric_limits<std::int64_t>::max() - dx)
                            || (dy < 0 && query_cell.y < std::numeric_limits<std::int64_t>::min() - dy)
                            || (dy > 0 && query_cell.y > std::numeric_limits<std::int64_t>::max() - dy)
                            || (dz < 0 && query_cell.z < std::numeric_limits<std::int64_t>::min() - dz)
                            || (dz > 0 && query_cell.z > std::numeric_limits<std::int64_t>::max() - dz))
                        {
                            continue;
                        }
                        const HostGridCell candidate{
                            query_cell.x + dx, query_cell.y + dy, query_cell.z + dz};
                        if (occupied_cells.find(candidate) != occupied_cells.end())
                        {
                            nearby_occupied_cell = true;
                            break;
                        }
                    }
                }
            }
        }
        if (!nearby_occupied_cell)
        {
            return false;
        }
    }
    return true;
}

} // namespace detail
} // namespace search
} // namespace plapoint

#endif
