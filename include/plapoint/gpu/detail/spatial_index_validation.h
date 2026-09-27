#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <plamatrix/internal/device/device_matrix.h>

namespace plapoint
{
namespace gpu
{
namespace detail
{

template <typename Scalar>
void validateRadiusArguments(
    const plamatrix::internal::ResidentMatrix<Scalar>& queries,
    Scalar radius,
    int maximum,
    std::uint64_t cloud_revision)
{
    if (cloud_revision == 0)
    {
        throw std::logic_error("GpuSpatialIndex must be built before querying");
    }
    if (queries.cols() != 3)
    {
        throw std::invalid_argument("GpuSpatialIndex queries must be Qx3");
    }
    if (!std::isfinite(radius) || radius < Scalar(0))
    {
        throw std::invalid_argument("GpuSpatialIndex radius must be finite and non-negative");
    }
    if (maximum <= 0)
    {
        throw std::invalid_argument("GpuSpatialIndex maximum result count must be positive");
    }
    if (queries.rows() > std::numeric_limits<int>::max())
    {
        throw std::overflow_error("GpuSpatialIndex query count exceeds int range");
    }
}

} // namespace detail
} // namespace gpu
} // namespace plapoint
