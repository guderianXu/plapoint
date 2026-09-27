#pragma once

#include <cstdint>

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/height_grid.h>
#include <plamatrix/internal/core/device.h>

namespace plapoint
{
namespace opencl
{

/// Return the number of successfully completed OpenCL height-grid aggregation launches.
std::uint64_t heightGridOpenClExecutionCount() noexcept;

/// Build a CPU-readable float height grid using private OpenCL aggregation buffers.
mesh::HeightGrid<float> buildHeightGrid(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>& cloud,
    const mesh::HeightGridOptions<float>& options = {});

/// Build a CPU-readable double height grid using private OpenCL aggregation buffers.
mesh::HeightGrid<double> buildHeightGrid(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>& cloud,
    const mesh::HeightGridOptions<double>& options = {});

} // namespace opencl
} // namespace plapoint
