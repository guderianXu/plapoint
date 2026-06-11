#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/height_grid.h>

namespace plapoint
{
namespace gpu
{

/// Build a CPU-readable height grid from a GPU-resident point cloud.
/// Point coordinates are consumed from PlaMatrix column-major device storage.
mesh::HeightGrid<float> buildHeightGrid(
    const PointCloud<float, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<float>& options = mesh::HeightGridOptions<float>{},
    cudaStream_t stream = 0);

/// Build a CPU-readable height grid from a GPU-resident point cloud.
/// Point coordinates are consumed from PlaMatrix column-major device storage.
mesh::HeightGrid<double> buildHeightGrid(
    const PointCloud<double, plamatrix::Device::GPU>& cloud,
    const mesh::HeightGridOptions<double>& options = mesh::HeightGridOptions<double>{},
    cudaStream_t stream = 0);

/// Fill height grid holes using the current CPU mesh implementation.
/// This wrapper keeps the GPU path API stable while full device-side fill support is added later.
void fillHoles(mesh::HeightGrid<float>& grid, int max_passes = 8);

/// Fill height grid holes using the current CPU mesh implementation.
/// This wrapper keeps the GPU path API stable while full device-side fill support is added later.
void fillHoles(mesh::HeightGrid<double>& grid, int max_passes = 8);

/// Convert a height grid to a CPU mesh while preserving CPU-supported source attributes.
/// The current mesh implementation preserves colors by transferring the GPU source cloud back to CPU.
PointCloud<float, plamatrix::Device::CPU> heightGridToMesh(
    const mesh::HeightGrid<float>& grid,
    const PointCloud<float, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options = mesh::HeightGridOptions<float>{});

/// Convert a height grid to a CPU mesh while preserving CPU-supported source attributes.
/// The current mesh implementation preserves colors by transferring the GPU source cloud back to CPU.
PointCloud<double, plamatrix::Device::CPU> heightGridToMesh(
    const mesh::HeightGrid<double>& grid,
    const PointCloud<double, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options = mesh::HeightGridOptions<double>{});

/// Convenience path: build a GPU height grid, fill holes on CPU, then emit a CPU mesh.
PointCloud<float, plamatrix::Device::CPU> heightGridToMesh(
    const PointCloud<float, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<float>& options = mesh::HeightGridOptions<float>{},
    int fill_passes = 8,
    cudaStream_t stream = 0);

/// Convenience path: build a GPU height grid, fill holes on CPU, then emit a CPU mesh.
PointCloud<double, plamatrix::Device::CPU> heightGridToMesh(
    const PointCloud<double, plamatrix::Device::GPU>& source_cloud,
    const mesh::HeightGridOptions<double>& options = mesh::HeightGridOptions<double>{},
    int fill_passes = 8,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
