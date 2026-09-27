#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <array>
#include <memory>

#include <cuda_runtime.h>

#include <plamatrix/internal/ops/indexing.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>

#include <plapoint/core/point_cloud.h>

namespace plapoint
{
namespace gpu
{

namespace marching_cubes_detail
{
struct WorkspaceAccess;
}

/// Move-only temporary storage for deterministic CUDA Marching Cubes.
/// A workspace is bound to one stream until closeAsyncAllocation(). The stream must remain alive
/// through close and one final synchronization that completes the enqueued frees.
template <typename Scalar>
class MarchingCubesGpuWorkspace
{
public:
    MarchingCubesGpuWorkspace() noexcept = default;
    ~MarchingCubesGpuWorkspace() noexcept;
    MarchingCubesGpuWorkspace(MarchingCubesGpuWorkspace&& other) noexcept;
    MarchingCubesGpuWorkspace& operator=(MarchingCubesGpuWorkspace&& other) noexcept;

    MarchingCubesGpuWorkspace(const MarchingCubesGpuWorkspace&) = delete;
    MarchingCubesGpuWorkspace& operator=(const MarchingCubesGpuWorkspace&) = delete;

    /// Enqueue release after the bound stream has completed all prior work.
    void closeAsyncAllocation();
    plamatrix::Index capacityCubes() const noexcept { return _capacityCubes; }

private:
    friend struct marching_cubes_detail::WorkspaceAccess;

    std::unique_ptr<plamatrix::internal::ResidentMatrix<plamatrix::Index>> _triangleCounts;
    std::unique_ptr<plamatrix::internal::ResidentMatrix<plamatrix::Index>> _triangleOffsets;
    std::unique_ptr<plamatrix::internal::ResidentMatrix<int>> _status;
    std::shared_ptr<plamatrix::internal::ExecutionContext> _context;
    plamatrix::internal::IndexingWorkspace _scanWorkspace;
    plamatrix::Index _capacityCubes = 0;
    cudaStream_t _stream = nullptr;
    bool _hasStream = false;
};

/// Extract a GPU-resident triangle soup from a device scalar field.
/// nx, ny, and nz are cube counts; field must be ((nx+1)*(ny+1)*(nz+1)) x 1
/// in x-fastest, then y, then z sample order. The call synchronizes stream before returning.
template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> marchingCubes(
    const plamatrix::internal::ResidentMatrix<Scalar>& field,
    int nx,
    int ny,
    int nz,
    const std::array<Scalar, 3>& min_corner,
    const std::array<Scalar, 3>& max_corner,
    Scalar iso,
    MarchingCubesGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
