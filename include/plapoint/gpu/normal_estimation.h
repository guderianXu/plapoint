#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/ops/small_matrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/spatial_index.h>

namespace plapoint
{
namespace gpu
{

template <typename Scalar>
class NormalEstimationGpuWorkspace
{
public:
    NormalEstimationGpuWorkspace() = default;
    NormalEstimationGpuWorkspace(const NormalEstimationGpuWorkspace&) = delete;
    NormalEstimationGpuWorkspace& operator=(const NormalEstimationGpuWorkspace&) = delete;
    NormalEstimationGpuWorkspace(NormalEstimationGpuWorkspace&&) noexcept = default;
    NormalEstimationGpuWorkspace& operator=(NormalEstimationGpuWorkspace&&) = default;

    /// Consume device-side eigensolver status after synchronizing the owning stream.
    void checkStatus()
    {
        _eigenWorkspace.checkStatus("estimateNormalsAsync");
    }

    /// Release stream reuse ownership after synchronizing and checking status.
    void resetStream()
    {
        _eigenWorkspace.reserveBytes(_eigenWorkspace.capacityBytes());
        _queryWorkspace = GpuSpatialQueryWorkspace<Scalar>();
    }

private:
    template <typename OtherScalar>
    friend void estimateNormalsAsync(
        const plapoint::internal::DeviceCloud<OtherScalar, plamatrix::internal::Device::GPU>&,
        const GpuSpatialIndex<OtherScalar>&,
        int,
        plamatrix::internal::ResidentMatrix<OtherScalar>&,
        NormalEstimationGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    void resize(plamatrix::Index rows, const std::shared_ptr<plamatrix::internal::ExecutionContext>& context)
    {
        if (!context)
        {
            throw std::invalid_argument("NormalEstimationGpuWorkspace requires a cloud execution context");
        }
        if (_context && _context != context)
        {
            _eigenWorkspace.reserveBytes(_eigenWorkspace.capacityBytes());
            _queryWorkspace = GpuSpatialQueryWorkspace<Scalar>();
            _covariances.reset();
            _eigenvalues.reset();
            _eigenvectors.reset();
        }
        _context = context;
        if (_covariances && _covariances->rows() == rows)
        {
            return;
        }
        _covariances = std::make_unique<plamatrix::internal::ResidentMatrix<Scalar>>(rows, 6, *context);
        _eigenvalues = std::make_unique<plamatrix::internal::ResidentMatrix<Scalar>>(rows, 3, *context);
        _eigenvectors = std::make_unique<plamatrix::internal::ResidentMatrix<Scalar>>(rows, 9, *context);
    }

    void bindStream(cudaStream_t stream)
    {
        if (_eigenWorkspace.capacityBytes() == 0)
        {
            _eigenWorkspace.reserveBytes(2 * sizeof(plamatrix::Index));
        }
        _eigenWorkspace.reserveBytesAsync(_eigenWorkspace.capacityBytes(), stream);
    }

    GpuSpatialQueryWorkspace<Scalar> _queryWorkspace;
    std::shared_ptr<plamatrix::internal::ExecutionContext> _context;
    std::unique_ptr<plamatrix::internal::ResidentMatrix<Scalar>> _covariances;
    std::unique_ptr<plamatrix::internal::ResidentMatrix<Scalar>> _eigenvalues;
    std::unique_ptr<plamatrix::internal::ResidentMatrix<Scalar>> _eigenvectors;
    plamatrix::internal::SymmetricEigh3x3Workspace _eigenWorkspace;
};

/// Enqueue indexed KNN normal estimation without staging point or neighbor vectors on the host.
/// The index must match the cloud. Synchronize stream and call workspace.checkStatus().
template <typename Scalar>
void estimateNormalsAsync(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    const GpuSpatialIndex<Scalar>& index,
    int k,
    plamatrix::internal::ResidentMatrix<Scalar>& normals,
    NormalEstimationGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream);

} // namespace gpu
} // namespace plapoint

#endif
