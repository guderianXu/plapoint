#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/small_matrix.h>

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
    }

private:
    template <typename OtherScalar>
    friend void estimateNormalsAsync(
        const PointCloud<OtherScalar, plamatrix::Device::GPU>&,
        const GpuSpatialIndex<OtherScalar>&,
        int,
        plamatrix::DenseMatrix<OtherScalar, plamatrix::Device::GPU>&,
        NormalEstimationGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    void resize(plamatrix::Index rows)
    {
        if (_covariances.rows() == rows)
        {
            return;
        }
        _covariances = plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>
            ::uninitialized(rows, 6);
        _eigenvalues = plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>
            ::uninitialized(rows, 3);
        _eigenvectors = plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>
            ::uninitialized(rows, 9);
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
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> _covariances;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> _eigenvalues;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> _eigenvectors;
    plamatrix::SymmetricEigh3x3Workspace _eigenWorkspace;
};

/// Enqueue indexed KNN normal estimation without staging point or neighbor vectors on the host.
/// The index must match the cloud. Synchronize stream and call workspace.checkStatus().
template <typename Scalar>
void estimateNormalsAsync(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    const GpuSpatialIndex<Scalar>& index,
    int k,
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& normals,
    NormalEstimationGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream);

} // namespace gpu
} // namespace plapoint

#endif
