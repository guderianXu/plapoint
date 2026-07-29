#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <stdexcept>

#include <cuda_runtime.h>

#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/vector.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/spatial_index.h>

namespace plapoint
{
namespace gpu
{

template <typename Scalar>
class NormalRefinementGpuWorkspace
{
public:
    NormalRefinementGpuWorkspace() = default;
    NormalRefinementGpuWorkspace(const NormalRefinementGpuWorkspace&) = delete;
    NormalRefinementGpuWorkspace& operator=(const NormalRefinementGpuWorkspace&) = delete;
    NormalRefinementGpuWorkspace(NormalRefinementGpuWorkspace&&) noexcept = default;
    NormalRefinementGpuWorkspace& operator=(NormalRefinementGpuWorkspace&&) noexcept = default;

    /// Release stream ownership after the bound stream has been synchronized.
    void resetStream()
    {
        if (!_streamBound)
        {
            return;
        }
        const cudaError_t status = cudaStreamQuery(_stream);
        if (status == cudaErrorNotReady)
        {
            throw std::logic_error(
                "NormalRefinementGpuWorkspace cannot reset while its stream has pending work");
        }
        PLAPOINT_CHECK_CUDA(status);

        _neighbors.indices.closeAsyncAllocation();
        _neighbors.squaredDistances.closeAsyncAllocation();
        _stream = nullptr;
        _streamBound = false;
    }

private:
    template <typename OtherScalar>
    friend void smoothNormalsAsync(
        const PointCloud<OtherScalar, plamatrix::Device::GPU>&,
        const GpuSpatialIndex<OtherScalar>&,
        int,
        plamatrix::DenseMatrix<OtherScalar, plamatrix::Device::GPU>&,
        NormalRefinementGpuWorkspace<OtherScalar>&,
        cudaStream_t);

    void bindStream(cudaStream_t stream)
    {
        if (!_streamBound)
        {
            _stream = stream;
            _streamBound = true;
            return;
        }
        if (_stream != stream)
        {
            throw std::logic_error(
                "NormalRefinementGpuWorkspace cannot be reused on another stream before resetStream()");
        }
    }

    GpuSpatialQueryWorkspace<Scalar> _queryWorkspace;
    GpuKnnSearchResult<Scalar> _neighbors;
    cudaStream_t _stream = nullptr;
    bool _streamBound = false;
};

/// Enqueue KNN normal averaging into an independent output matrix for 1 <= k <= 32.
/// Output must not alias the cloud's normal storage. Components are accumulated in double;
/// if any selected neighbor normal is non-finite, that point retains its source normal.
template <typename Scalar>
void smoothNormalsAsync(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    const GpuSpatialIndex<Scalar>& index,
    int k,
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& output,
    NormalRefinementGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream);

/// Enqueue in-place normal flips so finite rows point toward the supplied viewpoint.
template <typename Scalar>
void orientNormalsTowardViewpointAsync(
    PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    const plamatrix::Vec3<Scalar>& viewpoint,
    cudaStream_t stream);

} // namespace gpu
} // namespace plapoint

#endif
