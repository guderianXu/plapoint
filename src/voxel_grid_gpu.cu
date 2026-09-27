#include <cmath>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_grid.h>

#include "voxel_grouping_gpu_kernels.cuh"
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_matrix.h>

namespace plapoint
{
    namespace gpu
    {

        namespace
        {

            template <typename Scalar>
            int voxelGridDownsampleColumnMajorImpl(const Scalar* d_points,
                                                   int N,
                                                   Scalar leaf_x,
                                                   Scalar leaf_y,
                                                   Scalar leaf_z,
                                                   Scalar* d_out_points,
                                                   plamatrix::internal::ExecutionContext& context,
                                                   cudaStream_t stream)
            {
                if (N <= 0)
                {
                    return 0;
                }
                if (!d_points || !d_out_points)
                {
                    throw std::invalid_argument("VoxelGrid GPU: device pointers must not be null");
                }
                if (!std::isfinite(leaf_x) || !std::isfinite(leaf_y) || !std::isfinite(leaf_z) || leaf_x <= Scalar(0) ||
                    leaf_y <= Scalar(0) || leaf_z <= Scalar(0))
                {
                    throw std::invalid_argument("VoxelGrid GPU: leaf size must be positive");
                }
                return detail::groupVoxelCentroidsColumnMajor(d_points,
                                                              N,
                                                              Scalar(0),
                                                              Scalar(0),
                                                              Scalar(0),
                                                              leaf_x,
                                                              leaf_y,
                                                              leaf_z,
                                                              d_out_points,
                                                              nullptr,
                                                              "VoxelGrid GPU",
                                                              context,
                                                              stream);
            }

        } // namespace

        int voxelGridDownsampleColumnMajor(const float* d_points,
                                           int N,
                                           float leaf_x,
                                           float leaf_y,
                                           float leaf_z,
                                           float* d_out_points,
                                           cudaStream_t stream)
        {
            int device = 0;
            PLAPOINT_CHECK_CUDA(cudaGetDevice(&device));
            auto context =
                plamatrix::internal::ExecutionContext::create({plamatrix::internal::Backend::Cuda, static_cast<std::size_t>(device)});
            return voxelGridDownsampleColumnMajorImpl<float>(
                d_points, N, leaf_x, leaf_y, leaf_z, d_out_points, context, stream);
        }

        int voxelGridDownsampleColumnMajor(const double* d_points,
                                           int N,
                                           double leaf_x,
                                           double leaf_y,
                                           double leaf_z,
                                           double* d_out_points,
                                           cudaStream_t stream)
        {
            int device = 0;
            PLAPOINT_CHECK_CUDA(cudaGetDevice(&device));
            auto context =
                plamatrix::internal::ExecutionContext::create({plamatrix::internal::Backend::Cuda, static_cast<std::size_t>(device)});
            return voxelGridDownsampleColumnMajorImpl<double>(
                d_points, N, leaf_x, leaf_y, leaf_z, d_out_points, context, stream);
        }

        template <typename Scalar>
        int voxelGridDownsampleColumnMajorMatrixImpl(const plamatrix::internal::ResidentMatrix<Scalar>& points,
                                                     Scalar leaf_x,
                                                     Scalar leaf_y,
                                                     Scalar leaf_z,
                                                     plamatrix::internal::ResidentMatrix<Scalar>& out_points,
                                                     cudaStream_t stream)
        {
            out_points.validateContext(points.context());
            static_cast<void>(points.template view<plamatrix::internal::Device::GPU>());
            PLAPOINT_CHECK_CUDA(cudaSetDevice(static_cast<int>(points.context().device().index)));
            if (points.cols() != 3 || out_points.cols() != 3)
            {
                throw std::invalid_argument("VoxelGrid GPU PlaMatrix inputs must be Nx3");
            }
            if (points.rows() > std::numeric_limits<int>::max())
            {
                throw std::overflow_error("VoxelGrid GPU PlaMatrix point count exceeds int range");
            }
            if (out_points.rows() < points.rows())
            {
                throw std::invalid_argument("VoxelGrid GPU PlaMatrix output capacity must be at least input rows");
            }
            return voxelGridDownsampleColumnMajorImpl<Scalar>(points.data(),
                                                              static_cast<int>(points.rows()),
                                                              leaf_x,
                                                              leaf_y,
                                                              leaf_z,
                                                              out_points.data(),
                                                              points.context(),
                                                              stream);
        }

        int voxelGridDownsampleColumnMajor(const plamatrix::internal::ResidentMatrix<float>& points,
                                           float leaf_x,
                                           float leaf_y,
                                           float leaf_z,
                                           plamatrix::internal::ResidentMatrix<float>& out_points,
                                           cudaStream_t stream)
        {
            return voxelGridDownsampleColumnMajorMatrixImpl<float>(points, leaf_x, leaf_y, leaf_z, out_points, stream);
        }

        int voxelGridDownsampleColumnMajor(const plamatrix::internal::ResidentMatrix<double>& points,
                                           double leaf_x,
                                           double leaf_y,
                                           double leaf_z,
                                           plamatrix::internal::ResidentMatrix<double>& out_points,
                                           cudaStream_t stream)
        {
            return voxelGridDownsampleColumnMajorMatrixImpl<double>(points, leaf_x, leaf_y, leaf_z, out_points, stream);
        }

    } // namespace gpu
} // namespace plapoint
