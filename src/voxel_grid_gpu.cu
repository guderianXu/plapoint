#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_grid.h>

namespace plapoint
{
namespace gpu
{

namespace
{

struct VoxelKey
{
    int x;
    int y;
    int z;
};

struct VoxelKeyLess
{
    __host__ __device__ bool operator()(const VoxelKey& lhs, const VoxelKey& rhs) const
    {
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        return lhs.z < rhs.z;
    }
};

struct VoxelKeyEqual
{
    __host__ __device__ bool operator()(const VoxelKey& lhs, const VoxelKey& rhs) const
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

template <typename Scalar>
struct ComputeVoxelKey
{
    const Scalar* points;
    int point_count;
    Scalar leaf_x;
    Scalar leaf_y;
    Scalar leaf_z;

    __host__ __device__ VoxelKey operator()(int idx) const
    {
        const Scalar x = points[idx];
        const Scalar y = points[point_count + idx];
        const Scalar z = points[2 * point_count + idx];
        return {
            static_cast<int>(floor(static_cast<double>(x / leaf_x))),
            static_cast<int>(floor(static_cast<double>(y / leaf_y))),
            static_cast<int>(floor(static_cast<double>(z / leaf_z)))
        };
    }
};

template <typename Scalar, int Dim>
struct GatherCoordinate
{
    const Scalar* points;
    int point_count;

    __host__ __device__ Scalar operator()(int idx) const
    {
        return points[static_cast<int>(Dim) * point_count + idx];
    }
};

template <typename Scalar>
__global__ void writeCentroidsColumnMajor(
    const Scalar* sum_x,
    const Scalar* sum_y,
    const Scalar* sum_z,
    const int* counts,
    int voxel_count,
    Scalar* out_points)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= voxel_count)
    {
        return;
    }

    const Scalar inv_count = Scalar(1) / static_cast<Scalar>(counts[idx]);
    out_points[idx] = sum_x[idx] * inv_count;
    out_points[voxel_count + idx] = sum_y[idx] * inv_count;
    out_points[2 * voxel_count + idx] = sum_z[idx] * inv_count;
}

template <typename Scalar>
int voxelGridDownsampleColumnMajorImpl(const Scalar* d_points, int N,
                                       Scalar leaf_x, Scalar leaf_y, Scalar leaf_z,
                                       Scalar* d_out_points,
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
    if (!std::isfinite(leaf_x) || !std::isfinite(leaf_y) || !std::isfinite(leaf_z) ||
        leaf_x <= Scalar(0) || leaf_y <= Scalar(0) || leaf_z <= Scalar(0))
    {
        throw std::invalid_argument("VoxelGrid GPU: leaf size must be positive");
    }

    auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<int> indices(static_cast<std::size_t>(N));
    thrust::device_vector<VoxelKey> keys(static_cast<std::size_t>(N));
    thrust::sequence(policy, indices.begin(), indices.end(), 0);
    thrust::transform(policy, indices.begin(), indices.end(), keys.begin(),
                      ComputeVoxelKey<Scalar>{d_points, N, leaf_x, leaf_y, leaf_z});
    thrust::sort_by_key(policy, keys.begin(), keys.end(), indices.begin(), VoxelKeyLess{});

    thrust::device_vector<VoxelKey> unique_keys(static_cast<std::size_t>(N));
    thrust::device_vector<Scalar> sum_x(static_cast<std::size_t>(N));
    thrust::device_vector<Scalar> sum_y(static_cast<std::size_t>(N));
    thrust::device_vector<Scalar> sum_z(static_cast<std::size_t>(N));
    thrust::device_vector<int> counts(static_cast<std::size_t>(N));

    auto x_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinate<Scalar, 0>{d_points, N});
    auto y_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinate<Scalar, 1>{d_points, N});
    auto z_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinate<Scalar, 2>{d_points, N});

    auto reduced_x = thrust::reduce_by_key(policy,
                                           keys.begin(), keys.end(),
                                           x_values,
                                           unique_keys.begin(),
                                           sum_x.begin(),
                                           VoxelKeyEqual{},
                                           thrust::plus<Scalar>{});
    const int voxel_count = static_cast<int>(reduced_x.first - unique_keys.begin());

    thrust::reduce_by_key(policy,
                          keys.begin(), keys.end(),
                          y_values,
                          thrust::make_discard_iterator(),
                          sum_y.begin(),
                          VoxelKeyEqual{},
                          thrust::plus<Scalar>{});
    thrust::reduce_by_key(policy,
                          keys.begin(), keys.end(),
                          z_values,
                          thrust::make_discard_iterator(),
                          sum_z.begin(),
                          VoxelKeyEqual{},
                          thrust::plus<Scalar>{});
    thrust::reduce_by_key(policy,
                          keys.begin(), keys.end(),
                          thrust::make_constant_iterator(1),
                          thrust::make_discard_iterator(),
                          counts.begin(),
                          VoxelKeyEqual{},
                          thrust::plus<int>{});

    constexpr int block_size = 256;
    const int grid_size = (voxel_count + block_size - 1) / block_size;
    writeCentroidsColumnMajor<Scalar><<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(sum_x.data()),
        thrust::raw_pointer_cast(sum_y.data()),
        thrust::raw_pointer_cast(sum_z.data()),
        thrust::raw_pointer_cast(counts.data()),
        voxel_count,
        d_out_points);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return voxel_count;
}

} // namespace

int voxelGridDownsampleColumnMajor(const float* d_points, int N,
                                   float leaf_x, float leaf_y, float leaf_z,
                                   float* d_out_points,
                                   cudaStream_t stream)
{
    return voxelGridDownsampleColumnMajorImpl<float>(d_points, N, leaf_x, leaf_y, leaf_z, d_out_points, stream);
}

int voxelGridDownsampleColumnMajor(const double* d_points, int N,
                                   double leaf_x, double leaf_y, double leaf_z,
                                   double* d_out_points,
                                   cudaStream_t stream)
{
    return voxelGridDownsampleColumnMajorImpl<double>(d_points, N, leaf_x, leaf_y, leaf_z, d_out_points, stream);
}

} // namespace gpu
} // namespace plapoint
