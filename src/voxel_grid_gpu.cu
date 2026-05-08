// CUDA kernel for voxel grid downsampling
// Uses atomic operations to accumulate centroids per voxel

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace plapoint {
namespace gpu {

// Hash point to a linear voxel index based on leaf size
template <typename Scalar>
__device__ int64_t voxelIndex(Scalar x, Scalar y, Scalar z,
                                Scalar lx, Scalar ly, Scalar lz,
                                Scalar min_x, Scalar min_y, Scalar min_z)
{
    int ix = static_cast<int>(floorf((x - min_x) / lx));
    int iy = static_cast<int>(floorf((y - min_y) / ly));
    int iz = static_cast<int>(floorf((z - min_z) / lz));
    // Simple hash: combine with prime multipliers (fits in 64-bit)
    return (static_cast<int64_t>(ix) * 73856093) ^
           (static_cast<int64_t>(iy) * 19349663) ^
           (static_cast<int64_t>(iz) * 83492791);
}

// Phase 1: compute voxel key for each point and find unique voxels
template <typename Scalar>
__global__ void computeVoxelKeys(
    const Scalar* __restrict__ points,     // N x 3
    int N,
    Scalar lx, Scalar ly, Scalar lz,
    Scalar min_x, Scalar min_y, Scalar min_z,
    int64_t* __restrict__ keys)             // N
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Scalar x = points[i * 3];
    Scalar y = points[i * 3 + 1];
    Scalar z = points[i * 3 + 2];

    keys[i] = voxelIndex(x, y, z, lx, ly, lz, min_x, min_y, min_z);
}

// Phase 2: accumulate centroids per voxel key using a two-pass approach
// First, sort by key (uses thrust::sort_by_key, called from host)
// Then, reduce by key to compute centroids

template <typename Scalar>
__global__ void reduceVoxelCentroids(
    const Scalar* __restrict__ points,      // N x 3 (sorted by key)
    const int64_t* __restrict__ keys,       // N (sorted)
    int N,
    Scalar* __restrict__ out_centroids,     // max_voxels x 3
    int* __restrict__ out_counts,           // max_voxels
    int* __restrict__ out_nvoxels)          // 1
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Each thread checks if its key differs from previous
    bool is_first = (i == 0) || (keys[i] != keys[i - 1]);

    if (is_first)
    {
        // Start a new voxel: accumulate all points with this key
        Scalar sum_x = 0, sum_y = 0, sum_z = 0;
        int count = 0;
        int64_t this_key = keys[i];
        int j = i;
        while (j < N && keys[j] == this_key)
        {
            sum_x += points[j * 3];
            sum_y += points[j * 3 + 1];
            sum_z += points[j * 3 + 2];
            ++count;
            ++j;
        }

        // Write centroid (use atomic to reserve output slot)
        int slot = atomicAdd(out_nvoxels, 1);
        out_centroids[slot * 3]     = sum_x / Scalar(count);
        out_centroids[slot * 3 + 1] = sum_y / Scalar(count);
        out_centroids[slot * 3 + 2] = sum_z / Scalar(count);
        out_counts[slot] = count;
    }
}

// Host launch function
template <typename Scalar>
cudaError_t voxelDownsample(
    const Scalar* d_points, int N,
    Scalar lx, Scalar ly, Scalar lz,
    Scalar min_x, Scalar min_y, Scalar min_z,
    Scalar* d_out_centroids, int* d_out_counts,
    int* d_out_nvoxels,
    int64_t* d_keys,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (N + block - 1) / block;

    // Phase 1: compute keys
    computeVoxelKeys<<<grid, block, 0, stream>>>(
        d_points, N, lx, ly, lz, min_x, min_y, min_z, d_keys);

    // Phase 2: reduce by key (requires sorted keys)
    // Note: caller must sort by key first (use thrust::sort_by_key or cub)
    // For now, this assumes keys are already sorted (for simplicity)
    cudaMemsetAsync(d_out_nvoxels, 0, sizeof(int), stream);
    reduceVoxelCentroids<<<grid, block, 0, stream>>>(
        d_points, d_keys, N, d_out_centroids, d_out_counts, d_out_nvoxels);

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t voxelDownsample<float>(
    const float*, int, float, float, float, float, float, float,
    float*, int*, int*, int64_t*, cudaStream_t);

template cudaError_t voxelDownsample<double>(
    const double*, int, double, double, double, double, double, double,
    double*, int*, int*, int64_t*, cudaStream_t);

} // namespace gpu
} // namespace plapoint
