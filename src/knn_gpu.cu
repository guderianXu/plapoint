// CUDA kernel for brute-force batch KNN search
// Each block handles one query point, per-thread local top-K + block reduction

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <math_constants.h>

namespace plapoint {
namespace gpu {

// Per-thread maintains local top-K in ascending order (smallest distance at index 0)
// This makes it easy to compare against the K-th largest (which is at d[K-1])

template <int K>
__device__ void localTopKInsert(double* d, int* idx, double dist, int data_idx)
{
    if (!isfinite(dist)) return;

    // Sorted ascending: d[0] is smallest, d[K-1] is largest
    // If dist >= largest kept distance, skip
    if (dist >= d[K - 1]) return;

    // Find insertion position: first j where d[j] > dist
    int pos = 0;
    while (pos < K && d[pos] <= dist) ++pos;

    // pos is where dist should go; shift larger elements right
    for (int j = K - 1; j > pos; --j)
    {
        d[j]   = d[j - 1];
        idx[j] = idx[j - 1];
    }
    d[pos]   = dist;
    idx[pos] = data_idx;
}

__device__ __forceinline__ double knnInfinityDistance()
{
    return CUDART_INF;
}

template <typename Scalar>
__device__ Scalar maxKnnDistance();

template <>
__device__ float maxKnnDistance<float>()
{
    return FLT_MAX;
}

template <>
__device__ double maxKnnDistance<double>()
{
    return DBL_MAX;
}

template <typename Scalar>
__device__ Scalar clampOutputDistance(double dist)
{
    const Scalar max_dist = maxKnnDistance<Scalar>();
    if (!isfinite(dist) || dist >= static_cast<double>(max_dist))
    {
        return max_dist;
    }
    return static_cast<Scalar>(dist);
}

template <typename Scalar>
__device__ double squaredOutputDistance(
    Scalar qx, Scalar qy, Scalar qz,
    Scalar data_x, Scalar data_y, Scalar data_z)
{
    const double dx = static_cast<double>(qx) - static_cast<double>(data_x);
    const double dy = static_cast<double>(qy) - static_cast<double>(data_y);
    const double dz = static_cast<double>(qz) - static_cast<double>(data_z);
    const double scaled_distance = norm3d(dx, dy, dz);
    if (!isfinite(scaled_distance))
    {
        return DBL_MAX;
    }
    if (scaled_distance >= sqrt(DBL_MAX))
    {
        return DBL_MAX;
    }
    return scaled_distance * scaled_distance;
}

template <typename Scalar, int BLOCK_SIZE, int K>
__global__ void bruteForceKnnKernel(
    const Scalar* __restrict__ queries,
    const Scalar* __restrict__ data,
    int M, int N, int outputK,
    bool queries_column_major,
    bool data_column_major,
    bool output_column_major,
    int* __restrict__ out_indices,
    Scalar* __restrict__ out_dists)
{
    // Shared memory: each thread writes its local K, then thread 0 merges
    // Layout: BLOCK_SIZE * K double distances | BLOCK_SIZE * K indices
    extern __shared__ char smem[];
    double* s_dists = reinterpret_cast<double*>(smem);
    int*    s_inds  = reinterpret_cast<int*>(s_dists + BLOCK_SIZE * K);

    int tid = threadIdx.x;
    int query_idx = static_cast<int>(blockIdx.x);

    if (query_idx >= M) return;

    // Per-thread local top-K (registers)
    double local_key[K];
    int    local_idx[K];
    for (int j = 0; j < K; ++j)
    {
        local_key[j] = knnInfinityDistance();
        local_idx[j] = -1;
    }

    const size_t query_offset = static_cast<size_t>(query_idx) * 3u;
    Scalar qx = queries_column_major ? queries[query_idx] : queries[query_offset];
    Scalar qy = queries_column_major ? queries[static_cast<size_t>(M) + query_idx] : queries[query_offset + 1u];
    Scalar qz = queries_column_major ? queries[2u * static_cast<size_t>(M) + query_idx] : queries[query_offset + 2u];

    // Each thread processes strided data points, updating local top-K.
    const size_t n_size = static_cast<size_t>(N);
    for (size_t data_idx = static_cast<size_t>(tid); data_idx < n_size; data_idx += BLOCK_SIZE)
    {
        const int point_idx = static_cast<int>(data_idx);
        const size_t row_major_offset = data_idx * 3u;
        Scalar data_x = data_column_major ? data[data_idx] : data[row_major_offset];
        Scalar data_y = data_column_major ? data[n_size + data_idx] : data[row_major_offset + 1u];
        Scalar data_z = data_column_major ? data[2u * n_size + data_idx] : data[row_major_offset + 2u];
        const double dx = static_cast<double>(qx) - static_cast<double>(data_x);
        const double dy = static_cast<double>(qy) - static_cast<double>(data_y);
        const double dz = static_cast<double>(qz) - static_cast<double>(data_z);
        const double dist_key = norm3d(dx, dy, dz);
        localTopKInsert<K>(local_key, local_idx, dist_key, point_idx);
    }
    __syncthreads();

    // Each thread writes its local top-K to shared memory
    for (int j = 0; j < K; ++j)
    {
        s_dists[tid * K + j] = local_key[j];
        s_inds[tid * K + j]  = local_idx[j];
    }
    __syncthreads();

    // Thread 0 merges all threads' results into final top-K
    if (tid == 0)
    {
        double final_key[K];
        int    final_idx[K];
        for (int j = 0; j < K; ++j)
        {
            final_key[j] = knnInfinityDistance();
            final_idx[j] = -1;
        }

        for (int t = 0; t < BLOCK_SIZE; ++t)
        {
            for (int j = 0; j < K; ++j)
            {
                int idx = s_inds[t * K + j];
                if (idx >= 0)
                {
                    const double candidate_key = s_dists[t * K + j];
                    localTopKInsert<K>(final_key, final_idx, candidate_key, idx);
                }
            }
        }

        for (int k = 0; k < outputK; ++k)
        {
            const size_t output_offset = output_column_major
                ? static_cast<size_t>(query_idx) + static_cast<size_t>(k) * static_cast<size_t>(M)
                : static_cast<size_t>(query_idx) * static_cast<size_t>(outputK) + static_cast<size_t>(k);
            out_indices[output_offset] = final_idx[k];
            if (out_dists)
            {
                double out_dist = DBL_MAX;
                if (final_idx[k] >= 0)
                {
                    const int idx = final_idx[k];
                    const size_t row_major_offset = static_cast<size_t>(idx) * 3u;
                    const Scalar data_x = data_column_major ? data[static_cast<size_t>(idx)] : data[row_major_offset];
                    const Scalar data_y = data_column_major ? data[n_size + static_cast<size_t>(idx)] : data[row_major_offset + 1u];
                    const Scalar data_z = data_column_major ? data[2u * n_size + static_cast<size_t>(idx)] : data[row_major_offset + 2u];
                    out_dist = squaredOutputDistance(qx, qy, qz, data_x, data_y, data_z);
                }
                out_dists[output_offset] = clampOutputDistance<Scalar>(out_dist);
            }
        }
    }
}

template <typename Scalar>
cudaError_t launchBruteForceKnn(
    const Scalar* d_queries, const Scalar* d_data,
    int M, int N, int K,
    int* d_out_indices, Scalar* d_out_dists,
    bool queries_column_major,
    bool data_column_major,
    bool output_column_major,
    cudaStream_t stream)
{
    constexpr int kDefaultBlockSize = 256;
    constexpr int kLargeKBlockSize = 128;

    int K_template = K;
    if (K <= 1)      K_template = 1;
    else if (K <= 4)  K_template = 4;
    else if (K <= 8)  K_template = 8;
    else if (K <= 16) K_template = 16;
    else              K_template = 32;

    // Shared memory: BLOCK_SIZE * K_template double distances plus indices.
    const int block_size = (K_template <= 16) ? kDefaultBlockSize : kLargeKBlockSize;
    size_t smem = static_cast<size_t>(block_size) * static_cast<size_t>(K_template)
        * (sizeof(double) + sizeof(int));

    switch (K_template)
    {
        case 1:
            bruteForceKnnKernel<Scalar, kDefaultBlockSize, 1>
                <<<M, kDefaultBlockSize, smem, stream>>>(
                    d_queries, d_data, M, N, K, queries_column_major, data_column_major,
                    output_column_major,
                    d_out_indices, d_out_dists);
            break;
        case 4:
            bruteForceKnnKernel<Scalar, kDefaultBlockSize, 4>
                <<<M, kDefaultBlockSize, smem, stream>>>(
                    d_queries, d_data, M, N, K, queries_column_major, data_column_major,
                    output_column_major,
                    d_out_indices, d_out_dists);
            break;
        case 8:
            bruteForceKnnKernel<Scalar, kDefaultBlockSize, 8>
                <<<M, kDefaultBlockSize, smem, stream>>>(
                    d_queries, d_data, M, N, K, queries_column_major, data_column_major,
                    output_column_major,
                    d_out_indices, d_out_dists);
            break;
        case 16:
            bruteForceKnnKernel<Scalar, kDefaultBlockSize, 16>
                <<<M, kDefaultBlockSize, smem, stream>>>(
                    d_queries, d_data, M, N, K, queries_column_major, data_column_major,
                    output_column_major,
                    d_out_indices, d_out_dists);
            break;
        default:
            bruteForceKnnKernel<Scalar, kLargeKBlockSize, 32>
                <<<M, kLargeKBlockSize, smem, stream>>>(
                    d_queries, d_data, M, N, K, queries_column_major, data_column_major,
                    output_column_major,
                    d_out_indices, d_out_dists);
            break;
    }

    return cudaGetLastError();
}

template cudaError_t launchBruteForceKnn<float>(
    const float*, const float*, int, int, int, int*, float*, bool, bool, bool, cudaStream_t);

template cudaError_t launchBruteForceKnn<double>(
    const double*, const double*, int, int, int, int*, double*, bool, bool, bool, cudaStream_t);

} // namespace gpu
} // namespace plapoint
