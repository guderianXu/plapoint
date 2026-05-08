// CUDA kernel for brute-force batch KNN search
// Each block handles one query point, per-thread local top-K + block reduction

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace plapoint {
namespace gpu {

// Per-thread maintains local top-K in ascending order (smallest distance at index 0)
// This makes it easy to compare against the K-th largest (which is at d[K-1])

template <typename Scalar, int K>
__device__ void localTopKInsert(Scalar* d, int* idx, Scalar dist, int data_idx)
{
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

// Warp reduction: merge local top-K from all threads in a warp into shared memory
template <typename Scalar, int K>
__device__ void warpReduceTopK(
    volatile Scalar* s_dists, volatile int* s_inds, int tid,
    Scalar* local_d, int* local_idx)
{
    // Thread 0 of each warp writes its local best into shared
    // Then thread 0 of block merges all warp results
    // For simplicity: each thread's result is already in local arrays
    // We use a single-threaded merge at the end
}

template <typename Scalar, int BLOCK_SIZE, int K>
__global__ void bruteForceKnnKernel(
    const Scalar* __restrict__ queries,
    const Scalar* __restrict__ data,
    int M, int N,
    int* __restrict__ out_indices,
    Scalar* __restrict__ out_dists)
{
    // Shared memory: each thread writes its local K, then thread 0 merges
    // Layout: BLOCK_SIZE * K distances | BLOCK_SIZE * K indices
    extern __shared__ char smem[];
    Scalar* s_dists = reinterpret_cast<Scalar*>(smem);
    int*    s_inds  = reinterpret_cast<int*>(s_dists + BLOCK_SIZE * K);

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int query_idx = blockIdx.x;

    if (query_idx >= M) return;

    // Per-thread local top-K (registers)
    Scalar local_d[K];
    int    local_idx[K];
    for (int j = 0; j < K; ++j)
    {
        local_d[j]   = FLT_MAX;
        local_idx[j] = -1;
    }

    Scalar qx = queries[query_idx * 3];
    Scalar qy = queries[query_idx * 3 + 1];
    Scalar qz = queries[query_idx * 3 + 2];

    // Each thread processes strided data points, updating local top-K
    for (int i = tid; i < N; i += BLOCK_SIZE)
    {
        Scalar dx = qx - data[i * 3];
        Scalar dy = qy - data[i * 3 + 1];
        Scalar dz = qz - data[i * 3 + 2];
        Scalar dist = dx * dx + dy * dy + dz * dz;
        localTopKInsert<Scalar, K>(local_d, local_idx, dist, i);
    }
    __syncthreads();

    // Each thread writes its local top-K to shared memory
    for (int j = 0; j < K; ++j)
    {
        s_dists[tid * K + j] = local_d[j];
        s_inds[tid * K + j]  = local_idx[j];
    }
    __syncthreads();

    // Thread 0 merges all threads' results into final top-K
    if (tid == 0)
    {
        Scalar final_d[K];
        int    final_idx[K];
        for (int j = 0; j < K; ++j)
        {
            final_d[j]   = FLT_MAX;
            final_idx[j] = -1;
        }

        for (int t = 0; t < BLOCK_SIZE; ++t)
        {
            for (int j = 0; j < K; ++j)
            {
                int idx = s_inds[t * K + j];
                if (idx >= 0)
                    localTopKInsert<Scalar, K>(final_d, final_idx, s_dists[t * K + j], idx);
            }
        }

        for (int k = 0; k < K; ++k)
        {
            out_indices[query_idx * K + k] = final_idx[k];
            if (out_dists)
                out_dists[query_idx * K + k] = final_d[k];
        }
    }
}

template <typename Scalar>
cudaError_t launchBruteForceKnn(
    const Scalar* d_queries, const Scalar* d_data,
    int M, int N, int K,
    int* d_out_indices, Scalar* d_out_dists,
    cudaStream_t stream)
{
    const int BLOCK_SIZE = 256;
    const int NUM_WARPS = BLOCK_SIZE / 32;

    int K_template = K;
    if (K <= 1)      K_template = 1;
    else if (K <= 4)  K_template = 4;
    else if (K <= 8)  K_template = 8;
    else if (K <= 16) K_template = 16;
    else              K_template = 32;

    // Shared memory: BLOCK_SIZE * K_template per type
    size_t smem = BLOCK_SIZE * K_template * (sizeof(Scalar) + sizeof(int));

    switch (K_template)
    {
        case 1:
            bruteForceKnnKernel<Scalar, BLOCK_SIZE, 1>
                <<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists);
            break;
        case 4:
            bruteForceKnnKernel<Scalar, BLOCK_SIZE, 4>
                <<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists);
            break;
        case 8:
            bruteForceKnnKernel<Scalar, BLOCK_SIZE, 8>
                <<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists);
            break;
        case 16:
            bruteForceKnnKernel<Scalar, BLOCK_SIZE, 16>
                <<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists);
            break;
        default:
            bruteForceKnnKernel<Scalar, BLOCK_SIZE, 32>
                <<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists);
            break;
    }

    return cudaGetLastError();
}

template cudaError_t launchBruteForceKnn<float>(
    const float*, const float*, int, int, int, int*, float*, cudaStream_t);

template cudaError_t launchBruteForceKnn<double>(
    const double*, const double*, int, int, int, int*, double*, cudaStream_t);

} // namespace gpu
} // namespace plapoint
