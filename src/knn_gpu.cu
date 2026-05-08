// CUDA kernel for brute-force batch KNN search
// Each block handles one query point and finds its K nearest neighbors

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace plapoint {
namespace gpu {

// Shared memory layout per block: K distances + K indices (maintained as insertion-sorted lists)
// BlockDim.x threads cooperate to find K nearest for one query point

template <typename Scalar, int BLOCK_SIZE, int K>
__global__ void bruteForceKnnKernel(
    const Scalar* __restrict__ queries,    // M x 3
    const Scalar* __restrict__ data,       // N x 3
    int M, int N,
    int* __restrict__ out_indices,         // M x K
    Scalar* __restrict__ out_dists)         // M x K
{
    extern __shared__ char smem[];
    Scalar* s_dists = reinterpret_cast<Scalar*>(smem);
    int*    s_inds  = reinterpret_cast<int*>(s_dists + BLOCK_SIZE * K);

    int tid = threadIdx.x;
    int query_idx = blockIdx.x;

    if (query_idx >= M) return;

    // Initialize shared top-K with infinity
    for (int i = tid; i < K; i += BLOCK_SIZE)
    {
        s_dists[i] = FLT_MAX;
        s_inds[i]  = -1;
    }
    __syncthreads();

    Scalar qx = queries[query_idx * 3];
    Scalar qy = queries[query_idx * 3 + 1];
    Scalar qz = queries[query_idx * 3 + 2];

    // Each thread computes distances for a strided range of data points
    for (int i = tid; i < N; i += BLOCK_SIZE)
    {
        Scalar dx = qx - data[i * 3];
        Scalar dy = qy - data[i * 3 + 1];
        Scalar dz = qz - data[i * 3 + 2];
        Scalar dist = dx * dx + dy * dy + dz * dz;

        // Insert into sorted top-K (descending order, largest at position 0)
        // Find insertion position
        int pos = -1;
        for (int j = 0; j < K; ++j)
        {
            if (dist < s_dists[j])
            {
                pos = j;
                break;
            }
        }
        if (pos >= 0)
        {
            // Shift larger elements up, insert at pos
            for (int j = K - 1; j > pos; --j)
            {
                s_dists[j] = s_dists[j - 1];
                s_inds[j]  = s_inds[j - 1];
            }
            s_dists[pos] = dist;
            s_inds[pos]  = i;
        }
    }
    __syncthreads();

    // Write block's top-K to global memory (one thread writes)
    if (tid == 0)
    {
        for (int k = 0; k < K; ++k)
        {
            out_indices[query_idx * K + k] = s_inds[k];
            if (out_dists)
                out_dists[query_idx * K + k] = s_dists[k];
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

    // Determine the kernel template K (must match K_alloc in callers)
    // Round K up to the nearest supported value
    int K_template = K;
    if (K <= 1)      K_template = 1;
    else if (K <= 4)  K_template = 4;
    else if (K <= 8)  K_template = 8;
    else if (K <= 16) K_template = 16;
    else              K_template = 32;

    size_t smem = BLOCK_SIZE * K_template * (sizeof(Scalar) + sizeof(int));

    switch (K_template)
    {
        case 1:  bruteForceKnnKernel<Scalar, BLOCK_SIZE, 1><<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists); break;
        case 4:  bruteForceKnnKernel<Scalar, BLOCK_SIZE, 4><<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists); break;
        case 8:  bruteForceKnnKernel<Scalar, BLOCK_SIZE, 8><<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists); break;
        case 16: bruteForceKnnKernel<Scalar, BLOCK_SIZE, 16><<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists); break;
        default: bruteForceKnnKernel<Scalar, BLOCK_SIZE, 32><<<M, BLOCK_SIZE, smem, stream>>>(d_queries, d_data, M, N, d_out_indices, d_out_dists); break;
    }
    #undef LAUNCH_K

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t launchBruteForceKnn<float>(
    const float*, const float*, int, int, int, int*, float*, cudaStream_t);

template cudaError_t launchBruteForceKnn<double>(
    const double*, const double*, int, int, int, int*, double*, cudaStream_t);

} // namespace gpu
} // namespace plapoint
