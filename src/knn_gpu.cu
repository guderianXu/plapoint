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

    // Use specialized kernel for common K values
    size_t smem = BLOCK_SIZE * K * (sizeof(Scalar) + sizeof(int));

    #define LAUNCH_K(K_VAL) \
        bruteForceKnnKernel<Scalar, BLOCK_SIZE, K_VAL><<<M, BLOCK_SIZE, smem, stream>>>( \
            d_queries, d_data, M, N, d_out_indices, d_out_dists)

    switch (K)
    {
        case 1:  LAUNCH_K(1);  break;
        case 4:  LAUNCH_K(4);  break;
        case 8:  LAUNCH_K(8);  break;
        case 16: LAUNCH_K(16); break;
        case 32: LAUNCH_K(32); break;
        default:
        {
            // Generic kernel for arbitrary K
            int threads_per_block = 256;
            int blocks = M;
            size_t sm = threads_per_block * K * (sizeof(Scalar) + sizeof(int));
            bruteForceKnnKernel<Scalar, 256, 32><<<blocks, threads_per_block, sm, stream>>>(
                d_queries, d_data, M, N, d_out_indices, d_out_dists);
            break;
        }
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
