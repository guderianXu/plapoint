// CUDA kernel for brute-force batch KNN search
// Each block handles one query point, per-thread local top-K + block reduction

#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <math_constants.h>

#include <plapoint/gpu/detail/distance_key.cuh>

namespace plapoint {
namespace gpu {

// Per-thread maintains local top-K in ascending order (smallest distance at index 0)
// This makes it easy to compare against the K-th largest (which is at d[K-1])

template <int K>
__device__ void localTopKInsert(
    double* mantissas,
    int* exponents,
    int* indices,
    const detail::DistanceKey& distance,
    int data_idx)
{
    if (distance.exponent == INT_MAX) return;

    // Sorted ascending: d[0] is smallest, d[K-1] is largest
    // Equal distances are ordered by source index for backend-independent results.
    const auto better = [](const detail::DistanceKey& lhs, int lhs_idx,
                           const detail::DistanceKey& rhs, int rhs_idx)
    {
        return detail::distanceLess(lhs, rhs)
            || (!detail::distanceLess(rhs, lhs) && (rhs_idx < 0 || lhs_idx < rhs_idx));
    };
    const detail::DistanceKey last{exponents[K - 1], mantissas[K - 1]};
    if (!better(distance, data_idx, last, indices[K - 1])) return;

    // Find insertion position: first entry ordered after this candidate.
    int pos = 0;
    while (pos < K)
    {
        const detail::DistanceKey current{exponents[pos], mantissas[pos]};
        if (better(distance, data_idx, current, indices[pos]))
        {
            break;
        }
        ++pos;
    }

    // pos is where dist should go; shift larger elements right
    for (int j = K - 1; j > pos; --j)
    {
        mantissas[j] = mantissas[j - 1];
        exponents[j] = exponents[j - 1];
        indices[j] = indices[j - 1];
    }
    mantissas[pos] = distance.mantissa;
    exponents[pos] = distance.exponent;
    indices[pos] = data_idx;
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
    // Layout: BLOCK_SIZE * K mantissas | exponents | source indices
    extern __shared__ char smem[];
    double* s_dists = reinterpret_cast<double*>(smem);
    int* s_exponents = reinterpret_cast<int*>(s_dists + BLOCK_SIZE * K);
    int* s_inds = s_exponents + BLOCK_SIZE * K;

    int tid = threadIdx.x;
    int query_idx = static_cast<int>(blockIdx.x);

    if (query_idx >= M) return;

    // Per-thread local top-K (registers)
    double local_key[K];
    int local_exponent[K];
    int local_idx[K];
    for (int j = 0; j < K; ++j)
    {
        local_key[j] = DBL_MAX;
        local_exponent[j] = INT_MAX;
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
        const auto distance = detail::finiteDistanceKey(
            static_cast<double>(qx), static_cast<double>(qy), static_cast<double>(qz),
            static_cast<double>(data_x), static_cast<double>(data_y),
            static_cast<double>(data_z));
        localTopKInsert<K>(
            local_key, local_exponent, local_idx, distance, point_idx);
    }
    __syncthreads();

    // Each thread writes its local top-K to shared memory
    for (int j = 0; j < K; ++j)
    {
        s_dists[tid * K + j] = local_key[j];
        s_exponents[tid * K + j] = local_exponent[j];
        s_inds[tid * K + j] = local_idx[j];
    }
    __syncthreads();

    // Thread 0 merges all threads' results into final top-K
    if (tid == 0)
    {
        double final_key[K];
        int final_exponent[K];
        int final_idx[K];
        for (int j = 0; j < K; ++j)
        {
            final_key[j] = DBL_MAX;
            final_exponent[j] = INT_MAX;
            final_idx[j] = -1;
        }

        for (int t = 0; t < BLOCK_SIZE; ++t)
        {
            for (int j = 0; j < K; ++j)
            {
                int idx = s_inds[t * K + j];
                if (idx >= 0)
                {
                    const detail::DistanceKey candidate{
                        s_exponents[t * K + j], s_dists[t * K + j]};
                    localTopKInsert<K>(
                        final_key, final_exponent, final_idx, candidate, idx);
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
                Scalar out_dist = sizeof(Scalar) == sizeof(float) ? Scalar(FLT_MAX) : Scalar(DBL_MAX);
                if (final_idx[k] >= 0)
                {
                    const detail::DistanceKey distance{final_exponent[k], final_key[k]};
                    out_dist = detail::squaredOutputDistance<Scalar>(distance);
                }
                out_dists[output_offset] = out_dist;
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
    constexpr int kMediumKBlockSize = 128;
    constexpr int kLargeKBlockSize = 64;

    int K_template = K;
    if (K <= 1)      K_template = 1;
    else if (K <= 4)  K_template = 4;
    else if (K <= 8)  K_template = 8;
    else if (K <= 16) K_template = 16;
    else              K_template = 32;

    // Shared memory: BLOCK_SIZE * K_template mantissas plus two integer arrays.
    const int block_size = K_template <= 8
        ? kDefaultBlockSize
        : (K_template <= 16 ? kMediumKBlockSize : kLargeKBlockSize);
    size_t smem = static_cast<size_t>(block_size) * static_cast<size_t>(K_template)
        * (sizeof(double) + 2 * sizeof(int));

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
            bruteForceKnnKernel<Scalar, kMediumKBlockSize, 16>
                <<<M, kMediumKBlockSize, smem, stream>>>(
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
