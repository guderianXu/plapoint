// Implementation of GPU batch KNN — host-side wrapper around brute-force kernel

#include <plapoint/gpu/knn.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Forward-declare the kernel from knn_gpu.cu
namespace plapoint { namespace gpu {
template <typename Scalar>
cudaError_t launchBruteForceKnn(
    const Scalar* d_queries, const Scalar* d_data,
    int M, int N, int K,
    int* d_out_indices, Scalar* d_out_dists,
    cudaStream_t stream);
} }

namespace plapoint {
namespace gpu {

template <typename Scalar>
void batchKnnImpl(const Scalar* h_queries, int M,
                  const Scalar* h_data, int N, int K,
                  std::vector<int>& out_indices,
                  std::vector<Scalar>& out_dists)
{
    if (M <= 0 || N <= 0 || K <= 0)
    {
        out_indices.clear();
        out_dists.clear();
        return;
    }

    Scalar* d_queries = nullptr;
    Scalar* d_data    = nullptr;
    int*    d_indices = nullptr;
    Scalar* d_dists   = nullptr;

    cudaMalloc(&d_queries, M * 3 * sizeof(Scalar));
    cudaMalloc(&d_data,    N * 3 * sizeof(Scalar));
    cudaMalloc(&d_indices, M * K * sizeof(int));
    cudaMalloc(&d_dists,   M * K * sizeof(Scalar));

    cudaMemcpy(d_queries, h_queries, M * 3 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,    h_data,    N * 3 * sizeof(Scalar), cudaMemcpyHostToDevice);

    cudaError_t err = launchBruteForceKnn<Scalar>(
        d_queries, d_data, M, N, K, d_indices, d_dists, 0);

    if (err != cudaSuccess)
    {
        cudaFree(d_queries); cudaFree(d_data); cudaFree(d_indices); cudaFree(d_dists);
        throw std::runtime_error(std::string("GPU KNN failed: ") + cudaGetErrorString(err));
    }

    out_indices.resize(static_cast<std::size_t>(M * K));
    out_dists.resize(static_cast<std::size_t>(M * K));

    cudaMemcpy(out_indices.data(), d_indices, M * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_dists.data(),   d_dists,   M * K * sizeof(Scalar), cudaMemcpyDeviceToHost);

    cudaFree(d_queries); cudaFree(d_data); cudaFree(d_indices); cudaFree(d_dists);
}

// Explicit instantiations
template void batchKnnImpl<float>(
    const float*, int, const float*, int, int, std::vector<int>&, std::vector<float>&);
template void batchKnnImpl<double>(
    const double*, int, const double*, int, int, std::vector<int>&, std::vector<double>&);

void batchKnn(const float* h_queries, int M,
              const float* h_data, int N, int K,
              std::vector<int>& out_indices,
              std::vector<float>& out_dists)
{
    batchKnnImpl<float>(h_queries, M, h_data, N, K, out_indices, out_dists);
}

void batchKnn(const double* h_queries, int M,
              const double* h_data, int N, int K,
              std::vector<int>& out_indices,
              std::vector<double>& out_dists)
{
    batchKnnImpl<double>(h_queries, M, h_data, N, K, out_indices, out_dists);
}

} // namespace gpu
} // namespace plapoint
