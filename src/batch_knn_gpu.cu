// Implementation of GPU batch KNN — host-side wrapper around brute-force kernel

#include <plapoint/gpu/knn.h>
#include <plapoint/gpu/cuda_check.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
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
    if (K > 32)
    {
        throw std::invalid_argument("GPU KNN supports K in [1, 32]");
    }
    if (K > N)
    {
        throw std::invalid_argument("GPU KNN requires K <= N");
    }

    const int K_use = K;

    DeviceBuffer<Scalar> d_queries(static_cast<std::size_t>(M * 3));
    DeviceBuffer<Scalar> d_data(static_cast<std::size_t>(N * 3));
    DeviceBuffer<int> d_indices(static_cast<std::size_t>(M * K_use));
    DeviceBuffer<Scalar> d_dists(static_cast<std::size_t>(M * K_use));

    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_queries.get(), h_queries, M * 3 * sizeof(Scalar), cudaMemcpyHostToDevice));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_data.get(), h_data, N * 3 * sizeof(Scalar), cudaMemcpyHostToDevice));

    cudaError_t err = launchBruteForceKnn<Scalar>(
        d_queries.get(), d_data.get(), M, N, K_use, d_indices.get(), d_dists.get(), 0);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GPU KNN failed: ") + cudaGetErrorString(err));
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    out_indices.resize(static_cast<std::size_t>(M * K_use));
    out_dists.resize(static_cast<std::size_t>(M * K_use));

    PLAPOINT_CHECK_CUDA(cudaMemcpy(out_indices.data(), d_indices.get(), M * K_use * sizeof(int), cudaMemcpyDeviceToHost));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(out_dists.data(), d_dists.get(), M * K_use * sizeof(Scalar), cudaMemcpyDeviceToHost));
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

// Device-pointer version: no host roundtrip
template <typename Scalar>
void batchKnnDeviceImpl(const Scalar* d_queries, int M,
                        const Scalar* d_data, int N, int K,
                        int* d_indices, Scalar* d_dists)
{
    if (M <= 0 || N <= 0 || K <= 0)
    {
        return;
    }
    if (K > 32)
    {
        throw std::invalid_argument("GPU KNN supports K in [1, 32]");
    }
    if (K > N)
    {
        throw std::invalid_argument("GPU KNN requires K <= N");
    }
    const int K_use = K;
    cudaError_t err = launchBruteForceKnn<Scalar>(d_queries, d_data, M, N, K_use, d_indices, d_dists, 0);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GPU KNN failed: ") + cudaGetErrorString(err));
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaDeviceSynchronize());
}

void batchKnnDevice(const float* d_queries, int M,
                    const float* d_data, int N, int K,
                    int* d_indices, float* d_dists)
{
    batchKnnDeviceImpl<float>(d_queries, M, d_data, N, K, d_indices, d_dists);
}

void batchKnnDevice(const double* d_queries, int M,
                    const double* d_data, int N, int K,
                    int* d_indices, double* d_dists)
{
    batchKnnDeviceImpl<double>(d_queries, M, d_data, N, K, d_indices, d_dists);
}

} // namespace gpu
} // namespace plapoint
