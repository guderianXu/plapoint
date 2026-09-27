// Implementation of GPU batch KNN — host-side wrapper around brute-force kernel

#include <plapoint/gpu/knn.h>
#include <plapoint/gpu/cuda_check.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/core/backend.h>

// Forward-declare the kernel from knn_gpu.cu
namespace plapoint { namespace gpu {
template <typename Scalar>
cudaError_t launchBruteForceKnn(
    const Scalar* d_queries, const Scalar* d_data,
    int M, int N, int K,
    int* d_out_indices, Scalar* d_out_dists,
    bool queries_column_major,
    bool data_column_major,
    bool output_column_major,
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
    if (!h_queries || !h_data)
    {
        throw std::invalid_argument("GPU KNN host input pointers must not be null");
    }

    const int K_use = K;
    const std::size_t query_count = static_cast<std::size_t>(M);
    const std::size_t data_count = static_cast<std::size_t>(N);
    const std::size_t k_count = static_cast<std::size_t>(K_use);
    const std::size_t query_scalars = detail::checkedSizeProduct(query_count, 3, "GPU KNN query buffer");
    const std::size_t data_scalars = detail::checkedSizeProduct(data_count, 3, "GPU KNN data buffer");
    const std::size_t result_count =
        detail::checkedSizeProduct(query_count, k_count, "GPU KNN result buffer");
    const std::size_t query_bytes = detail::checkedByteCount<Scalar>(query_scalars, "GPU KNN query buffer");
    const std::size_t data_bytes = detail::checkedByteCount<Scalar>(data_scalars, "GPU KNN data buffer");
    const std::size_t index_bytes = detail::checkedByteCount<int>(result_count, "GPU KNN index buffer");
    const std::size_t dist_bytes = detail::checkedByteCount<Scalar>(result_count, "GPU KNN distance buffer");

    DeviceBuffer<Scalar> d_queries(query_scalars);
    DeviceBuffer<Scalar> d_data(data_scalars);
    DeviceBuffer<int> d_indices(result_count);
    DeviceBuffer<Scalar> d_dists(result_count);

    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_queries.get(), h_queries, query_bytes, cudaMemcpyHostToDevice));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_data.get(), h_data, data_bytes, cudaMemcpyHostToDevice));

    cudaError_t err = launchBruteForceKnn<Scalar>(
        d_queries.get(), d_data.get(), M, N, K_use, d_indices.get(), d_dists.get(), false, false, false, 0);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GPU KNN failed: ") + cudaGetErrorString(err));
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    out_indices.resize(result_count);
    out_dists.resize(result_count);

    PLAPOINT_CHECK_CUDA(cudaMemcpy(out_indices.data(), d_indices.get(), index_bytes, cudaMemcpyDeviceToHost));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(out_dists.data(), d_dists.get(), dist_bytes, cudaMemcpyDeviceToHost));
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

template <typename Queries, typename Data, typename Indices, typename Distances>
void validateMatrixKnnInputs(
    const Queries& queries,
    const Data& data,
    int K,
    const Indices& out_indices,
    const Distances& out_dists)
{
    if (queries.cols() != 3 || data.cols() != 3)
    {
        throw std::invalid_argument("GPU KNN PlaMatrix inputs must be Mx3 and Nx3");
    }
    if (K <= 0 || K > 32)
    {
        throw std::invalid_argument("GPU KNN supports K in [1, 32]");
    }
    if (data.rows() <= 0 || K > data.rows())
    {
        throw std::invalid_argument("GPU KNN requires K <= N");
    }
    if (out_indices.rows() != queries.rows() || out_indices.cols() != K ||
        out_dists.rows() != queries.rows() || out_dists.cols() != K)
    {
        throw std::invalid_argument("GPU KNN PlaMatrix outputs must be MxK");
    }
    if (queries.rows() > std::numeric_limits<int>::max() ||
        data.rows() > std::numeric_limits<int>::max())
    {
        throw std::overflow_error("GPU KNN PlaMatrix inputs exceed int range");
    }
}

template <typename Scalar>
void batchKnnMatrixImpl(
    const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& queries,
    const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& data,
    int K,
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>& out_indices,
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& out_dists)
{
    validateMatrixKnnInputs(queries, data, K, out_indices, out_dists);
    auto context = plamatrix::internal::ExecutionContext::create({plamatrix::internal::Backend::Cuda, 0});
    auto d_queries = plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(queries, context);
    auto d_data = plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(data, context);
    plamatrix::internal::ResidentMatrix<int> d_indices(queries.rows(), K, context);
    plamatrix::internal::ResidentMatrix<Scalar> d_dists(queries.rows(), K, context);
    batchKnnDevice(d_queries, d_data, K, d_indices, d_dists);
    out_indices = d_indices.toHostMatrix();
    out_dists = d_dists.toHostMatrix();
}

void batchKnn(
    const plamatrix::MatrixXf& queries,
    const plamatrix::MatrixXf& data,
    int K,
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>& out_indices,
    plamatrix::MatrixXf& out_dists)
{
    batchKnnMatrixImpl<float>(queries, data, K, out_indices, out_dists);
}

void batchKnn(
    const plamatrix::MatrixXd& queries,
    const plamatrix::MatrixXd& data,
    int K,
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>& out_indices,
    plamatrix::MatrixXd& out_dists)
{
    batchKnnMatrixImpl<double>(queries, data, K, out_indices, out_dists);
}

// Device-pointer version: no host roundtrip
template <typename Scalar>
void batchKnnDeviceImpl(const Scalar* d_queries, int M,
                        const Scalar* d_data, int N, int K,
                        int* d_indices, Scalar* d_dists,
                        bool queries_column_major,
                        bool data_column_major,
                        bool output_column_major,
                        cudaStream_t stream,
                        bool synchronize)
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
    if (!d_queries || !d_data || !d_indices || !d_dists)
    {
        throw std::invalid_argument("GPU KNN device pointers must not be null");
    }

    const int K_use = K;
    cudaError_t err = launchBruteForceKnn<Scalar>(
        d_queries, d_data, M, N, K_use, d_indices, d_dists,
        queries_column_major, data_column_major, output_column_major, stream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GPU KNN failed: ") + cudaGetErrorString(err));
    }
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    if (synchronize)
    {
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
}

void batchKnnDevice(const float* d_queries, int M,
                    const float* d_data, int N, int K,
                    int* d_indices, float* d_dists)
{
    batchKnnDeviceImpl<float>(d_queries, M, d_data, N, K, d_indices, d_dists, false, false, false, 0, true);
}

void batchKnnDevice(const double* d_queries, int M,
                    const double* d_data, int N, int K,
                    int* d_indices, double* d_dists)
{
    batchKnnDeviceImpl<double>(d_queries, M, d_data, N, K, d_indices, d_dists, false, false, false, 0, true);
}

void batchKnnDeviceAsync(const float* d_queries, int M,
                         const float* d_data, int N, int K,
                         int* d_indices, float* d_dists,
                         cudaStream_t stream)
{
    batchKnnDeviceImpl<float>(d_queries, M, d_data, N, K, d_indices, d_dists, false, false, false, stream, false);
}

void batchKnnDeviceAsync(const double* d_queries, int M,
                         const double* d_data, int N, int K,
                         int* d_indices, double* d_dists,
                         cudaStream_t stream)
{
    batchKnnDeviceImpl<double>(d_queries, M, d_data, N, K, d_indices, d_dists, false, false, false, stream, false);
}

void batchKnnDeviceColumnMajor(const float* d_queries, int M,
                               const float* d_data, int N, int K,
                               int* d_indices, float* d_dists)
{
    batchKnnDeviceImpl<float>(d_queries, M, d_data, N, K, d_indices, d_dists, false, true, false, 0, true);
}

void batchKnnDeviceColumnMajor(const double* d_queries, int M,
                               const double* d_data, int N, int K,
                               int* d_indices, double* d_dists)
{
    batchKnnDeviceImpl<double>(d_queries, M, d_data, N, K, d_indices, d_dists, false, true, false, 0, true);
}

void batchKnnDeviceColumnMajorAsync(const float* d_queries, int M,
                                    const float* d_data, int N, int K,
                                    int* d_indices, float* d_dists,
                                    cudaStream_t stream)
{
    batchKnnDeviceImpl<float>(d_queries, M, d_data, N, K, d_indices, d_dists, false, true, false, stream, false);
}

void batchKnnDeviceColumnMajorAsync(const double* d_queries, int M,
                                    const double* d_data, int N, int K,
                                    int* d_indices, double* d_dists,
                                    cudaStream_t stream)
{
    batchKnnDeviceImpl<double>(d_queries, M, d_data, N, K, d_indices, d_dists, false, true, false, stream, false);
}

template <typename Scalar>
void batchKnnDeviceMatrixImpl(
    const plamatrix::internal::ResidentMatrix<Scalar>& queries,
    const plamatrix::internal::ResidentMatrix<Scalar>& data,
    int K,
    plamatrix::internal::ResidentMatrix<int>& out_indices,
    plamatrix::internal::ResidentMatrix<Scalar>& out_dists,
    cudaStream_t stream,
    bool synchronize)
{
    validateMatrixKnnInputs(queries, data, K, out_indices, out_dists);
    data.validateContext(queries.context());
    out_indices.validateContext(queries.context());
    out_dists.validateContext(queries.context());
    batchKnnDeviceImpl<Scalar>(
        queries.data(),
        static_cast<int>(queries.rows()),
        data.data(),
        static_cast<int>(data.rows()),
        K,
        out_indices.data(),
        out_dists.data(),
        true,
        true,
        true,
        stream,
        synchronize);
}

void batchKnnDevice(
    const plamatrix::internal::ResidentMatrix<float>& queries,
    const plamatrix::internal::ResidentMatrix<float>& data,
    int K,
    plamatrix::internal::ResidentMatrix<int>& out_indices,
    plamatrix::internal::ResidentMatrix<float>& out_dists)
{
    batchKnnDeviceMatrixImpl<float>(queries, data, K, out_indices, out_dists, 0, true);
}

void batchKnnDevice(
    const plamatrix::internal::ResidentMatrix<double>& queries,
    const plamatrix::internal::ResidentMatrix<double>& data,
    int K,
    plamatrix::internal::ResidentMatrix<int>& out_indices,
    plamatrix::internal::ResidentMatrix<double>& out_dists)
{
    batchKnnDeviceMatrixImpl<double>(queries, data, K, out_indices, out_dists, 0, true);
}

void batchKnnDeviceAsync(
    const plamatrix::internal::ResidentMatrix<float>& queries,
    const plamatrix::internal::ResidentMatrix<float>& data,
    int K,
    plamatrix::internal::ResidentMatrix<int>& out_indices,
    plamatrix::internal::ResidentMatrix<float>& out_dists,
    cudaStream_t stream)
{
    batchKnnDeviceMatrixImpl<float>(queries, data, K, out_indices, out_dists, stream, false);
}

void batchKnnDeviceAsync(
    const plamatrix::internal::ResidentMatrix<double>& queries,
    const plamatrix::internal::ResidentMatrix<double>& data,
    int K,
    plamatrix::internal::ResidentMatrix<int>& out_indices,
    plamatrix::internal::ResidentMatrix<double>& out_dists,
    cudaStream_t stream)
{
    batchKnnDeviceMatrixImpl<double>(queries, data, K, out_indices, out_dists, stream, false);
}

} // namespace gpu
} // namespace plapoint
