#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace plapoint {
namespace gpu {

namespace detail {

inline std::size_t checkedSizeProduct(std::size_t lhs, std::size_t rhs, const char* label)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw std::overflow_error(std::string(label) + " size exceeds size_t range");
    }
    return lhs * rhs;
}

template <typename T>
inline std::size_t checkedByteCount(std::size_t count, const char* label)
{
    return checkedSizeProduct(count, sizeof(T), label);
}

} // namespace detail

/// Batch KNN on GPU with host pointers (copies data to/from GPU).
/// If fewer than K finite distances exist for a query, remaining slots use
/// index -1 and std::numeric_limits<Scalar>::max() as sentinels.
/// Treat the index as authoritative: valid neighbors can also report max distance
/// when their squared distance exceeds Scalar's representable range.
void batchKnn(const float* h_queries, int M,
              const float* h_data, int N, int K,
              std::vector<int>& out_indices,
              std::vector<float>& out_dists);

void batchKnn(const double* h_queries, int M,
              const double* h_data, int N, int K,
              std::vector<int>& out_indices,
              std::vector<double>& out_dists);

/// Batch KNN on GPU with device pointers (no host roundtrip).
/// Results are written to d_out_indices and d_out_dists (pre-allocated on device).
/// Missing finite neighbors are represented as index -1 and max distance.
/// Treat the index as authoritative; a valid index may have a clamped max distance.
void batchKnnDevice(const float* d_queries, int M,
                    const float* d_data, int N, int K,
                    int* d_out_indices, float* d_out_dists);

void batchKnnDevice(const double* d_queries, int M,
                    const double* d_data, int N, int K,
                    int* d_out_indices, double* d_out_dists);

#ifdef PLAPOINT_WITH_CUDA
/// Async device-pointer batch KNN on the caller-provided CUDA stream.
/// The caller owns stream synchronization before reading output buffers.
void batchKnnDeviceAsync(const float* d_queries, int M,
                         const float* d_data, int N, int K,
                         int* d_out_indices, float* d_out_dists,
                         cudaStream_t stream);

void batchKnnDeviceAsync(const double* d_queries, int M,
                         const double* d_data, int N, int K,
                         int* d_out_indices, double* d_out_dists,
                         cudaStream_t stream);
#endif

/// Batch KNN on GPU with row-major queries and column-major Nx3 point data.
/// This matches PlaMatrix DenseMatrix device storage: [x0..xN-1, y0..yN-1, z0..zN-1].
/// Missing finite neighbors are represented as index -1 and max distance.
/// Treat the index as authoritative; a valid index may have a clamped max distance.
void batchKnnDeviceColumnMajor(const float* d_queries, int M,
                               const float* d_data, int N, int K,
                               int* d_out_indices, float* d_out_dists);

void batchKnnDeviceColumnMajor(const double* d_queries, int M,
                               const double* d_data, int N, int K,
                               int* d_out_indices, double* d_out_dists);

#ifdef PLAPOINT_WITH_CUDA
/// Async column-major device-pointer batch KNN on the caller-provided CUDA stream.
/// The caller owns stream synchronization before reading output buffers.
void batchKnnDeviceColumnMajorAsync(const float* d_queries, int M,
                                    const float* d_data, int N, int K,
                                    int* d_out_indices, float* d_out_dists,
                                    cudaStream_t stream);

void batchKnnDeviceColumnMajorAsync(const double* d_queries, int M,
                                    const double* d_data, int N, int K,
                                    int* d_out_indices, double* d_out_dists,
                                    cudaStream_t stream);
#endif

} // namespace gpu
} // namespace plapoint
