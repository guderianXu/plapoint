#pragma once

#include <cstdint>
#include <vector>

namespace plapoint {
namespace gpu {

/// Batch KNN on GPU with host pointers (copies data to/from GPU).
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
void batchKnnDevice(const float* d_queries, int M,
                    const float* d_data, int N, int K,
                    int* d_out_indices, float* d_out_dists);

void batchKnnDevice(const double* d_queries, int M,
                    const double* d_data, int N, int K,
                    int* d_out_indices, double* d_out_dists);

} // namespace gpu
} // namespace plapoint
