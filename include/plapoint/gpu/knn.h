#pragma once

#include <cstdint>
#include <vector>

namespace plapoint {
namespace gpu {

/// Batch K-nearest neighbor search on GPU (brute-force).
/// Copies data to GPU, runs kernel, copies results back.
/// @tparam Scalar  float or double
/// @param h_queries  M x 3 query points (host)
/// @param M          number of query points
/// @param h_data     N x 3 data points (host)
/// @param N          number of data points
/// @param K          number of neighbors per query
/// @param out_indices  output: M x K neighbor indices (sorted by distance)
/// @param out_dists    output: M x K squared distances (optional, pass empty to skip)
void batchKnn(const float* h_queries, int M,
              const float* h_data, int N, int K,
              std::vector<int>& out_indices,
              std::vector<float>& out_dists);

void batchKnn(const double* h_queries, int M,
              const double* h_data, int N, int K,
              std::vector<int>& out_indices,
              std::vector<double>& out_dists);

} // namespace gpu
} // namespace plapoint
