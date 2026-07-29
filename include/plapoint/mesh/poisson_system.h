#pragma once

#include <vector>

#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/sparse/csr_matrix.h>

namespace plapoint
{
namespace mesh
{

template <typename Scalar>
struct PoissonSystem
{
    plamatrix::CSRMatrix<Scalar, plamatrix::Device::CPU> matrix;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> rhs;
    std::vector<int> leafNodes;
};

} // namespace mesh
} // namespace plapoint
