#pragma once

#include <vector>

#include <plamatrix/dense/matrix.h>
#include <plamatrix/sparse/sparse_matrix.h>

namespace plapoint
{
namespace mesh
{

template <typename Scalar>
struct PoissonSystem
{
    plamatrix::SparseMatrix<Scalar, plamatrix::RowMajor, plamatrix::Index> matrix;
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, 1> rhs;
    std::vector<int> leafNodes;
};

} // namespace mesh
} // namespace plapoint
