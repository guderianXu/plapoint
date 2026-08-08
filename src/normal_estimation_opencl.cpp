#include <plapoint/opencl/normal_estimation.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <plamatrix/ops/decomposition.h>
#include <plamatrix/ops/point_cloud.h>

#include <plapoint/opencl/preprocessing.h>

namespace plapoint
{
namespace opencl
{
namespace
{

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> estimateNormalsImpl(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    int k)
{
    if (k < 3 || k > 32)
    {
        throw std::invalid_argument("OpenCL normal estimation requires 3 <= k <= 32");
    }

    const OpenClKnnResult neighbors = knnSearch(input, k);
    if (std::find(neighbors.finiteQueries.begin(), neighbors.finiteQueries.end(), 0)
        != neighbors.finiteQueries.end())
    {
        throw std::invalid_argument("OpenCL normal estimation requires finite input points");
    }
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(
        static_cast<plamatrix::Index>(input.size()), 3);
    normals.fill(Scalar(0));
    const auto& points = input.points();
    const auto row_count = static_cast<std::int64_t>(input.size());
    int failure_slot_count = 1;
#ifdef _OPENMP
    failure_slot_count = omp_get_max_threads();
    if (failure_slot_count < 1)
    {
        failure_slot_count = 1;
    }
#endif
    std::vector<std::int64_t> failure_rows(
        static_cast<std::size_t>(failure_slot_count), row_count);
    std::vector<std::exception_ptr> failures(
        static_cast<std::size_t>(failure_slot_count));
    std::atomic<std::int64_t> first_failed_row{row_count};
    const auto record_failure = [&](std::int64_t row_value) noexcept
    {
        int worker_index = 0;
#ifdef _OPENMP
        worker_index = omp_get_thread_num();
#endif
        const auto failure_slot = static_cast<std::size_t>(worker_index);
        if (row_value < failure_rows[failure_slot])
        {
            failures[failure_slot] = std::current_exception();
            failure_rows[failure_slot] = row_value;
        }

        std::int64_t previous = first_failed_row.load(std::memory_order_relaxed);
        while (row_value < previous
               && !first_failed_row.compare_exchange_weak(
                   previous,
                   row_value,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed))
        {
        }
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t row_value = 0; row_value < row_count; ++row_value)
    {
        if (row_value > first_failed_row.load(std::memory_order_relaxed))
        {
            continue;
        }
        try
        {
            const auto row = static_cast<std::size_t>(row_value);
            if (!neighbors.finiteQueries[row])
            {
                continue;
            }
            int neighbor_count = 0;
            for (int column = 0; column < neighbors.k; ++column)
            {
                if (neighbors.indices[row * static_cast<std::size_t>(neighbors.k)
                                      + static_cast<std::size_t>(column)] >= 0)
                {
                    ++neighbor_count;
                }
            }
            if (neighbor_count < 3)
            {
                continue;
            }

            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> neighborhood(
                neighbor_count, 3);
            int output_row = 0;
            for (int column = 0; column < neighbors.k; ++column)
            {
                const int point_index = neighbors.indices[
                    row * static_cast<std::size_t>(neighbors.k)
                    + static_cast<std::size_t>(column)];
                if (point_index < 0)
                {
                    continue;
                }
                neighborhood(output_row, 0) = points(point_index, 0);
                neighborhood(output_row, 1) = points(point_index, 1);
                neighborhood(output_row, 2) = points(point_index, 2);
                ++output_row;
            }
            auto covariance = plamatrix::covarianceMatrix(neighborhood);
            auto [left, singular_values, right_transpose] = plamatrix::svd(covariance);
            (void)left;
            (void)singular_values;
            normals(static_cast<plamatrix::Index>(row), 0) = right_transpose.getValue(2, 0);
            normals(static_cast<plamatrix::Index>(row), 1) = right_transpose.getValue(2, 1);
            normals(static_cast<plamatrix::Index>(row), 2) = right_transpose.getValue(2, 2);
        }
        catch (...)
        {
            record_failure(row_value);
        }
    }

    std::int64_t selected_failure_row = row_count;
    std::exception_ptr selected_failure;
    for (std::size_t failure_slot = 0; failure_slot < failures.size(); ++failure_slot)
    {
        if (failure_rows[failure_slot] < selected_failure_row)
        {
            selected_failure_row = failure_rows[failure_slot];
            selected_failure = failures[failure_slot];
        }
    }
    if (selected_failure)
    {
        std::rethrow_exception(selected_failure);
    }
    return normals;
}

} // namespace

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> estimateNormals(
    const PointCloud<float, plamatrix::Device::CPU>& input,
    int k)
{
    return estimateNormalsImpl(input, k);
}

plamatrix::DenseMatrix<double, plamatrix::Device::CPU> estimateNormals(
    const PointCloud<double, plamatrix::Device::CPU>& input,
    int k)
{
    return estimateNormalsImpl(input, k);
}

} // namespace opencl
} // namespace plapoint
