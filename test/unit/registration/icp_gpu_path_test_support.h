#pragma once

// Shared data builders and test-only access needed by the themed CUDA ICP test translation units.
#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <plamatrix/plamatrix.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/icp_testing.h>

#define private public
#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/icp.h>
#include <plapoint/registration/icp.h>
#undef private

namespace
{

// Mirrors the production GPU ICP spatial-grid target-count threshold.
constexpr int kMinTargetSpatialGridRowsForTesting = 128;

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeNonCollinearPoints()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 0.0f); points.setValue(2, 1, 1.0f); points.setValue(2, 2, 0.0f);
    points.setValue(3, 0, 0.0f); points.setValue(3, 1, 0.0f); points.setValue(3, 2, 1.0f);
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslationTransform(
    float tx,
    float ty,
    float tz)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> transform(4, 4);
    transform.fill(0.0f);
    transform.setValue(0, 0, 1.0f);
    transform.setValue(1, 1, 1.0f);
    transform.setValue(2, 2, 1.0f);
    transform.setValue(3, 3, 1.0f);
    transform.setValue(0, 3, tx);
    transform.setValue(1, 3, ty);
    transform.setValue(2, 3, tz);
    return transform;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedNonCollinearPoints(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& source,
    float tx,
    float ty,
    float tz)
{
    auto transform = makeTranslationTransform(tx, ty, tz);
    return plamatrix::transformPoints(transform, source);
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeCompactNonCollinearGridPoints(int count)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 8;
        const int y = (i / 8) % 8;
        const int z = i / 64;
        points.setValue(i, 0, static_cast<float>(x) * 0.25f);
        points.setValue(i, 1, static_cast<float>(y) * 0.25f);
        points.setValue(i, 2, static_cast<float>(z) * 0.25f);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> padTargetWithNonFiniteRows(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& points,
    int min_rows = kMinTargetSpatialGridRowsForTesting)
{
    const int rows = static_cast<int>(points.rows());
    const int cols = static_cast<int>(points.cols());
    const int padded_rows = std::max(rows, min_rows);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> padded(padded_rows, cols);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int row = 0; row < padded_rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            padded.setValue(row, col, row < rows ? points.getValue(row, col) : nan);
        }
    }
    return padded;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeGridPoints(int count)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 257;
        const int y = (i / 257) % 251;
        const int z = (i / (257 * 251)) % 241;
        points.setValue(i, 0, static_cast<float>(x) * 0.01f);
        points.setValue(i, 1, static_cast<float>(y) * 0.01f);
        points.setValue(i, 2, static_cast<float>(z) * 0.01f);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedGridPoints(
    int count,
    float tx,
    float ty,
    float tz)
{
    auto points = makeGridPoints(count);
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, points.getValue(i, 0) + tx);
        points.setValue(i, 1, points.getValue(i, 1) + ty);
        points.setValue(i, 2, points.getValue(i, 2) + tz);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeBinaryGridPoints(int count)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 257;
        const int y = (i / 257) % 251;
        const int z = (i / (257 * 251)) % 241;
        points.setValue(i, 0, static_cast<float>(x) * 0.125f);
        points.setValue(i, 1, static_cast<float>(y) * 0.125f);
        points.setValue(i, 2, static_cast<float>(z) * 0.125f);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedBinaryGridPoints(
    int count,
    float tx,
    float ty,
    float tz)
{
    auto points = makeBinaryGridPoints(count);
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, points.getValue(i, 0) + tx);
        points.setValue(i, 1, points.getValue(i, 1) + ty);
        points.setValue(i, 2, points.getValue(i, 2) + tz);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedPerturbedGridPoints(
    int count,
    float tx,
    float ty,
    float tz)
{
    auto points = makeTranslatedGridPoints(count, tx, ty, tz);
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, points.getValue(i, 0) + static_cast<float>((i % 7) - 3) * 0.0002f);
        points.setValue(i, 1, points.getValue(i, 1) + static_cast<float>(((i / 7) % 5) - 2) * 0.00015f);
        points.setValue(i, 2, points.getValue(i, 2) + static_cast<float>(((i / 35) % 3) - 1) * 0.0001f);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeCollinearPoints()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 2.0f); points.setValue(2, 1, 0.0f); points.setValue(2, 2, 0.0f);
    points.setValue(3, 0, 3.0f); points.setValue(3, 1, 0.0f); points.setValue(3, 2, 0.0f);
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> multiplyCpu4x4(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& A,
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& B)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> C(4, 4);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                sum += static_cast<double>(A.getValue(row, k)) * static_cast<double>(B.getValue(k, col));
            }
            C.setValue(row, col, static_cast<float>(sum));
        }
    }
    return C;
}

plapoint::gpu::IcpCorrespondenceStats<float> makeMatchedStats(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& source,
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& target)
{
    plapoint::gpu::IcpCorrespondenceStats<float> stats;
    stats.active_count = static_cast<int>(source.rows());

    double src_sum[3]{};
    double tgt_sum[3]{};
    double cross_sum[9]{};
    double src_outer_sum[9]{};
    double tgt_outer_sum[9]{};

    for (plamatrix::Index row = 0; row < source.rows(); ++row)
    {
        const double src_values[3]{
            static_cast<double>(source.getValue(row, 0)),
            static_cast<double>(source.getValue(row, 1)),
            static_cast<double>(source.getValue(row, 2))};
        const double tgt_values[3]{
            static_cast<double>(target.getValue(row, 0)),
            static_cast<double>(target.getValue(row, 1)),
            static_cast<double>(target.getValue(row, 2))};

        for (int r = 0; r < 3; ++r)
        {
            src_sum[r] += src_values[r];
            tgt_sum[r] += tgt_values[r];
            const double residual = src_values[r] - tgt_values[r];
            stats.residual_sq_sum += residual * residual;
            for (int c = 0; c < 3; ++c)
            {
                cross_sum[r * 3 + c] += src_values[r] * tgt_values[c];
                src_outer_sum[r * 3 + c] += src_values[r] * src_values[c];
                tgt_outer_sum[r * 3 + c] += tgt_values[r] * tgt_values[c];
            }
        }
    }

    const double inv_count = 1.0 / static_cast<double>(stats.active_count);
    for (int c = 0; c < 3; ++c)
    {
        stats.src_centroid[c] = src_sum[c] * inv_count;
        stats.tgt_centroid[c] = tgt_sum[c] * inv_count;
    }
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            const int idx = r * 3 + c;
            stats.cross_covariance[idx] = cross_sum[idx] - src_sum[r] * tgt_sum[c] * inv_count;
            stats.src_covariance[idx] = src_outer_sum[idx] - src_sum[r] * src_sum[c] * inv_count;
            stats.tgt_covariance[idx] = tgt_outer_sum[idx] - tgt_sum[r] * tgt_sum[c] * inv_count;
        }
    }
    return stats;
}

} // namespace


#endif // PLAPOINT_WITH_CUDA
