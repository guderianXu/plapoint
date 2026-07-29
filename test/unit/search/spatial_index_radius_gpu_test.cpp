#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <plamatrix/plamatrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/spatial_index.h>

namespace
{

template <typename Scalar>
using CpuMatrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

template <typename Scalar>
plapoint::PointCloud<Scalar, plamatrix::Device::GPU> makeCloud(
    const std::vector<Scalar>& row_major_points)
{
    plapoint::PointCloud<Scalar, plamatrix::Device::CPU> cloud(row_major_points.size() / 3);
    auto& points = cloud.points();
    for (std::size_t row = 0; row < row_major_points.size() / 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            points(static_cast<plamatrix::Index>(row), column) =
                row_major_points[row * 3 + static_cast<std::size_t>(column)];
        }
    }
    return cloud.toGpu();
}

template <typename Scalar>
CpuMatrix<Scalar> makeQueries(int count)
{
    CpuMatrix<Scalar> queries(count, 3);
    for (int row = 0; row < count; ++row)
    {
        queries(row, 0) = static_cast<Scalar>((row % 17) * 0.2 - 1.5);
        queries(row, 1) = static_cast<Scalar>((row % 11) * 0.15 - 0.75);
        queries(row, 2) = static_cast<Scalar>((row % 7) * 0.1 - 0.3);
    }
    return queries;
}

template <typename Scalar>
std::vector<plamatrix::Index> referenceCounts(
    const CpuMatrix<Scalar>& points,
    const CpuMatrix<Scalar>& queries,
    Scalar radius,
    int max_count)
{
    std::vector<plamatrix::Index> counts(static_cast<std::size_t>(queries.rows()), 0);
    const long double radius_value = static_cast<long double>(radius);
    for (plamatrix::Index query = 0; query < queries.rows(); ++query)
    {
        for (plamatrix::Index point = 0; point < points.rows(); ++point)
        {
            const long double dx = static_cast<long double>(queries(query, 0))
                - static_cast<long double>(points(point, 0));
            const long double dy = static_cast<long double>(queries(query, 1))
                - static_cast<long double>(points(point, 1));
            const long double dz = static_cast<long double>(queries(query, 2))
                - static_cast<long double>(points(point, 2));
            if (std::isfinite(dx) && std::isfinite(dy) && std::isfinite(dz)
                && std::hypot(std::hypot(dx, dy), dz) <= radius_value)
            {
                ++counts[static_cast<std::size_t>(query)];
                if (counts[static_cast<std::size_t>(query)] == max_count)
                {
                    break;
                }
            }
        }
    }
    return counts;
}

template <typename Scalar>
class SpatialIndexRadiusGpuTest : public ::testing::Test
{
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SpatialIndexRadiusGpuTest, ScalarTypes);

TYPED_TEST(SpatialIndexRadiusGpuTest, RadiusCountsMatchBruteForceAndSaturate)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    std::vector<Scalar> point_values;
    point_values.reserve(240 * 3);
    std::mt19937 generator(7);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    for (int point = 0; point < 240; ++point)
    {
        point_values.push_back(static_cast<Scalar>(distribution(generator)));
        point_values.push_back(static_cast<Scalar>(distribution(generator)));
        point_values.push_back(static_cast<Scalar>(distribution(generator)));
    }

    auto cloud = makeCloud(point_values);
    const auto cloud_cpu = cloud.toCpu();
    const auto& points_cpu = cloud_cpu.pointsCpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(cloud, Scalar(0.5));
    plapoint::gpu::GpuSpatialQueryWorkspace<Scalar> workspace;

    for (int query_count : {1, 31, 32, 33, 1000})
    {
        auto queries_cpu = makeQueries<Scalar>(query_count);
        auto queries_gpu = queries_cpu.toGpu();
        auto counts_gpu = index.radiusCountAsync(
            queries_gpu, Scalar(0.8), 7, workspace, nullptr);
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
        const auto counts_cpu = counts_gpu.toCpu();
        const auto expected = referenceCounts(points_cpu, queries_cpu, Scalar(0.8), 7);
        for (int query = 0; query < query_count; ++query)
        {
            EXPECT_EQ(counts_cpu(query, 0), expected[static_cast<std::size_t>(query)]);
        }
    }
}

TYPED_TEST(SpatialIndexRadiusGpuTest, RadiusSearchUsesDistanceThenSourceIndexOrder)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    auto cloud = makeCloud<Scalar>({
        Scalar(0), Scalar(0), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(-1), Scalar(0), Scalar(0),
        Scalar(0), Scalar(1), Scalar(0),
        Scalar(0), Scalar(-1), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0),
        nan, Scalar(0), Scalar(0)
    });
    CpuMatrix<Scalar> queries_cpu(3, 3);
    queries_cpu.fill(Scalar(0));
    queries_cpu(1, 0) = Scalar(10);
    queries_cpu(2, 0) = nan;
    auto queries_gpu = queries_cpu.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(cloud, Scalar(1));
    plapoint::gpu::GpuSpatialQueryWorkspace<Scalar> workspace;

    auto result = index.radiusSearchAsync(
        queries_gpu, Scalar(1), 4, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    const auto indices = result.indices.toCpu();
    const auto distances = result.squaredDistances.toCpu();
    const auto counts = result.counts.toCpu();

    EXPECT_EQ(counts(0, 0), 4);
    EXPECT_EQ(indices(0, 0), 0);
    EXPECT_EQ(indices(0, 1), 1);
    EXPECT_EQ(indices(0, 2), 2);
    EXPECT_EQ(indices(0, 3), 3);
    EXPECT_EQ(distances(0, 0), Scalar(0));
    EXPECT_EQ(distances(0, 1), Scalar(1));
    EXPECT_EQ(distances(0, 2), Scalar(1));
    EXPECT_EQ(distances(0, 3), Scalar(1));
    EXPECT_EQ(counts(1, 0), 0);
    EXPECT_EQ(counts(2, 0), 0);
    for (int neighbor = 0; neighbor < 4; ++neighbor)
    {
        EXPECT_EQ(indices(1, neighbor), -1);
        EXPECT_EQ(indices(2, neighbor), -1);
    }
}

TYPED_TEST(SpatialIndexRadiusGpuTest, RadiusSearchRunsOnNonDefaultStream)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cloud = makeCloud<Scalar>({
        Scalar(0), Scalar(0), Scalar(0),
        Scalar(0), Scalar(0), Scalar(0),
        Scalar(2), Scalar(0), Scalar(0)
    });
    CpuMatrix<Scalar> queries_cpu(1, 3);
    queries_cpu.fill(Scalar(0));
    auto queries_gpu = queries_cpu.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(cloud, Scalar(1));
    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CpuMatrix<plamatrix::Index> indices(1, 3);
    CpuMatrix<plamatrix::Index> counts(1, 1);
    plamatrix::Index saturated_count = -1;
    {
        plapoint::gpu::GpuSpatialQueryWorkspace<Scalar> workspace;
        auto count_result = index.radiusCountAsync(
            queries_gpu, Scalar(0), 3, workspace, stream);
        auto result = index.radiusSearchAsync(
            queries_gpu, Scalar(0), 3, workspace, stream);
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
        saturated_count = count_result.toCpu()(0, 0);
        indices = result.indices.toCpu();
        counts = result.counts.toCpu();
    }
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));

    EXPECT_EQ(counts(0, 0), 2);
    EXPECT_EQ(saturated_count, 2);
    EXPECT_EQ(indices(0, 0), 0);
    EXPECT_EQ(indices(0, 1), 1);
}

TYPED_TEST(SpatialIndexRadiusGpuTest, EqualFloatRoundedDistancesUseSourceIndexTieBreak)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cloud = makeCloud<Scalar>({
        Scalar(0.3), Scalar(0), Scalar(0),
        Scalar(-0.3), Scalar(0), Scalar(0)
    });
    CpuMatrix<Scalar> queries_cpu(1, 3);
    queries_cpu.fill(Scalar(0));
    auto queries_gpu = queries_cpu.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(cloud, Scalar(0.25));
    plapoint::gpu::GpuSpatialQueryWorkspace<Scalar> workspace;

    auto result = index.radiusSearchAsync(
        queries_gpu, Scalar(1), 2, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    const auto indices = result.indices.toCpu();

    EXPECT_EQ(indices(0, 0), 0);
    EXPECT_EQ(indices(0, 1), 1);
}

TEST(SpatialIndexRadiusGpuTest, HugeDoubleRadiusRejectsPointOutsideEuclideanBall)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    const double maximum = std::numeric_limits<double>::max();
    auto cloud = makeCloud<double>({0.8 * maximum, 0.8 * maximum, 0.0});
    CpuMatrix<double> queries_cpu(1, 3);
    queries_cpu.fill(0.0);
    auto queries_gpu = queries_cpu.toGpu();
    plapoint::gpu::GpuSpatialIndex<double> index;
    index.build(cloud, maximum);
    plapoint::gpu::GpuSpatialQueryWorkspace<double> workspace;

    auto counts = index.radiusCountAsync(
        queries_gpu, maximum, 1, workspace, nullptr).toCpu();
    auto result = index.radiusSearchAsync(
        queries_gpu, maximum, 1, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));

    EXPECT_EQ(counts(0, 0), 0);
    EXPECT_EQ(result.counts.toCpu()(0, 0), 0);
    EXPECT_EQ(result.indices.toCpu()(0, 0), -1);
}

TYPED_TEST(SpatialIndexRadiusGpuTest, IndexOwnsPointSnapshotAfterCloudDestruction)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    {
        auto cloud = makeCloud<Scalar>({
            Scalar(0), Scalar(0), Scalar(0),
            Scalar(1), Scalar(0), Scalar(0)});
        index.build(cloud, Scalar(1));
    }
    CpuMatrix<Scalar> queries_cpu(1, 3);
    queries_cpu.fill(Scalar(0));
    auto queries_gpu = queries_cpu.toGpu();
    plapoint::gpu::GpuSpatialQueryWorkspace<Scalar> workspace;

    auto result = index.radiusSearchAsync(
        queries_gpu, Scalar(0), 1, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));

    EXPECT_EQ(result.counts.toCpu()(0, 0), 1);
    EXPECT_EQ(result.indices.toCpu()(0, 0), 0);
}

} // namespace

#endif
