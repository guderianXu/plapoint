#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <plamatrix/plamatrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/spatial_index.h>
#include <plapoint/search/kdtree.h>

namespace
{

template <typename Scalar>
using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

template <typename T>
std::vector<T> download(const T* device_data, std::size_t count)
{
    std::vector<T> result(count);
    if (count != 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpy(
            result.data(), device_data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return result;
}

template <typename Scalar>
plapoint::PointCloud<Scalar, plamatrix::Device::GPU> makeCloud(
    const std::vector<Scalar>& row_major_points)
{
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    const auto point_count = row_major_points.size() / 3;
    CpuCloud cpu_cloud(point_count);
    auto& points = cpu_cloud.points();
    for (std::size_t row = 0; row < point_count; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            points(static_cast<plamatrix::Index>(row), column) =
                row_major_points[row * 3 + static_cast<std::size_t>(column)];
        }
    }
    return cpu_cloud.toGpu();
}

template <typename Scalar>
class SpatialIndexGpuTest : public ::testing::Test
{
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SpatialIndexGpuTest, ScalarTypes);

TYPED_TEST(SpatialIndexGpuTest, BuildsEmptyCloudAndRejectsInvalidCellSizesTransactionally)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cloud = makeCloud<Scalar>({});
    plapoint::gpu::GpuSpatialIndex<Scalar> index;

    EXPECT_FALSE(index.matches(cloud, Scalar(1)));
    index.build(cloud, Scalar(1));
    EXPECT_TRUE(index.matches(cloud, Scalar(1)));
    EXPECT_EQ(index.finitePointCount(), 0);
    EXPECT_EQ(index.cellCount(), 0);
    EXPECT_EQ(index.cellSize(), Scalar(1));

    for (Scalar invalid : {
             Scalar(0), Scalar(-1), std::numeric_limits<Scalar>::infinity(),
             std::numeric_limits<Scalar>::quiet_NaN()})
    {
        EXPECT_THROW(index.build(cloud, invalid), std::invalid_argument);
        EXPECT_TRUE(index.matches(cloud, Scalar(1)));
    }
}

TYPED_TEST(SpatialIndexGpuTest, CompactsFinitePointsAndBuildsStableCellRanges)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using Index = plamatrix::Index;
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    auto cloud = makeCloud<Scalar>({
        Scalar(-1), Scalar(-1), Scalar(-1),
        Scalar(0), Scalar(0), Scalar(0),
        Scalar(0.999), Scalar(0), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(-1), Scalar(-1), Scalar(-1),
        infinity, Scalar(0), Scalar(0)
    });
    plapoint::gpu::GpuSpatialIndex<Scalar> index;

    index.build(cloud, Scalar(1));

    EXPECT_EQ(index.finitePointCount(), 5);
    EXPECT_EQ(index.cellCount(), 3);
    EXPECT_EQ(download(index.sortedPointIndicesData(), 5),
              (std::vector<Index>{0, 4, 1, 2, 3}));
    EXPECT_EQ(download(index.cellOffsetsData(), 3),
              (std::vector<Index>{0, 2, 4}));
    EXPECT_EQ(download(index.cellCountsData(), 3),
              (std::vector<Index>{2, 2, 1}));

    const auto keys = download(index.uniqueCellKeysData(), 3);
    ASSERT_EQ(keys.size(), 3U);
    EXPECT_LT(keys[0], keys[1]);
    EXPECT_LT(keys[1], keys[2]);
}

TYPED_TEST(SpatialIndexGpuTest, RebuildIsDeterministicAndRevisionInvalidatesCache)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cloud = makeCloud<Scalar>({
        Scalar(-2), Scalar(0), Scalar(0),
        Scalar(3), Scalar(1), Scalar(0),
        Scalar(3), Scalar(1), Scalar(0),
        Scalar(0), Scalar(-1), Scalar(2)
    });
    plapoint::gpu::GpuSpatialIndex<Scalar> index;

    index.build(cloud, Scalar(0.5));
    const auto first_indices = download(
        index.sortedPointIndicesData(), static_cast<std::size_t>(index.finitePointCount()));
    const auto first_keys = download(
        index.uniqueCellKeysData(), static_cast<std::size_t>(index.cellCount()));
    const auto revision = cloud.pointsRevision();

    index.build(cloud, Scalar(0.5));
    EXPECT_EQ(download(index.sortedPointIndicesData(), first_indices.size()), first_indices);
    EXPECT_EQ(download(index.uniqueCellKeysData(), first_keys.size()), first_keys);
    EXPECT_TRUE(index.matches(cloud, Scalar(0.5)));
    EXPECT_EQ(cloud.pointsRevision(), revision);

    cloud.points().setValue(0, 0, Scalar(-3));
    EXPECT_FALSE(index.matches(cloud, Scalar(0.5)));
}

TYPED_TEST(SpatialIndexGpuTest, RejectsPositiveInt64CellBoundary)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    const Scalar outside_cell_range = std::ldexp(Scalar(1), 63);
    auto cloud = makeCloud<Scalar>({
        outside_cell_range, Scalar(0), Scalar(0)
    });
    plapoint::gpu::GpuSpatialIndex<Scalar> index;

    EXPECT_THROW(index.build(cloud, Scalar(1)), std::overflow_error);
    EXPECT_FALSE(index.matches(cloud, Scalar(1)));
}

TYPED_TEST(SpatialIndexGpuTest, MutableGpuPointAliasConservativelyInvalidatesMatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cloud = makeCloud<Scalar>({
        Scalar(0), Scalar(0), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0)});
    auto& retained_alias = cloud.points();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(cloud, Scalar(1));

    retained_alias.setValue(0, 0, Scalar(2));

    EXPECT_FALSE(index.matches(cloud, Scalar(1)));
}

TYPED_TEST(SpatialIndexGpuTest, AdaptiveKnnMatchesCpuForEverySupportedK)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    constexpr int point_count = 1024;
    constexpr int query_count = 8;
    auto cpu_cloud = std::make_shared<CpuCloud>(point_count);
    for (int i = 0; i < point_count; ++i)
    {
        cpu_cloud->points().setValue(i, 0, Scalar(i % 32) * Scalar(1.07));
        cpu_cloud->points().setValue(i, 1, Scalar((i / 32) % 16) * Scalar(0.91));
        cpu_cloud->points().setValue(i, 2, Scalar(i / 512) * Scalar(1.31));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::CPU> cpu_tree;
    cpu_tree.setInputCloud(cpu_cloud);
    cpu_tree.build();
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> gpu_tree;
    gpu_tree.setInputCloud(gpu_cloud);
    gpu_tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(query_count, 3);
    for (int i = 0; i < query_count; ++i)
    {
        queries.setValue(i, 0, Scalar(i * 3) + Scalar(0.173));
        queries.setValue(i, 1, Scalar(i) + Scalar(0.287));
        queries.setValue(i, 2, Scalar(0.419));
    }

    for (int k = 1; k <= 32; ++k)
    {
        EXPECT_EQ(gpu_tree.batchNearestKSearch(queries, k),
                  cpu_tree.batchNearestKSearch(queries, k)) << "k=" << k;
        EXPECT_EQ(gpu_tree.lastNeighborBackend(),
                  plapoint::gpu::GpuNeighborBackend::UniformGrid);
    }
}

TYPED_TEST(SpatialIndexGpuTest, AdaptiveKnnOrdersCellFaceAndEqualDistanceTiesBySourceIndex)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    auto cpu_cloud = std::make_shared<CpuCloud>(512);
    for (int i = 0; i < 512; ++i)
    {
        const Scalar coordinate = Scalar(i / 2 + 1);
        cpu_cloud->points().setValue(i, 0, (i % 2 == 0) ? -coordinate : coordinate);
        cpu_cloud->points().setValue(i, 1, Scalar(0));
        cpu_cloud->points().setValue(i, 2, Scalar(0));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(8, 3);
    for (int i = 0; i < 8; ++i)
    {
        queries.setValue(i, 0, Scalar(0));
        queries.setValue(i, 1, Scalar(0));
        queries.setValue(i, 2, Scalar(0));
    }
    const auto result = tree.batchNearestKSearch(queries, 4);

    EXPECT_EQ(result.front(), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::UniformGrid);
}

TYPED_TEST(SpatialIndexGpuTest, AdaptiveKnnFallsBackForSmallWorkAndPathologicalOccupancy)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    auto cpu_cloud = std::make_shared<CpuCloud>(4096);
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> one_query(1, 3);
    for (int column = 0; column < 3; ++column)
    {
        one_query.setValue(0, column, Scalar(0));
    }
    tree.batchNearestKSearch(one_query, 1);
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::BruteForce);

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> many_queries(8, 3);
    for (int row = 0; row < 8; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            many_queries.setValue(row, column, Scalar(0));
        }
    }
    tree.batchNearestKSearch(many_queries, 1);
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::BruteForce);
}

TYPED_TEST(SpatialIndexGpuTest, AdaptiveKnnFallsBackWhenQueriesNeedWideEmptyShells)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    auto cpu_cloud = std::make_shared<CpuCloud>(1024);
    for (int row = 0; row < 1024; ++row)
    {
        cpu_cloud->points().setValue(row, 0, Scalar(row % 32));
        cpu_cloud->points().setValue(row, 1, Scalar(row / 32));
        cpu_cloud->points().setValue(row, 2, Scalar(0));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(8, 3);
    for (int row = 0; row < 8; ++row)
    {
        queries.setValue(row, 0, Scalar(15));
        queries.setValue(row, 1, Scalar(15));
        queries.setValue(row, 2, Scalar(31));
    }

    const auto result = tree.batchNearestKSearch(queries, 8);

    EXPECT_EQ(result.size(), 8u);
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::BruteForce);
}

TEST(SpatialIndexGpuTest, AdaptiveKnnFallsBackForWideInteriorEmptyShells)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    constexpr int point_count = 8192;
    auto cpu_cloud = std::make_shared<CpuCloud>(point_count);
    for (int row = 0; row < point_count; ++row)
    {
        const bool positive = (row % 2) != 0;
        const int local = row / 2;
        const Scalar edge_offset = Scalar(local % 11);
        cpu_cloud->points().setValue(
            row, 0, positive ? Scalar(90) + edge_offset : Scalar(-100) + edge_offset);
        cpu_cloud->points().setValue(row, 1, Scalar((local / 11) % 64));
        cpu_cloud->points().setValue(row, 2, Scalar((local / (11 * 8)) % 64));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> query(1, 3);
    query.setValue(0, 0, Scalar(0));
    query.setValue(0, 1, Scalar(31));
    query.setValue(0, 2, Scalar(2));

    const auto result = tree.batchNearestKSearch(query, 8);

    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::BruteForce);
}

TYPED_TEST(SpatialIndexGpuTest, AdaptiveKnnRebuildsAfterMutablePointAccess)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    auto cpu_cloud = std::make_shared<CpuCloud>(1024);
    for (int i = 0; i < 1024; ++i)
    {
        cpu_cloud->points().setValue(i, 0, Scalar(i));
        cpu_cloud->points().setValue(i, 1, Scalar(i % 7));
        cpu_cloud->points().setValue(i, 2, Scalar(i % 11));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(8, 3);
    for (int row = 0; row < 8; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            queries.setValue(row, column, Scalar(0));
        }
    }
    EXPECT_EQ(tree.batchNearestKSearch(queries, 1).front().front(), 0);
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::UniformGrid);

    gpu_cloud->points().setValue(0, 0, Scalar(500));
    EXPECT_NE(tree.batchNearestKSearch(queries, 1).front().front(), 0);
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::UniformGrid);
}

TEST(SpatialIndexGpuTest, IndexedKnnKeepsFiniteNeighborsWithUnrepresentableDistance)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    const double maximum = std::numeric_limits<double>::max();
    auto cloud = makeCloud<double>({
        maximum, maximum, 0.0,
        1.0, 0.0, 0.0});
    plapoint::gpu::GpuSpatialIndex<double> index;
    index.build(cloud, maximum);
    plamatrix::DenseMatrix<double, plamatrix::Device::CPU> query_cpu(1, 3);
    query_cpu.fill(0.0);
    auto query_gpu = query_cpu.toGpu();
    plapoint::gpu::GpuSpatialQueryWorkspace<double> workspace;

    auto result = index.knnSearchAsync(query_gpu, 2, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    const auto indices = result.indices.toCpu();

    EXPECT_EQ(indices(0, 0), 1);
    EXPECT_EQ(indices(0, 1), 0);
}

TYPED_TEST(SpatialIndexGpuTest, BuildsCorrectlyOnNonBlockingStream)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cloud = makeCloud<Scalar>({
        Scalar(-2), Scalar(0), Scalar(0),
        Scalar(-1), Scalar(0), Scalar(0),
        Scalar(0), Scalar(0), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(2), Scalar(0), Scalar(0)});
    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(cloud, Scalar(1), stream);
    EXPECT_EQ(index.finitePointCount(), 5);
    EXPECT_EQ(download(index.sortedPointIndicesData(), 5),
              (std::vector<plamatrix::Index>{0, 1, 2, 3, 4}));

    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
}

TYPED_TEST(SpatialIndexGpuTest, RetainedMutableAliasStaysConservativelyInvalidAfterSetPoints)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cloud = makeCloud<Scalar>({
        Scalar(0), Scalar(0), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0)});
    auto& retained_alias = cloud.points();
    auto replacement_cloud = makeCloud<Scalar>({
        Scalar(2), Scalar(0), Scalar(0),
        Scalar(3), Scalar(0), Scalar(0)});
    cloud.setPoints(std::move(replacement_cloud.points()));

    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(cloud, Scalar(1));
    retained_alias.setValue(0, 0, Scalar(10));

    EXPECT_FALSE(index.matches(cloud, Scalar(1)));
}

TYPED_TEST(SpatialIndexGpuTest, BruteForceFallbackOrdersEqualDistancesBySourceIndex)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    auto cpu_cloud = std::make_shared<CpuCloud>(1024);
    for (int row = 0; row < 1024; ++row)
    {
        cpu_cloud->points().setValue(row, 0, row % 2 == 0 ? Scalar(-1) : Scalar(1));
        cpu_cloud->points().setValue(row, 1, Scalar(0));
        cpu_cloud->points().setValue(row, 2, Scalar(0));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> query(1, 3);
    query.fill(Scalar(0));

    EXPECT_EQ(tree.batchNearestKSearch(query, 8).front(),
              (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::BruteForce);
}

TYPED_TEST(SpatialIndexGpuTest, KGreaterThan32RefreshesCpuFallbackAfterPointReplacement)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    CpuCloud cpu_cloud(64);
    for (int row = 0; row < 64; ++row)
    {
        cpu_cloud.points().setValue(row, 0, Scalar(row));
        cpu_cloud.points().setValue(row, 1, Scalar(0));
        cpu_cloud.points().setValue(row, 2, Scalar(0));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    CpuCloud replacement(64);
    for (int row = 0; row < 64; ++row)
    {
        replacement.points().setValue(row, 0, Scalar(row + 100));
        replacement.points().setValue(row, 1, Scalar(0));
        replacement.points().setValue(row, 2, Scalar(0));
    }
    replacement.points().setValue(63, 0, Scalar(0));
    auto replacement_gpu = replacement.toGpu();
    gpu_cloud->setPoints(std::move(replacement_gpu.points()));
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> query(1, 3);
    query.fill(Scalar(0));

    EXPECT_EQ(tree.batchNearestKSearch(query, 33).front().front(), 63);
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::CpuBruteForce);
}

TEST(SpatialIndexGpuTest, ExtremeFiniteCoordinatesFallBackInsteadOfThrowing)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = double;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    CpuCloud cpu_cloud(64);
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    for (int row = 0; row < 64; ++row)
    {
        cpu_cloud.points().setValue(row, 0, row % 2 == 0 ? -maximum : maximum);
        cpu_cloud.points().setValue(row, 1, Scalar(0));
        cpu_cloud.points().setValue(row, 2, Scalar(0));
    }
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(64, 3);
    queries.fill(Scalar(0));

    EXPECT_NO_THROW(tree.batchNearestKSearch(queries, 4));
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::BruteForce);
}

TEST(SpatialIndexGpuTest, CloudIdentityPreventsAbaMatchAfterDeviceAddressReuse)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Cloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;
    auto original = std::make_unique<Cloud>(makeCloud<float>({
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f}));
    const float* original_data = static_cast<const Cloud&>(*original).points().data();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.build(*original, 1.0f);
    original.reset();

    bool reused_address = false;
    for (int attempt = 0; attempt < 256; ++attempt)
    {
        Cloud candidate = makeCloud<float>({
            10.0f, 0.0f, 0.0f,
            11.0f, 0.0f, 0.0f});
        if (static_cast<const Cloud&>(candidate).points().data() == original_data)
        {
            reused_address = true;
            EXPECT_FALSE(index.matches(candidate, 1.0f));
            break;
        }
    }
    EXPECT_TRUE(reused_address) << "CUDA allocator did not reuse the released test allocation";
}

TEST(SpatialIndexGpuTest, QueryWorkspaceCanMoveAcrossDestroyedNonBlockingStreams)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cloud = makeCloud<float>({
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f});
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.build(cloud, 1.0f);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> query_cpu(1, 3);
    query_cpu.fill(0.0f);
    auto query_gpu = query_cpu.toGpu();
    plapoint::gpu::GpuSpatialQueryWorkspace<float> workspace;

    for (int iteration = 0; iteration < 2; ++iteration)
    {
        cudaStream_t stream = nullptr;
        PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto result = index.knnSearchAsync(query_gpu, 2, workspace, stream);
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
        const auto indices = result.indices.toCpu();
        EXPECT_EQ(indices(0, 0), 0);
        EXPECT_EQ(indices(0, 1), 1);
        result.indices.closeAsyncAllocation();
        result.squaredDistances.closeAsyncAllocation();
        PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
    }
}

TYPED_TEST(SpatialIndexGpuTest, CpuFallbackRefreshesEveryQueryAfterMutableAliasEscapes)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    using CpuCloudType = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloudType = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    CpuCloudType cpu_cloud(64);
    for (int row = 0; row < 64; ++row)
    {
        cpu_cloud.points().setValue(row, 0, Scalar(row + 100));
        cpu_cloud.points().setValue(row, 1, Scalar(0));
        cpu_cloud.points().setValue(row, 2, Scalar(0));
    }
    auto gpu_cloud = std::make_shared<GpuCloudType>(cpu_cloud.toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    auto& retained_alias = gpu_cloud->points();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> query(1, 3);
    query.fill(Scalar(0));

    retained_alias.setValue(63, 0, Scalar(0));
    EXPECT_EQ(tree.batchNearestKSearch(query, 33).front().front(), 63);
    retained_alias.setValue(63, 0, Scalar(200));
    retained_alias.setValue(62, 0, Scalar(0));
    EXPECT_EQ(tree.batchNearestKSearch(query, 33).front().front(), 62);
}

TEST(SpatialIndexGpuTest, CpuFallbackRefreshesAfterCloudMoveAssignmentWithSameRevision)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using CpuCloudType = plapoint::PointCloud<double, plamatrix::Device::CPU>;
    using GpuCloudType = plapoint::PointCloud<double, plamatrix::Device::GPU>;
    CpuCloudType original(64);
    CpuCloudType replacement(64);
    for (int row = 0; row < 64; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            original.points().setValue(row, column, column == 0 ? row + 100.0 : 0.0);
            replacement.points().setValue(row, column, column == 0 ? row + 200.0 : 0.0);
        }
    }
    replacement.points().setValue(63, 0, 0.0);
    auto gpu_cloud = std::make_shared<GpuCloudType>(original.toGpu());
    plapoint::search::KdTree<double, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    *gpu_cloud = replacement.toGpu();
    plamatrix::DenseMatrix<double, plamatrix::Device::CPU> query(1, 3);
    query.fill(0.0);

    EXPECT_EQ(tree.batchNearestKSearch(query, 33).front().front(), 63);
}

TEST(SpatialIndexGpuTest, IndexedAndBruteKnnPreserveTinyDifferenceAtHugeCommonCoordinate)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    const double maximum = std::numeric_limits<double>::max();
    const double tiny = std::numeric_limits<double>::denorm_min();
    auto cloud = makeCloud<double>({
        maximum, 2.0 * tiny, 0.0,
        maximum, tiny, 0.0});
    plapoint::gpu::GpuSpatialIndex<double> index;
    index.build(cloud, maximum);
    plamatrix::DenseMatrix<double, plamatrix::Device::CPU> query(1, 3);
    query.setValue(0, 0, maximum);
    query.setValue(0, 1, 0.0);
    query.setValue(0, 2, 0.0);
    auto gpu_query = query.toGpu();
    plapoint::gpu::GpuSpatialQueryWorkspace<double> workspace;
    auto indexed = index.knnSearchAsync(gpu_query, 2, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    const auto indexed_indices = indexed.indices.toCpu();
    EXPECT_EQ(indexed_indices(0, 0), 1);
    EXPECT_EQ(indexed_indices(0, 1), 0);

    auto shared_cloud = std::make_shared<GpuCloud<double>>(std::move(cloud));
    plapoint::search::KdTree<double, plamatrix::Device::GPU> tree;
    tree.setInputCloud(shared_cloud);
    tree.build();
    EXPECT_EQ(tree.batchNearestKSearch(query, 2).front(), (std::vector<int>{1, 0}));
    EXPECT_EQ(tree.lastNeighborBackend(), plapoint::gpu::GpuNeighborBackend::BruteForce);
}

TEST(SpatialIndexGpuTest, BruteKnnOrdersFiniteDistancesBeyondDoubleRange)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    const double maximum = std::numeric_limits<double>::max();
    auto gpu_cloud = std::make_shared<GpuCloud<double>>(makeCloud<double>({
        -maximum, 0.0, 0.0,
        -maximum / 2.0, 0.0, 0.0}));
    plapoint::search::KdTree<double, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    plamatrix::DenseMatrix<double, plamatrix::Device::CPU> query(1, 3);
    query.setValue(0, 0, maximum);
    query.setValue(0, 1, 0.0);
    query.setValue(0, 2, 0.0);

    EXPECT_EQ(tree.batchNearestKSearch(query, 2).front(), (std::vector<int>{1, 0}));
}

} // namespace

#endif
