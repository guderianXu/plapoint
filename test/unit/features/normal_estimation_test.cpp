#include <gtest/gtest.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif

static bool hasCudaDevice()
{
#ifdef PLAPOINT_WITH_CUDA
    return plapoint::gpu::hasUsableCudaDevice();
#else
    return false;
#endif
}

TEST(NormalEstimationTest, PlaneNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // Points on the XY plane (z=0) => normals should be approximately (0,0,±1)
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(9, 3);
    int idx = 0;
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
        {
            mat.setValue(idx, 0, Scalar(x));
            mat.setValue(idx, 1, Scalar(y));
            mat.setValue(idx, 2, 0);
            ++idx;
        }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(8);

    auto normals = ne.compute();
    EXPECT_EQ(normals.rows(), 9);
    EXPECT_EQ(normals.cols(), 3);

    // Center point normal should be approximately (0,0,1) or (0,0,-1)
    Scalar z = normals.getValue(4, 2);
    EXPECT_GT(std::abs(z), Scalar(0.9));
}

TEST(NormalEstimationTest, AutoUsesCpuForSmallNeighborhoodWork)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(9, 3);
    int row = 0;
    for (int x = 0; x < 3; ++x)
    {
        for (int y = 0; y < 3; ++y)
        {
            points.setValue(row, 0, static_cast<Scalar>(x));
            points.setValue(row, 1, static_cast<Scalar>(y));
            points.setValue(row, 2, Scalar(0));
            ++row;
        }
    }

    const Cloud cloud(std::move(points));
    plapoint::ProcessingReport report;
    const auto normals = plapoint::estimateNormals(
        cloud, 8, plapoint::ProcessingDevice::Auto, &report);

    EXPECT_EQ(normals.rows(), static_cast<plamatrix::Index>(cloud.size()));
    EXPECT_EQ(report.requestedDevice, plapoint::ProcessingDevice::Auto);
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(report.neighborBackend, plapoint::ProcessingNeighborBackend::CpuKdTree);
    EXPECT_FALSE(report.usedFallback);
    EXPECT_TRUE(report.fallbackReason.empty());
    EXPECT_FALSE(report.selectionReason.empty());
}

TEST(NormalEstimationTest, ThrowsIfNoInput)
{
    plapoint::NormalEstimation<float, plamatrix::Device::CPU> ne;
    EXPECT_THROW(ne.compute(), std::runtime_error);
}

TEST(NormalEstimationTest, ThrowsIfNoSearchMethod)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(1);

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> ne;
    ne.setInputCloud(cloud);

    EXPECT_THROW(ne.compute(), std::runtime_error);
}

TEST(NormalEstimationTest, EmptyInputReturnsEmptyNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);
    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(3);

    auto normals = ne.compute();

    EXPECT_EQ(normals.rows(), 0);
    EXPECT_EQ(normals.cols(), 3);
}

TEST(NormalEstimationTest, KGreaterThanPointCountLeavesSmallNeighborhoodNormalsZero)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    mat.setValue(0, 0, 0.0f); mat.setValue(0, 1, 0.0f); mat.setValue(0, 2, 0.0f);
    mat.setValue(1, 0, 1.0f); mat.setValue(1, 1, 0.0f); mat.setValue(1, 2, 0.0f);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(8);

    auto normals = ne.compute();

    ASSERT_EQ(normals.rows(), 2);
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_FLOAT_EQ(normals.getValue(i, 0), 0.0f);
        EXPECT_FLOAT_EQ(normals.getValue(i, 1), 0.0f);
        EXPECT_FLOAT_EQ(normals.getValue(i, 2), 0.0f);
    }
}

TEST(NormalEstimationTest, DegenerateRepeatedNeighborhoodProducesFiniteNormal)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        mat.setValue(i, 0, 1.0f);
        mat.setValue(i, 1, 1.0f);
        mat.setValue(i, 2, 1.0f);
    }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(3);

    auto normals = ne.compute();

    ASSERT_EQ(normals.rows(), 3);
    for (int i = 0; i < 3; ++i)
    {
        const Scalar nx = normals.getValue(i, 0);
        const Scalar ny = normals.getValue(i, 1);
        const Scalar nz = normals.getValue(i, 2);
        EXPECT_TRUE(std::isfinite(nx));
        EXPECT_TRUE(std::isfinite(ny));
        EXPECT_TRUE(std::isfinite(nz));
    }
}

TEST(NormalEstimationTest, RejectsInvalidKSearch)
{
    plapoint::NormalEstimation<float, plamatrix::Device::CPU> ne;
    EXPECT_THROW(ne.setKSearch(-1), std::invalid_argument);
    EXPECT_THROW(ne.setKSearch(0), std::invalid_argument);
    EXPECT_THROW(ne.setKSearch(2), std::invalid_argument);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(NormalEstimationTest, GpuPlaneNormalsMatchCpuLayout)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal estimation test";
    }

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(9, 3);
    int idx = 0;
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
        {
            mat.setValue(idx, 0, Scalar(x));
            mat.setValue(idx, 1, Scalar(y));
            mat.setValue(idx, 2, 0);
            ++idx;
        }
    Cloud cpu_cloud(std::move(mat));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    tree->setInputCloud(gpu_cloud);
    tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::GPU> ne;
    ne.setInputCloud(gpu_cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(8);

    auto normals = ne.compute().toCpu();
    ASSERT_EQ(normals.rows(), 9);
    ASSERT_EQ(normals.cols(), 3);

    EXPECT_GT(std::abs(normals.getValue(4, 2)), Scalar(0.9));
}

TEST(NormalEstimationTest, GpuPlaneNormalsMatchCpuForEveryPoint)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal estimation test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(9, 3);
    int idx = 0;
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
        {
            mat.setValue(idx, 0, Scalar(x));
            mat.setValue(idx, 1, Scalar(y));
            mat.setValue(idx, 2, 0);
            ++idx;
        }
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(mat));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    auto cpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();

    auto gpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    gpu_tree->setInputCloud(gpu_cloud);
    gpu_tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> cpu_ne;
    cpu_ne.setInputCloud(cpu_cloud);
    cpu_ne.setSearchMethod(cpu_tree);
    cpu_ne.setKSearch(8);

    plapoint::NormalEstimation<Scalar, plamatrix::Device::GPU> gpu_ne;
    gpu_ne.setInputCloud(gpu_cloud);
    gpu_ne.setSearchMethod(gpu_tree);
    gpu_ne.setKSearch(8);

    auto cpu_normals = cpu_ne.compute();
    auto gpu_normals = gpu_ne.compute().toCpu();

    ASSERT_EQ(gpu_normals.rows(), cpu_normals.rows());
    ASSERT_EQ(gpu_normals.cols(), cpu_normals.cols());
    for (plamatrix::Index i = 0; i < cpu_normals.rows(); ++i)
    {
        EXPECT_NEAR(std::abs(gpu_normals.getValue(i, 0)),
                    std::abs(cpu_normals.getValue(i, 0)), Scalar(1e-5));
        EXPECT_NEAR(std::abs(gpu_normals.getValue(i, 1)),
                    std::abs(cpu_normals.getValue(i, 1)), Scalar(1e-5));
        EXPECT_NEAR(std::abs(gpu_normals.getValue(i, 2)),
                    std::abs(cpu_normals.getValue(i, 2)), Scalar(1e-5));
    }
}

TEST(NormalEstimationTest, GpuMatchesCpuForKGreaterThanPointCount)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal estimation test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    mat.setValue(0, 0, 0.0f); mat.setValue(0, 1, 0.0f); mat.setValue(0, 2, 0.0f);
    mat.setValue(1, 0, 1.0f); mat.setValue(1, 1, 0.0f); mat.setValue(1, 2, 0.0f);
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(mat));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    auto cpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();

    auto gpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    gpu_tree->setInputCloud(gpu_cloud);
    gpu_tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> cpu_ne;
    cpu_ne.setInputCloud(cpu_cloud);
    cpu_ne.setSearchMethod(cpu_tree);
    cpu_ne.setKSearch(8);

    plapoint::NormalEstimation<Scalar, plamatrix::Device::GPU> gpu_ne;
    gpu_ne.setInputCloud(gpu_cloud);
    gpu_ne.setSearchMethod(gpu_tree);
    gpu_ne.setKSearch(8);

    auto cpu_normals = cpu_ne.compute();
    auto gpu_normals = gpu_ne.compute().toCpu();

    ASSERT_EQ(gpu_normals.rows(), cpu_normals.rows());
    for (plamatrix::Index i = 0; i < cpu_normals.rows(); ++i)
    {
        EXPECT_FLOAT_EQ(gpu_normals.getValue(i, 0), cpu_normals.getValue(i, 0));
        EXPECT_FLOAT_EQ(gpu_normals.getValue(i, 1), cpu_normals.getValue(i, 1));
        EXPECT_FLOAT_EQ(gpu_normals.getValue(i, 2), cpu_normals.getValue(i, 2));
    }
}

TEST(NormalEstimationTest, GpuUsesHostFallbackForKAboveIndexedLimit)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal estimation test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(40, 3);
    for (int index = 0; index < 40; ++index)
    {
        points.setValue(index, 0, Scalar(index % 8));
        points.setValue(index, 1, Scalar(index / 8));
        points.setValue(index, 2, Scalar(0));
    }
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(points));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    auto cpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();
    auto gpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    gpu_tree->setInputCloud(gpu_cloud);
    gpu_tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> cpu_estimator;
    cpu_estimator.setInputCloud(cpu_cloud);
    cpu_estimator.setSearchMethod(cpu_tree);
    cpu_estimator.setKSearch(33);
    plapoint::NormalEstimation<Scalar, plamatrix::Device::GPU> gpu_estimator;
    gpu_estimator.setInputCloud(gpu_cloud);
    gpu_estimator.setSearchMethod(gpu_tree);
    gpu_estimator.setKSearch(33);

    const auto cpu_normals = cpu_estimator.compute();
    const auto gpu_normals = gpu_estimator.compute().toCpu();
    ASSERT_EQ(gpu_normals.rows(), cpu_normals.rows());
    for (plamatrix::Index row = 0; row < cpu_normals.rows(); ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            EXPECT_NEAR(
                std::abs(gpu_normals(row, column)),
                std::abs(cpu_normals(row, column)),
                Scalar(1e-5));
        }
    }
}

TEST(NormalEstimationTest, ExplicitGpuRejectsKAboveIndexedLimit)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal estimation test";
    }

    using Scalar = float;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(40, 3);
    for (int index = 0; index < 40; ++index)
    {
        points.setValue(index, 0, Scalar(index % 8));
        points.setValue(index, 1, Scalar(index / 8));
        points.setValue(index, 2, Scalar(0));
    }
    plapoint::PointCloud<Scalar, plamatrix::Device::CPU> cloud(std::move(points));
    EXPECT_THROW(
        plapoint::estimateNormals(cloud, 33, plapoint::ProcessingDevice::GPU),
        std::invalid_argument);
}

TEST(NormalEstimationTest, ExplicitGpuReportsUniformGridBackend)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal estimation test";
    }

    using Scalar = float;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(40, 3);
    for (int index = 0; index < 40; ++index)
    {
        points.setValue(index, 0, Scalar(index % 8));
        points.setValue(index, 1, Scalar(index / 8));
        points.setValue(index, 2, Scalar(0));
    }
    const plapoint::PointCloud<Scalar, plamatrix::Device::CPU> cloud(std::move(points));
    plapoint::ProcessingReport report;

    const auto normals = plapoint::estimateNormals(
        cloud, 8, plapoint::ProcessingDevice::GPU, &report);

    EXPECT_EQ(normals.rows(), 40);
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::GPU);
    EXPECT_EQ(report.neighborBackend, plapoint::ProcessingNeighborBackend::GpuUniformGrid);
    EXPECT_FALSE(report.usedFallback);
}
#endif
