#include <gtest/gtest.h>
#include <plapoint/features/normal_refinement.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>

static bool hasCudaDeviceForNormalRefinement()
{
    return plapoint::gpu::hasUsableCudaDevice();
}
#endif

TEST(NormalRefinementTest, SmoothNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    pts.setValue(0,0,0); pts.setValue(0,1,0); pts.setValue(0,2,0);
    pts.setValue(1,0,1); pts.setValue(1,1,0); pts.setValue(1,2,0);
    pts.setValue(2,0,0); pts.setValue(2,1,1); pts.setValue(2,2,0);
    pts.setValue(3,0,1); pts.setValue(3,1,1); pts.setValue(3,2,0);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    Matrix normals(4, 3);
    normals.setValue(0,0,0.1f); normals.setValue(0,1,0.1f); normals.setValue(0,2,1.0f);
    normals.setValue(1,0,0.1f); normals.setValue(1,1,0.0f); normals.setValue(1,2,0.9f);
    normals.setValue(2,0,0.0f); normals.setValue(2,1,0.1f); normals.setValue(2,2,1.1f);
    normals.setValue(3,0,0.0f); normals.setValue(3,1,0.0f); normals.setValue(3,2,-1.0f);
    cloud->setNormals(std::move(normals));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> nr;
    nr.setInputCloud(cloud);
    nr.setSearchMethod(tree);
    nr.smooth(4);

    auto* ns = cloud->normals();
    for (int i = 0; i < 4; ++i)
    {
        Scalar nx = ns->getValue(i, 0), ny = ns->getValue(i, 1), nz = ns->getValue(i, 2);
        Scalar len = std::sqrt(nx*nx + ny*ny + nz*nz);
        EXPECT_NEAR(len, Scalar(1), Scalar(1e-4));
        EXPECT_GT(nz / len, Scalar(0));
    }
}

TEST(NormalRefinementTest, OrientConsistently)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(2, 3);
    pts.setValue(0,0,-1); pts.setValue(0,1,0); pts.setValue(0,2,0);
    pts.setValue(1,0, 1); pts.setValue(1,1,0); pts.setValue(1,2,0);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    Matrix normals(2, 3);
    // Both normals pointing right (+x direction)
    normals.setValue(0,0,1); normals.setValue(0,1,0); normals.setValue(0,2,0);
    normals.setValue(1,0,1); normals.setValue(1,1,0); normals.setValue(1,2,0);
    cloud->setNormals(std::move(normals));

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> nr;
    nr.setInputCloud(cloud);
    // Viewpoint far to the left => normals should flip toward viewpoint
    nr.orientConsistently({Scalar(-10), 0, 0});

    // Both normals were (1,0,0). Left point at (-1,0,0): should flip to point toward (-10,0,0)
    EXPECT_LT(cloud->normals()->getValue(0, 0), 0);
    // Right point at (1,0,0): stays pointing toward (-10,0,0) => normal becomes (-1,0,0)
    EXPECT_LT(cloud->normals()->getValue(1, 0), 0);
}

TEST(NormalRefinementTest, RejectsInvalidSmoothK)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(1);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(1, 3);
    normals.setValue(0, 0, 0);
    normals.setValue(0, 1, 0);
    normals.setValue(0, 2, 1);
    cloud->setNormals(std::move(normals));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> nr;
    nr.setInputCloud(cloud);
    nr.setSearchMethod(tree);

    EXPECT_THROW(nr.smooth(0), std::invalid_argument);
    EXPECT_THROW(nr.smooth(-1), std::invalid_argument);
}

TEST(NormalRefinementTest, SmoothThrowsForMissingInputs)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> no_cloud;
    EXPECT_THROW(no_cloud.smooth(1), std::runtime_error);

    auto cloud = std::make_shared<Cloud>(1);
    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> no_tree;
    no_tree.setInputCloud(cloud);
    EXPECT_THROW(no_tree.smooth(1), std::runtime_error);

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();
    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> no_normals;
    no_normals.setInputCloud(cloud);
    no_normals.setSearchMethod(tree);
    EXPECT_THROW(no_normals.smooth(1), std::runtime_error);
}

TEST(NormalRefinementTest, SmoothKGreaterThanPointCountUsesAvailableNeighbors)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(2, 3);
    pts.setValue(0, 0, 0); pts.setValue(0, 1, 0); pts.setValue(0, 2, 0);
    pts.setValue(1, 0, 1); pts.setValue(1, 1, 0); pts.setValue(1, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    Matrix normals(2, 3);
    normals.setValue(0, 0, 1); normals.setValue(0, 1, 0); normals.setValue(0, 2, 0);
    normals.setValue(1, 0, 0); normals.setValue(1, 1, 1); normals.setValue(1, 2, 0);
    cloud->setNormals(std::move(normals));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> nr;
    nr.setInputCloud(cloud);
    nr.setSearchMethod(tree);
    nr.smooth(10);

    const Scalar expected = Scalar(1) / std::sqrt(Scalar(2));
    ASSERT_TRUE(cloud->hasNormals());
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_NEAR(cloud->normals()->getValue(i, 0), expected, Scalar(1e-6));
        EXPECT_NEAR(cloud->normals()->getValue(i, 1), expected, Scalar(1e-6));
        EXPECT_NEAR(cloud->normals()->getValue(i, 2), Scalar(0), Scalar(1e-6));
    }
}

TEST(NormalRefinementTest, EmptyCloudWithNormalsSmoothsToEmptyNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);
    Matrix normals(0, 3);
    cloud->setNormals(std::move(normals));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> nr;
    nr.setInputCloud(cloud);
    nr.setSearchMethod(tree);

    EXPECT_NO_THROW(nr.smooth(1));
    ASSERT_TRUE(cloud->hasNormals());
    EXPECT_EQ(cloud->normals()->rows(), 0);
    EXPECT_EQ(cloud->normals()->cols(), 3);
}

TEST(NormalRefinementTest, OrientConsistentlyIgnoresMissingCloudOrNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> no_cloud;
    EXPECT_NO_THROW(no_cloud.orientConsistently({0, 0, 1}));

    auto cloud = std::make_shared<Cloud>(1);
    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> no_normals;
    no_normals.setInputCloud(cloud);
    EXPECT_NO_THROW(no_normals.orientConsistently({0, 0, 1}));
    EXPECT_FALSE(cloud->hasNormals());
}

#ifdef PLAPOINT_WITH_CUDA
TEST(NormalRefinementTest, GpuSmoothAndOrientUpdatesNormals)
{
    if (!hasCudaDeviceForNormalRefinement())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal refinement test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    pts.setValue(0, 0, 0); pts.setValue(0, 1, 0); pts.setValue(0, 2, 0);
    pts.setValue(1, 0, 1); pts.setValue(1, 1, 0); pts.setValue(1, 2, 0);
    pts.setValue(2, 0, 0); pts.setValue(2, 1, 1); pts.setValue(2, 2, 0);
    pts.setValue(3, 0, 1); pts.setValue(3, 1, 1); pts.setValue(3, 2, 0);
    CpuCloud cpu_cloud(std::move(pts));

    Matrix normals(4, 3);
    normals.setValue(0, 0, 0.1f); normals.setValue(0, 1, 0.1f); normals.setValue(0, 2, 1.0f);
    normals.setValue(1, 0, 0.1f); normals.setValue(1, 1, 0.0f); normals.setValue(1, 2, 0.9f);
    normals.setValue(2, 0, 0.0f); normals.setValue(2, 1, 0.1f); normals.setValue(2, 2, 1.1f);
    normals.setValue(3, 0, 0.0f); normals.setValue(3, 1, 0.0f); normals.setValue(3, 2, -1.0f);
    cpu_cloud.setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    tree->setInputCloud(gpu_cloud);
    tree->build();

    plapoint::NormalRefinement<Scalar, plamatrix::Device::GPU> nr;
    nr.setInputCloud(gpu_cloud);
    nr.setSearchMethod(tree);
    nr.smooth(4);
    nr.orientConsistently({0, 0, 10});

    auto refined = gpu_cloud->toCpu();
    ASSERT_TRUE(refined.hasNormals());
    for (int i = 0; i < 4; ++i)
    {
        Scalar nx = refined.normals()->getValue(i, 0);
        Scalar ny = refined.normals()->getValue(i, 1);
        Scalar nz = refined.normals()->getValue(i, 2);
        Scalar len = std::sqrt(nx * nx + ny * ny + nz * nz);
        EXPECT_NEAR(len, Scalar(1), Scalar(1e-4));
        EXPECT_GT(nz, Scalar(0));
    }
}

TEST(NormalRefinementTest, GpuSmoothDoesNotRetainBatchKnnWorkspace)
{
    if (!hasCudaDeviceForNormalRefinement())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal refinement test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(6, 3);
    for (int i = 0; i < 6; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i % 2));
        pts.setValue(i, 2, Scalar(0));
    }
    CpuCloud cpu_cloud(std::move(pts));

    Matrix normals(6, 3);
    for (int i = 0; i < 6; ++i)
    {
        normals.setValue(i, 0, Scalar(i % 3) * Scalar(0.1));
        normals.setValue(i, 1, Scalar((i + 1) % 3) * Scalar(0.1));
        normals.setValue(i, 2, Scalar(1));
    }
    cpu_cloud.setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    tree->setInputCloud(gpu_cloud);
    tree->build();

    ASSERT_EQ(tree->gpuBatchQueryScalarCapacityForTesting(), 0u);
    ASSERT_EQ(tree->gpuBatchResultCapacityForTesting(), 0u);

    plapoint::NormalRefinement<Scalar, plamatrix::Device::GPU> nr;
    nr.setInputCloud(gpu_cloud);
    nr.setSearchMethod(tree);
    nr.smooth(4);

    EXPECT_EQ(tree->gpuBatchQueryScalarCapacityForTesting(), 0u);
    EXPECT_EQ(tree->gpuBatchResultCapacityForTesting(), 0u);
}

TEST(NormalRefinementTest, GpuSmoothMatchesCpuForKGreaterThanPointCount)
{
    if (!hasCudaDeviceForNormalRefinement())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU normal refinement test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(2, 3);
    pts.setValue(0, 0, 0); pts.setValue(0, 1, 0); pts.setValue(0, 2, 0);
    pts.setValue(1, 0, 1); pts.setValue(1, 1, 0); pts.setValue(1, 2, 0);

    Matrix normals(2, 3);
    normals.setValue(0, 0, 1); normals.setValue(0, 1, 0); normals.setValue(0, 2, 0);
    normals.setValue(1, 0, 0); normals.setValue(1, 1, 1); normals.setValue(1, 2, 0);

    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(pts));
    cpu_cloud->setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    auto cpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();

    auto gpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    gpu_tree->setInputCloud(gpu_cloud);
    gpu_tree->build();

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> cpu_nr;
    cpu_nr.setInputCloud(cpu_cloud);
    cpu_nr.setSearchMethod(cpu_tree);
    cpu_nr.smooth(10);

    plapoint::NormalRefinement<Scalar, plamatrix::Device::GPU> gpu_nr;
    gpu_nr.setInputCloud(gpu_cloud);
    gpu_nr.setSearchMethod(gpu_tree);
    gpu_nr.smooth(10);

    auto gpu_cpu = gpu_cloud->toCpu();
    ASSERT_TRUE(cpu_cloud->hasNormals());
    ASSERT_TRUE(gpu_cpu.hasNormals());
    ASSERT_EQ(gpu_cpu.normals()->rows(), cpu_cloud->normals()->rows());
    for (plamatrix::Index i = 0; i < cpu_cloud->normals()->rows(); ++i)
    {
        EXPECT_NEAR(gpu_cpu.normals()->getValue(i, 0), cpu_cloud->normals()->getValue(i, 0), Scalar(1e-6));
        EXPECT_NEAR(gpu_cpu.normals()->getValue(i, 1), cpu_cloud->normals()->getValue(i, 1), Scalar(1e-6));
        EXPECT_NEAR(gpu_cpu.normals()->getValue(i, 2), cpu_cloud->normals()->getValue(i, 2), Scalar(1e-6));
    }
}
#endif
