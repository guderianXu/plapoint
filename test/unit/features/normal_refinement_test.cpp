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

TEST(NormalRefinementTest, GpuSmoothUsesBatchKnnWorkspace)
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

    EXPECT_GE(tree->gpuBatchQueryScalarCapacityForTesting(), 18u);
    EXPECT_GE(tree->gpuBatchResultCapacityForTesting(), 24u);
}
#endif
