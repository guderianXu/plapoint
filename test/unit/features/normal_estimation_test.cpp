#include <gtest/gtest.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

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

TEST(NormalEstimationTest, ThrowsIfNoInput)
{
    plapoint::NormalEstimation<float, plamatrix::Device::CPU> ne;
    EXPECT_THROW(ne.compute(), std::runtime_error);
}

TEST(NormalEstimationTest, RejectsInvalidKSearch)
{
    plapoint::NormalEstimation<float, plamatrix::Device::CPU> ne;
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
#endif
