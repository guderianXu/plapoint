#include <gtest/gtest.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <limits>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>

static bool hasCudaDeviceForSOR()
{
    return plapoint::gpu::hasUsableCudaDevice();
}
#endif

TEST(SORTest, RemovesSingleOutlier)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // 10 points clustered at origin + 1 far outlier
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(11, 3);
    for (int i = 0; i < 10; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * 0.01f);
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
    }
    mat.setValue(10, 0, 100); mat.setValue(10, 1, 0); mat.setValue(10, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(4);
    sor.setStddevMulThresh(Scalar(1.0));

    Cloud output;
    sor.filter(output);
    EXPECT_EQ(output.size(), 10u);
}

TEST(SORTest, ThrowsIfNoSearchMethod)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    mat.fill(0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);

    Cloud output;
    EXPECT_THROW(sor.filter(output), std::runtime_error);
}

TEST(SORTest, EmptyInputReturnsEmptyOutput)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);
    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);

    Cloud output;
    ASSERT_NO_THROW(sor.filter(output));
    EXPECT_EQ(output.size(), 0u);
}

TEST(SORTest, RejectsInvalidParameters)
{
    plapoint::StatisticalOutlierRemoval<float, plamatrix::Device::CPU> sor;

    EXPECT_THROW(sor.setMeanK(0), std::invalid_argument);
    EXPECT_THROW(sor.setMeanK(std::numeric_limits<int>::max()), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(-0.1f), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(std::numeric_limits<float>::infinity()), std::invalid_argument);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(SORTest, GpuInputUsesBatchKnnWorkspace)
{
    if (!hasCudaDeviceForSOR())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU SOR test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(12, 3);
    for (int i = 0; i < 11; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * Scalar(0.01));
        mat.setValue(i, 1, Scalar(i % 3) * Scalar(0.01));
        mat.setValue(i, 2, Scalar(0));
    }
    mat.setValue(11, 0, Scalar(10));
    mat.setValue(11, 1, Scalar(10));
    mat.setValue(11, 2, Scalar(0));

    CpuCloud cpu_cloud(std::move(mat));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    tree->setInputCloud(gpu_cloud);
    tree->build();

    ASSERT_EQ(tree->gpuBatchQueryScalarCapacityForTesting(), 0u);
    ASSERT_EQ(tree->gpuBatchResultCapacityForTesting(), 0u);

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::GPU> sor;
    sor.setInputCloud(gpu_cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(4);
    sor.setStddevMulThresh(Scalar(1));

    GpuCloud output;
    sor.filter(output);

    EXPECT_GE(tree->gpuBatchQueryScalarCapacityForTesting(), 36u);
    EXPECT_GE(tree->gpuBatchResultCapacityForTesting(), 60u);
}
#endif
