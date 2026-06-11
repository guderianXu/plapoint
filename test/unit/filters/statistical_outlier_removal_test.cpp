#include <gtest/gtest.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdint>
#include <limits>
#include <vector>

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

TEST(SORTest, ReportsRemovedIndicesWithFilteredOutput)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    for (int i = 0; i < 4; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * Scalar(0.01));
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
    }
    mat.setValue(4, 0, 100);
    mat.setValue(4, 1, 0);
    mat.setValue(4, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(0.5));

    Cloud output;
    std::vector<int> removed_indices;
    sor.filter(output, removed_indices);

    ASSERT_EQ(output.size(), 4u);
    ASSERT_EQ(removed_indices.size(), 1u);
    EXPECT_EQ(removed_indices[0], 4);
}

TEST(SORTest, RemovedIndexOnlyOverloadIncludesNonFiniteInputPoints)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    for (int i = 0; i < 4; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * Scalar(0.01));
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
    }
    mat.setValue(4, 0, std::numeric_limits<Scalar>::quiet_NaN());
    mat.setValue(4, 1, 0);
    mat.setValue(4, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(10));

    std::vector<int> removed_indices;
    sor.filter(removed_indices);

    ASSERT_EQ(removed_indices.size(), 1u);
    EXPECT_EQ(removed_indices[0], 4);
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

TEST(SORTest, SinglePointWithKGreaterThanPointCountIsKept)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    mat.setValue(0, 0, 1.0f);
    mat.setValue(0, 1, 2.0f);
    mat.setValue(0, 2, 3.0f);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(8);
    sor.setStddevMulThresh(Scalar(0));

    Cloud output;
    sor.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_FLOAT_EQ(output.points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(output.points().getValue(0, 2), 3.0f);
}

TEST(SORTest, RepeatedPointsWithZeroMeanDistanceAreKept)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        mat.setValue(i, 0, 2.0f);
        mat.setValue(i, 1, -1.0f);
        mat.setValue(i, 2, 0.5f);
    }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(10);
    sor.setStddevMulThresh(Scalar(0));

    Cloud output;
    sor.filter(output);

    EXPECT_EQ(output.size(), 3u);
}

TEST(SORTest, CopiesNormalsForInliers)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    auto normals = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    for (int i = 0; i < 4; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * Scalar(0.01));
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
        normals.setValue(i, 0, Scalar(i + 1));
        normals.setValue(i, 1, 0);
        normals.setValue(i, 2, Scalar(10 + i));
    }
    mat.setValue(4, 0, 100);
    mat.setValue(4, 1, 0);
    mat.setValue(4, 2, 0);
    normals.setValue(4, 0, 99);
    normals.setValue(4, 1, 0);
    normals.setValue(4, 2, 99);
    auto cloud = std::make_shared<Cloud>(std::move(mat));
    cloud->setNormals(std::move(normals));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(0.5));

    Cloud output;
    sor.filter(output);

    ASSERT_TRUE(output.hasNormals());
    ASSERT_EQ(output.size(), 4u);
    EXPECT_FLOAT_EQ(output.normals()->getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.normals()->getValue(3, 0), 4.0f);
    EXPECT_FLOAT_EQ(output.normals()->getValue(3, 2), 13.0f);
}

TEST(SORTest, CopiesColorsAndIntensitiesForInliers)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(5, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(5, 1);
    for (int i = 0; i < 4; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * Scalar(0.01));
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
    }
    mat.setValue(4, 0, 100);
    mat.setValue(4, 1, 0);
    mat.setValue(4, 2, 0);
    for (int i = 0; i < 5; ++i)
    {
        colors.setValue(i, 0, static_cast<std::uint8_t>(70 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(80 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(90 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(3000 + i));
    }
    auto cloud = std::make_shared<Cloud>(std::move(mat));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(0.5));

    Cloud output;
    sor.filter(output);

    ASSERT_EQ(output.size(), 4u);
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    EXPECT_EQ(output.colors()->getValue(0, 0), 70);
    EXPECT_EQ(output.colors()->getValue(3, 0), 73);
    EXPECT_EQ(output.colors()->getValue(3, 2), 93);
    EXPECT_EQ(output.intensities()->getValue(0, 0), 3000);
    EXPECT_EQ(output.intensities()->getValue(3, 0), 3003);
}

TEST(SORTest, RemovesNonFiniteInputPointsAndPreservesAttributes)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(5, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(5, 1);
    for (int i = 0; i < 4; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * Scalar(0.01));
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
    }
    mat.setValue(4, 0, std::numeric_limits<Scalar>::quiet_NaN());
    mat.setValue(4, 1, 0);
    mat.setValue(4, 2, 0);
    for (int i = 0; i < 5; ++i)
    {
        colors.setValue(i, 0, static_cast<std::uint8_t>(10 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(20 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(30 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(100 + i));
    }
    auto cloud = std::make_shared<Cloud>(std::move(mat));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(10));

    Cloud output;
    ASSERT_NO_THROW(sor.filter(output));

    ASSERT_EQ(output.size(), 4u);
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    EXPECT_FLOAT_EQ(output.points().getValue(3, 0), 0.03f);
    EXPECT_EQ(output.colors()->getValue(3, 0), 13);
    EXPECT_EQ(output.colors()->getValue(3, 2), 33);
    EXPECT_EQ(output.intensities()->getValue(3, 0), 103);
}

TEST(SORTest, KeepsFiniteExtremeDistancePointsWhenDistancesWouldSquareOverflow)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    mat.setValue(0, 0, 0);
    mat.setValue(0, 1, 0);
    mat.setValue(0, 2, 0);
    mat.setValue(1, 0, std::numeric_limits<Scalar>::max());
    mat.setValue(1, 1, 0);
    mat.setValue(1, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(1);
    sor.setStddevMulThresh(Scalar(0));

    Cloud output;
    ASSERT_NO_THROW(sor.filter(output));

    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output.points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.points().getValue(1, 0), std::numeric_limits<Scalar>::max());
}

TEST(SORTest, RejectsInvalidParameters)
{
    plapoint::StatisticalOutlierRemoval<float, plamatrix::Device::CPU> sor;

    EXPECT_THROW(sor.setMeanK(-1), std::invalid_argument);
    EXPECT_THROW(sor.setMeanK(0), std::invalid_argument);
    EXPECT_THROW(sor.setMeanK(std::numeric_limits<int>::max()), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(-0.1f), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(std::numeric_limits<float>::infinity()), std::invalid_argument);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(SORTest, GpuInputUsesCudaMaskWithoutKdTreeWorkspace)
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

    EXPECT_EQ(output.toCpu().size(), 11u);
    EXPECT_EQ(tree->gpuBatchQueryScalarCapacityForTesting(), 0u);
    EXPECT_EQ(tree->gpuBatchResultCapacityForTesting(), 0u);
}

TEST(SORTest, GpuMatchesCpuAndCopiesNormals)
{
    if (!hasCudaDeviceForSOR())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU SOR test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(6, 3);
    auto normals = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(6, 3);
    for (int i = 0; i < 5; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * Scalar(0.02));
        mat.setValue(i, 1, Scalar(i % 2) * Scalar(0.01));
        mat.setValue(i, 2, 0);
        normals.setValue(i, 0, Scalar(i + 1));
        normals.setValue(i, 1, 0);
        normals.setValue(i, 2, Scalar(20 + i));
    }
    mat.setValue(5, 0, 50);
    mat.setValue(5, 1, 50);
    mat.setValue(5, 2, 0);
    normals.setValue(5, 0, 99);
    normals.setValue(5, 1, 0);
    normals.setValue(5, 2, 99);

    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(mat));
    cpu_cloud->setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    auto cpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();

    auto gpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    gpu_tree->setInputCloud(gpu_cloud);
    gpu_tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_sor;
    cpu_sor.setInputCloud(cpu_cloud);
    cpu_sor.setSearchMethod(cpu_tree);
    cpu_sor.setMeanK(3);
    cpu_sor.setStddevMulThresh(Scalar(0.75));
    CpuCloud cpu_output;
    cpu_sor.filter(cpu_output);

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::GPU> gpu_sor;
    gpu_sor.setInputCloud(gpu_cloud);
    gpu_sor.setSearchMethod(gpu_tree);
    gpu_sor.setMeanK(3);
    gpu_sor.setStddevMulThresh(Scalar(0.75));
    GpuCloud gpu_output;
    gpu_sor.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    ASSERT_TRUE(gpu_output_cpu.hasNormals());
    ASSERT_TRUE(cpu_output.hasNormals());
    for (std::size_t i = 0; i < cpu_output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().getValue(static_cast<plamatrix::Index>(i), 0),
                        cpu_output.points().getValue(static_cast<plamatrix::Index>(i), 0));
        EXPECT_FLOAT_EQ(gpu_output_cpu.normals()->getValue(static_cast<plamatrix::Index>(i), 0),
                        cpu_output.normals()->getValue(static_cast<plamatrix::Index>(i), 0));
        EXPECT_FLOAT_EQ(gpu_output_cpu.normals()->getValue(static_cast<plamatrix::Index>(i), 2),
                        cpu_output.normals()->getValue(static_cast<plamatrix::Index>(i), 2));
    }
}
#endif
