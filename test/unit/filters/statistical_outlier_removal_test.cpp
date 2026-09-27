#include <gtest/gtest.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    // 10 points clustered at origin + 1 far outlier
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(11, 3);
    for (int i = 0; i < 10; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * 0.01f;
        mat.operator()(i, 1) = 0;
        mat.operator()(i, 2) = 0;
    }
    mat.operator()(10, 0) = 100; mat.operator()(10, 1) = 0; mat.operator()(10, 2) = 0;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    for (int i = 0; i < 4; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * Scalar(0.01);
        mat.operator()(i, 1) = 0;
        mat.operator()(i, 2) = 0;
    }
    mat.operator()(4, 0) = 100;
    mat.operator()(4, 1) = 0;
    mat.operator()(4, 2) = 0;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    for (int i = 0; i < 4; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * Scalar(0.01);
        mat.operator()(i, 1) = 0;
        mat.operator()(i, 2) = 0;
    }
    mat.operator()(4, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    mat.operator()(4, 1) = 0;
    mat.operator()(4, 2) = 0;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    mat.setConstant(0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
    sor.setInputCloud(cloud);

    Cloud output;
    EXPECT_THROW(sor.filter(output), std::runtime_error);
}

TEST(SORTest, EmptyInputReturnsEmptyOutput)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);
    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);

    Cloud output;
    ASSERT_NO_THROW(sor.filter(output));
    EXPECT_EQ(output.size(), 0u);
}

TEST(SORTest, SinglePointWithKGreaterThanPointCountIsKept)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    mat.operator()(0, 0) = 1.0f;
    mat.operator()(0, 1) = 2.0f;
    mat.operator()(0, 2) = 3.0f;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(8);
    sor.setStddevMulThresh(Scalar(0));

    Cloud output;
    sor.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_FLOAT_EQ(output.points().operator()(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.points().operator()(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(output.points().operator()(0, 2), 3.0f);
}

TEST(SORTest, RepeatedPointsWithZeroMeanDistanceAreKept)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        mat.operator()(i, 0) = 2.0f;
        mat.operator()(i, 1) = -1.0f;
        mat.operator()(i, 2) = 0.5f;
    }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    auto normals = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    for (int i = 0; i < 4; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * Scalar(0.01);
        mat.operator()(i, 1) = 0;
        mat.operator()(i, 2) = 0;
        normals.operator()(i, 0) = Scalar(i + 1);
        normals.operator()(i, 1) = 0;
        normals.operator()(i, 2) = Scalar(10 + i);
    }
    mat.operator()(4, 0) = 100;
    mat.operator()(4, 1) = 0;
    mat.operator()(4, 2) = 0;
    normals.operator()(4, 0) = 99;
    normals.operator()(4, 1) = 0;
    normals.operator()(4, 2) = 99;
    auto cloud = std::make_shared<Cloud>(std::move(mat));
    cloud->setNormals(std::move(normals));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(0.5));

    Cloud output;
    sor.filter(output);

    ASSERT_TRUE(output.hasNormals());
    ASSERT_EQ(output.size(), 4u);
    EXPECT_FLOAT_EQ(output.normals()->operator()(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.normals()->operator()(3, 0), 4.0f);
    EXPECT_FLOAT_EQ(output.normals()->operator()(3, 2), 13.0f);
}

TEST(SORTest, CopiesColorsAndIntensitiesForInliers)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(5, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(5, 1);
    for (int i = 0; i < 4; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * Scalar(0.01);
        mat.operator()(i, 1) = 0;
        mat.operator()(i, 2) = 0;
    }
    mat.operator()(4, 0) = 100;
    mat.operator()(4, 1) = 0;
    mat.operator()(4, 2) = 0;
    for (int i = 0; i < 5; ++i)
    {
        colors.operator()(i, 0) = static_cast<std::uint8_t>(70 + i);
        colors.operator()(i, 1) = static_cast<std::uint8_t>(80 + i);
        colors.operator()(i, 2) = static_cast<std::uint8_t>(90 + i);
        intensities.operator()(i, 0) = static_cast<std::uint16_t>(3000 + i);
    }
    auto cloud = std::make_shared<Cloud>(std::move(mat));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(0.5));

    Cloud output;
    sor.filter(output);

    ASSERT_EQ(output.size(), 4u);
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    EXPECT_EQ(output.colors()->operator()(0, 0), 70);
    EXPECT_EQ(output.colors()->operator()(3, 0), 73);
    EXPECT_EQ(output.colors()->operator()(3, 2), 93);
    EXPECT_EQ(output.intensities()->operator()(0, 0), 3000);
    EXPECT_EQ(output.intensities()->operator()(3, 0), 3003);
}

TEST(SORTest, RemovesNonFiniteInputPointsAndPreservesAttributes)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(5, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(5, 1);
    for (int i = 0; i < 4; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * Scalar(0.01);
        mat.operator()(i, 1) = 0;
        mat.operator()(i, 2) = 0;
    }
    mat.operator()(4, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    mat.operator()(4, 1) = 0;
    mat.operator()(4, 2) = 0;
    for (int i = 0; i < 5; ++i)
    {
        colors.operator()(i, 0) = static_cast<std::uint8_t>(10 + i);
        colors.operator()(i, 1) = static_cast<std::uint8_t>(20 + i);
        colors.operator()(i, 2) = static_cast<std::uint8_t>(30 + i);
        intensities.operator()(i, 0) = static_cast<std::uint16_t>(100 + i);
    }
    auto cloud = std::make_shared<Cloud>(std::move(mat));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(2);
    sor.setStddevMulThresh(Scalar(10));

    Cloud output;
    ASSERT_NO_THROW(sor.filter(output));

    ASSERT_EQ(output.size(), 4u);
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    EXPECT_FLOAT_EQ(output.points().operator()(3, 0), 0.03f);
    EXPECT_EQ(output.colors()->operator()(3, 0), 13);
    EXPECT_EQ(output.colors()->operator()(3, 2), 33);
    EXPECT_EQ(output.intensities()->operator()(3, 0), 103);
}

TEST(SORTest, KeepsFiniteExtremeDistancePointsWhenDistancesWouldSquareOverflow)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    mat.operator()(0, 0) = 0;
    mat.operator()(0, 1) = 0;
    mat.operator()(0, 2) = 0;
    mat.operator()(1, 0) = std::numeric_limits<Scalar>::max();
    mat.operator()(1, 1) = 0;
    mat.operator()(1, 2) = 0;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(1);
    sor.setStddevMulThresh(Scalar(0));

    Cloud output;
    ASSERT_NO_THROW(sor.filter(output));

    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output.points().operator()(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.points().operator()(1, 0), std::numeric_limits<Scalar>::max());
}

TEST(SORTest, RejectsInvalidParameters)
{
    plapoint::StatisticalOutlierRemoval<float, plamatrix::internal::Device::CPU> sor;

    EXPECT_THROW(sor.setMeanK(-1), std::invalid_argument);
    EXPECT_THROW(sor.setMeanK(0), std::invalid_argument);
    EXPECT_THROW(sor.setMeanK(std::numeric_limits<int>::max()), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(-0.1f), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(sor.setStddevMulThresh(std::numeric_limits<float>::infinity()), std::invalid_argument);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(SORTest, GpuIndexedPathDoesNotRequireHostBackedKdTree)
{
    if (!hasCudaDeviceForSOR())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU SOR test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(12, 3);
    for (int i = 0; i < 11; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * Scalar(0.01);
        mat.operator()(i, 1) = Scalar(i % 3) * Scalar(0.01);
        mat.operator()(i, 2) = Scalar(0);
    }
    mat.operator()(11, 0) = Scalar(10);
    mat.operator()(11, 1) = Scalar(10);
    mat.operator()(11, 2) = Scalar(0);

    CpuCloud cpu_cloud(std::move(mat));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::GPU> sor;
    sor.setInputCloud(gpu_cloud);
    sor.setMeanK(4);
    sor.setStddevMulThresh(Scalar(1));

    GpuCloud output;
    sor.filter(output);

    EXPECT_EQ(output.toCpu().size(), 11u);
    EXPECT_EQ(sor.lastGpuBackend(), plapoint::gpu::GpuOutlierRemovalBackend::UniformGrid);
}

TEST(SORTest, GpuMatchesCpuAndCopiesNormals)
{
    if (!hasCudaDeviceForSOR())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU SOR test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(6, 3);
    auto normals = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(6, 3);
    for (int i = 0; i < 5; ++i)
    {
        mat.operator()(i, 0) = Scalar(i) * Scalar(0.02);
        mat.operator()(i, 1) = Scalar(i % 2) * Scalar(0.01);
        mat.operator()(i, 2) = 0;
        normals.operator()(i, 0) = Scalar(i + 1);
        normals.operator()(i, 1) = 0;
        normals.operator()(i, 2) = Scalar(20 + i);
    }
    mat.operator()(5, 0) = 50;
    mat.operator()(5, 1) = 50;
    mat.operator()(5, 2) = 0;
    normals.operator()(5, 0) = 99;
    normals.operator()(5, 1) = 0;
    normals.operator()(5, 2) = 99;

    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(mat));
    cpu_cloud->setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    auto cpu_tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();

    auto gpu_tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::GPU>>();
    gpu_tree->setInputCloud(gpu_cloud);
    gpu_tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> cpu_sor;
    cpu_sor.setInputCloud(cpu_cloud);
    cpu_sor.setSearchMethod(cpu_tree);
    cpu_sor.setMeanK(3);
    cpu_sor.setStddevMulThresh(Scalar(0.75));
    CpuCloud cpu_output;
    cpu_sor.filter(cpu_output);

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::GPU> gpu_sor;
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
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(static_cast<plamatrix::Index>(i), 0),
                        cpu_output.points().operator()(static_cast<plamatrix::Index>(i), 0));
        EXPECT_FLOAT_EQ(gpu_output_cpu.normals()->operator()(static_cast<plamatrix::Index>(i), 0),
                        cpu_output.normals()->operator()(static_cast<plamatrix::Index>(i), 0));
        EXPECT_FLOAT_EQ(gpu_output_cpu.normals()->operator()(static_cast<plamatrix::Index>(i), 2),
                        cpu_output.normals()->operator()(static_cast<plamatrix::Index>(i), 2));
    }
}

TEST(SORTest, GpuIndexedPathPreservesAllAttributesAndReportsRemovedIndices)
{
    if (!hasCudaDeviceForSOR())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU SOR test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;
    constexpr int count = 7;
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> points(count, 3);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(count, 3);
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(count, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(count, 1);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> fields(count, 2);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> texture_coords(count, 2);
    for (int i = 0; i < count; ++i)
    {
        const Scalar x =
            i < 5 ? Scalar(i) * Scalar(0.01) : (i == 5 ? Scalar(100) : std::numeric_limits<Scalar>::quiet_NaN());
        points.operator()(i, 0) = x;
        points.operator()(i, 1) = 0;
        points.operator()(i, 2) = 0;
        normals.operator()(i, 0) = Scalar(10 + i);
        normals.operator()(i, 1) = Scalar(20 + i);
        normals.operator()(i, 2) = Scalar(30 + i);
        colors.operator()(i, 0) = static_cast<std::uint8_t>(40 + i);
        colors.operator()(i, 1) = static_cast<std::uint8_t>(50 + i);
        colors.operator()(i, 2) = static_cast<std::uint8_t>(60 + i);
        intensities.operator()(i, 0) = static_cast<std::uint16_t>(700 + i);
        fields.operator()(i, 0) = Scalar(100 + i);
        fields.operator()(i, 1) = Scalar(200 + i);
        texture_coords.operator()(i, 0) = Scalar(i) + Scalar(0.1);
        texture_coords.operator()(i, 1) = Scalar(i) + Scalar(0.2);
    }
    auto cpu_input = std::make_shared<CpuCloud>(std::move(points));
    cpu_input->setNormals(std::move(normals));
    cpu_input->setColors(std::move(colors));
    cpu_input->setIntensities(std::move(intensities));
    cpu_input->setScalarFields({"score", "frame"}, std::move(fields));
    cpu_input->setTextureCoords(std::move(texture_coords));
    auto gpu_input = std::make_shared<GpuCloud>(cpu_input->toGpu());
    auto cpu_tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    auto gpu_tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::GPU>>();
    cpu_tree->setInputCloud(cpu_input);
    gpu_tree->setInputCloud(gpu_input);
    cpu_tree->build();
    gpu_tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> cpu_filter;
    cpu_filter.setInputCloud(cpu_input);
    cpu_filter.setSearchMethod(cpu_tree);
    cpu_filter.setMeanK(2);
    cpu_filter.setStddevMulThresh(Scalar(0.5));
    CpuCloud cpu_output;
    std::vector<int> cpu_removed;
    cpu_filter.filter(cpu_output, cpu_removed);

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::GPU> gpu_filter;
    gpu_filter.setInputCloud(gpu_input);
    gpu_filter.setSearchMethod(gpu_tree);
    gpu_filter.setMeanK(2);
    gpu_filter.setStddevMulThresh(Scalar(0.5));
    GpuCloud gpu_output;
    std::vector<int> gpu_removed;
    gpu_filter.filter(gpu_output, gpu_removed);
    const auto output = gpu_output.toCpu();

    EXPECT_EQ(gpu_removed, cpu_removed);
    EXPECT_EQ(gpu_removed, (std::vector<int>{5, 6}));
    EXPECT_EQ(gpu_filter.lastGpuBackend(), plapoint::gpu::GpuOutlierRemovalBackend::UniformGrid);
    EXPECT_EQ(gpu_filter.gpuIndexBuildCount(), 1u);
    ASSERT_EQ(output.size(), cpu_output.size());
    ASSERT_TRUE(output.hasNormals());
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    ASSERT_TRUE(output.hasScalarFields());
    ASSERT_TRUE(output.hasTextureCoords());
    EXPECT_EQ(output.scalarFieldNames(), (std::vector<std::string>{"score", "frame"}));
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_FLOAT_EQ(output.points().operator()(i, 0), cpu_output.points().operator()(i, 0));
        EXPECT_FLOAT_EQ(output.normals()->operator()(i, 2), Scalar(30 + i));
        EXPECT_EQ(output.colors()->operator()(i, 1), 50 + i);
        EXPECT_EQ(output.intensities()->operator()(i, 0), 700 + i);
        EXPECT_FLOAT_EQ(output.scalarFields()->operator()(i, 0), Scalar(100 + i));
        EXPECT_FLOAT_EQ(output.scalarFields()->operator()(i, 1), Scalar(200 + i));
        EXPECT_FLOAT_EQ(output.textureCoords()->operator()(i, 0), Scalar(i) + Scalar(0.1));
        EXPECT_FLOAT_EQ(output.textureCoords()->operator()(i, 1), Scalar(i) + Scalar(0.2));
    }
}

TEST(SORTest, GpuMeanKBoundaryReportsIndexedAndCompatibilityBackends)
{
    if (!hasCudaDeviceForSOR())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU SOR test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> points(33, 3);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(33, 3);
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(33, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(33, 1);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> fields(33, 1);
    for (int i = 0; i < 33; ++i)
    {
        points.operator()(i, 0) = Scalar(i);
        points.operator()(i, 1) = Scalar(i % 3);
        points.operator()(i, 2) = 0;
        normals.operator()(i, 0) = Scalar(i + 1);
        normals.operator()(i, 1) = Scalar(0);
        normals.operator()(i, 2) = Scalar(100 + i);
        colors.operator()(i, 0) = static_cast<std::uint8_t>(i);
        colors.operator()(i, 1) = static_cast<std::uint8_t>(i + 1);
        colors.operator()(i, 2) = static_cast<std::uint8_t>(i + 2);
        intensities.operator()(i, 0) = static_cast<std::uint16_t>(1000 + i);
        fields.operator()(i, 0) = Scalar(2000 + i);
    }
    auto cpu_input = std::make_shared<CpuCloud>(std::move(points));
    cpu_input->setNormals(std::move(normals));
    cpu_input->setColors(std::move(colors));
    cpu_input->setIntensities(std::move(intensities));
    cpu_input->setScalarFields({"source"}, std::move(fields));
    auto gpu_input = std::make_shared<GpuCloud>(cpu_input->toGpu());
    auto tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::GPU>>();
    tree->setInputCloud(gpu_input);
    tree->build();
    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::GPU> filter;
    filter.setInputCloud(gpu_input);
    filter.setSearchMethod(tree);
    filter.setStddevMulThresh(Scalar(1));

    filter.setMeanK(31);
    GpuCloud indexed_output;
    filter.filter(indexed_output);
    EXPECT_EQ(filter.lastGpuBackend(), plapoint::gpu::GpuOutlierRemovalBackend::UniformGrid);
    EXPECT_TRUE(filter.lastGpuFallbackReason().empty());

    filter.setMeanK(32);
    GpuCloud fallback_output;
    std::vector<int> fallback_removed;
    filter.filter(fallback_output, fallback_removed);
    EXPECT_EQ(filter.lastGpuBackend(), plapoint::gpu::GpuOutlierRemovalBackend::CpuCompatibility);
    EXPECT_EQ(filter.lastGpuFallbackReason(), "mean_k + 1 exceeds indexed KNN limit 32");

    auto cpu_tree = std::make_shared<plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_input);
    cpu_tree->build();
    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::internal::Device::CPU> cpu_filter;
    cpu_filter.setInputCloud(cpu_input);
    cpu_filter.setSearchMethod(cpu_tree);
    cpu_filter.setMeanK(32);
    cpu_filter.setStddevMulThresh(Scalar(1));
    CpuCloud cpu_output;
    std::vector<int> cpu_removed;
    cpu_filter.filter(cpu_output, cpu_removed);

    const auto fallback_cpu = fallback_output.toCpu();
    EXPECT_EQ(fallback_removed, cpu_removed);
    ASSERT_EQ(fallback_cpu.size(), cpu_output.size());
    ASSERT_TRUE(fallback_cpu.hasNormals());
    ASSERT_TRUE(fallback_cpu.hasColors());
    ASSERT_TRUE(fallback_cpu.hasIntensities());
    ASSERT_TRUE(fallback_cpu.hasScalarFields());
    for (plamatrix::Index row = 0; row < fallback_cpu.points().rows(); ++row)
    {
        EXPECT_FLOAT_EQ(fallback_cpu.points().operator()(row, 0), cpu_output.points().operator()(row, 0));
        EXPECT_FLOAT_EQ(fallback_cpu.normals()->operator()(row, 2), cpu_output.normals()->operator()(row, 2));
        EXPECT_EQ(fallback_cpu.colors()->operator()(row, 1), cpu_output.colors()->operator()(row, 1));
        EXPECT_EQ(fallback_cpu.intensities()->operator()(row, 0), cpu_output.intensities()->operator()(row, 0));
        EXPECT_FLOAT_EQ(
            fallback_cpu.scalarFields()->operator()(row, 0),
            cpu_output.scalarFields()->operator()(row, 0));
    }

    filter.setMeanK(31);
    GpuCloud indexed_again_output;
    filter.filter(indexed_again_output);
    EXPECT_EQ(filter.lastGpuBackend(), plapoint::gpu::GpuOutlierRemovalBackend::UniformGrid);
    EXPECT_TRUE(filter.lastGpuFallbackReason().empty());
    EXPECT_EQ(indexed_again_output.size(), indexed_output.size());
}
#endif
