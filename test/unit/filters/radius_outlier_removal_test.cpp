#include <gtest/gtest.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>

static bool hasCudaDeviceForRadiusOutlierRemoval()
{
    return plapoint::gpu::hasUsableCudaDevice();
}
#endif

TEST(RadiusOutlierRemovalTest, RemovesIsolatedPoint)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(11, 3);
    for (int i = 0; i < 10; ++i) { pts.setValue(i, 0, Scalar(i)*0.01f); pts.setValue(i, 1, 0); pts.setValue(i, 2, 0); }
    pts.setValue(10, 0, 100); pts.setValue(10, 1, 0); pts.setValue(10, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(Scalar(1.0));
    ror.setMinNeighbors(2);

    Cloud output;
    ror.filter(output);
    EXPECT_EQ(output.size(), 10u);
}

TEST(RadiusOutlierRemovalTest, ReportsRemovedIndicesWithFilteredOutput)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    pts.setValue(0, 0, 0.0f); pts.setValue(0, 1, 0.0f); pts.setValue(0, 2, 0.0f);
    pts.setValue(1, 0, 0.1f); pts.setValue(1, 1, 0.0f); pts.setValue(1, 2, 0.0f);
    pts.setValue(2, 0, 0.2f); pts.setValue(2, 1, 0.0f); pts.setValue(2, 2, 0.0f);
    pts.setValue(3, 0, 10.0f); pts.setValue(3, 1, 0.0f); pts.setValue(3, 2, 0.0f);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(0.25f);
    ror.setMinNeighbors(2);

    Cloud output;
    std::vector<int> removed_indices;
    ror.filter(output, removed_indices);

    ASSERT_EQ(output.size(), 3u);
    ASSERT_EQ(removed_indices.size(), 1u);
    EXPECT_EQ(removed_indices[0], 3);
}

TEST(RadiusOutlierRemovalTest, RemovedIndexOnlyOverloadIncludesNonFiniteInputPoints)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    pts.setValue(0, 0, 0.0f); pts.setValue(0, 1, 0.0f); pts.setValue(0, 2, 0.0f);
    pts.setValue(1, 0, 0.1f); pts.setValue(1, 1, 0.0f); pts.setValue(1, 2, 0.0f);
    pts.setValue(2, 0, std::numeric_limits<Scalar>::quiet_NaN());
    pts.setValue(2, 1, 0.0f);
    pts.setValue(2, 2, 0.0f);
    pts.setValue(3, 0, 10.0f); pts.setValue(3, 1, 0.0f); pts.setValue(3, 2, 0.0f);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(0.25f);
    ror.setMinNeighbors(2);

    std::vector<int> removed_indices;
    ror.filter(removed_indices);

    ASSERT_EQ(removed_indices.size(), 2u);
    EXPECT_EQ(removed_indices[0], 2);
    EXPECT_EQ(removed_indices[1], 3);
}

TEST(RadiusOutlierRemovalTest, EmptyInputReturnsEmptyOutput)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(Scalar(1.0));
    ror.setMinNeighbors(1);

    Cloud output;
    ror.filter(output);

    EXPECT_EQ(output.size(), 0u);
    EXPECT_EQ(output.points().rows(), 0);
    EXPECT_EQ(output.points().cols(), 3);
}

TEST(RadiusOutlierRemovalTest, SinglePointHonorsSelfNeighborCount)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto pts = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    pts.setValue(0, 0, 1.0f);
    pts.setValue(0, 1, 2.0f);
    pts.setValue(0, 2, 3.0f);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> keep_self;
    keep_self.setInputCloud(cloud);
    keep_self.setRadius(0.0f);
    keep_self.setMinNeighbors(1);

    Cloud kept;
    keep_self.filter(kept);
    EXPECT_EQ(kept.size(), 1u);

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> require_neighbor;
    require_neighbor.setInputCloud(cloud);
    require_neighbor.setRadius(0.0f);
    require_neighbor.setMinNeighbors(2);

    Cloud removed;
    require_neighbor.filter(removed);
    EXPECT_EQ(removed.size(), 0u);
}

TEST(RadiusOutlierRemovalTest, ZeroRadiusKeepsOnlyRepeatedPoints)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(3, 3);
    pts.setValue(0, 0, 1.0f); pts.setValue(0, 1, 1.0f); pts.setValue(0, 2, 1.0f);
    pts.setValue(1, 0, 1.0f); pts.setValue(1, 1, 1.0f); pts.setValue(1, 2, 1.0f);
    pts.setValue(2, 0, 2.0f); pts.setValue(2, 1, 2.0f); pts.setValue(2, 2, 2.0f);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(0.0f);
    ror.setMinNeighbors(2);

    Cloud output;
    ror.filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output.points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.points().getValue(1, 0), 1.0f);
}

TEST(RadiusOutlierRemovalTest, RemovesNonFinitePoints)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    pts.setValue(0, 0, 0.0f); pts.setValue(0, 1, 0.0f); pts.setValue(0, 2, 0.0f);
    pts.setValue(1, 0, 0.1f); pts.setValue(1, 1, 0.0f); pts.setValue(1, 2, 0.0f);
    pts.setValue(2, 0, std::numeric_limits<Scalar>::infinity());
    pts.setValue(2, 1, 0.0f);
    pts.setValue(2, 2, 0.0f);
    pts.setValue(3, 0, std::numeric_limits<Scalar>::quiet_NaN());
    pts.setValue(3, 1, 0.0f);
    pts.setValue(3, 2, 0.0f);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(0.2f);
    ror.setMinNeighbors(2);

    Cloud output;
    ror.filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output.points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.points().getValue(1, 0), 0.1f);
}

TEST(RadiusOutlierRemovalTest, CopiesColorsAndIntensitiesForInliers)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(4, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(4, 1);
    for (int i = 0; i < 4; ++i)
    {
        pts.setValue(i, 0, i < 3 ? Scalar(i) * Scalar(0.1) : Scalar(10));
        pts.setValue(i, 1, 0);
        pts.setValue(i, 2, 0);
        colors.setValue(i, 0, static_cast<std::uint8_t>(40 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(50 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(60 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(2000 + i));
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(0.25f);
    ror.setMinNeighbors(2);

    Cloud output;
    ror.filter(output);

    ASSERT_EQ(output.size(), 3u);
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    EXPECT_EQ(output.colors()->getValue(0, 0), 40);
    EXPECT_EQ(output.colors()->getValue(1, 0), 41);
    EXPECT_EQ(output.colors()->getValue(2, 2), 62);
    EXPECT_EQ(output.intensities()->getValue(0, 0), 2000);
    EXPECT_EQ(output.intensities()->getValue(1, 0), 2001);
    EXPECT_EQ(output.intensities()->getValue(2, 0), 2002);
}

TEST(RadiusOutlierRemovalTest, RejectsInvalidParameters)
{
    plapoint::RadiusOutlierRemoval<float, plamatrix::Device::CPU> ror;

    EXPECT_THROW(ror.setRadius(-0.1f), std::invalid_argument);
    EXPECT_THROW(ror.setRadius(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(ror.setRadius(std::numeric_limits<float>::infinity()), std::invalid_argument);
    EXPECT_THROW(ror.setMinNeighbors(0), std::invalid_argument);
    EXPECT_THROW(ror.setMinNeighbors(-1), std::invalid_argument);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(RadiusOutlierRemovalTest, GpuInputProducesGpuOutput)
{
    if (!hasCudaDeviceForRadiusOutlierRemoval())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU radius outlier removal test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(4, 3);
    pts.setValue(0, 0, 0.0f); pts.setValue(0, 1, 0.0f); pts.setValue(0, 2, 0.0f);
    pts.setValue(1, 0, 0.1f); pts.setValue(1, 1, 0.0f); pts.setValue(1, 2, 0.0f);
    pts.setValue(2, 0, 0.2f); pts.setValue(2, 1, 0.0f); pts.setValue(2, 2, 0.0f);
    pts.setValue(3, 0, 10.0f); pts.setValue(3, 1, 0.0f); pts.setValue(3, 2, 0.0f);

    CpuCloud cpu_cloud(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::GPU> ror;
    ror.setInputCloud(gpu_cloud);
    ror.setRadius(0.25f);
    ror.setMinNeighbors(2);

    GpuCloud output;
    ror.filter(output);
    auto cpu_output = output.toCpu();

    EXPECT_EQ(cpu_output.size(), 3u);
}

TEST(RadiusOutlierRemovalTest, GpuMatchesCpuAndCopiesNormals)
{
    if (!hasCudaDeviceForRadiusOutlierRemoval())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU radius outlier removal test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(4, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        pts.setValue(i, 0, i < 3 ? Scalar(i) * Scalar(0.1) : Scalar(10));
        pts.setValue(i, 1, 0);
        pts.setValue(i, 2, 0);
        normals.setValue(i, 0, Scalar(i + 1));
        normals.setValue(i, 1, 0);
        normals.setValue(i, 2, Scalar(10 + i));
    }
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(pts));
    cpu_cloud->setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_ror;
    cpu_ror.setInputCloud(cpu_cloud);
    cpu_ror.setRadius(0.25f);
    cpu_ror.setMinNeighbors(2);
    CpuCloud cpu_output;
    cpu_ror.filter(cpu_output);

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::GPU> gpu_ror;
    gpu_ror.setInputCloud(gpu_cloud);
    gpu_ror.setRadius(0.25f);
    gpu_ror.setMinNeighbors(2);
    GpuCloud gpu_output;
    gpu_ror.filter(gpu_output);
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


TEST(RadiusOutlierRemovalTest, GpuIndexedPathMatchesCpuRemovedIndicesAndPreservesAllAttributes)
{
    if (!hasCudaDeviceForRadiusOutlierRemoval())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU radius outlier test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    constexpr int count = 6;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(count, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(count, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(count, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(count, 1);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> fields(count, 2);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> texture_coords(count, 2);
    const Scalar x[count] = {0, 0, 1, 2, 10, std::numeric_limits<Scalar>::quiet_NaN()};
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, x[i]);
        points.setValue(i, 1, 0);
        points.setValue(i, 2, 0);
        normals.setValue(i, 0, Scalar(10 + i));
        normals.setValue(i, 1, Scalar(20 + i));
        normals.setValue(i, 2, Scalar(30 + i));
        colors.setValue(i, 0, static_cast<std::uint8_t>(40 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(50 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(60 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(700 + i));
        fields.setValue(i, 0, Scalar(100 + i));
        fields.setValue(i, 1, Scalar(200 + i));
        texture_coords.setValue(i, 0, Scalar(i) + Scalar(0.1));
        texture_coords.setValue(i, 1, Scalar(i) + Scalar(0.2));
    }
    auto cpu_input = std::make_shared<CpuCloud>(std::move(points));
    cpu_input->setNormals(std::move(normals));
    cpu_input->setColors(std::move(colors));
    cpu_input->setIntensities(std::move(intensities));
    cpu_input->setScalarFields({"score", "frame"}, std::move(fields));
    cpu_input->setTextureCoords(std::move(texture_coords));
    auto gpu_input = std::make_shared<GpuCloud>(cpu_input->toGpu());

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_filter;
    cpu_filter.setInputCloud(cpu_input);
    cpu_filter.setRadius(Scalar(1));
    cpu_filter.setMinNeighbors(2);
    CpuCloud cpu_output;
    std::vector<int> cpu_removed;
    cpu_filter.filter(cpu_output, cpu_removed);

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::GPU> gpu_filter;
    gpu_filter.setInputCloud(gpu_input);
    gpu_filter.setRadius(Scalar(1));
    gpu_filter.setMinNeighbors(2);
    GpuCloud gpu_output;
    std::vector<int> gpu_removed;
    gpu_filter.filter(gpu_output, gpu_removed);
    const auto output = gpu_output.toCpu();

    EXPECT_EQ(gpu_removed, cpu_removed);
    EXPECT_EQ(gpu_removed, (std::vector<int>{4, 5}));
    EXPECT_EQ(gpu_filter.lastGpuBackend(), plapoint::gpu::GpuOutlierRemovalBackend::UniformGrid);
    EXPECT_EQ(gpu_filter.gpuIndexBuildCount(), 1u);
    ASSERT_EQ(output.size(), cpu_output.size());
    ASSERT_TRUE(output.hasNormals());
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    ASSERT_TRUE(output.hasScalarFields());
    ASSERT_TRUE(output.hasTextureCoords());
    EXPECT_EQ(output.scalarFieldNames(), (std::vector<std::string>{"score", "frame"}));
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(output.points().getValue(i, 0), cpu_output.points().getValue(i, 0));
        EXPECT_FLOAT_EQ(output.normals()->getValue(i, 2), Scalar(30 + i));
        EXPECT_EQ(output.colors()->getValue(i, 1), 50 + i);
        EXPECT_EQ(output.intensities()->getValue(i, 0), 700 + i);
        EXPECT_FLOAT_EQ(output.scalarFields()->getValue(i, 0), Scalar(100 + i));
        EXPECT_FLOAT_EQ(output.scalarFields()->getValue(i, 1), Scalar(200 + i));
        EXPECT_FLOAT_EQ(output.textureCoords()->getValue(i, 0), Scalar(i) + Scalar(0.1));
        EXPECT_FLOAT_EQ(output.textureCoords()->getValue(i, 1), Scalar(i) + Scalar(0.2));
    }

    GpuCloud repeated_output;
    gpu_filter.filter(repeated_output);
    EXPECT_EQ(gpu_filter.gpuIndexBuildCount(), 1u);
}
#endif
