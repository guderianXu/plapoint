#include <gtest/gtest.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <limits>

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
#endif
