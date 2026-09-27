#include <gtest/gtest.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>

static bool hasCudaDeviceForVoxelGrid()
{
    return plapoint::gpu::hasUsableCudaDevice();
}
#endif

namespace
{

    plamatrix::MatrixXf makeSignedVoxelPoints()
    {
        plamatrix::MatrixXf points(5, 3);
        points.operator()(0, 0) = 1.2f;
        points.operator()(0, 1) = 0.0f;
        points.operator()(0, 2) = 0.0f;
        points.operator()(1, 0) = -0.8f;
        points.operator()(1, 1) = 0.0f;
        points.operator()(1, 2) = 0.0f;
        points.operator()(2, 0) = 0.2f;
        points.operator()(2, 1) = 0.0f;
        points.operator()(2, 2) = 0.0f;
        points.operator()(3, 0) = -1.2f;
        points.operator()(3, 1) = 0.0f;
        points.operator()(3, 2) = 0.0f;
        points.operator()(4, 0) = 1.8f;
        points.operator()(4, 1) = 0.0f;
        points.operator()(4, 2) = 0.0f;
        return points;
    }

} // namespace

TEST(VoxelGridTest, DownsamplesUniformGrid)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    // 8 points forming a 2x2x2 cube
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(8, 3);
    int idx = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
            {
                mat.operator()(idx, 0) = Scalar(x);
                mat.operator()(idx, 1) = Scalar(y);
                mat.operator()(idx, 2) = Scalar(z);
                ++idx;
            }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(2.0), Scalar(2.0), Scalar(2.0));

    Cloud output;
    vg.filter(output);
    // All 8 points in one voxel => 1 centroid
    EXPECT_EQ(output.size(), 1u);
}

TEST(VoxelGridTest, PreservesSinglePoint)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    mat.operator()(0, 0) = 1.0f;
    mat.operator()(0, 1) = 2.0f;
    mat.operator()(0, 2) = 3.0f;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(0.1), Scalar(0.1), Scalar(0.1));

    Cloud output;
    vg.filter(output);
    EXPECT_EQ(output.size(), 1u);
}

TEST(VoxelGridTest, EmptyInputReturnsEmptyOutput)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(0.5), Scalar(0.5), Scalar(0.5));

    Cloud output;
    vg.filter(output);

    EXPECT_EQ(output.size(), 0u);
    EXPECT_EQ(output.points().rows(), 0);
    EXPECT_EQ(output.points().cols(), 3);
}

TEST(VoxelGridTest, AveragesRepeatedPointsIntoSingleCentroid)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
    mat.operator()(0, 0) = 0.0f;
    mat.operator()(0, 1) = 0.0f;
    mat.operator()(0, 2) = 0.0f;
    mat.operator()(1, 0) = 0.0f;
    mat.operator()(1, 1) = 0.0f;
    mat.operator()(1, 2) = 0.0f;
    mat.operator()(2, 0) = 0.3f;
    mat.operator()(2, 1) = 0.6f;
    mat.operator()(2, 2) = 0.9f;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_NEAR(output.points().operator()(0, 0), 0.1f, 1e-6f);
    EXPECT_NEAR(output.points().operator()(0, 1), 0.2f, 1e-6f);
    EXPECT_NEAR(output.points().operator()(0, 2), 0.3f, 1e-6f);
}

TEST(VoxelGridTest, OutputsCentroidsInDeterministicVoxelKeyOrder)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = makeSignedVoxelPoints();
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 4u);
    EXPECT_FLOAT_EQ(output.points().operator()(0, 0), -1.2f);
    EXPECT_FLOAT_EQ(output.points().operator()(1, 0), -0.8f);
    EXPECT_FLOAT_EQ(output.points().operator()(2, 0), 0.2f);
    EXPECT_FLOAT_EQ(output.points().operator()(3, 0), 1.5f);
}

TEST(VoxelGridTest, ThrowsOnZeroLeafSize)
{
    plapoint::VoxelGrid<float, plamatrix::internal::Device::CPU> vg;
    EXPECT_THROW(vg.setLeafSize(0, 1, 1), std::invalid_argument);
}

TEST(VoxelGridTest, PclContractLeafSizeUsesPlaMatrixVectors)
{
    plapoint::VoxelGrid<float, plamatrix::internal::Device::CPU> vg;
    vg.setLeafSize(plamatrix::Vector4f(0.25f, 0.5f, 1.0f, 1.0f));

    const plamatrix::Vector3f leaf_size = vg.getLeafSize();
    EXPECT_FLOAT_EQ(leaf_size(0), 0.25f);
    EXPECT_FLOAT_EQ(leaf_size(1), 0.5f);
    EXPECT_FLOAT_EQ(leaf_size(2), 1.0f);
    EXPECT_THROW(vg.setLeafSize(plamatrix::Vector4f(0.0f, 0.5f, 1.0f, 1.0f)),
                 std::invalid_argument);
}

TEST(VoxelGridTest, RejectsNegativeNaNAndInfiniteLeafSizes)
{
    plapoint::VoxelGrid<float, plamatrix::internal::Device::CPU> vg;

    EXPECT_THROW(vg.setLeafSize(-1.0f, 1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(1.0f, -1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(1.0f, 1.0f, -1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(1.0f, std::numeric_limits<float>::infinity(), 1.0f), std::invalid_argument);
}

TEST(VoxelGridTest, RejectsNonFiniteInputCoordinates)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    mat.operator()(0, 0) = 0.0f;
    mat.operator()(0, 1) = 0.0f;
    mat.operator()(0, 2) = 0.0f;
    mat.operator()(1, 0) = std::numeric_limits<float>::quiet_NaN();
    mat.operator()(1, 1) = 1.0f;
    mat.operator()(1, 2) = 2.0f;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    EXPECT_THROW(vg.filter(output), std::invalid_argument);
}

TEST(VoxelGridTest, RejectsVoxelKeysOutsideIntRange)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    mat.operator()(0, 0) = 1.0e20f;
    mat.operator()(0, 1) = 0.0f;
    mat.operator()(0, 2) = 0.0f;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    EXPECT_THROW(vg.filter(output), std::out_of_range);
}

TEST(VoxelGridTest, KeepsCentroidFiniteWhenFloatSumsWouldOverflow)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    mat.operator()(0, 0) = 3.0e38f;
    mat.operator()(0, 1) = 0.0f;
    mat.operator()(0, 2) = 0.0f;
    mat.operator()(1, 0) = 3.0e38f;
    mat.operator()(1, 1) = 0.0f;
    mat.operator()(1, 2) = 0.0f;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0e38f, 1.0f, 1.0f);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_TRUE(std::isfinite(output.points().operator()(0, 0)));
    EXPECT_FLOAT_EQ(output.points().operator()(0, 0), 3.0e38f);
}

TEST(VoxelGridTest, KeepsDoubleCentroidFiniteWhenDoubleSumsWouldOverflow)
{
    using Scalar = double;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    mat.operator()(0, 0) = 1.0e308;
    mat.operator()(0, 1) = 0.0;
    mat.operator()(0, 2) = 0.0;
    mat.operator()(1, 0) = 1.0e308;
    mat.operator()(1, 1) = 0.0;
    mat.operator()(1, 2) = 0.0;
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0e308, 1.0, 1.0);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_TRUE(std::isfinite(output.points().operator()(0, 0)));
    EXPECT_DOUBLE_EQ(output.points().operator()(0, 0), 1.0e308);
}

TEST(VoxelGridTest, AveragesNormalsColorsAndIntensitiesPerVoxel)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
    pts.operator()(0, 0) = 0.0f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    pts.operator()(1, 0) = 0.2f;
    pts.operator()(1, 1) = 0.0f;
    pts.operator()(1, 2) = 0.0f;
    pts.operator()(2, 0) = 2.0f;
    pts.operator()(2, 1) = 0.0f;
    pts.operator()(2, 2) = 0.0f;
    auto normals = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
    normals.operator()(0, 0) = 1.0f;
    normals.operator()(0, 1) = 0.0f;
    normals.operator()(0, 2) = 0.0f;
    normals.operator()(1, 0) = 0.0f;
    normals.operator()(1, 1) = 1.0f;
    normals.operator()(1, 2) = 0.0f;
    normals.operator()(2, 0) = 0.0f;
    normals.operator()(2, 1) = 0.0f;
    normals.operator()(2, 2) = 1.0f;
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(3, 3);
    colors.operator()(0, 0) = 10;
    colors.operator()(0, 1) = 20;
    colors.operator()(0, 2) = 30;
    colors.operator()(1, 0) = 14;
    colors.operator()(1, 1) = 24;
    colors.operator()(1, 2) = 34;
    colors.operator()(2, 0) = 100;
    colors.operator()(2, 1) = 110;
    colors.operator()(2, 2) = 120;
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(3, 1);
    intensities.operator()(0, 0) = 1000;
    intensities.operator()(1, 0) = 1004;
    intensities.operator()(2, 0) = 3000;

    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(normals));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 2u);
    ASSERT_TRUE(output.hasNormals());
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    EXPECT_FLOAT_EQ(output.normals()->operator()(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(output.normals()->operator()(0, 1), 0.5f);
    EXPECT_EQ(output.colors()->operator()(0, 0), 12);
    EXPECT_EQ(output.colors()->operator()(0, 1), 22);
    EXPECT_EQ(output.colors()->operator()(0, 2), 32);
    EXPECT_EQ(output.intensities()->operator()(0, 0), 1002);
    EXPECT_FLOAT_EQ(output.normals()->operator()(1, 2), 1.0f);
    EXPECT_EQ(output.colors()->operator()(1, 0), 100);
    EXPECT_EQ(output.intensities()->operator()(1, 0), 3000);
}

TEST(VoxelGridTest, AveragesScalarFieldsPerVoxel)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
    pts.operator()(0, 0) = 0.0f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    pts.operator()(1, 0) = 0.2f;
    pts.operator()(1, 1) = 0.0f;
    pts.operator()(1, 2) = 0.0f;
    pts.operator()(2, 0) = 2.0f;
    pts.operator()(2, 1) = 0.0f;
    pts.operator()(2, 2) = 0.0f;
    auto scalar_fields = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 2);
    scalar_fields.operator()(0, 0) = 1.0f;
    scalar_fields.operator()(0, 1) = 10.0f;
    scalar_fields.operator()(1, 0) = 3.0f;
    scalar_fields.operator()(1, 1) = 14.0f;
    scalar_fields.operator()(2, 0) = 20.0f;
    scalar_fields.operator()(2, 1) = 100.0f;

    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setScalarFields({"error", "confidence"}, std::move(scalar_fields));

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 2u);
    ASSERT_TRUE(output.hasScalarFields());
    EXPECT_EQ(output.scalarFieldNames(), (std::vector<std::string>{"error", "confidence"}));
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(0, 1), 12.0f);
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(1, 0), 20.0f);
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(1, 1), 100.0f);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(VoxelGridTest, GpuInputProducesGpuOutput)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> pts(3, 3);
    pts.operator()(0, 0) = 0.0f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    pts.operator()(1, 0) = 0.2f;
    pts.operator()(1, 1) = 0.0f;
    pts.operator()(1, 2) = 0.0f;
    pts.operator()(2, 0) = 2.0f;
    pts.operator()(2, 1) = 0.0f;
    pts.operator()(2, 2) = 0.0f;

    CpuCloud cpu_cloud(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    GpuCloud output;
    vg.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 2u);
    EXPECT_FLOAT_EQ(cpu_output.points().operator()(0, 0), 0.1f);
    EXPECT_FLOAT_EQ(cpu_output.points().operator()(1, 0), 2.0f);
}

TEST(VoxelGridTest, GpuVoxelGridHelperWritesPlaMatrixOutput)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> input_cpu(3, 3);
    input_cpu.operator()(0, 0) = 0.0f;
    input_cpu.operator()(0, 1) = 0.0f;
    input_cpu.operator()(0, 2) = 0.0f;
    input_cpu.operator()(1, 0) = 0.2f;
    input_cpu.operator()(1, 1) = 0.0f;
    input_cpu.operator()(1, 2) = 0.0f;
    input_cpu.operator()(2, 0) = 2.0f;
    input_cpu.operator()(2, 1) = 0.0f;
    input_cpu.operator()(2, 2) = 0.0f;
    auto context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
    auto input_gpu = plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(input_cpu, context);
    plamatrix::internal::ResidentMatrix<Scalar> output_gpu(3, 3, context);

    const int centroid_count = plapoint::gpu::voxelGridDownsampleColumnMajor(
        input_gpu, 1.0f, 1.0f, 1.0f, output_gpu);

    const auto output_cpu = output_gpu.toHostMatrix();
    ASSERT_EQ(centroid_count, 2);
    EXPECT_FLOAT_EQ(output_cpu.operator()(0, 0), 0.1f);
    EXPECT_FLOAT_EQ(output_cpu.operator()(1, 0), 2.0f);
}

TEST(VoxelGridTest, GpuMatchesCpuForNegativeCoordinatesAndSortedOutput)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto cpu_cloud = std::make_shared<CpuCloud>(makeSignedVoxelPoints());
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> cpu_vg;
    cpu_vg.setInputCloud(cpu_cloud);
    cpu_vg.setLeafSize(1.0f, 1.0f, 1.0f);
    CpuCloud cpu_output;
    cpu_vg.filter(cpu_output);

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> gpu_vg;
    gpu_vg.setInputCloud(gpu_cloud);
    gpu_vg.setLeafSize(1.0f, 1.0f, 1.0f);
    GpuCloud gpu_output;
    gpu_vg.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    for (std::size_t i = 0; i < cpu_output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(static_cast<plamatrix::Index>(i), 0),
                        cpu_output.points().operator()(static_cast<plamatrix::Index>(i), 0));
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(static_cast<plamatrix::Index>(i), 1),
                        cpu_output.points().operator()(static_cast<plamatrix::Index>(i), 1));
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(static_cast<plamatrix::Index>(i), 2),
                        cpu_output.points().operator()(static_cast<plamatrix::Index>(i), 2));
    }
}

TEST(VoxelGridTest, GpuEmptyInputMatchesCpu)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto cpu_cloud = std::make_shared<CpuCloud>(0);
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> cpu_vg;
    cpu_vg.setInputCloud(cpu_cloud);
    cpu_vg.setLeafSize(0.5f, 0.5f, 0.5f);
    CpuCloud cpu_output;
    cpu_vg.filter(cpu_output);

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> gpu_vg;
    gpu_vg.setInputCloud(gpu_cloud);
    gpu_vg.setLeafSize(0.5f, 0.5f, 0.5f);
    GpuCloud gpu_output;
    gpu_vg.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    EXPECT_EQ(gpu_output_cpu.size(), cpu_output.size());
    EXPECT_EQ(gpu_output_cpu.points().rows(), cpu_output.points().rows());
    EXPECT_EQ(gpu_output_cpu.points().cols(), cpu_output.points().cols());
}

TEST(VoxelGridTest, GpuRejectsNonFiniteInputCoordinates)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    pts.operator()(0, 0) = 0.0f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    pts.operator()(1, 0) = std::numeric_limits<float>::infinity();
    pts.operator()(1, 1) = 1.0f;
    pts.operator()(1, 2) = 2.0f;
    CpuCloud cpu_cloud(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    GpuCloud output;
    EXPECT_THROW(vg.filter(output), std::invalid_argument);
}

TEST(VoxelGridTest, GpuRejectsVoxelKeysOutsideIntRange)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    pts.operator()(0, 0) = 1.0e20f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    CpuCloud cpu_cloud(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    GpuCloud output;
    EXPECT_THROW(vg.filter(output), std::out_of_range);
}

TEST(VoxelGridTest, GpuKeepsCentroidFiniteWhenFloatSumsWouldOverflow)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    pts.operator()(0, 0) = 3.0e38f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    pts.operator()(1, 0) = 3.0e38f;
    pts.operator()(1, 1) = 0.0f;
    pts.operator()(1, 2) = 0.0f;
    CpuCloud cpu_cloud(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0e38f, 1.0f, 1.0f);

    GpuCloud output;
    vg.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 1u);
    EXPECT_TRUE(std::isfinite(cpu_output.points().operator()(0, 0)));
    EXPECT_FLOAT_EQ(cpu_output.points().operator()(0, 0), 3.0e38f);
}

TEST(VoxelGridTest, GpuKeepsDoubleCentroidFiniteWhenDoubleSumsWouldOverflow)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = double;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    pts.operator()(0, 0) = 1.0e308;
    pts.operator()(0, 1) = 0.0;
    pts.operator()(0, 2) = 0.0;
    pts.operator()(1, 0) = 1.0e308;
    pts.operator()(1, 1) = 0.0;
    pts.operator()(1, 2) = 0.0;
    CpuCloud cpu_cloud(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0e308, 1.0, 1.0);

    GpuCloud output;
    vg.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 1u);
    EXPECT_TRUE(std::isfinite(cpu_output.points().operator()(0, 0)));
    EXPECT_DOUBLE_EQ(cpu_output.points().operator()(0, 0), 1.0e308);
}

TEST(VoxelGridTest, GpuMatchesCpuForDoubleMixedMagnitudeCentroid)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = double;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    const auto makePoints = []
    {
        auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
        pts.operator()(0, 0) = 1.0e16;
        pts.operator()(0, 1) = 0.0;
        pts.operator()(0, 2) = 0.0;
        pts.operator()(1, 0) = 1.0;
        pts.operator()(1, 1) = 0.0;
        pts.operator()(1, 2) = 0.0;
        pts.operator()(2, 0) = 1.0;
        pts.operator()(2, 1) = 0.0;
        pts.operator()(2, 2) = 0.0;
        return pts;
    };

    auto cpu_cloud = std::make_shared<CpuCloud>(makePoints());
    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> cpu_vg;
    cpu_vg.setInputCloud(cpu_cloud);
    cpu_vg.setLeafSize(1.0e17, 1.0, 1.0);
    CpuCloud cpu_output;
    cpu_vg.filter(cpu_output);

    CpuCloud source_cpu(makePoints());
    auto gpu_cloud = std::make_shared<GpuCloud>(source_cpu.toGpu());
    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> gpu_vg;
    gpu_vg.setInputCloud(gpu_cloud);
    gpu_vg.setLeafSize(1.0e17, 1.0, 1.0);
    GpuCloud gpu_output;
    gpu_vg.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    ASSERT_EQ(cpu_output.size(), 1u);
    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    EXPECT_DOUBLE_EQ(gpu_output_cpu.points().operator()(0, 0),
                     cpu_output.points().operator()(0, 0));
}

TEST(VoxelGridTest, GpuUsesCpuConsistentDoublePrecisionVoxelBoundary)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    const auto makePoints = []
    {
        auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
        const Scalar boundary_below = std::nextafter(0.05f, 0.0f);
        pts.operator()(0, 0) = boundary_below;
        pts.operator()(0, 1) = 0.0f;
        pts.operator()(0, 2) = 0.0f;
        pts.operator()(1, 0) = 0.041f;
        pts.operator()(1, 1) = 0.0f;
        pts.operator()(1, 2) = 0.0f;
        return pts;
    };

    auto cpu_cloud = std::make_shared<CpuCloud>(makePoints());
    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::CPU> cpu_vg;
    cpu_vg.setInputCloud(cpu_cloud);
    cpu_vg.setLeafSize(0.01f, 1.0f, 1.0f);
    CpuCloud cpu_output;
    cpu_vg.filter(cpu_output);

    CpuCloud source_cpu(makePoints());
    auto gpu_cloud = std::make_shared<GpuCloud>(source_cpu.toGpu());
    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> gpu_vg;
    gpu_vg.setInputCloud(gpu_cloud);
    gpu_vg.setLeafSize(0.01f, 1.0f, 1.0f);
    GpuCloud gpu_output;
    gpu_vg.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    ASSERT_EQ(cpu_output.size(), 1u);
    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(0, 0), cpu_output.points().operator()(0, 0));
}

TEST(VoxelGridTest, GpuPreservesAveragedAttributesWhenInputHasPointAttributes)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    pts.operator()(0, 0) = 0.0f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    pts.operator()(1, 0) = 0.2f;
    pts.operator()(1, 1) = 0.0f;
    pts.operator()(1, 2) = 0.0f;
    auto normals = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    normals.operator()(0, 0) = 1.0f;
    normals.operator()(0, 1) = 0.0f;
    normals.operator()(0, 2) = 0.0f;
    normals.operator()(1, 0) = 0.0f;
    normals.operator()(1, 1) = 1.0f;
    normals.operator()(1, 2) = 0.0f;
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(2, 3);
    colors.operator()(0, 0) = 40;
    colors.operator()(0, 1) = 50;
    colors.operator()(0, 2) = 60;
    colors.operator()(1, 0) = 44;
    colors.operator()(1, 1) = 54;
    colors.operator()(1, 2) = 64;
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(2, 1);
    intensities.operator()(0, 0) = 2000;
    intensities.operator()(1, 0) = 2004;

    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(pts));
    cpu_cloud->setNormals(std::move(normals));
    cpu_cloud->setColors(std::move(colors));
    cpu_cloud->setIntensities(std::move(intensities));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    GpuCloud output;
    vg.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 1u);
    ASSERT_TRUE(cpu_output.hasNormals());
    ASSERT_TRUE(cpu_output.hasColors());
    ASSERT_TRUE(cpu_output.hasIntensities());
    EXPECT_FLOAT_EQ(cpu_output.normals()->operator()(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(cpu_output.normals()->operator()(0, 1), 0.5f);
    EXPECT_EQ(cpu_output.colors()->operator()(0, 0), 42);
    EXPECT_EQ(cpu_output.colors()->operator()(0, 2), 62);
    EXPECT_EQ(cpu_output.intensities()->operator()(0, 0), 2002);
}

TEST(VoxelGridTest, GpuPreservesAveragedScalarFieldsWhenInputHasOnlyScalarFields)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    auto pts = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    pts.operator()(0, 0) = 0.0f;
    pts.operator()(0, 1) = 0.0f;
    pts.operator()(0, 2) = 0.0f;
    pts.operator()(1, 0) = 0.2f;
    pts.operator()(1, 1) = 0.0f;
    pts.operator()(1, 2) = 0.0f;
    auto scalar_fields = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 1);
    scalar_fields.operator()(0, 0) = 1.0f;
    scalar_fields.operator()(1, 0) = 3.0f;

    CpuCloud cpu_cloud(std::move(pts));
    cpu_cloud.setScalarFields({"error"}, std::move(scalar_fields));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::internal::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    GpuCloud output;
    vg.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 1u);
    ASSERT_TRUE(cpu_output.hasScalarField("error"));
    EXPECT_FLOAT_EQ(cpu_output.scalarFields()->operator()(0, 0), 2.0f);
}
#endif
