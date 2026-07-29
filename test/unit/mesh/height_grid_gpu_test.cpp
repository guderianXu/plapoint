#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/height_grid.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/height_grid.h>
#endif

namespace
{

using Scalar = float;
using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
using CpuMatrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

#ifdef PLAPOINT_WITH_CUDA
using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

bool hasCudaDevice()
{
    return plapoint::gpu::hasUsableCudaDevice();
}

#define SKIP_IF_NO_GPU() \
    do \
    { \
        if (!hasCudaDevice()) \
        { \
            GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU height grid test"; \
        } \
    } while (0)
#endif

CpuCloud makeSmallTerrainCloud()
{
    CpuMatrix points(5, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 1.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 2.0f);
    points.setValue(2, 0, 0.0f); points.setValue(2, 1, 1.0f); points.setValue(2, 2, 3.0f);
    points.setValue(3, 0, 1.0f); points.setValue(3, 1, 1.0f); points.setValue(3, 2, 4.0f);
    points.setValue(4, 0, 0.25f); points.setValue(4, 1, 0.25f); points.setValue(4, 2, 5.0f);

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(5, 3);
    colors.setValue(0, 0, 10); colors.setValue(0, 1, 20); colors.setValue(0, 2, 30);
    colors.setValue(1, 0, 40); colors.setValue(1, 1, 50); colors.setValue(1, 2, 60);
    colors.setValue(2, 0, 70); colors.setValue(2, 1, 80); colors.setValue(2, 2, 90);
    colors.setValue(3, 0, 100); colors.setValue(3, 1, 110); colors.setValue(3, 2, 120);
    colors.setValue(4, 0, 130); colors.setValue(4, 1, 140); colors.setValue(4, 2, 150);

    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(5, 1);
    intensities.setValue(0, 0, 1000);
    intensities.setValue(1, 0, 2000);
    intensities.setValue(2, 0, 3000);
    intensities.setValue(3, 0, 4000);
    intensities.setValue(4, 0, 5000);

    CpuCloud cloud(std::move(points));
    cloud.setColors(std::move(colors));
    cloud.setIntensities(std::move(intensities));
    return cloud;
}

plapoint::mesh::HeightGridOptions<Scalar> smallGridOptions()
{
    plapoint::mesh::HeightGridOptions<Scalar> options;
    options.width = 3;
    options.height = 3;
    options.padding = 0.0f;
    options.maxFillPassForFaces = 1;
    return options;
}

void expectGridNear(
    const plapoint::mesh::HeightGrid<Scalar>& actual,
    const plapoint::mesh::HeightGrid<Scalar>& expected)
{
    ASSERT_EQ(actual.width, expected.width);
    ASSERT_EQ(actual.height, expected.height);
    EXPECT_NEAR(actual.minX, expected.minX, 1.0e-6f);
    EXPECT_NEAR(actual.minY, expected.minY, 1.0e-6f);
    EXPECT_NEAR(actual.stepX, expected.stepX, 1.0e-6f);
    EXPECT_NEAR(actual.stepY, expected.stepY, 1.0e-6f);
    ASSERT_EQ(actual.heights.size(), expected.heights.size());
    ASSERT_EQ(actual.weights.size(), expected.weights.size());
    ASSERT_EQ(actual.valid.size(), expected.valid.size());
    ASSERT_EQ(actual.fillPass.size(), expected.fillPass.size());
    ASSERT_EQ(actual.colors.size(), expected.colors.size());

    for (std::size_t i = 0; i < expected.heights.size(); ++i)
    {
        EXPECT_NEAR(actual.heights[i], expected.heights[i], 1.0e-5f) << "cell " << i;
        EXPECT_NEAR(actual.weights[i], expected.weights[i], 1.0e-5f) << "cell " << i;
        EXPECT_EQ(actual.valid[i], expected.valid[i]) << "cell " << i;
        EXPECT_EQ(actual.fillPass[i], expected.fillPass[i]) << "cell " << i;
    }

    EXPECT_EQ(actual.colors, expected.colors);
}

} // namespace

#ifdef PLAPOINT_WITH_CUDA

TEST(HeightGridGpuTest, BuildHeightGridMatchesCpuOnSmallCloud)
{
    SKIP_IF_NO_GPU();

    const CpuCloud cpu_cloud = makeSmallTerrainCloud();
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    const auto options = smallGridOptions();

    const auto expected = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
    const auto actual = plapoint::gpu::buildHeightGrid(gpu_cloud, options);

    expectGridNear(actual, expected);
}

TEST(HeightGridGpuTest, HeightGridToMeshPreservesColorsAndIntensitiesThroughGpuPath)
{
    SKIP_IF_NO_GPU();

    const CpuCloud cpu_cloud = makeSmallTerrainCloud();
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    const auto options = smallGridOptions();

    auto expected_grid = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
    plapoint::mesh::fillHoles(expected_grid, 1);
    const auto expected_mesh = plapoint::mesh::heightGridToMesh(expected_grid, cpu_cloud, options);

    auto actual_grid = plapoint::gpu::buildHeightGrid(gpu_cloud, options);
    plapoint::gpu::fillHoles(actual_grid, 1);
    const auto actual_mesh = plapoint::gpu::heightGridToMesh(actual_grid, gpu_cloud, options);

    ASSERT_TRUE(actual_mesh.hasColors());
    ASSERT_TRUE(actual_mesh.hasIntensities());
    ASSERT_TRUE(actual_mesh.hasFaces());
    ASSERT_TRUE(expected_mesh.hasColors());
    ASSERT_TRUE(expected_mesh.hasIntensities());
    ASSERT_TRUE(expected_mesh.hasFaces());
    ASSERT_EQ(actual_mesh.size(), expected_mesh.size());
    ASSERT_EQ(actual_mesh.faces()->rows(), expected_mesh.faces()->rows());

    for (std::size_t i = 0; i < expected_mesh.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        EXPECT_NEAR(actual_mesh.points().getValue(row, 0), expected_mesh.points().getValue(row, 0), 1.0e-5f);
        EXPECT_NEAR(actual_mesh.points().getValue(row, 1), expected_mesh.points().getValue(row, 1), 1.0e-5f);
        EXPECT_NEAR(actual_mesh.points().getValue(row, 2), expected_mesh.points().getValue(row, 2), 1.0e-5f);
        EXPECT_EQ(actual_mesh.colors()->getValue(row, 0), expected_mesh.colors()->getValue(row, 0));
        EXPECT_EQ(actual_mesh.colors()->getValue(row, 1), expected_mesh.colors()->getValue(row, 1));
        EXPECT_EQ(actual_mesh.colors()->getValue(row, 2), expected_mesh.colors()->getValue(row, 2));
        EXPECT_EQ(actual_mesh.intensities()->getValue(row, 0), expected_mesh.intensities()->getValue(row, 0));
    }
}

TEST(HeightGridGpuTest, BuildHeightGridHandlesEmptyCloudAndRejectsInvalidOptions)
{
    SKIP_IF_NO_GPU();

    const CpuCloud empty_cpu;
    const GpuCloud empty_gpu = empty_cpu.toGpu();
    const auto empty_grid = plapoint::gpu::buildHeightGrid(empty_gpu, smallGridOptions());
    EXPECT_EQ(empty_grid.width, 0);
    EXPECT_EQ(empty_grid.height, 0);
    EXPECT_TRUE(empty_grid.heights.empty());
    EXPECT_TRUE(empty_grid.valid.empty());

    const GpuCloud gpu_cloud = makeSmallTerrainCloud().toGpu();

    auto bad_padding = smallGridOptions();
    bad_padding.padding = -0.25f;
    EXPECT_THROW(
        (void)plapoint::gpu::buildHeightGrid(gpu_cloud, bad_padding),
        std::invalid_argument);

    auto bad_size = smallGridOptions();
    bad_size.width = 1;
    EXPECT_THROW(
        (void)plapoint::gpu::buildHeightGrid(gpu_cloud, bad_size),
        std::invalid_argument);
}

TEST(HeightGridGpuTest, DeviceGridBuildAndDownloadMatchCpuOnNonDefaultStream)
{
    SKIP_IF_NO_GPU();

    const CpuCloud cpu_cloud = makeSmallTerrainCloud();
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    auto options = smallGridOptions();
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = 1.0f;
    options.minY = 0.0f;
    options.maxY = 1.0f;

    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        plapoint::gpu::HeightGridGpuWorkspace<Scalar> workspace;
        const auto device_grid = plapoint::gpu::buildHeightGridDeviceAsync(
            gpu_cloud, options, workspace, stream);
        EXPECT_EQ(device_grid.width, options.width);
        EXPECT_EQ(device_grid.height, options.height);
        EXPECT_TRUE(device_grid.hasColors());
        EXPECT_GE(workspace.cellCapacity(), std::size_t{9});

        const auto actual = plapoint::gpu::downloadHeightGrid(device_grid, stream);
        const auto expected = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
        expectGridNear(actual, expected);
    }
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST(HeightGridGpuTest, DeviceHoleFillMatchesCpuForMultiplePassesColorsAndMetadata)
{
    SKIP_IF_NO_GPU();

    const CpuCloud cpu_cloud = makeSmallTerrainCloud();
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    auto options = smallGridOptions();
    options.width = 5;
    options.height = 5;
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = 1.0f;
    options.minY = 0.0f;
    options.maxY = 1.0f;
    options.useBilinearSplat = false;

    auto expected = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
    plapoint::mesh::fillHoles(expected, 4, 2, 1);

    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        plapoint::gpu::HeightGridGpuWorkspace<Scalar> workspace;
        auto device_grid = plapoint::gpu::buildHeightGridDeviceAsync(
            gpu_cloud, options, workspace, stream);
        plapoint::gpu::fillHolesAsync(device_grid, 4, 2, 1, workspace, stream);
        const auto actual = plapoint::gpu::downloadHeightGrid(device_grid, stream);
        expectGridNear(actual, expected);

        const auto capacity = workspace.cellCapacity();
        plapoint::gpu::fillHolesAsync(device_grid, 2, 2, 2, workspace, stream);
        EXPECT_EQ(workspace.cellCapacity(), capacity);
    }
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST(HeightGridGpuTest, DeviceHoleFillHandlesEmptyGridAndRejectsInvalidShape)
{
    SKIP_IF_NO_GPU();

    plapoint::gpu::HeightGridGpuWorkspace<Scalar> workspace;
    plapoint::gpu::GpuHeightGrid<Scalar> empty;
    EXPECT_NO_THROW(plapoint::gpu::fillHolesAsync(empty, 3, 1, 1, workspace, nullptr));
    const auto downloaded = plapoint::gpu::downloadHeightGrid(empty);
    EXPECT_EQ(downloaded.width, 0);
    EXPECT_EQ(downloaded.height, 0);

    plapoint::gpu::GpuHeightGrid<Scalar> invalid;
    invalid.width = 2;
    invalid.height = 2;
    EXPECT_THROW(
        (void)plapoint::gpu::downloadHeightGrid(invalid),
        std::invalid_argument);
}

TEST(HeightGridGpuTest, DeviceBuildSupportsMinMaxAndSkippedNonFiniteBounds)
{
    SKIP_IF_NO_GPU();

    CpuMatrix points(4, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 3.0f);
    points.setValue(1, 0, 0.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 1.0f);
    points.setValue(2, 0, 1.0f); points.setValue(2, 1, 1.0f); points.setValue(2, 2, 5.0f);
    points.setValue(3, 0, std::numeric_limits<float>::quiet_NaN());
    points.setValue(3, 1, 0.5f); points.setValue(3, 2, 9.0f);
    const CpuCloud cpu_cloud(std::move(points));
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    auto options = smallGridOptions();
    options.useBilinearSplat = false;
    options.skipNonFinite = true;
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = 1.0f;
    options.minY = 0.0f;
    options.maxY = 1.0f;

    for (const auto aggregation : {
             plapoint::mesh::ElevationAggregation::Min,
             plapoint::mesh::ElevationAggregation::Max})
    {
        options.elevationAggregation = aggregation;
        const auto expected = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
        plapoint::gpu::HeightGridGpuWorkspace<Scalar> workspace;
        const auto device = plapoint::gpu::buildHeightGridDeviceAsync(
            gpu_cloud, options, workspace, nullptr);
        const auto actual = plapoint::gpu::downloadHeightGrid(device);
        expectGridNear(actual, expected);
    }
}

TEST(HeightGridGpuTest, DoubleDeviceBuildAndFillMatchCpu)
{
    SKIP_IF_NO_GPU();

    using DoubleMatrix = plamatrix::DenseMatrix<double, plamatrix::Device::CPU>;
    using DoubleCpuCloud = plapoint::PointCloud<double, plamatrix::Device::CPU>;
    DoubleMatrix points(4, 3);
    points.setValue(0, 0, 0.0); points.setValue(0, 1, 0.0); points.setValue(0, 2, 1.0);
    points.setValue(1, 0, 1.0); points.setValue(1, 1, 0.0); points.setValue(1, 2, 2.0);
    points.setValue(2, 0, 0.0); points.setValue(2, 1, 1.0); points.setValue(2, 2, 3.0);
    points.setValue(3, 0, 1.0); points.setValue(3, 1, 1.0); points.setValue(3, 2, 4.0);
    const DoubleCpuCloud cpu_cloud(std::move(points));
    const auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::mesh::HeightGridOptions<double> options;
    options.width = 5;
    options.height = 5;
    options.padding = 0.0;
    options.useBilinearSplat = false;
    options.useExplicitBounds = true;
    options.minX = 0.0;
    options.maxX = 1.0;
    options.minY = 0.0;
    options.maxY = 1.0;
    auto expected = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
    plapoint::mesh::fillHoles(expected, 3, 1, 1);

    plapoint::gpu::HeightGridGpuWorkspace<double> workspace;
    auto device = plapoint::gpu::buildHeightGridDeviceAsync(
        gpu_cloud, options, workspace, nullptr);
    plapoint::gpu::fillHolesAsync(device, 3, 1, 1, workspace, nullptr);
    const auto actual = plapoint::gpu::downloadHeightGrid(device);
    ASSERT_EQ(actual.valid, expected.valid);
    ASSERT_EQ(actual.fillPass, expected.fillPass);
    ASSERT_EQ(actual.heights.size(), expected.heights.size());
    for (std::size_t i = 0; i < expected.heights.size(); ++i)
    {
        EXPECT_NEAR(actual.heights[i], expected.heights[i], 1.0e-12) << "cell " << i;
    }
}

TEST(HeightGridGpuTest, WorkspaceRequiresExplicitResetBeforeCrossStreamReuse)
{
    SKIP_IF_NO_GPU();

    const GpuCloud cloud = makeSmallTerrainCloud().toGpu();
    auto options = smallGridOptions();
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = 1.0f;
    options.minY = 0.0f;
    options.maxY = 1.0f;
    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);
    {
        plapoint::gpu::HeightGridGpuWorkspace<Scalar> workspace;
        auto first = plapoint::gpu::buildHeightGridDeviceAsync(
            cloud, options, workspace, first_stream);
        (void)plapoint::gpu::downloadHeightGrid(first, first_stream);
        EXPECT_THROW(
            (void)plapoint::gpu::buildHeightGridDeviceAsync(
                cloud, options, workspace, second_stream),
            std::logic_error);
        workspace.resetStream(second_stream);
        auto second = plapoint::gpu::buildHeightGridDeviceAsync(
            cloud, options, workspace, second_stream);
        (void)plapoint::gpu::downloadHeightGrid(second, second_stream);
    }
    EXPECT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
}

TEST(HeightGridGpuTest, DeviceAsyncRequiresExplicitBoundsAndDefersStrictInputStatus)
{
    SKIP_IF_NO_GPU();

    const GpuCloud normal_cloud = makeSmallTerrainCloud().toGpu();
    plapoint::gpu::HeightGridGpuWorkspace<Scalar> workspace;
    EXPECT_THROW(
        (void)plapoint::gpu::buildHeightGridDeviceAsync(
            normal_cloud, smallGridOptions(), workspace, nullptr),
        std::invalid_argument);

    CpuMatrix points(2, 3);
    points.fill(0.0f);
    points.setValue(1, 0, std::numeric_limits<float>::quiet_NaN());
    const GpuCloud invalid_cloud = CpuCloud(std::move(points)).toGpu();
    auto options = smallGridOptions();
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = 1.0f;
    options.minY = 0.0f;
    options.maxY = 1.0f;
    plapoint::gpu::GpuHeightGrid<Scalar> device;
    EXPECT_NO_THROW(device = plapoint::gpu::buildHeightGridDeviceAsync(
        invalid_cloud, options, workspace, nullptr));
    EXPECT_THROW(
        (void)plapoint::gpu::downloadHeightGrid(device),
        std::invalid_argument);
}

TEST(HeightGridGpuTest, DeviceGridRejectsCrossStreamConsumptionUntilSynchronized)
{
    SKIP_IF_NO_GPU();

    const GpuCloud cloud = makeSmallTerrainCloud().toGpu();
    auto options = smallGridOptions();
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = 1.0f;
    options.minY = 0.0f;
    options.maxY = 1.0f;
    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
    {
        plapoint::gpu::HeightGridGpuWorkspace<Scalar> build_workspace;
        plapoint::gpu::HeightGridGpuWorkspace<Scalar> fill_workspace;
        auto device = plapoint::gpu::buildHeightGridDeviceAsync(
            cloud, options, build_workspace, producer);
        EXPECT_THROW(
            plapoint::gpu::fillHolesAsync(
                device, 1, 1, 1, fill_workspace, consumer),
            std::logic_error);
        EXPECT_THROW(
            (void)plapoint::gpu::downloadHeightGrid(device, consumer),
            std::logic_error);

        device.synchronize(producer);
        EXPECT_NO_THROW(plapoint::gpu::fillHolesAsync(
            device, 1, 1, 1, fill_workspace, consumer));
        const auto downloaded = plapoint::gpu::downloadHeightGrid(device, consumer);
        EXPECT_EQ(downloaded.width, options.width);
    }
    EXPECT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(producer), cudaSuccess);
}

TEST(HeightGridGpuTest, MovedFromWorkspaceCanBeReusedAndOddColorPassMatchesCpu)
{
    SKIP_IF_NO_GPU();

    const CpuCloud cpu_cloud = makeSmallTerrainCloud();
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    auto options = smallGridOptions();
    options.width = 5;
    options.height = 5;
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = 1.0f;
    options.minY = 0.0f;
    options.maxY = 1.0f;
    options.useBilinearSplat = false;
    auto expected = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
    plapoint::mesh::fillHoles(expected, 3, 1, 1);

    plapoint::gpu::HeightGridGpuWorkspace<Scalar> original;
    auto device = plapoint::gpu::buildHeightGridDeviceAsync(
        gpu_cloud, options, original, nullptr);
    plapoint::gpu::HeightGridGpuWorkspace<Scalar> moved(std::move(original));
    EXPECT_EQ(original.cellCapacity(), std::size_t{0});
    plapoint::gpu::fillHolesAsync(device, 3, 1, 1, moved, nullptr);
    expectGridNear(plapoint::gpu::downloadHeightGrid(device), expected);

    auto second = plapoint::gpu::buildHeightGridDeviceAsync(
        gpu_cloud, options, original, nullptr);
    EXPECT_GT(original.cellCapacity(), std::size_t{0});
    (void)plapoint::gpu::downloadHeightGrid(second);
}

#else

TEST(HeightGridGpuTest, SkipsWhenBuiltWithoutCuda)
{
    GTEST_SKIP() << "PlaPoint was built without CUDA support";
}

#endif // PLAPOINT_WITH_CUDA
