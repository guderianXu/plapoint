#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>

#include <utility>

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

TEST(PointCloudTest, CpuCreation)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(100);
    EXPECT_EQ(cloud.size(), 100);
    EXPECT_EQ(cloud.points().rows(), 100);
    EXPECT_EQ(cloud.points().cols(), 3);
}

TEST(PointCloudTest, DefaultConstructsEmptyNx3Cloud)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud;

    EXPECT_EQ(cloud.size(), 0);
    EXPECT_EQ(cloud.points().rows(), 0);
    EXPECT_EQ(cloud.points().cols(), 3);
    EXPECT_THROW((void)cloud[0], std::out_of_range);
}

TEST(PointCloudTest, GpuTransfer)
{
    if (!hasCudaDevice()) { GTEST_SKIP() << "No CUDA device, skipping GPU transfer test"; }
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cpu_cloud(10);
    cpu_cloud.points().setConstant(1.0f);

    auto gpu_cloud = cpu_cloud.toGpu();
    EXPECT_EQ(gpu_cloud.size(), 10);

    auto cpu_cloud_back = gpu_cloud.toCpu();
    EXPECT_FLOAT_EQ(cpu_cloud_back.points().operator()(0, 0), 1.0f);
}

TEST(PointCloudTest, PointsCpuReturnsCpuPointStorage)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(2);
    cloud.points().operator()(0, 0) = 1.0f;
    cloud.points().operator()(1, 0) = 2.0f;

    const auto& points = cloud.pointsCpu();
    ASSERT_EQ(points.rows(), 2);
    ASSERT_EQ(points.cols(), 3);
    EXPECT_FLOAT_EQ(points.operator()(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(points.operator()(1, 0), 2.0f);
}

TEST(PointCloudTest, MutablePointAccessIncrementsPointsVersion)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;

    Cloud cloud(2);
    const auto initial_version = cloud.pointsVersion();

    const auto& const_cloud = static_cast<const Cloud&>(cloud);
    (void)const_cloud.points();
    EXPECT_EQ(cloud.pointsVersion(), initial_version);

    cloud.points().operator()(0, 0) = 3.0f;
    EXPECT_GT(cloud.pointsVersion(), initial_version);

    const auto after_mutable_access = cloud.pointsVersion();
    (void)cloud.pointsCpu();
    EXPECT_EQ(cloud.pointsVersion(), after_mutable_access);
}

TEST(PointCloudTest, PointRevisionTracksOnlyPositionMutations)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::MatrixXf;

    Cloud cloud(2);
    const auto initial_revision = cloud.pointsRevision();
    EXPECT_NE(initial_revision, 0U);

    const auto& read_only_points = std::as_const(cloud).points();
    static_cast<void>(read_only_points);
    EXPECT_EQ(cloud.pointsRevision(), initial_revision);

    Matrix normals(2, 3);
    normals.setConstant(1.0f);
    cloud.setNormals(std::move(normals));
    EXPECT_EQ(cloud.pointsRevision(), initial_revision);

    auto& mutable_points = cloud.points();
    EXPECT_GT(cloud.pointsRevision(), initial_revision);
    mutable_points(0, 0) = 3.0f;

    const auto before_replacement = cloud.pointsRevision();
    Matrix replacement(2, 3);
    replacement.setConstant(2.0f);
    cloud.setPoints(std::move(replacement));
    EXPECT_GT(cloud.pointsRevision(), before_replacement);
    EXPECT_EQ(cloud.size(), 2U);
    EXPECT_FLOAT_EQ(std::as_const(cloud).points()(1, 2), 2.0f);
}

TEST(PointCloudTest, ScopedPointEditInvalidatesAtBeginAndEndWithoutEscapingAlias)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;

    Cloud cloud(2);
    const auto initial_revision = cloud.pointsRevision();
    EXPECT_TRUE(cloud.pointCachesReusable());
    EXPECT_FALSE(cloud.hasActivePointEdit());
    EXPECT_FALSE(cloud.hasUntrackedMutablePointAlias());

    std::uint64_t editing_revision = 0;
    {
        auto edit = cloud.editPoints();
        editing_revision = cloud.pointsRevision();
        EXPECT_GT(editing_revision, initial_revision);
        EXPECT_TRUE(cloud.hasActivePointEdit());
        EXPECT_FALSE(cloud.pointCachesReusable());
        edit->operator()(1, 2) = 7.0f;
    }

    EXPECT_GT(cloud.pointsRevision(), editing_revision);
    EXPECT_FALSE(cloud.hasActivePointEdit());
    EXPECT_TRUE(cloud.pointCachesReusable());
    EXPECT_FALSE(cloud.hasUntrackedMutablePointAlias());
    EXPECT_FLOAT_EQ(std::as_const(cloud).points()(1, 2), 7.0f);
}

TEST(PointCloudTest, LegacyMutablePointAccessMarksUntrackedAlias)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;

    Cloud cloud(1);
    EXPECT_FALSE(cloud.hasUntrackedMutablePointAlias());

    auto& points = cloud.points();
    points(0, 0) = 1.0f;

    EXPECT_TRUE(cloud.hasUntrackedMutablePointAlias());
    EXPECT_FALSE(cloud.pointCachesReusable());
}

TEST(PointCloudTest, SetPointsCopyOwnsIndependentStorageAndAdvancesRevision)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::MatrixXf;

    Cloud cloud(2);
    Matrix replacement(2, 3);
    replacement(0, 0) = 1.0f;
    replacement(1, 0) = 2.0f;
    replacement(0, 1) = 3.0f;
    replacement(1, 1) = 4.0f;
    replacement(0, 2) = 5.0f;
    replacement(1, 2) = 6.0f;

    const auto revision = cloud.pointsRevision();
    const auto identity = cloud.pointsIdentity();
    cloud.setPoints(std::as_const(replacement));

    const auto& stored = std::as_const(cloud).points();
    EXPECT_GT(cloud.pointsRevision(), revision);
    EXPECT_EQ(cloud.pointsIdentity(), identity);
    EXPECT_NE(stored.data(), replacement.data());
    EXPECT_FLOAT_EQ(stored(1, 2), 6.0f);

    replacement(1, 2) = 99.0f;
    EXPECT_FLOAT_EQ(std::as_const(cloud).points()(1, 2), 6.0f);
}

TEST(PointCloudTest, SetPointsCopyHandlesEmptyNx3Matrix)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::MatrixXf;

    Cloud cloud;
    Matrix empty(0, 3);
    const auto revision = cloud.pointsRevision();

    cloud.setPoints(std::as_const(empty));

    EXPECT_EQ(cloud.size(), 0U);
    EXPECT_EQ(std::as_const(cloud).points().cols(), 3);
    EXPECT_EQ(std::as_const(cloud).points().data(), nullptr);
    EXPECT_GT(cloud.pointsRevision(), revision);
}

TEST(PointCloudTest, InvalidSetPointsCopyKeepsRevisionStorageAndReportsShape)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::MatrixXf;

    Cloud cloud(2);
    Matrix invalid(2, 4);
    const auto revision = cloud.pointsRevision();
    const auto* const storage = std::as_const(cloud).points().data();

    try
    {
        cloud.setPoints(std::as_const(invalid));
        FAIL() << "Expected invalid copy replacement shape to be rejected";
    }
    catch (const std::runtime_error& ex)
    {
        EXPECT_NE(std::string(ex.what()).find("DeviceCloud requires Nx3 matrix"), std::string::npos);
    }

    EXPECT_EQ(cloud.pointsRevision(), revision);
    EXPECT_EQ(std::as_const(cloud).points().data(), storage);
}

TEST(PointCloudTest, FailedPointReplacementKeepsRevisionAndStorage)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::MatrixXf;

    Cloud cloud(2);
    const auto revision = cloud.pointsRevision();
    const auto* const data = std::as_const(cloud).points().data();
    Matrix invalid(2, 4);

    EXPECT_THROW(cloud.setPoints(std::move(invalid)), std::runtime_error);
    EXPECT_EQ(cloud.pointsRevision(), revision);
    EXPECT_EQ(std::as_const(cloud).points().data(), data);
}

TEST(PointCloudTest, PointReplacementRejectsCountThatInvalidatesAttributes)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::MatrixXf;

    Cloud cloud(2);
    Matrix normals(2, 3);
    normals.setConstant(1.0f);
    cloud.setNormals(std::move(normals));
    const auto revision = cloud.pointsRevision();
    const auto* const data = std::as_const(cloud).points().data();
    Matrix replacement(3, 3);

    EXPECT_THROW(cloud.setPoints(std::move(replacement)), std::runtime_error);
    EXPECT_EQ(cloud.pointsRevision(), revision);
    EXPECT_EQ(std::as_const(cloud).points().data(), data);
    ASSERT_NE(cloud.normals(), nullptr);
    EXPECT_EQ(cloud.normals()->rows(), 2);
}

TEST(PointCloudTest, MoveAssignmentReplacesPointRevisionIdentity)
{
    using Cloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;

    Cloud source(3);
    static_cast<void>(source.points());
    static_cast<void>(source.points());
    const auto source_revision = source.pointsRevision();
    const auto* const source_data = std::as_const(source).points().data();

    Cloud destination(1);
    const auto destination_revision = destination.pointsRevision();
    destination = std::move(source);

    EXPECT_NE(destination.pointsRevision(), destination_revision);
    EXPECT_EQ(destination.pointsRevision(), source_revision);
    EXPECT_EQ(std::as_const(destination).points().data(), source_data);
}

#ifdef PLAPOINT_WITH_CUDA
static void writeGpuPoint(plamatrix::internal::ResidentMatrix<float>& points,
                          plamatrix::Index row, int column, float value)
{
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        points.data() + static_cast<std::size_t>(column) * points.rows() + row,
        &value, sizeof(value), cudaMemcpyHostToDevice));
}

TEST(PointCloudTest, GpuPointsCpuCachesAndInvalidatesOnMutablePointAccess)
{
    if (!hasCudaDevice()) { GTEST_SKIP() << "No CUDA device, skipping GPU point CPU cache test"; }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cpu_cloud(2);
    cpu_cloud.points().operator()(0, 0) = 1.0f;
    cpu_cloud.points().operator()(1, 0) = 2.0f;
    auto gpu_cloud = cpu_cloud.toGpu();

    const auto& first = gpu_cloud.pointsCpu();
    const auto& second = gpu_cloud.pointsCpu();
    EXPECT_EQ(first.data(), second.data());
    EXPECT_FLOAT_EQ(second.operator()(1, 0), 2.0f);

    writeGpuPoint(gpu_cloud.points(), 1, 0, 9.0f);
    const auto& refreshed = gpu_cloud.pointsCpu();
    EXPECT_FLOAT_EQ(refreshed.operator()(1, 0), 9.0f);
}

TEST(PointCloudTest, GpuPointsCpuRefreshesAfterRetainedMutableAliasWrites)
{
    if (!hasCudaDevice()) { GTEST_SKIP() << "No CUDA device, skipping GPU point CPU cache test"; }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cpu_cloud(2);
    auto gpu_cloud = cpu_cloud.toGpu();
    auto& retained_points = gpu_cloud.points();
    writeGpuPoint(retained_points, 1, 0, 3.0f);
    EXPECT_FLOAT_EQ(gpu_cloud.pointsCpu().operator()(1, 0), 3.0f);

    writeGpuPoint(retained_points, 1, 0, 9.0f);
    EXPECT_FLOAT_EQ(gpu_cloud.pointsCpu().operator()(1, 0), 9.0f);
}

TEST(PointCloudTest, GpuSetPointsCopyInvalidatesCpuMirrorAndOwnsDeviceStorage)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU point copy test";
    }

    using CpuMatrix = plamatrix::MatrixXf;

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> initial(2);
    auto gpu_cloud = initial.toGpu();
    static_cast<void>(gpu_cloud.pointsCpu());

    CpuMatrix replacement_cpu(2, 3);
    replacement_cpu.operator()(0, 0) = 1.0f;
    replacement_cpu.operator()(1, 0) = 2.0f;
    replacement_cpu.operator()(0, 1) = 3.0f;
    replacement_cpu.operator()(1, 1) = 4.0f;
    replacement_cpu.operator()(0, 2) = 5.0f;
    replacement_cpu.operator()(1, 2) = 6.0f;
    auto replacement_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(
        replacement_cpu, gpu_cloud.executionContext());

    const auto revision = gpu_cloud.pointsRevision();
    gpu_cloud.setPoints(std::as_const(replacement_gpu));

    EXPECT_GT(gpu_cloud.pointsRevision(), revision);
    EXPECT_NE(std::as_const(gpu_cloud).points().data(), replacement_gpu.data());
    EXPECT_FLOAT_EQ(gpu_cloud.pointsCpu().operator()(1, 2), 6.0f);

    writeGpuPoint(replacement_gpu, 1, 2, 99.0f);
    const auto stored_cpu = gpu_cloud.toCpu();
    EXPECT_FLOAT_EQ(stored_cpu.points().operator()(1, 2), 6.0f);
}

TEST(PointCloudTest, GpuSetPointsCopyHandlesEmptyNx3Matrix)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping empty GPU point copy test";
    }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> empty_cpu;
    auto cloud = empty_cpu.toGpu();
    auto empty_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(
        std::as_const(empty_cpu).points(), cloud.executionContext());
    const auto revision = cloud.pointsRevision();

    cloud.setPoints(std::as_const(empty_gpu));

    EXPECT_EQ(cloud.size(), 0U);
    EXPECT_EQ(std::as_const(cloud).points().cols(), 3);
    EXPECT_EQ(std::as_const(cloud).points().data(), nullptr);
    EXPECT_GT(cloud.pointsRevision(), revision);
}
#endif

TEST(PointCloudTest, MoveFromMatrix)
{
    plamatrix::MatrixXf mat(50, 3);
    mat.setConstant(3.0f);
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(std::move(mat));
    EXPECT_EQ(cloud.size(), 50);
    EXPECT_FLOAT_EQ(cloud.points().operator()(0, 0), 3.0f);
}

TEST(PointCloudTest, RejectsNonNx3Matrix)
{
    using CloudType = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    plamatrix::MatrixXf bad(5, 4);
    EXPECT_THROW(CloudType cloud(std::move(bad)), std::runtime_error);
}
