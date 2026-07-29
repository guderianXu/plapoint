#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

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
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(100);
    EXPECT_EQ(cloud.size(), 100);
    EXPECT_EQ(cloud.points().rows(), 100);
    EXPECT_EQ(cloud.points().cols(), 3);
}

TEST(PointCloudTest, DefaultConstructsEmptyNx3Cloud)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud;

    EXPECT_EQ(cloud.size(), 0);
    EXPECT_EQ(cloud.points().rows(), 0);
    EXPECT_EQ(cloud.points().cols(), 3);
    EXPECT_THROW((void)cloud[0], std::out_of_range);
}

TEST(PointCloudTest, GpuTransfer)
{
    if (!hasCudaDevice()) { GTEST_SKIP() << "No CUDA device, skipping GPU transfer test"; }
    plapoint::PointCloud<float, plamatrix::Device::CPU> cpu_cloud(10);
    cpu_cloud.points().fill(1.0f);

    auto gpu_cloud = cpu_cloud.toGpu();
    EXPECT_EQ(gpu_cloud.size(), 10);

    auto cpu_cloud_back = gpu_cloud.toCpu();
    EXPECT_FLOAT_EQ(cpu_cloud_back.points().getValue(0, 0), 1.0f);
}

TEST(PointCloudTest, PointsCpuReturnsCpuPointStorage)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(2);
    cloud.points().setValue(0, 0, 1.0f);
    cloud.points().setValue(1, 0, 2.0f);

    const auto& points = cloud.pointsCpu();
    ASSERT_EQ(points.rows(), 2);
    ASSERT_EQ(points.cols(), 3);
    EXPECT_FLOAT_EQ(points.getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(points.getValue(1, 0), 2.0f);
}

TEST(PointCloudTest, MutablePointAccessIncrementsPointsVersion)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

    Cloud cloud(2);
    const auto initial_version = cloud.pointsVersion();

    const auto& const_cloud = static_cast<const Cloud&>(cloud);
    (void)const_cloud.points();
    EXPECT_EQ(cloud.pointsVersion(), initial_version);

    cloud.points().setValue(0, 0, 3.0f);
    EXPECT_GT(cloud.pointsVersion(), initial_version);

    const auto after_mutable_access = cloud.pointsVersion();
    (void)cloud.pointsCpu();
    EXPECT_EQ(cloud.pointsVersion(), after_mutable_access);
}

TEST(PointCloudTest, PointRevisionTracksOnlyPositionMutations)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

    Cloud cloud(2);
    const auto initial_revision = cloud.pointsRevision();
    EXPECT_NE(initial_revision, 0U);

    const auto& read_only_points = std::as_const(cloud).points();
    static_cast<void>(read_only_points);
    EXPECT_EQ(cloud.pointsRevision(), initial_revision);

    Matrix normals(2, 3);
    normals.fill(1.0f);
    cloud.setNormals(std::move(normals));
    EXPECT_EQ(cloud.pointsRevision(), initial_revision);

    auto& mutable_points = cloud.points();
    EXPECT_GT(cloud.pointsRevision(), initial_revision);
    mutable_points(0, 0) = 3.0f;

    const auto before_replacement = cloud.pointsRevision();
    Matrix replacement(2, 3);
    replacement.fill(2.0f);
    cloud.setPoints(std::move(replacement));
    EXPECT_GT(cloud.pointsRevision(), before_replacement);
    EXPECT_EQ(cloud.size(), 2U);
    EXPECT_FLOAT_EQ(std::as_const(cloud).points()(1, 2), 2.0f);
}

TEST(PointCloudTest, FailedPointReplacementKeepsRevisionAndStorage)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

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
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

    Cloud cloud(2);
    Matrix normals(2, 3);
    normals.fill(1.0f);
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
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

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
TEST(PointCloudTest, GpuPointsCpuCachesAndInvalidatesOnMutablePointAccess)
{
    if (!hasCudaDevice()) { GTEST_SKIP() << "No CUDA device, skipping GPU point CPU cache test"; }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cpu_cloud(2);
    cpu_cloud.points().setValue(0, 0, 1.0f);
    cpu_cloud.points().setValue(1, 0, 2.0f);
    auto gpu_cloud = cpu_cloud.toGpu();

    const auto& first = gpu_cloud.pointsCpu();
    const auto& second = gpu_cloud.pointsCpu();
    EXPECT_EQ(first.data(), second.data());
    EXPECT_FLOAT_EQ(second.getValue(1, 0), 2.0f);

    gpu_cloud.points().setValue(1, 0, 9.0f);
    const auto& refreshed = gpu_cloud.pointsCpu();
    EXPECT_FLOAT_EQ(refreshed.getValue(1, 0), 9.0f);
}

TEST(PointCloudTest, GpuPointsCpuRefreshesAfterRetainedMutableAliasWrites)
{
    if (!hasCudaDevice()) { GTEST_SKIP() << "No CUDA device, skipping GPU point CPU cache test"; }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cpu_cloud(2);
    auto gpu_cloud = cpu_cloud.toGpu();
    auto& retained_points = gpu_cloud.points();
    retained_points.setValue(1, 0, 3.0f);
    EXPECT_FLOAT_EQ(gpu_cloud.pointsCpu().getValue(1, 0), 3.0f);

    retained_points.setValue(1, 0, 9.0f);
    EXPECT_FLOAT_EQ(gpu_cloud.pointsCpu().getValue(1, 0), 9.0f);
}
#endif

TEST(PointCloudTest, MoveFromMatrix)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> mat(50, 3);
    mat.fill(3.0f);
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(mat));
    EXPECT_EQ(cloud.size(), 50);
    EXPECT_FLOAT_EQ(cloud.points().getValue(0, 0), 3.0f);
}

TEST(PointCloudTest, RejectsNonNx3Matrix)
{
    using CloudType = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> bad(5, 4);
    EXPECT_THROW(CloudType cloud(std::move(bad)), std::runtime_error);
}
