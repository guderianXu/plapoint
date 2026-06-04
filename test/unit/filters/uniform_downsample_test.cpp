#include <gtest/gtest.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>

static bool hasCudaDeviceForUniformDownsample()
{
    return plapoint::gpu::hasUsableCudaDevice();
}
#endif

TEST(UniformDownsampleTest, KeepsEveryNthPoint)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(10, 3);
    for (int i = 0; i < 10; ++i) { pts.setValue(i, 0, Scalar(i)); pts.setValue(i, 1, 0); pts.setValue(i, 2, 0); }
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(3);

    Cloud output;
    ud.filter(output);
    // Every 3rd: indices 0, 3, 6, 9 = 4 points
    EXPECT_EQ(output.size(), 4u);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(UniformDownsampleTest, GpuInputKeepsEveryNthPointAndNormals)
{
    if (!hasCudaDeviceForUniformDownsample())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU uniform downsample test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(5, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, 0);
        pts.setValue(i, 2, 0);
        normals.setValue(i, 0, 0);
        normals.setValue(i, 1, 0);
        normals.setValue(i, 2, Scalar(i + 1));
    }
    CpuCloud cpu_cloud(std::move(pts));
    cpu_cloud.setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::UniformDownsample<Scalar, plamatrix::Device::GPU> ud;
    ud.setInputCloud(gpu_cloud);
    ud.setStep(2);

    GpuCloud output;
    ud.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 3u);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(2, 0), 4.0f);
    ASSERT_TRUE(cpu_output.hasNormals());
    EXPECT_FLOAT_EQ(cpu_output.normals()->getValue(2, 2), 5.0f);
}
#endif
