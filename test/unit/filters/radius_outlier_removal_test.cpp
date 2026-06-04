#include <gtest/gtest.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

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
#endif
