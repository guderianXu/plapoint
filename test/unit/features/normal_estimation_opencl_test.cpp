#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_OPENCL

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <plapoint/features/normal_estimation.h>
#include <plapoint/opencl/opencl_runtime.h>
#include <plamatrix/internal/core/device.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif

namespace
{

    plapoint::GeometryCloud<float> makePlane()
    {
        constexpr int side = 6;
        plamatrix::MatrixXf points(side * side, 3);
        for (int y = 0; y < side; ++y)
        {
            for (int x = 0; x < side; ++x)
            {
                const int row = y * side + x;
                points(row, 0) = static_cast<float>(x) * 0.1f;
                points(row, 1) = static_cast<float>(y) * 0.1f;
                points(row, 2) = 0.25f * points(row, 0) - 0.1f * points(row, 1);
            }
        }
        return plapoint::GeometryCloud<float>(std::move(points));
    }

} // namespace

TEST(NormalEstimationOpenClTest, MatchesCpuPlaneNormals)
{
    if (!plapoint::opencl::hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU device";
    }
    const auto cloud = makePlane();
    const auto cpu = plapoint::estimateNormals(
        cloud, 8, plapoint::ProcessingDevice::CPU);
    plapoint::ProcessingReport report;
    const auto result = plapoint::estimateNormals(
        cloud, 8, plapoint::ProcessingDevice::OpenCL, &report);

    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::OpenCL);
    EXPECT_EQ(report.neighborBackend, plapoint::ProcessingNeighborBackend::OpenClUniformGrid);
    EXPECT_FALSE(report.usedFallback);
    for (plamatrix::Index row = 0; row < result.rows(); ++row)
    {
        const float dot = result(row, 0) * cpu(row, 0)
            + result(row, 1) * cpu(row, 1)
            + result(row, 2) * cpu(row, 2);
        EXPECT_NEAR(std::abs(dot), 1.0f, 1e-4f);
    }
}

TEST(NormalEstimationOpenClTest, RejectsNonFiniteInputLikeCpuBackend)
{
    if (!plapoint::opencl::hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU device";
    }
    auto cloud = makePlane();
    cloud.points()(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(
        plapoint::estimateNormals(cloud, 8, plapoint::ProcessingDevice::CPU),
        std::invalid_argument);
    EXPECT_THROW(
        plapoint::estimateNormals(cloud, 8, plapoint::ProcessingDevice::OpenCL),
        std::invalid_argument);
}

#endif // PLAPOINT_WITH_OPENCL
