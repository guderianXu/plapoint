#include <gtest/gtest.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

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
