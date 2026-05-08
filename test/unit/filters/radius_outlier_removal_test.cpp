#include <gtest/gtest.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

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
