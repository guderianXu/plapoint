#include <gtest/gtest.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(NormalEstimationTest, PlaneNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // Points on the XY plane (z=0) => normals should be approximately (0,0,±1)
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(9, 3);
    int idx = 0;
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
        {
            mat.setValue(idx, 0, Scalar(x));
            mat.setValue(idx, 1, Scalar(y));
            mat.setValue(idx, 2, 0);
            ++idx;
        }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(8);

    auto normals = ne.compute();
    EXPECT_EQ(normals.rows(), 9);
    EXPECT_EQ(normals.cols(), 3);

    // Center point normal should be approximately (0,0,1) or (0,0,-1)
    Scalar z = normals.getValue(4, 2);
    EXPECT_GT(std::abs(z), Scalar(0.9));
}

TEST(NormalEstimationTest, ThrowsIfNoInput)
{
    plapoint::NormalEstimation<float, plamatrix::Device::CPU> ne;
    EXPECT_THROW(ne.compute(), std::runtime_error);
}
