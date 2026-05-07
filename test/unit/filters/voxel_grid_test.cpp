#include <gtest/gtest.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(VoxelGridTest, DownsamplesUniformGrid)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // 8 points forming a 2x2x2 cube
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(8, 3);
    int idx = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
            {
                mat.setValue(idx, 0, Scalar(x));
                mat.setValue(idx, 1, Scalar(y));
                mat.setValue(idx, 2, Scalar(z));
                ++idx;
            }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
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
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    mat.setValue(0, 0, 1.0f);
    mat.setValue(0, 1, 2.0f);
    mat.setValue(0, 2, 3.0f);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(0.1), Scalar(0.1), Scalar(0.1));

    Cloud output;
    vg.filter(output);
    EXPECT_EQ(output.size(), 1u);
}

TEST(VoxelGridTest, ThrowsOnZeroLeafSize)
{
    plapoint::VoxelGrid<float, plamatrix::Device::CPU> vg;
    EXPECT_THROW(vg.setLeafSize(0, 1, 1), std::invalid_argument);
}
