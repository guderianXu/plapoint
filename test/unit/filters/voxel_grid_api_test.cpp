#include <gtest/gtest.h>
#include <plapoint/filters/voxel_grid.h>

#include <memory>
#include <type_traits>

TEST(VoxelGridApiTest, PointXYZInputAndOutputUsePclSignatures)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    Cloud::Ptr cloud = std::make_shared<Cloud>();
    cloud->push_back(plapoint::PointXYZ(0.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(0.2f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(2.0f, 0.0f, 0.0f));
    cloud->header.frame_id = "test_frame";

    plapoint::VoxelGrid<plapoint::PointXYZ> voxel;
    static_assert(std::is_same_v<decltype(voxel.getLeafSize()), Eigen::Vector3f>);
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(1.0f, 1.0f, 1.0f);
    Cloud output;
    voxel.filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_NEAR(output.points[0].x, 0.1f, 1.0e-6f);
    EXPECT_FLOAT_EQ(output.points[1].x, 2.0f);
    EXPECT_EQ(output.width, 2u);
    EXPECT_EQ(output.height, 1u);
    EXPECT_EQ(output.header.frame_id, "test_frame");
}

TEST(VoxelGridApiTest, PointXYZRGBPreservesAveragedColors)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZRGB>;
    auto cloud = std::make_shared<Cloud>();
    cloud->push_back(plapoint::PointXYZRGB(0.0f, 0.0f, 0.0f, 10, 20, 30));
    cloud->push_back(plapoint::PointXYZRGB(0.2f, 0.0f, 0.0f, 30, 40, 50));

    plapoint::VoxelGrid<plapoint::PointXYZRGB> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
    Cloud output;
    voxel.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].r, 20);
    EXPECT_EQ(output[0].g, 30);
    EXPECT_EQ(output[0].b, 40);
    EXPECT_FLOAT_EQ(voxel.getLeafSize()(0), 1.0f);
    EXPECT_FLOAT_EQ(voxel.getLeafSize()(2), 1.0f);
}

TEST(VoxelGridApiTest, PointXYZdPreservesLargeCoordinatePrecision)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZd>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(1000000000.125, 0.0, 0.0);
    cloud->emplace_back(1000000000.375, 0.0, 0.0);
    cloud->emplace_back(1000000002.25, 0.0, 0.0);

    plapoint::VoxelGrid<plapoint::PointXYZd> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(1.0f, 1.0f, 1.0f);
    Cloud output;
    voxel.filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_DOUBLE_EQ(output[0].x, 1000000000.25);
    EXPECT_DOUBLE_EQ(output[1].x, 1000000002.25);
}

TEST(VoxelGridApiTest, PointNormalAggregatesNormalAndCurvature)
{
    using Cloud = plapoint::PointCloud<plapoint::PointNormal>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f);
    cloud->emplace_back(0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.4f);

    plapoint::VoxelGrid<plapoint::PointNormal> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(1.0f, 1.0f, 1.0f);
    Cloud output;
    voxel.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_NEAR(output[0].x, 0.1f, 1.0e-6f);
    EXPECT_FLOAT_EQ(output[0].normal_z, 1.0f);
    EXPECT_NEAR(output[0].curvature, 0.3f, 1.0e-6f);
}

TEST(VoxelGridApiTest, PolymorphicFilterHonorsInputIndices)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    cloud->emplace_back(0.2f, 0.0f, 0.0f);
    cloud->emplace_back(2.0f, 0.0f, 0.0f);

    auto voxel = std::make_shared<plapoint::VoxelGrid<Point>>();
    voxel->setLeafSize(1.0f, 1.0f, 1.0f);
    plapoint::Filter<Point>::Ptr filter = voxel;
    filter->setInputCloud(cloud);
    filter->setIndices(std::make_shared<const plapoint::Indices>(plapoint::Indices{0, 2}));
    Cloud output;
    filter->filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output[0].x, 0.0f);
    EXPECT_FLOAT_EQ(output[1].x, 2.0f);
}

TEST(VoxelGridApiTest, LeafLayoutMinimumCountAndFieldFilterMatchPclOptions)
{
    using Point = plapoint::PointXYZI;
    using Cloud = plapoint::PointCloud<Point>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f, 1.0f);
    cloud->emplace_back(0.2f, 0.0f, 0.0f, 3.0f);
    cloud->emplace_back(1.2f, 0.0f, 0.0f, 10.0f);

    plapoint::VoxelGrid<Point> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(1.0f, 1.0f, 1.0f);
    voxel.setMinimumPointsNumberPerVoxel(2);
    voxel.setDownsampleAllData(true);
    voxel.setSaveLeafLayout(true);
    voxel.setFilterFieldName("intensity");
    voxel.setFilterLimits(0.0, 5.0);
    Cloud output;
    voxel.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_NEAR(output[0].x, 0.1f, 1.0e-6f);
    EXPECT_NEAR(output[0].intensity, 2.0f, 1.0e-6f);
    EXPECT_EQ(voxel.getCentroidIndex(output[0]), 0);
    EXPECT_EQ(voxel.getCentroidIndexAt(voxel.getGridCoordinates(0.1f, 0.0f, 0.0f)), 0);
    EXPECT_FALSE(voxel.getLeafLayout().empty());
    EXPECT_TRUE(voxel.getSaveLeafLayout());
    EXPECT_EQ(voxel.getMinimumPointsNumberPerVoxel(), 2u);

    voxel.setFilterLimitsNegative(true);
    voxel.setMinimumPointsNumberPerVoxel(1);
    voxel.filter(output);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_FLOAT_EQ(output[0].x, 1.2f);
}
