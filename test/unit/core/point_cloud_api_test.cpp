#include <gtest/gtest.h>
#include <plapoint/exceptions.h>
#include <plapoint/point_cloud.h>

#include <memory>
#include <type_traits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

TEST(PointCloudApiTest, PointXYZCloudExposesPclPointAndCloudShape)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    Cloud::Ptr cloud = std::make_shared<Cloud>();
    cloud->push_back(plapoint::PointXYZ(1.0f, 2.0f, 3.0f));
    cloud->emplace_back(4.0f, 5.0f, 6.0f);

    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_EQ(cloud->width, 2u);
    EXPECT_EQ(cloud->height, 1u);
    EXPECT_FALSE(cloud->isOrganized());
    EXPECT_FLOAT_EQ(cloud->points[0].x, 1.0f);
    EXPECT_FLOAT_EQ((*cloud)[1].z, 6.0f);

    const Cloud::ConstPtr copy = cloud->makeShared();
    EXPECT_EQ(copy->points[0].y, 2.0f);
}

TEST(PointCloudApiTest, OrganizedCloudIndexesByColumnAndRow)
{
    plapoint::PointCloud<plapoint::PointXYZ> cloud(2u, 2u);
    ASSERT_EQ(cloud.points.size(), 4u);
    EXPECT_TRUE(cloud.isOrganized());

    cloud.at(1u, 1u) = plapoint::PointXYZ(7.0f, 8.0f, 9.0f);
    EXPECT_FLOAT_EQ(cloud.points[3].z, 9.0f);
    EXPECT_FLOAT_EQ(cloud(1u, 1u).z, 9.0f);
    EXPECT_THROW(cloud.at(2u, 1u), std::out_of_range);

    cloud.resize(2u);
    EXPECT_EQ(cloud.width, 2u);
    EXPECT_EQ(cloud.height, 1u);
}

TEST(PointCloudApiTest, ContainerAccessAndOrganizationFollowPclSemantics)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    static_assert(std::is_same_v<Cloud::value_type, plapoint::PointXYZ>);
    static_assert(std::is_same_v<Cloud::reference, plapoint::PointXYZ&>);
    static_assert(std::is_same_v<Cloud::VectorType,
                                 std::vector<plapoint::PointXYZ, Eigen::aligned_allocator<plapoint::PointXYZ>>>);

    Cloud cloud(2u, 2u);
    cloud.resize(4u);
    EXPECT_TRUE(cloud.isOrganized());
    EXPECT_EQ(cloud.data(), cloud.points.data());
    EXPECT_EQ(&cloud.front(), &cloud.points.front());
    EXPECT_EQ(&cloud.back(), &cloud.points.back());
    EXPECT_EQ(cloud.rbegin(), cloud.points.rbegin());

    cloud.is_dense = false;
    cloud.clear();
    EXPECT_FALSE(cloud.is_dense);
    EXPECT_EQ(cloud.height, 0u);

    cloud.push_back(plapoint::PointXYZ(1.0f, 2.0f, 3.0f));
    EXPECT_THROW(cloud.at(0, 0), plapoint::UnorganizedPointCloudException);
    cloud.resize(2u, 2u);
    EXPECT_TRUE(cloud.isOrganized());
}

TEST(PointCloudApiTest, SubsetPreservesHeaderAndSensorPose)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    Cloud cloud;
    cloud.header.frame_id = "camera";
    cloud.sensor_origin_ = Eigen::Vector4f(1.0f, 2.0f, 3.0f, 0.0f);
    cloud.sensor_orientation_ = Eigen::Quaternionf::Identity();
    cloud.push_back(plapoint::PointXYZ(1.0f, 0.0f, 0.0f));
    cloud.push_back(plapoint::PointXYZ(2.0f, 0.0f, 0.0f));

    Cloud subset(cloud, plapoint::Indices{1});
    ASSERT_EQ(subset.size(), 1u);
    EXPECT_FLOAT_EQ(subset[0].x, 2.0f);
    EXPECT_EQ(subset.header.frame_id, "camera");
    EXPECT_TRUE(subset.sensor_origin_.isApprox(cloud.sensor_origin_));
}

TEST(PointCloudApiTest, UserDefinedPointTypeUsesPointContainer)
{
    struct PointWithIntensity
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float intensity = 0.0f;
    };
    plapoint::PointCloud<PointWithIntensity> cloud;
    cloud.push_back(PointWithIntensity{1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_EQ(cloud.size(), 1u);
    EXPECT_FLOAT_EQ(cloud.points[0].intensity, 4.0f);
}

TEST(PointCloudApiTest, ConcatenationKeepsPclMetadataRules)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    Cloud first;
    first.header.stamp = 10;
    first.header.frame_id = "map";
    first.push_back(plapoint::PointXYZ(1.0f, 0.0f, 0.0f));

    Cloud second;
    second.header.stamp = 20;
    second.is_dense = false;
    second.push_back(plapoint::PointXYZ(2.0f, 0.0f, 0.0f));

    const Cloud combined = first + second;
    ASSERT_EQ(combined.size(), 2u);
    EXPECT_EQ(combined.header.stamp, 20u);
    EXPECT_EQ(combined.header.frame_id, "map");
    EXPECT_FALSE(combined.is_dense);
    EXPECT_EQ(combined.width, 2u);
    EXPECT_EQ(combined.height, 1u);
    EXPECT_EQ(first.size(), 1u);
}

TEST(PointCloudApiTest, NormalAndColorTypesKeepNamedFields)
{
    plapoint::PointCloud<plapoint::PointXYZRGB> colors;
    colors.push_back(plapoint::PointXYZRGB(1.0f, 2.0f, 3.0f, 10, 20, 30));
    EXPECT_EQ(colors[0].g, 20);

    plapoint::PointCloud<plapoint::Normal> normals(1u);
    normals[0].normal_z = 1.0f;
    normals[0].curvature = 0.2f;
    EXPECT_FLOAT_EQ(normals.points[0].normal_z, 1.0f);
    EXPECT_FLOAT_EQ(normals.points[0].curvature, 0.2f);
}

TEST(PointCloudApiTest, PointTypesExposePclLayoutAndEigenMaps)
{
    plapoint::PointXYZ xyz(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(xyz.data[3], 1.0f);
    xyz.getVector3fMap() = Eigen::Vector3f(4.0f, 5.0f, 6.0f);
    EXPECT_FLOAT_EQ(xyz.x, 4.0f);
    EXPECT_FLOAT_EQ(xyz.getVector4fMap().w(), 1.0f);

    plapoint::PointXYZRGB rgb(1.0f, 2.0f, 3.0f, 10, 20, 30);
    EXPECT_EQ(rgb.rgba, 0xff0a141eu);
    EXPECT_TRUE(rgb.getRGBVector3i().isApprox(Eigen::Vector3i(10, 20, 30)));

    plapoint::Normal normal(0.0f, 0.0f, 1.0f, 0.25f);
    normal.getNormalVector3fMap() = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(normal.normal_x, 1.0f);
    EXPECT_FLOAT_EQ(normal.curvature, 0.25f);

    plapoint::PointNormal point_normal(1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 1.0f, 0.5f);
    EXPECT_FLOAT_EQ(point_normal.data[3], 1.0f);
    EXPECT_FLOAT_EQ(point_normal.getNormalVector4fMap().w(), 0.0f);
    EXPECT_FLOAT_EQ(point_normal.curvature, 0.5f);
}

TEST(PointCloudApiTest, PointContainerMutationSignaturesPreserveShapeRules)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    static_assert(std::is_same_v<decltype(std::declval<const Cloud&>().max_size()), plapoint::index_t>);

    Cloud cloud(2u, 2u);
    cloud.resize(4, Point(1.0f, 0.0f, 0.0f));
    EXPECT_TRUE(cloud.isOrganized());
    cloud.assign(2, 2, Point(2.0f, 0.0f, 0.0f));
    EXPECT_TRUE(cloud.isOrganized());
    EXPECT_FLOAT_EQ(cloud[3].x, 2.0f);

    cloud.assign({Point(1.0f, 0.0f, 0.0f), Point(2.0f, 0.0f, 0.0f),
                  Point(3.0f, 0.0f, 0.0f), Point(4.0f, 0.0f, 0.0f)}, 2);
    EXPECT_EQ(cloud.width, 2u);
    EXPECT_EQ(cloud.height, 2u);
    cloud.insert(cloud.begin() + 1, Point(5.0f, 0.0f, 0.0f));
    EXPECT_EQ(cloud.width, 5u);
    EXPECT_EQ(cloud.height, 1u);
    cloud.erase(cloud.begin() + 1, cloud.begin() + 3);
    EXPECT_EQ(cloud.width, 3u);
    EXPECT_EQ(cloud.height, 1u);
    EXPECT_FLOAT_EQ(cloud[1].x, 3.0f);

    cloud.assign(2, Point(7.0f, 0.0f, 0.0f));
    cloud.transient_push_back(Point(8.0f, 0.0f, 0.0f));
    EXPECT_EQ(cloud.width, 2u);
    EXPECT_EQ(cloud.points.size(), 3u);
    cloud.width = 3;
    cloud.transient_emplace_back(9.0f, 0.0f, 0.0f);
    EXPECT_EQ(cloud.width, 3u);
    cloud.width = 4;
    EXPECT_EQ(cloud.size(), 4u);
}

TEST(PointCloudApiTest, FloatMatrixMapExposesPointStorage)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    Cloud cloud;
    cloud.emplace_back(1.0f, 2.0f, 3.0f);
    cloud.emplace_back(4.0f, 5.0f, 6.0f);

    auto coordinates = cloud.getMatrixXfMap(3, 4, 0);
    EXPECT_EQ(coordinates.rows(), 3);
    EXPECT_EQ(coordinates.cols(), 2);
    EXPECT_FLOAT_EQ(coordinates(1, 1), 5.0f);
    coordinates(2, 0) = 7.0f;
    EXPECT_FLOAT_EQ(cloud[0].z, 7.0f);

    const Cloud& const_cloud = cloud;
    const auto mapped = const_cloud.getMatrixXfMap(3, 4, 0);
    EXPECT_FLOAT_EQ(mapped(2, 0), 7.0f);
    EXPECT_EQ(cloud.getMatrixXfMap().cols(), 2);
    EXPECT_THROW(cloud.getMatrixXfMap(3, 3, 1), std::invalid_argument);
}
