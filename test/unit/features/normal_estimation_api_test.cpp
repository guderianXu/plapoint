#include <gtest/gtest.h>
#include <plapoint/features/normal_3d.h>

#include <cmath>
#include <limits>
#include <memory>

TEST(NormalEstimationApiTest, ComputesPointNormalAndCurvatureUsingPclSignatures)
{
    using InputCloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<InputCloud>();
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            cloud->emplace_back(static_cast<float>(x), static_cast<float>(y), 1.0f);
        }
    }

    plapoint::NormalEstimation<plapoint::PointXYZ, plapoint::PointNormal> estimation;
    estimation.setInputCloud(cloud);
    estimation.setKSearch(9);
    plapoint::PointCloud<plapoint::PointNormal> output;
    estimation.compute(output);

    ASSERT_EQ(output.size(), 9u);
    EXPECT_TRUE(output.is_dense);
    EXPECT_NEAR(output.points[4].normal_x, 0.0f, 1.0e-5f);
    EXPECT_NEAR(output.points[4].normal_y, 0.0f, 1.0e-5f);
    EXPECT_NEAR(output.points[4].normal_z, -1.0f, 1.0e-5f);
    EXPECT_NEAR(output.points[4].curvature, 0.0f, 1.0e-5f);
    EXPECT_FLOAT_EQ(output.points[4].x, 1.0f);
}

TEST(NormalEstimationApiTest, RadiusSearchMarksInsufficientNeighborhoodAsInvalid)
{
    using InputCloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<InputCloud>();
    cloud->push_back(plapoint::PointXYZ(0.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(10.0f, 0.0f, 0.0f));

    auto tree = std::make_shared<plapoint::search::KdTree<plapoint::PointXYZ>>();
    ASSERT_TRUE(tree->setInputCloud(cloud));
    plapoint::Feature<plapoint::PointXYZ, plapoint::Normal>::Ptr estimation =
        std::make_shared<plapoint::NormalEstimation<plapoint::PointXYZ, plapoint::Normal>>();
    estimation->setInputCloud(cloud);
    estimation->setSearchMethod(tree);
    estimation->setRadiusSearch(1.0);
    EXPECT_EQ(estimation->getSearchMethod(), tree);
    EXPECT_DOUBLE_EQ(estimation->getSearchParameter(), 1.0);
    plapoint::PointCloud<plapoint::Normal> output;
    estimation->compute(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_FALSE(output.is_dense);
    EXPECT_TRUE(std::isnan(output.points[0].normal_x));
    EXPECT_TRUE(std::isnan(output.points[1].curvature));
}

TEST(NormalEstimationApiTest, NonFiniteInputProducesInvalidNormal)
{
    using InputCloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<InputCloud>();
    cloud->push_back(plapoint::PointXYZ(0.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(1.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(0.0f, 1.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f));
    cloud->is_dense = false;

    plapoint::NormalEstimation<plapoint::PointXYZ> estimation;
    estimation.setInputCloud(cloud);
    estimation.setKSearch(3);
    plapoint::PointCloud<plapoint::Normal> output;
    estimation.compute(output);

    ASSERT_EQ(output.size(), 4u);
    EXPECT_FALSE(output.is_dense);
    EXPECT_TRUE(std::isnan(output[3].normal_x));
}

TEST(NormalEstimationApiTest, InitializesProvidedSearchTreeFromInputCloud)
{
    using InputCloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<InputCloud>();
    cloud->emplace_back(0.0f, 0.0f, 1.0f);
    cloud->emplace_back(1.0f, 0.0f, 1.0f);
    cloud->emplace_back(0.0f, 1.0f, 1.0f);

    auto tree = std::make_shared<plapoint::search::KdTree<plapoint::PointXYZ>>();
    plapoint::NormalEstimation<plapoint::PointXYZ> estimation;
    estimation.setInputCloud(cloud);
    estimation.setSearchMethod(tree);
    estimation.setKSearch(3);
    plapoint::PointCloud<plapoint::Normal> output;
    estimation.compute(output);

    ASSERT_EQ(output.size(), 3u);
    EXPECT_EQ(tree->getInputCloud(), cloud);
    EXPECT_TRUE(std::isfinite(output[0].normal_z));
}

TEST(NormalEstimationApiTest, SearchSurfaceIndicesAndSensorViewpointFollowPclCalls)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto input = std::make_shared<Cloud>();
    input->header.frame_id = "camera";
    input->sensor_origin_ = Eigen::Vector4f(0.0f, 0.0f, 2.0f, 0.0f);
    input->emplace_back(0.0f, 0.0f, 1.0f);
    input->emplace_back(8.0f, 8.0f, 8.0f);

    auto surface = std::make_shared<Cloud>();
    surface->emplace_back(0.0f, 0.0f, 1.0f);
    surface->emplace_back(1.0f, 0.0f, 1.0f);
    surface->emplace_back(0.0f, 1.0f, 1.0f);
    auto indices = std::make_shared<const plapoint::Indices>(plapoint::Indices{0});

    plapoint::NormalEstimation<plapoint::PointXYZ, plapoint::PointNormal> estimation;
    estimation.setInputCloud(input);
    estimation.setSearchSurface(surface);
    estimation.setIndices(indices);
    estimation.setKSearch(3);
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    estimation.getViewPoint(x, y, z);
    EXPECT_FLOAT_EQ(z, 2.0f);

    plapoint::PointCloud<plapoint::PointNormal> output;
    estimation.compute(output);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_GT(output[0].normal_z, 0.9f);
    EXPECT_EQ(output.header.frame_id, "camera");
    EXPECT_TRUE(output.sensor_origin_.isApprox(input->sensor_origin_));
}

TEST(NormalEstimationApiTest, FreeNormalUtilitiesMatchPclCallShape)
{
    plapoint::PointCloud<plapoint::PointXYZ> cloud;
    cloud.emplace_back(0.0f, 0.0f, 1.0f);
    cloud.emplace_back(1.0f, 0.0f, 1.0f);
    cloud.emplace_back(0.0f, 1.0f, 1.0f);

    Eigen::Vector4f plane;
    float curvature = 1.0f;
    ASSERT_TRUE(plapoint::computePointNormal(cloud, plane, curvature));
    EXPECT_NEAR(std::abs(plane.z()), 1.0f, 1.0e-5f);
    EXPECT_NEAR(curvature, 0.0f, 1.0e-5f);

    Eigen::Vector3f normal(0.0f, 0.0f, -1.0f);
    plapoint::flipNormalTowardsViewpoint(cloud[0], 0.0f, 0.0f, 2.0f, normal);
    EXPECT_GT(normal.z(), 0.0f);

    plapoint::PointCloud<plapoint::Normal> normals;
    normals.emplace_back(0.0f, 0.0f, 1.0f);
    normals.emplace_back(0.0f, 0.0f, 1.0f);
    normal = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    ASSERT_TRUE(plapoint::flipNormalTowardsNormalsMean(normals, plapoint::Indices{0, 1}, normal));
    EXPECT_GT(normal.z(), 0.0f);
}
