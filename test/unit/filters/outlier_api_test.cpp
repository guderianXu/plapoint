#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/search/search.h>

TEST(OutlierApiTest, RadiusFilterKeepsFieldsAndSupportsNegativeSelection)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZRGB>;
    auto cloud = std::make_shared<Cloud>();
    cloud->header.frame_id = "sensor";
    cloud->emplace_back(0.0f, 0.0f, 0.0f, 10, 20, 30);
    cloud->emplace_back(0.1f, 0.0f, 0.0f, 40, 50, 60);
    cloud->emplace_back(10.0f, 0.0f, 0.0f, 70, 80, 90);

    plapoint::RadiusOutlierRemoval<plapoint::PointXYZRGB> filter(true);
    filter.setInputCloud(cloud);
    filter.setRadiusSearch(0.5);
    filter.setMinNeighborsInRadius(1);
    Cloud output;
    filter.filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_EQ(output[0].r, 10);
    EXPECT_EQ(output[1].g, 50);
    EXPECT_EQ(output.header.frame_id, "sensor");
    EXPECT_EQ(*filter.getRemovedIndices(), (plapoint::Indices{2}));

    filter.setNegative(true);
    filter.filter(output);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].b, 90);
    EXPECT_EQ(*filter.getRemovedIndices(), (plapoint::Indices{0, 1}));
}

TEST(OutlierApiTest, StatisticalFilterUsesPclParameterNamesAndIndicesOutput)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    cloud->emplace_back(0.1f, 0.0f, 0.0f);
    cloud->emplace_back(0.2f, 0.0f, 0.0f);
    cloud->emplace_back(0.3f, 0.0f, 0.0f);
    cloud->emplace_back(10.0f, 0.0f, 0.0f);

    plapoint::StatisticalOutlierRemoval<plapoint::PointXYZ> filter(true);
    filter.setInputCloud(cloud);
    filter.setMeanK(2);
    filter.setStddevMulThresh(0.5);
    EXPECT_EQ(filter.getMeanK(), 2);
    EXPECT_DOUBLE_EQ(filter.getStddevMulThresh(), 0.5);
    plapoint::Indices retained;
    filter.filter(retained);
    EXPECT_EQ(retained, (plapoint::Indices{0, 1, 2, 3}));
    EXPECT_EQ(*filter.getRemovedIndices(), (plapoint::Indices{4}));
}

TEST(OutlierApiTest, KeepOrganizedRetainsShapeAndMarksRemovedPoint)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<Cloud>(2u, 2u);
    cloud->points[0] = plapoint::PointXYZ(0.0f, 0.0f, 0.0f);
    cloud->points[1] = plapoint::PointXYZ(0.1f, 0.0f, 0.0f);
    cloud->points[2] = plapoint::PointXYZ(0.2f, 0.0f, 0.0f);
    cloud->points[3] = plapoint::PointXYZ(10.0f, 0.0f, 0.0f);

    plapoint::RadiusOutlierRemoval<plapoint::PointXYZ> filter;
    filter.setInputCloud(cloud);
    filter.setRadiusSearch(0.5);
    filter.setMinNeighborsInRadius(1);
    filter.setKeepOrganized(true);
    Cloud output;
    filter.filter(output);

    EXPECT_EQ(output.width, 2u);
    EXPECT_EQ(output.height, 2u);
    EXPECT_EQ(output.size(), 4u);
    EXPECT_FALSE(output.is_dense);
    EXPECT_TRUE(std::isnan(output.points[3].x));
}

TEST(OutlierApiTest, SearchInjectionAndInputIndicesUseTheFullCloudAsTheSearchSurface)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    auto cloud = std::make_shared<Cloud>();
    for (const float x : {0.0f, 0.1f, 0.2f, 2.0f, 100.0f})
    {
        cloud->emplace_back(x, 0.0f, 0.0f);
    }
    auto subset = std::make_shared<const plapoint::Indices>(plapoint::Indices{0, 3});
    plapoint::search::Search<Point>::Ptr searcher = std::make_shared<plapoint::search::KdTree<Point>>();

    plapoint::StatisticalOutlierRemoval<Point> statistical;
    statistical.setInputCloud(cloud);
    statistical.setIndices(subset);
    statistical.setSearchMethod(searcher);
    statistical.setMeanK(1);
    statistical.setStddevMulThresh(0.0);
    plapoint::Indices retained;
    statistical.filter(retained);
    EXPECT_EQ(retained, (plapoint::Indices{0}));

    plapoint::RadiusOutlierRemoval<Point> radius;
    radius.setInputCloud(cloud);
    radius.setIndices(subset);
    radius.setSearchMethod(searcher);
    radius.setRadiusSearch(0.25);
    radius.setMinNeighborsInRadius(1);
    radius.filter(retained);
    EXPECT_EQ(retained, (plapoint::Indices{0}));
}
