#include <gtest/gtest.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <algorithm>
#include <limits>
#include <vector>

class KdTreeTest : public ::testing::Test
{
protected:
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using KdTree = plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>;

    void SetUp() override
    {
        auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(6, 3);
        // 6 points: two clusters at origin and (10,10,10)
        mat.setValue(0, 0, 0); mat.setValue(0, 1, 0); mat.setValue(0, 2, 0);
        mat.setValue(1, 0, 1); mat.setValue(1, 1, 0); mat.setValue(1, 2, 0);
        mat.setValue(2, 0, 0); mat.setValue(2, 1, 1); mat.setValue(2, 2, 0);
        mat.setValue(3, 0, 10); mat.setValue(3, 1, 10); mat.setValue(3, 2, 10);
        mat.setValue(4, 0, 11); mat.setValue(4, 1, 10); mat.setValue(4, 2, 10);
        mat.setValue(5, 0, 10); mat.setValue(5, 1, 11); mat.setValue(5, 2, 10);
        cloud = std::make_shared<Cloud>(std::move(mat));
    }

    std::shared_ptr<Cloud> cloud;
};

TEST_F(KdTreeTest, BuildAndSize)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();
    // tree built without exception
}

TEST_F(KdTreeTest, ThrowsIfNoInput)
{
    KdTree tree;
    EXPECT_THROW(tree.build(), std::runtime_error);
}

TEST_F(KdTreeTest, NearestKSearchSinglePoint)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    auto results = tree.nearestKSearch(query, 3);
    ASSERT_EQ(results.size(), 3u);
    // points 0,1,2 are closest to origin
}

TEST_F(KdTreeTest, EmptyCloudBuildsAndReturnsNoSearchResults)
{
    auto empty_cloud = std::make_shared<Cloud>(0);

    KdTree tree;
    tree.setInputCloud(empty_cloud);
    EXPECT_NO_THROW(tree.build());

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    EXPECT_TRUE(tree.nearestKSearch(query, 3).empty());
    EXPECT_TRUE(tree.radiusSearch(query, Scalar(1)).empty());

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(2, 3);
    queries.fill(0);
    const auto batch_results = tree.batchNearestKSearch(queries, 3);
    ASSERT_EQ(batch_results.size(), 2u);
    EXPECT_TRUE(batch_results[0].empty());
    EXPECT_TRUE(batch_results[1].empty());
}

TEST_F(KdTreeTest, NearestKSearchClampsKToPointCount)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    auto results = tree.nearestKSearch(query, 50);
    std::sort(results.begin(), results.end());

    EXPECT_EQ(results, (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST_F(KdTreeTest, NearestKSearchReturnsDuplicatePointTiesBeforeFartherPoint)
{
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    mat.setValue(0, 0, 0); mat.setValue(0, 1, 0); mat.setValue(0, 2, 0);
    mat.setValue(1, 0, 0); mat.setValue(1, 1, 0); mat.setValue(1, 2, 0);
    mat.setValue(2, 0, 0); mat.setValue(2, 1, 0); mat.setValue(2, 2, 0);
    mat.setValue(3, 0, 5); mat.setValue(3, 1, 0); mat.setValue(3, 2, 0);
    auto duplicate_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(duplicate_cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    auto results = tree.nearestKSearch(query, 3);
    std::sort(results.begin(), results.end());

    EXPECT_EQ(results, (std::vector<int>{0, 1, 2}));
}

TEST_F(KdTreeTest, NearestKSearchKeepsExtremeButFiniteDistance)
{
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    mat.setValue(0, 0, Scalar(0));
    mat.setValue(0, 1, Scalar(0));
    mat.setValue(0, 2, Scalar(0));
    auto extreme_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(extreme_cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{max_value, 0, 0};
    const auto results = tree.nearestKSearch(query, 1);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], 0);
}

TEST_F(KdTreeTest, RadiusSearch)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    auto results = tree.radiusSearch(query, Scalar(2.0));
    // Should find points 0,1,2 within radius 2 of origin
    ASSERT_EQ(results.size(), 3u);
}

TEST_F(KdTreeTest, RadiusSearchIncludesPointsExactlyOnBoundary)
{
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    mat.setValue(0, 0, 0);       mat.setValue(0, 1, 0); mat.setValue(0, 2, 0);
    mat.setValue(1, 0, 1);       mat.setValue(1, 1, 0); mat.setValue(1, 2, 0);
    mat.setValue(2, 0, 0);       mat.setValue(2, 1, 1); mat.setValue(2, 2, 0);
    mat.setValue(3, 0, 1.0001f); mat.setValue(3, 1, 0); mat.setValue(3, 2, 0);
    auto boundary_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(boundary_cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    auto results = tree.radiusSearch(query, Scalar(1));
    std::sort(results.begin(), results.end());

    EXPECT_EQ(results, (std::vector<int>{0, 1, 2}));
}

TEST_F(KdTreeTest, RadiusSearchRejectsNegativeRadius)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    EXPECT_THROW(tree.radiusSearch(query, Scalar(-1)), std::invalid_argument);
}

TEST_F(KdTreeTest, RadiusSearchUsesFiniteDistanceForExtremeCoordinates)
{
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    mat.setValue(0, 0, -max_value); mat.setValue(0, 1, 0); mat.setValue(0, 2, 0);
    mat.setValue(1, 0, Scalar(0)); mat.setValue(1, 1, 0); mat.setValue(1, 2, 0);
    auto extreme_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(extreme_cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{max_value, 0, 0};
    const auto results = tree.radiusSearch(query, max_value);

    EXPECT_EQ(std::find(results.begin(), results.end(), 0), results.end());
    EXPECT_NE(std::find(results.begin(), results.end(), 1), results.end());
}

TEST_F(KdTreeTest, RadiusSearchTraversesBothSidesWhenSplitDistanceIsNonFinite)
{
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(3, 3);
    mat.setValue(0, 0, Scalar(0)); mat.setValue(0, 1, Scalar(0)); mat.setValue(0, 2, Scalar(0));
    mat.setValue(1, 0, nan); mat.setValue(1, 1, Scalar(0)); mat.setValue(1, 2, Scalar(0));
    mat.setValue(2, 0, nan); mat.setValue(2, 1, Scalar(1)); mat.setValue(2, 2, Scalar(0));
    auto mixed_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(mixed_cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{Scalar(0), Scalar(0), Scalar(0)};
    const auto results = tree.radiusSearch(query, Scalar(0.5));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], 0);
}

TEST_F(KdTreeTest, BatchNearestKSearchRejectsNon3ColumnQueries)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 2);
    queries(0, 0) = Scalar(0);
    queries(0, 1) = Scalar(0);

    EXPECT_THROW(tree.batchNearestKSearch(queries, 1), std::invalid_argument);
}

TEST_F(KdTreeTest, BatchNearestKSearchWithoutInputReturnsEmptyRows)
{
    KdTree tree;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(2, 3);

    auto results = tree.batchNearestKSearch(queries, 2);
    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0].empty());
    EXPECT_TRUE(results[1].empty());
}

TEST_F(KdTreeTest, BatchNearestKSearchClampsKToPointCount)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = Scalar(0);
    queries(0, 1) = Scalar(0);
    queries(0, 2) = Scalar(0);

    auto results = tree.batchNearestKSearch(queries, 50);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), cloud->size());
}

TEST_F(KdTreeTest, BatchNearestKSearchMatchesIndividualSearchOrderAcrossRows)
{
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    mat.setValue(0, 0, Scalar(0));  mat.setValue(0, 1, Scalar(0)); mat.setValue(0, 2, Scalar(0));
    mat.setValue(1, 0, Scalar(2));  mat.setValue(1, 1, Scalar(0)); mat.setValue(1, 2, Scalar(0));
    mat.setValue(2, 0, Scalar(5));  mat.setValue(2, 1, Scalar(0)); mat.setValue(2, 2, Scalar(0));
    mat.setValue(3, 0, Scalar(9));  mat.setValue(3, 1, Scalar(0)); mat.setValue(3, 2, Scalar(0));
    mat.setValue(4, 0, Scalar(14)); mat.setValue(4, 1, Scalar(0)); mat.setValue(4, 2, Scalar(0));
    auto line_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(line_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(3, 3);
    queries.setValue(0, 0, Scalar(4));  queries.setValue(0, 1, Scalar(0)); queries.setValue(0, 2, Scalar(0));
    queries.setValue(1, 0, Scalar(13)); queries.setValue(1, 1, Scalar(0)); queries.setValue(1, 2, Scalar(0));
    queries.setValue(2, 0, Scalar(0.25)); queries.setValue(2, 1, Scalar(0)); queries.setValue(2, 2, Scalar(0));

    const auto results = tree.batchNearestKSearch(queries, 3);

    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], (std::vector<int>{2, 1, 0}));
    EXPECT_EQ(results[1], (std::vector<int>{4, 3, 2}));
    EXPECT_EQ(results[2], (std::vector<int>{0, 1, 2}));
}

TEST_F(KdTreeTest, BatchNearestKSearchDropsInvalidInfiniteDistances)
{
    constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    mat.setValue(0, 0, infinity);
    mat.setValue(0, 1, Scalar(0));
    mat.setValue(0, 2, Scalar(0));
    auto far_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(far_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = Scalar(0);
    queries(0, 1) = Scalar(0);
    queries(0, 2) = Scalar(0);

    const auto results = tree.batchNearestKSearch(queries, 1);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST_F(KdTreeTest, BatchNearestKSearchKeepsExtremeButFiniteDistance)
{
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    mat.setValue(0, 0, Scalar(0));
    mat.setValue(0, 1, Scalar(0));
    mat.setValue(0, 2, Scalar(0));
    auto extreme_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(extreme_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = max_value;
    queries(0, 1) = Scalar(0);
    queries(0, 2) = Scalar(0);

    const auto results = tree.batchNearestKSearch(queries, 1);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0], 0);
}

TEST_F(KdTreeTest, BatchNearestKSearchSkipsInvalidDistanceAndReturnsFiniteCandidate)
{
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    mat.setValue(0, 0, std::numeric_limits<Scalar>::quiet_NaN());
    mat.setValue(0, 1, Scalar(0));
    mat.setValue(0, 2, Scalar(0));
    mat.setValue(1, 0, Scalar(0));
    mat.setValue(1, 1, Scalar(0));
    mat.setValue(1, 2, Scalar(0));
    auto mixed_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(mixed_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = Scalar(0);
    queries(0, 1) = Scalar(0);
    queries(0, 2) = Scalar(0);

    const auto results = tree.batchNearestKSearch(queries, 1);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0], 1);
}

TEST(KdTreeSizeCheckTest, CheckedSizeProductRejectsOverflow)
{
    EXPECT_EQ(plapoint::search::detail::checkedSizeProduct(7, 3, "test buffer"), 21u);
    EXPECT_THROW(
        plapoint::search::detail::checkedSizeProduct(
            std::numeric_limits<std::size_t>::max(), 2, "test buffer"),
        std::overflow_error);
}

TEST(KdTreeOrderingTest, PointCoordinateLessHandlesNonFiniteValuesDeterministically)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(plapoint::search::detail::pointCoordinateLess(0.0f, 0, nan, 1));
    EXPECT_FALSE(plapoint::search::detail::pointCoordinateLess(nan, 1, 0.0f, 0));
    EXPECT_TRUE(plapoint::search::detail::pointCoordinateLess(nan, 1, nan, 2));
    EXPECT_FALSE(plapoint::search::detail::pointCoordinateLess(nan, 2, nan, 1));
}
