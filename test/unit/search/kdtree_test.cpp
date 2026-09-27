#include <gtest/gtest.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

class KdTreeTest : public ::testing::Test
{
protected:
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using KdTree = plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>;

    void SetUp() override
    {
        auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(6, 3);
        // 6 points: two clusters at origin and (10,10,10)
        mat.operator()(0, 0) = 0;
        mat.operator()(0, 1) = 0;
        mat.operator()(0, 2) = 0;
        mat.operator()(1, 0) = 1;
        mat.operator()(1, 1) = 0;
        mat.operator()(1, 2) = 0;
        mat.operator()(2, 0) = 0; mat.operator()(2, 1) = 1; mat.operator()(2, 2) = 0;
        mat.operator()(3, 0) = 10; mat.operator()(3, 1) = 10; mat.operator()(3, 2) = 10;
        mat.operator()(4, 0) = 11; mat.operator()(4, 1) = 10; mat.operator()(4, 2) = 10;
        mat.operator()(5, 0) = 10; mat.operator()(5, 1) = 11; mat.operator()(5, 2) = 10;
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

TEST_F(KdTreeTest, SetInputCloudReplacesBuiltTreeAndSearchesWithoutBuild)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    auto replacement_points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    replacement_points.operator()(0, 0) = Scalar(100);
    replacement_points.operator()(0, 1) = Scalar(100);
    replacement_points.operator()(0, 2) = Scalar(100);
    replacement_points.operator()(1, 0) = Scalar(101);
    replacement_points.operator()(1, 1) = Scalar(100);
    replacement_points.operator()(1, 2) = Scalar(100);
    auto replacement_cloud = std::make_shared<Cloud>(std::move(replacement_points));

    EXPECT_TRUE(tree.setInputCloud(replacement_cloud));

    plamatrix::Matrix<Scalar, 3, 1> old_query{0, 0, 0};
    EXPECT_TRUE(tree.radiusSearch(old_query, Scalar(1)).empty());

    plamatrix::Matrix<Scalar, 3, 1> new_query{100, 100, 100};
    const auto rebuilt_results = tree.nearestKSearch(new_query, 1);
    ASSERT_EQ(rebuilt_results.size(), 1u);
    EXPECT_EQ(rebuilt_results[0], 0);
}

TEST_F(KdTreeTest, PointMutationInvalidatesAndRebuildsTreeOnNextSearch)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();
    ASSERT_TRUE(tree.isBuilt());

    cloud->points().operator()(0, 0) = Scalar(100);

    EXPECT_FALSE(tree.isBuilt());
    const plamatrix::Matrix<Scalar, 3, 1> query{100, 0, 0};
    EXPECT_EQ(tree.nearestKSearch(query, 1), (std::vector<int>{0}));
    ASSERT_TRUE(tree.isBuilt());
    EXPECT_EQ(tree.radiusSearch(query, Scalar(1)), (std::vector<int>{0}));
}

TEST(KdTreeSnapshotTest, RetainedPointAliasMutationRefreshesCpuSnapshot)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Tree = plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>;

    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
    points.setConstant(Scalar(0));
    points.operator()(1, 0) = Scalar(10);
    points.operator()(2, 0) = Scalar(20);
    auto retained_alias_cloud = std::make_shared<Cloud>(std::move(points));
    auto& retained_alias = retained_alias_cloud->points();

    Tree tree;
    tree.setInputCloud(retained_alias_cloud);
    tree.build();

    retained_alias.operator()(0, 0) = Scalar(100);

    ASSERT_FALSE(tree.isBuilt());
    EXPECT_EQ(
        tree.nearestKSearch(plamatrix::Matrix<Scalar, 3, 1>{100, 0, 0}, 1),
        (std::vector<int>{0}));
    EXPECT_TRUE(tree.isBuilt());

    retained_alias.operator()(0, 0) = Scalar(-100);
    EXPECT_FALSE(tree.isBuilt());
    EXPECT_EQ(
        tree.nearestKSearch(plamatrix::Matrix<Scalar, 3, 1>{20, 0, 0}, 1),
        (std::vector<int>{2}));
}

TEST(KdTreeSnapshotTest, ScopedPointEditInvalidatesOnceAndRestoresCacheReuse)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Tree = plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>;

    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    points.setConstant(Scalar(0));
    points.operator()(1, 0) = Scalar(10);
    auto cloud = std::make_shared<Cloud>(std::move(points));

    Tree tree;
    tree.setInputCloud(cloud);
    tree.build();
    ASSERT_TRUE(tree.isBuilt());

    {
        auto edit = cloud->editPoints();
        EXPECT_FALSE(tree.isBuilt());
        edit->operator()(0, 0) = Scalar(20);
    }

    EXPECT_FALSE(tree.isBuilt());
    EXPECT_EQ(
        tree.nearestKSearch(plamatrix::Matrix<Scalar, 3, 1>{20, 0, 0}, 1),
        (std::vector<int>{0}));
    EXPECT_TRUE(tree.isBuilt());
    EXPECT_TRUE(cloud->pointCachesReusable());
}

TEST_F(KdTreeTest, SearchAfterSetInputCloudBuildsOnDemand)
{
    KdTree tree;
    EXPECT_TRUE(tree.setInputCloud(cloud));

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    EXPECT_FALSE(tree.isBuilt());
    EXPECT_EQ(tree.nearestKSearch(query, 1), (std::vector<int>{0}));
    EXPECT_TRUE(tree.isBuilt());
    EXPECT_EQ(tree.radiusSearch(query, Scalar(1)).size(), 3u);

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(1, 3);
    queries.setConstant(0);
    EXPECT_EQ(tree.batchNearestKSearch(queries, 1), (std::vector<std::vector<int>>{{0}}));
}

TEST_F(KdTreeTest, NullInputReportsFailureAndClearsSearchState)
{
    KdTree tree;
    EXPECT_TRUE(tree.setInputCloud(cloud));
    EXPECT_FALSE(tree.isBuilt());
    EXPECT_FALSE(tree.setInputCloud(nullptr));

    const auto query = plamatrix::Vector3f::Zero();
    EXPECT_THROW(tree.nearestKSearch(query, 1), std::runtime_error);
}

TEST_F(KdTreeTest, PclContractSubsetSearchReturnsOriginalCloudIndices)
{
    KdTree tree;
    auto subset = std::make_shared<const KdTree::Indices>(KdTree::Indices{1, 3, 5});
    ASSERT_TRUE(tree.setInputCloud(cloud, subset));
    EXPECT_EQ(tree.getInputCloud(), cloud);
    EXPECT_EQ(tree.getIndices(), subset);

    const plamatrix::Vector3f query(0.0f, 0.0f, 0.0f);
    std::vector<int> indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(query, 3, indices, squared_distances), 3);
    EXPECT_EQ(indices, (std::vector<int>{1, 3, 5}));
    EXPECT_EQ(squared_distances, (std::vector<float>{1.0f, 300.0f, 321.0f}));
    EXPECT_EQ(tree.radiusSearch(query, 2.0, indices, squared_distances), 1);
    EXPECT_EQ(indices, (std::vector<int>{1}));
    EXPECT_EQ(squared_distances, (std::vector<float>{1.0f}));
}

TEST_F(KdTreeTest, InvalidSubsetPreservesPreviousInput)
{
    KdTree tree;
    ASSERT_TRUE(tree.setInputCloud(cloud));
    const auto invalid = std::make_shared<const KdTree::Indices>(KdTree::Indices{6});
    EXPECT_THROW(tree.setInputCloud(cloud, invalid), std::out_of_range);
    EXPECT_EQ(tree.getInputCloud(), cloud);
    EXPECT_FALSE(tree.getIndices());
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

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    auto results = tree.nearestKSearch(query, 3);
    ASSERT_EQ(results.size(), 3u);
    // points 0,1,2 are closest to origin
}

TEST_F(KdTreeTest, NearestKSearchFillsPclContractIndicesAndSquaredDistances)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    const auto query = plamatrix::Vector3f::Zero();
    std::vector<int> indices{99};
    std::vector<Scalar> squared_distances{Scalar(-1)};
    const int found = tree.nearestKSearch(query, 3, indices, squared_distances);

    EXPECT_EQ(found, 3);
    EXPECT_EQ(indices, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(squared_distances, (std::vector<Scalar>{0, 1, 1}));

    EXPECT_EQ(tree.nearestKSearch(query, 0, indices, squared_distances), 0);
    EXPECT_TRUE(indices.empty());
    EXPECT_TRUE(squared_distances.empty());
}

TEST_F(KdTreeTest, PclContractSearchPreservesOutputsOnInvalidQuery)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    const plamatrix::Vector3f query(
        std::numeric_limits<Scalar>::quiet_NaN(), Scalar(0), Scalar(0));
    std::vector<int> indices{42};
    std::vector<Scalar> squared_distances{Scalar(7)};
    EXPECT_THROW(tree.nearestKSearch(query, 3, indices, squared_distances), std::invalid_argument);
    EXPECT_THROW(tree.radiusSearch(query, Scalar(1), indices, squared_distances), std::invalid_argument);
    EXPECT_EQ(indices, (std::vector<int>{42}));
    EXPECT_EQ(squared_distances, (std::vector<Scalar>{7}));
}

TEST_F(KdTreeTest, EmptyCloudBuildsAndReturnsNoSearchResults)
{
    auto empty_cloud = std::make_shared<Cloud>(0);

    KdTree tree;
    tree.setInputCloud(empty_cloud);
    EXPECT_NO_THROW(tree.build());

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    EXPECT_TRUE(tree.nearestKSearch(query, 3).empty());
    EXPECT_TRUE(tree.radiusSearch(query, Scalar(1)).empty());

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(2, 3);
    queries.setConstant(0);
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

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    auto results = tree.nearestKSearch(query, 50);
    std::sort(results.begin(), results.end());

    EXPECT_EQ(results, (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST_F(KdTreeTest, NearestKSearchReturnsDuplicatePointTiesBeforeFartherPoint)
{
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(4, 3);
    mat.operator()(0, 0) = 0;
    mat.operator()(0, 1) = 0;
    mat.operator()(0, 2) = 0;
    mat.operator()(1, 0) = 0;
    mat.operator()(1, 1) = 0;
    mat.operator()(1, 2) = 0;
    mat.operator()(2, 0) = 0;
    mat.operator()(2, 1) = 0;
    mat.operator()(2, 2) = 0;
    mat.operator()(3, 0) = 5; mat.operator()(3, 1) = 0; mat.operator()(3, 2) = 0;
    auto duplicate_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(duplicate_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    auto results = tree.nearestKSearch(query, 3);
    std::sort(results.begin(), results.end());

    EXPECT_EQ(results, (std::vector<int>{0, 1, 2}));
}

TEST_F(KdTreeTest, NearestKSearchBreaksDistanceTiesByInputIndex)
{
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(4, 3);
    mat.operator()(0, 0) = -1;
    mat.operator()(0, 1) = 0;
    mat.operator()(0, 2) = 0;
    mat.operator()(1, 0) = 1;
    mat.operator()(1, 1) = 0;
    mat.operator()(1, 2) = 0;
    mat.operator()(2, 0) = 0;
    mat.operator()(2, 1) = -1;
    mat.operator()(2, 2) = 0;
    mat.operator()(3, 0) = 0;  mat.operator()(3, 1) = 1; mat.operator()(3, 2) = 0;
    auto tied_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(tied_cloud);
    tree.build();

    const plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    EXPECT_EQ(tree.nearestKSearch(query, 1), (std::vector<int>{0}));
    EXPECT_EQ(tree.nearestKSearch(query, 3), (std::vector<int>{0, 1, 2}));
}

TEST_F(KdTreeTest, NearestKSearchKeepsExtremeButFiniteDistance)
{
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    mat.operator()(0, 0) = Scalar(0);
    mat.operator()(0, 1) = Scalar(0);
    mat.operator()(0, 2) = Scalar(0);
    auto extreme_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(extreme_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{max_value, 0, 0};
    const auto results = tree.nearestKSearch(query, 1);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], 0);
}

TEST_F(KdTreeTest, PclContractSearchKeepsNeighborWhenSquaredDistanceOverflows)
{
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();
    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    points.setConstant(Scalar(0));
    auto extreme_cloud = std::make_shared<Cloud>(std::move(points));

    KdTree tree;
    tree.setInputCloud(extreme_cloud);
    tree.build();

    const plamatrix::Vector3f query(max_value, Scalar(0), Scalar(0));
    std::vector<int> indices;
    std::vector<Scalar> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(query, 1, indices, squared_distances), 1);
    EXPECT_EQ(indices, (std::vector<int>{0}));
    ASSERT_EQ(squared_distances.size(), 1u);
    EXPECT_TRUE(std::isinf(squared_distances[0]));
}

TEST_F(KdTreeTest, RadiusSearch)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    auto results = tree.radiusSearch(query, Scalar(2.0));
    // Should find points 0,1,2 within radius 2 of origin
    ASSERT_EQ(results.size(), 3u);
}

TEST_F(KdTreeTest, RadiusSearchFillsPclContractDistancesAndAppliesMaxNn)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    const auto query = plamatrix::Vector3f::Zero();
    std::vector<int> indices;
    std::vector<Scalar> squared_distances;
    EXPECT_EQ(tree.radiusSearch(query, Scalar(2), indices, squared_distances, 2), 2);
    EXPECT_EQ(indices, (std::vector<int>{0, 1}));
    EXPECT_EQ(squared_distances, (std::vector<Scalar>{0, 1}));

    EXPECT_EQ(tree.radiusSearch(query, Scalar(2), indices, squared_distances), 3);
    EXPECT_EQ(indices, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(squared_distances, (std::vector<Scalar>{0, 1, 1}));
}

TEST_F(KdTreeTest, PclContractRadiusUsesDoublePrecision)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    const auto query = plamatrix::Vector3f::Zero();
    const double radius_just_below_one = std::nextafter(1.0, 0.0);
    std::vector<int> indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.radiusSearch(query, radius_just_below_one, indices, squared_distances), 1);
    EXPECT_EQ(indices, (std::vector<int>{0}));
    EXPECT_EQ(squared_distances, (std::vector<float>{0.0f}));
}

TEST(KdTreePclContractTest, DoubleCloudReturnsFloatSquaredDistances)
{
    using Cloud = plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>;
    using Tree = plapoint::search::internal::DeviceKdTree<double, plamatrix::internal::Device::CPU>;

    auto points = plamatrix::MatrixXd(2, 3);
    points.setConstant(0.0);
    points(1, 0) = 2.0;
    auto cloud = std::make_shared<Cloud>(std::move(points));
    Tree tree;
    tree.setInputCloud(cloud);
    tree.build();

    const auto query = plamatrix::Vector3d::Zero();
    std::vector<int> indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(query, 2, indices, squared_distances), 2);
    EXPECT_EQ(indices, (std::vector<int>{0, 1}));
    EXPECT_EQ(squared_distances, (std::vector<float>{0.0f, 4.0f}));
}

TEST_F(KdTreeTest, RadiusSearchIncludesPointsExactlyOnBoundary)
{
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(4, 3);
    mat.operator()(0, 0) = 0;
    mat.operator()(0, 1) = 0;
    mat.operator()(0, 2) = 0;
    mat.operator()(1, 0) = 1;
    mat.operator()(1, 1) = 0;
    mat.operator()(1, 2) = 0;
    mat.operator()(2, 0) = 0;
    mat.operator()(2, 1) = 1;
    mat.operator()(2, 2) = 0;
    mat.operator()(3, 0) = 1.0001f; mat.operator()(3, 1) = 0; mat.operator()(3, 2) = 0;
    auto boundary_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(boundary_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    auto results = tree.radiusSearch(query, Scalar(1));
    std::sort(results.begin(), results.end());

    EXPECT_EQ(results, (std::vector<int>{0, 1, 2}));
}

TEST_F(KdTreeTest, RadiusSearchRejectsNegativeRadius)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{0, 0, 0};
    EXPECT_THROW(tree.radiusSearch(query, Scalar(-1)), std::invalid_argument);
}

TEST_F(KdTreeTest, NearestKSearchRejectsNonFiniteQuery)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{std::numeric_limits<Scalar>::quiet_NaN(), 0, 0};
    EXPECT_THROW(tree.nearestKSearch(query, 1), std::invalid_argument);
}

TEST_F(KdTreeTest, RadiusSearchRejectsNonFiniteQuery)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{0, std::numeric_limits<Scalar>::infinity(), 0};
    EXPECT_THROW(tree.radiusSearch(query, Scalar(1)), std::invalid_argument);
}

TEST_F(KdTreeTest, BatchNearestKSearchRejectsNonFiniteQueries)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(2, 3);
    queries.setConstant(0);
    queries.operator()(1, 2) = std::numeric_limits<Scalar>::quiet_NaN();

    EXPECT_THROW(tree.batchNearestKSearch(queries, 1), std::invalid_argument);
}

TEST_F(KdTreeTest, RadiusSearchUsesFiniteDistanceForExtremeCoordinates)
{
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    mat.operator()(0, 0) = -max_value;
    mat.operator()(0, 1) = 0;
    mat.operator()(0, 2) = 0;
    mat.operator()(1, 0) = Scalar(0);
    mat.operator()(1, 1) = 0;
    mat.operator()(1, 2) = 0;
    auto extreme_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(extreme_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{max_value, 0, 0};
    const auto results = tree.radiusSearch(query, max_value);

    EXPECT_EQ(std::find(results.begin(), results.end(), 0), results.end());
    EXPECT_NE(std::find(results.begin(), results.end(), 1), results.end());
}

TEST_F(KdTreeTest, RadiusSearchTraversesBothSidesWhenSplitDistanceIsNonFinite)
{
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(3, 3);
    mat.operator()(0, 0) = Scalar(0);
    mat.operator()(0, 1) = Scalar(0);
    mat.operator()(0, 2) = Scalar(0);
    mat.operator()(1, 0) = nan;
    mat.operator()(1, 1) = Scalar(0);
    mat.operator()(1, 2) = Scalar(0);
    mat.operator()(2, 0) = nan;
    mat.operator()(2, 1) = Scalar(1);
    mat.operator()(2, 2) = Scalar(0);
    auto mixed_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(mixed_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, 3, 1> query{Scalar(0), Scalar(0), Scalar(0)};
    const auto results = tree.radiusSearch(query, Scalar(0.5));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], 0);
}

TEST_F(KdTreeTest, BatchNearestKSearchRejectsNon3ColumnQueries)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(1, 2);
    queries(0, 0) = Scalar(0);
    queries(0, 1) = Scalar(0);

    EXPECT_THROW(tree.batchNearestKSearch(queries, 1), std::invalid_argument);
}

TEST_F(KdTreeTest, BatchNearestKSearchWithoutBuildThrows)
{
    KdTree tree;
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(2, 3);
    queries.setConstant(0);

    EXPECT_THROW(tree.batchNearestKSearch(queries, 2), std::runtime_error);
}

TEST_F(KdTreeTest, BatchNearestKSearchClampsKToPointCount)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(1, 3);
    queries(0, 0) = Scalar(0);
    queries(0, 1) = Scalar(0);
    queries(0, 2) = Scalar(0);

    auto results = tree.batchNearestKSearch(queries, 50);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), cloud->size());
}

TEST_F(KdTreeTest, BatchNearestKSearchMatchesIndividualSearchOrderAcrossRows)
{
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(5, 3);
    mat.operator()(0, 0) = Scalar(0);
    mat.operator()(0, 1) = Scalar(0);
    mat.operator()(0, 2) = Scalar(0);
    mat.operator()(1, 0) = Scalar(2);
    mat.operator()(1, 1) = Scalar(0);
    mat.operator()(1, 2) = Scalar(0);
    mat.operator()(2, 0) = Scalar(5);
    mat.operator()(2, 1) = Scalar(0);
    mat.operator()(2, 2) = Scalar(0);
    mat.operator()(3, 0) = Scalar(9);  mat.operator()(3, 1) = Scalar(0); mat.operator()(3, 2) = Scalar(0);
    mat.operator()(4, 0) = Scalar(14); mat.operator()(4, 1) = Scalar(0); mat.operator()(4, 2) = Scalar(0);
    auto line_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(line_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(3, 3);
    queries.operator()(0, 0) = Scalar(4);
    queries.operator()(0, 1) = Scalar(0);
    queries.operator()(0, 2) = Scalar(0);
    queries.operator()(1, 0) = Scalar(13);
    queries.operator()(1, 1) = Scalar(0);
    queries.operator()(1, 2) = Scalar(0);
    queries.operator()(2, 0) = Scalar(0.25);
    queries.operator()(2, 1) = Scalar(0);
    queries.operator()(2, 2) = Scalar(0);

    const auto results = tree.batchNearestKSearch(queries, 3);

    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], (std::vector<int>{2, 1, 0}));
    EXPECT_EQ(results[1], (std::vector<int>{4, 3, 2}));
    EXPECT_EQ(results[2], (std::vector<int>{0, 1, 2}));
}

TEST_F(KdTreeTest, BatchNearestKSearchDropsInvalidInfiniteDistances)
{
    constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    mat.operator()(0, 0) = infinity;
    mat.operator()(0, 1) = Scalar(0);
    mat.operator()(0, 2) = Scalar(0);
    auto far_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(far_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(1, 3);
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

    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    mat.operator()(0, 0) = Scalar(0);
    mat.operator()(0, 1) = Scalar(0);
    mat.operator()(0, 2) = Scalar(0);
    auto extreme_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(extreme_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(1, 3);
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
    auto mat = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    mat.operator()(0, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    mat.operator()(0, 1) = Scalar(0);
    mat.operator()(0, 2) = Scalar(0);
    mat.operator()(1, 0) = Scalar(0);
    mat.operator()(1, 1) = Scalar(0);
    mat.operator()(1, 2) = Scalar(0);
    auto mixed_cloud = std::make_shared<Cloud>(std::move(mat));

    KdTree tree;
    tree.setInputCloud(mixed_cloud);
    tree.build();

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> queries(1, 3);
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
