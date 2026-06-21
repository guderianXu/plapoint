#include <gtest/gtest.h>

#include <plapoint/search/spatial_kdtree.h>

#include <algorithm>
#include <vector>

TEST(SpatialKdTreeTest, FindsNearestAndRadiusMatchesInTwoDimensions)
{
    using Tree = plapoint::search::SpatialKdTree<2, double>;
    std::vector<Tree::Point> points = {
        {{0.0, 0.0}, 10},
        {{5.0, 0.0}, 20},
        {{1.0, 1.0}, 30},
        {{8.0, 8.0}, 40},
    };

    Tree tree(points);

    double distance = 0.0;
    EXPECT_EQ(tree.nearest({0.9, 1.1}, &distance), 30);
    EXPECT_NEAR(distance, 0.141421356, 1e-6);

    auto within = tree.radiusSearch({0.0, 0.0}, 1.5);
    std::sort(within.begin(), within.end());
    ASSERT_EQ(within.size(), 2u);
    EXPECT_EQ(within[0], 10);
    EXPECT_EQ(within[1], 30);
}

TEST(SpatialKdTreeTest, SkipsSourcePointForNearestByPointIndex)
{
    using Tree = plapoint::search::SpatialKdTree<3, double>;
    std::vector<Tree::CoordinateArray> points = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},
    };

    Tree tree(points);

    double distance = 0.0;
    EXPECT_EQ(tree.nearestByPointIndex(0, &distance), 1);
    EXPECT_NEAR(distance, 1.0, 1e-12);
    EXPECT_EQ(tree.nearestByPointIndex(10), -1);
}

TEST(SpatialKdTreeTest, ReportsKNearestAndRadiusCount)
{
    using Tree = plapoint::search::SpatialKdTree<2, float>;
    std::vector<Tree::CoordinateArray> points = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 2.0f},
        {4.0f, 4.0f},
    };

    Tree tree(points);

    const auto nearest = tree.kNearest({0.0f, 0.0f}, 3);
    ASSERT_EQ(nearest.size(), 3u);
    EXPECT_EQ(nearest[0].index, 0);
    EXPECT_EQ(nearest[1].index, 1);
    EXPECT_EQ(nearest[2].index, 2);

    EXPECT_EQ(tree.radiusCount({0.0f, 0.0f}, 2.1f), 3);
    EXPECT_EQ(tree.radiusCount({0.0f, 0.0f}, 2.1f, 2), 2);
}
