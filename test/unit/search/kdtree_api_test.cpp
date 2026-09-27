#include <gtest/gtest.h>
#include <plapoint/kdtree/kdtree_flann.h>
#include <plapoint/search/kdtree.h>

#include <array>
#include <future>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

TEST(KdTreeApiTest, PointXYZQueryUsesPclSignature)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    using Tree = plapoint::search::KdTree<plapoint::PointXYZ>;
    static_assert(std::is_base_of_v<plapoint::search::Search<plapoint::PointXYZ>, Tree>);

    Cloud::Ptr cloud = std::make_shared<Cloud>();
    cloud->push_back(plapoint::PointXYZ(0.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(1.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZ(2.0f, 0.0f, 0.0f));

    Tree tree;
    ASSERT_TRUE(tree.setInputCloud(cloud));
    EXPECT_EQ(tree.getInputCloud(), cloud);

    std::vector<int> indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(plapoint::PointXYZ(0.0f, 0.0f, 0.0f), 2, indices, squared_distances), 2);
    EXPECT_EQ(indices, (std::vector<int>{0, 1}));
    EXPECT_EQ(squared_distances, (std::vector<float>{0.0f, 1.0f}));

    EXPECT_EQ(tree.radiusSearch(cloud->points[0], 1.0, indices, squared_distances), 2);
    EXPECT_EQ(indices, (std::vector<int>{0, 1}));
}

TEST(KdTreeApiTest, KdTreeFlannNameUsesTheExistingSearchBackend)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    plapoint::KdTreeFLANN<plapoint::PointXYZ> tree;
    tree.setInputCloud(cloud);
    plapoint::Indices indices;
    std::vector<float> distances;
    EXPECT_EQ(tree.nearestKSearch(cloud->points[0], 1, indices, distances), 1);
    EXPECT_EQ(indices, (plapoint::Indices{0}));
}

TEST(KdTreeApiTest, SubsetResultsKeepOriginalIndices)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZRGB>;
    using Tree = plapoint::search::KdTree<plapoint::PointXYZRGB>;

    auto cloud = std::make_shared<Cloud>();
    cloud->push_back(plapoint::PointXYZRGB(0.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZRGB(1.0f, 0.0f, 0.0f));
    cloud->push_back(plapoint::PointXYZRGB(2.0f, 0.0f, 0.0f));
    auto subset = std::make_shared<const Tree::Indices>(Tree::Indices{1, 2});

    Tree tree;
    ASSERT_TRUE(tree.setInputCloud(cloud, subset));
    std::vector<int> indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(cloud->points[0], 2, indices, squared_distances), 2);
    EXPECT_EQ(indices, (std::vector<int>{1, 2}));
    EXPECT_EQ(squared_distances, (std::vector<float>{1.0f, 4.0f}));
}

TEST(KdTreeApiTest, DoublePrecisionPointExtensionPreservesLargeCoordinates)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZd>;
    auto cloud = std::make_shared<Cloud>();
    cloud->push_back(plapoint::PointXYZd(1.0e9, 0.0, 0.0));
    cloud->push_back(plapoint::PointXYZd(1.0e9 + 0.25, 0.0, 0.0));

    plapoint::search::KdTree<plapoint::PointXYZd> tree;
    ASSERT_TRUE(tree.setInputCloud(cloud));
    std::vector<int> indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(plapoint::PointXYZd(1.0e9 + 0.24, 0.0, 0.0), 1, indices, squared_distances), 1);
    EXPECT_EQ(indices, (std::vector<int>{1}));
    EXPECT_NEAR(squared_distances[0], 0.0001f, 1.0e-5f);
}

TEST(KdTreeApiTest, NonFiniteCloudPointsAreExcludedFromSearch)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
    cloud->emplace_back(1.0f, 0.0f, 0.0f);
    cloud->emplace_back(2.0f, 0.0f, 0.0f);
    cloud->is_dense = false;

    plapoint::search::KdTree<plapoint::PointXYZ> tree;
    ASSERT_TRUE(tree.setInputCloud(cloud));
    std::vector<int> indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(plapoint::PointXYZ(0.0f, 0.0f, 0.0f), 3, indices, squared_distances), 2);
    EXPECT_EQ(indices, (std::vector<int>{1, 2}));
    EXPECT_EQ(squared_distances, (std::vector<float>{1.0f, 4.0f}));
}

TEST(KdTreeApiTest, IndexAndBatchOverloadsUseInputSubsetPositions)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    cloud->emplace_back(1.0f, 0.0f, 0.0f);
    cloud->emplace_back(2.0f, 0.0f, 0.0f);
    auto subset = std::make_shared<const plapoint::Indices>(plapoint::Indices{1, 2});

    plapoint::search::KdTree<plapoint::PointXYZ> tree(true);
    ASSERT_TRUE(tree.setInputCloud(cloud, subset));
    tree.setSortedResults(false);
    tree.setEpsilon(0.1f);
    EXPECT_FALSE(tree.getSortedResults());
    EXPECT_FLOAT_EQ(tree.getEpsilon(), 0.1f);

    plapoint::Indices indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(tree.nearestKSearch(0, 2, indices, squared_distances), 2);
    EXPECT_EQ(indices, (plapoint::Indices{1, 2}));
    EXPECT_EQ(tree.radiusSearch(*cloud, 0, 1.0, indices, squared_distances), 1);
    EXPECT_EQ(indices, (plapoint::Indices{1}));

    std::vector<plapoint::Indices> batch_indices;
    std::vector<std::vector<float>> batch_distances;
    tree.nearestKSearch(*cloud, plapoint::Indices{0, 2}, 1, batch_indices, batch_distances);
    ASSERT_EQ(batch_indices.size(), 2u);
    EXPECT_EQ(batch_indices[0], (plapoint::Indices{1}));
    EXPECT_EQ(batch_indices[1], (plapoint::Indices{2}));
}

TEST(KdTreeApiTest, SearchBaseExposesSortedAndBatchQueries)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    cloud->emplace_back(1.0f, 0.0f, 0.0f);
    cloud->emplace_back(2.0f, 0.0f, 0.0f);

    plapoint::search::Search<Point>::Ptr search = std::make_shared<plapoint::search::KdTree<Point>>();
    ASSERT_TRUE(search->setInputCloud(cloud));
    EXPECT_EQ(search->getName(), "KdTree");
    EXPECT_TRUE(search->getSortedResults());
    search->setSortedResults(false);
    EXPECT_FALSE(search->getSortedResults());

    std::vector<plapoint::Indices> indices;
    std::vector<std::vector<float>> squared_distances;
    search->nearestKSearch(*cloud, plapoint::Indices{0, 2}, 1, indices, squared_distances);
    ASSERT_EQ(indices.size(), 2u);
    EXPECT_EQ(indices[0], (plapoint::Indices{0}));
    EXPECT_EQ(indices[1], (plapoint::Indices{2}));
    search->radiusSearch(*cloud, plapoint::Indices{1}, 1.0, indices, squared_distances);
    ASSERT_EQ(indices.size(), 1u);
    EXPECT_EQ(indices[0], (plapoint::Indices{1, 0, 2}));
    EXPECT_EQ(squared_distances[0], (std::vector<float>{0.0f, 1.0f, 1.0f}));
}

TEST(KdTreeApiTest, SearchBaseAcceptsDifferentQueryPointType)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    cloud->emplace_back(2.0f, 0.0f, 0.0f);

    plapoint::search::Search<plapoint::PointXYZ>::Ptr search =
        std::make_shared<plapoint::search::KdTree<plapoint::PointXYZ>>();
    ASSERT_TRUE(search->setInputCloud(cloud));
    const plapoint::PointXYZRGB query(1.75f, 0.0f, 0.0f, 10, 20, 30);
    plapoint::Indices indices;
    std::vector<float> squared_distances;
    EXPECT_EQ(search->nearestKSearchT(query, 1, indices, squared_distances), 1);
    EXPECT_EQ(indices, (plapoint::Indices{1}));
    EXPECT_FLOAT_EQ(squared_distances[0], 0.0625f);
    EXPECT_EQ(search->radiusSearchT(query, 0.3, indices, squared_distances), 1);
    EXPECT_EQ(indices, (plapoint::Indices{1}));
}

TEST(KdTreeApiTest, IndependentFirstQueriesCanReadPreparedTreeConcurrently)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    auto cloud = std::make_shared<Cloud>();
    for (int index = 0; index < 64; ++index)
    {
        cloud->emplace_back(static_cast<float>(index), 0.0f, 0.0f);
    }

    plapoint::search::KdTree<Point> tree;
    ASSERT_TRUE(tree.setInputCloud(cloud));
    const auto& shared_tree = tree;
    std::promise<void> begin;
    const std::shared_future<void> gate = begin.get_future().share();
    std::array<std::future<bool>, 8> queries;
    for (auto& result : queries)
    {
        result = std::async(std::launch::async,
                            [&shared_tree, gate]()
                            {
                                gate.wait();
                                std::vector<int> indices;
                                std::vector<float> distances;
                                const Point query(17.2f, 0.0f, 0.0f);
                                if (shared_tree.nearestKSearch(query, 2, indices, distances) != 2 ||
                                    indices != std::vector<int>{17, 18})
                                {
                                    return false;
                                }
                                return shared_tree.radiusSearch(query, 0.3, indices, distances) == 1 &&
                                       indices == std::vector<int>{17};
                            });
    }
    begin.set_value();
    for (auto& result : queries)
    {
        EXPECT_TRUE(result.get());
    }
}
