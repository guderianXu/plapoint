#include <gtest/gtest.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

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
