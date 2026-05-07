#include <gtest/gtest.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(SORTest, RemovesSingleOutlier)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // 10 points clustered at origin + 1 far outlier
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(11, 3);
    for (int i = 0; i < 10; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * 0.01f);
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
    }
    mat.setValue(10, 0, 100); mat.setValue(10, 1, 0); mat.setValue(10, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(4);
    sor.setStddevMulThresh(Scalar(1.0));

    Cloud output;
    sor.filter(output);
    EXPECT_EQ(output.size(), 10u);
}

TEST(SORTest, ThrowsIfNoSearchMethod)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    mat.fill(0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);

    Cloud output;
    EXPECT_THROW(sor.filter(output), std::runtime_error);
}
