#include <gtest/gtest.h>
#include <plapoint/registration/icp.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(ICPTest, IdentityAlignment)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // Same cloud as source and target => identity transform
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(10, 3);
    for (int i = 0; i < 10; ++i)
    {
        mat.setValue(i, 0, Scalar(i));
        mat.setValue(i, 1, Scalar(i % 3));
        mat.setValue(i, 2, Scalar(i % 2));
    }

    // Copy for target
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(10, 3);
    for (int i = 0; i < 10; ++i)
    {
        tgt_mat.setValue(i, 0, Scalar(i));
        tgt_mat.setValue(i, 1, Scalar(i % 3));
        tgt_mat.setValue(i, 2, Scalar(i % 2));
    }
    auto source = std::make_shared<Cloud>(std::move(mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(10);

    Cloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());

    const auto& T = icp.getFinalTransformation();
    EXPECT_NEAR(T.getValue(0, 0), Scalar(1), Scalar(1e-3));
    EXPECT_NEAR(T.getValue(0, 3), Scalar(0), Scalar(1e-3));
}

TEST(ICPTest, ThrowsIfNoInput)
{
    plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
    plapoint::PointCloud<float, plamatrix::Device::CPU> output;
    EXPECT_THROW(icp.align(output), std::runtime_error);
}

TEST(ICPTest, ThrowsForEmptySourceOrTarget)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto empty = std::make_shared<Cloud>(0);

    auto one_point_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    one_point_mat.setValue(0, 0, 1);
    one_point_mat.setValue(0, 1, 2);
    one_point_mat.setValue(0, 2, 3);
    auto one_point = std::make_shared<Cloud>(std::move(one_point_mat));

    Cloud output;

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> empty_source_icp;
    empty_source_icp.setInputSource(empty);
    empty_source_icp.setInputTarget(one_point);
    EXPECT_THROW(empty_source_icp.align(output), std::invalid_argument);

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> empty_target_icp;
    empty_target_icp.setInputSource(one_point);
    empty_target_icp.setInputTarget(empty);
    EXPECT_THROW(empty_target_icp.align(output), std::invalid_argument);
}

TEST(ICPTest, RejectsInvalidIterationAndEpsilonParameters)
{
    plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
    EXPECT_THROW(icp.setMaxIterations(-1), std::invalid_argument);
    EXPECT_THROW(icp.setTransformationEpsilon(0.0f), std::invalid_argument);
}
