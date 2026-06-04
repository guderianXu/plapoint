#include <gtest/gtest.h>
#include <plapoint/registration/icp.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <limits>

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
    EXPECT_NEAR(icp.getFitnessScore(), Scalar(1), Scalar(1e-6));
    EXPECT_NEAR(icp.getFinalRmse(), Scalar(0), Scalar(1e-6));
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
    EXPECT_THROW(icp.setTransformationEpsilon(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
}

TEST(ICPTest, RejectsInvalidCorrespondenceDistance)
{
    plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
    EXPECT_THROW(icp.setMaxCorrespondenceDistance(0.0f), std::invalid_argument);
    EXPECT_THROW(icp.setMaxCorrespondenceDistance(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(icp.setMinFitnessScore(-0.1f), std::invalid_argument);
    EXPECT_THROW(icp.setMinFitnessScore(1.1f), std::invalid_argument);
    EXPECT_THROW(icp.setMinFitnessScore(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
}

TEST(ICPTest, MaxCorrespondenceDistanceIgnoresFarSourceOutlier)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    src_mat.setValue(0, 0, 0);   src_mat.setValue(0, 1, 0);   src_mat.setValue(0, 2, 0);
    src_mat.setValue(1, 0, 1);   src_mat.setValue(1, 1, 0);   src_mat.setValue(1, 2, 0);
    src_mat.setValue(2, 0, 0);   src_mat.setValue(2, 1, 1);   src_mat.setValue(2, 2, 0);
    src_mat.setValue(3, 0, 0);   src_mat.setValue(3, 1, 0);   src_mat.setValue(3, 2, 1);
    src_mat.setValue(4, 0, 100); src_mat.setValue(4, 1, 100); src_mat.setValue(4, 2, 100);

    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    tgt_mat.setValue(0, 0, 0); tgt_mat.setValue(0, 1, 0); tgt_mat.setValue(0, 2, 0);
    tgt_mat.setValue(1, 0, 1); tgt_mat.setValue(1, 1, 0); tgt_mat.setValue(1, 2, 0);
    tgt_mat.setValue(2, 0, 0); tgt_mat.setValue(2, 1, 1); tgt_mat.setValue(2, 2, 0);
    tgt_mat.setValue(3, 0, 0); tgt_mat.setValue(3, 1, 0); tgt_mat.setValue(3, 2, 1);

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(5);
    icp.setMaxCorrespondenceDistance(Scalar(0.25));

    Cloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getFitnessScore(), Scalar(0.8), Scalar(1e-6));
    EXPECT_NEAR(icp.getFinalRmse(), Scalar(0), Scalar(1e-6));
}

TEST(ICPTest, RejectsTooFewCorrespondencesAfterDistanceFiltering)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    src_mat.setValue(0, 0, 0);   src_mat.setValue(0, 1, 0);   src_mat.setValue(0, 2, 0);
    src_mat.setValue(1, 0, 1);   src_mat.setValue(1, 1, 0);   src_mat.setValue(1, 2, 0);
    src_mat.setValue(2, 0, 100); src_mat.setValue(2, 1, 100); src_mat.setValue(2, 2, 100);
    src_mat.setValue(3, 0, 101); src_mat.setValue(3, 1, 101); src_mat.setValue(3, 2, 101);

    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    tgt_mat.setValue(0, 0, 0); tgt_mat.setValue(0, 1, 0); tgt_mat.setValue(0, 2, 0);
    tgt_mat.setValue(1, 0, 1); tgt_mat.setValue(1, 1, 0); tgt_mat.setValue(1, 2, 0);
    tgt_mat.setValue(2, 0, 0); tgt_mat.setValue(2, 1, 1); tgt_mat.setValue(2, 2, 0);
    tgt_mat.setValue(3, 0, 0); tgt_mat.setValue(3, 1, 0); tgt_mat.setValue(3, 2, 1);

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(Scalar(0.25));

    Cloud output;
    EXPECT_THROW(icp.align(output), std::runtime_error);
}

TEST(ICPTest, RejectsCollinearCorrespondenceGeometry)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        src_mat.setValue(i, 0, Scalar(i));
        src_mat.setValue(i, 1, Scalar(0));
        src_mat.setValue(i, 2, Scalar(0));
        tgt_mat.setValue(i, 0, Scalar(i));
        tgt_mat.setValue(i, 1, Scalar(0));
        tgt_mat.setValue(i, 2, Scalar(0));
    }

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);

    Cloud output;
    EXPECT_THROW(icp.align(output), std::runtime_error);
}

TEST(ICPTest, DoesNotConvergeWhenFitnessBelowMinimum)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(10, 3);
    src_mat.setValue(0, 0, 0);   src_mat.setValue(0, 1, 0);   src_mat.setValue(0, 2, 0);
    src_mat.setValue(1, 0, 1);   src_mat.setValue(1, 1, 0);   src_mat.setValue(1, 2, 0);
    src_mat.setValue(2, 0, 0);   src_mat.setValue(2, 1, 1);   src_mat.setValue(2, 2, 0);
    src_mat.setValue(3, 0, 0);   src_mat.setValue(3, 1, 0);   src_mat.setValue(3, 2, 1);
    for (int i = 4; i < 10; ++i)
    {
        src_mat.setValue(i, 0, Scalar(100 + i));
        src_mat.setValue(i, 1, Scalar(100 + i));
        src_mat.setValue(i, 2, Scalar(100 + i));
    }

    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    tgt_mat.setValue(0, 0, 0); tgt_mat.setValue(0, 1, 0); tgt_mat.setValue(0, 2, 0);
    tgt_mat.setValue(1, 0, 1); tgt_mat.setValue(1, 1, 0); tgt_mat.setValue(1, 2, 0);
    tgt_mat.setValue(2, 0, 0); tgt_mat.setValue(2, 1, 1); tgt_mat.setValue(2, 2, 0);
    tgt_mat.setValue(3, 0, 0); tgt_mat.setValue(3, 1, 0); tgt_mat.setValue(3, 2, 1);

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(3);
    icp.setMaxCorrespondenceDistance(Scalar(0.25));
    icp.setMinFitnessScore(Scalar(0.5));

    Cloud output;
    icp.align(output);

    EXPECT_FALSE(icp.hasConverged());
    EXPECT_NEAR(icp.getFitnessScore(), Scalar(0.4), Scalar(1e-6));
}

TEST(ICPTest, FinalRmseReflectsResidualAfterLastStep)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    const Scalar target_points[4][3] = {
        {0, 0, 0},
        {10, 0, 0},
        {0, 10, 0},
        {0, 0, 10},
    };
    for (int i = 0; i < 4; ++i)
    {
        tgt_mat.setValue(i, 0, target_points[i][0]);
        tgt_mat.setValue(i, 1, target_points[i][1]);
        tgt_mat.setValue(i, 2, target_points[i][2]);
        src_mat.setValue(i, 0, target_points[i][0] + Scalar(1));
        src_mat.setValue(i, 1, target_points[i][1] + Scalar(2));
        src_mat.setValue(i, 2, target_points[i][2] + Scalar(3));
    }

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);
    icp.setMaxCorrespondenceDistance(Scalar(5));

    Cloud output;
    icp.align(output);

    const auto& T = icp.getFinalTransformation();
    EXPECT_NEAR(T.getValue(0, 3), Scalar(-1), Scalar(1e-4));
    EXPECT_NEAR(T.getValue(1, 3), Scalar(-2), Scalar(1e-4));
    EXPECT_NEAR(T.getValue(2, 3), Scalar(-3), Scalar(1e-4));
    EXPECT_NEAR(icp.getFinalRmse(), Scalar(0), Scalar(1e-4));
}
