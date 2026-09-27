#include <gtest/gtest.h>
#include <plapoint/registration/correspondence_estimation.h>
#include <plapoint/registration/correspondence_rejection_distance.h>
#include <plapoint/registration/icp.h>
#include <plapoint/registration/transformation_estimation_svd.h>

#include <functional>
#include <limits>
#include <memory>
#include <type_traits>

TEST(IcpApiTest, PclPointTypesAndEigenTransformation)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto source = std::make_shared<Cloud>();
    source->push_back(plapoint::PointXYZ(0.0f, 0.0f, 0.0f));
    source->push_back(plapoint::PointXYZ(1.0f, 0.0f, 0.0f));
    source->push_back(plapoint::PointXYZ(0.0f, 1.0f, 0.0f));
    source->push_back(plapoint::PointXYZ(0.0f, 0.0f, 1.0f));

    auto target = std::make_shared<Cloud>(*source);
    target->points[1].x = 1.1f;

    plapoint::IterativeClosestPoint<plapoint::PointXYZ, plapoint::PointXYZ> icp;
    static_assert(std::is_same_v<decltype(icp.getFinalTransformation()), Eigen::Matrix4f>);
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(20);
    icp.setMaxCorrespondenceDistance(2.0);
    icp.setTransformationEpsilon(1.0e-8);
    icp.setTransformationRotationEpsilon(0.99999);
    icp.setEuclideanFitnessEpsilon(1.0e-6);
    EXPECT_DOUBLE_EQ(icp.getTransformationEpsilon(), 1.0e-8);
    EXPECT_DOUBLE_EQ(icp.getTransformationRotationEpsilon(), 0.99999);
    EXPECT_DOUBLE_EQ(icp.getEuclideanFitnessEpsilon(), 1.0e-6);
    EXPECT_FALSE(icp.getUseReciprocalCorrespondences());

    Cloud output;
    icp.align(output, Eigen::Matrix4f::Identity());

    ASSERT_EQ(output.size(), source->size());
    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getInlierFraction(), 1.0f, 1.0e-5f);
    EXPECT_GT(icp.getFitnessScore(), 0.0);
    EXPECT_LT(icp.getFitnessScore(), 0.01);
    EXPECT_EQ(icp.getFinalTransformation().rows(), 4);
    EXPECT_EQ(icp.getFinalTransformation().cols(), 4);
    EXPECT_EQ(icp.getFitnessScore(0.0), std::numeric_limits<double>::max());
    EXPECT_DOUBLE_EQ(icp.getFitnessScore(std::vector<float>{1.0f, 2.0f}, std::vector<float>{0.5f, 1.5f}), 0.5);
}

TEST(IcpApiTest, FitnessScoreUsesIndicesEvenWhenSubsetCountEqualsSourceCount)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    auto source = std::make_shared<Cloud>();
    source->emplace_back(0.0f, 0.0f, 0.0f);
    source->emplace_back(1.0f, 0.0f, 0.0f);
    source->emplace_back(0.0f, 1.0f, 0.0f);
    source->emplace_back(0.0f, 0.0f, 1.0f);
    source->header.frame_id = "source";
    source->sensor_origin_ = Eigen::Vector4f(1.0f, 2.0f, 3.0f, 0.0f);

    auto target = std::make_shared<Cloud>(*source);
    target->points[3].z = 2.0f;
    auto indices = std::make_shared<const std::vector<int>>(std::vector<int>{0, 1, 2, 2});

    plapoint::IterativeClosestPoint<plapoint::PointXYZ, plapoint::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setIndices(indices);
    icp.setMaximumIterations(0);
    Cloud output;
    icp.align(output);

    ASSERT_EQ(output.size(), indices->size());
    EXPECT_EQ(output.header.frame_id, "source");
    EXPECT_TRUE(output.sensor_origin_.isApprox(source->sensor_origin_));
    EXPECT_NEAR(icp.getFitnessScore(std::numeric_limits<double>::max(), true), 0.0, 1.0e-6);
    EXPECT_GT(icp.getFitnessScore(), 0.1);
}

TEST(IcpApiTest, RegistrationBaseDispatchesAlignment)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    cloud->emplace_back(1.0f, 0.0f, 0.0f);
    cloud->emplace_back(0.0f, 1.0f, 0.0f);
    cloud->emplace_back(0.0f, 0.0f, 1.0f);

    plapoint::Registration<Point, Point>::Ptr registration =
        std::make_shared<plapoint::IterativeClosestPoint<Point, Point>>();
    registration->setInputSource(cloud);
    registration->setInputTarget(cloud);
    EXPECT_EQ(registration->getClassName(), "IterativeClosestPoint");
    registration->setMaximumIterations(5);
    Cloud output;
    registration->align(output);

    EXPECT_EQ(output.size(), cloud->size());
    EXPECT_TRUE(registration->hasConverged());
    EXPECT_NEAR(registration->getFitnessScore(), 0.0, 1.0e-6);
    EXPECT_NEAR(registration->getFitnessScore(std::numeric_limits<double>::max(), true), 0.0, 1.0e-6);
}

TEST(IcpApiTest, PointXYZdKeepsLargeCoordinateTranslation)
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZd>;
    auto source = std::make_shared<Cloud>();
    source->emplace_back(1.0e9, 0.0, 0.0);
    source->emplace_back(1.0e9 + 10.0, 0.0, 0.0);
    source->emplace_back(1.0e9, 10.0, 0.0);
    source->emplace_back(1.0e9, 0.0, 10.0);

    auto target = std::make_shared<Cloud>();
    for (const auto& point : source->points)
    {
        target->emplace_back(point.x + 0.25, point.y - 0.125, point.z + 0.5);
    }

    plapoint::IterativeClosestPoint<plapoint::PointXYZd, plapoint::PointXYZd, double> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(20);
    Cloud output;
    icp.align(output);

    ASSERT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalTransformation()(0, 3), 0.25, 1.0e-3);
    EXPECT_NEAR(icp.getFinalTransformation()(1, 3), -0.125, 1.0e-3);
    EXPECT_NEAR(icp.getFinalTransformation()(2, 3), 0.5, 1.0e-3);
    EXPECT_NEAR(output[0].x, target->points[0].x, 1.0e-3);
}

TEST(IcpApiTest, RegistrationComponentsUsePclContracts)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    auto source = std::make_shared<Cloud>();
    source->emplace_back(0.0f, 0.0f, 0.0f);
    source->emplace_back(1.0f, 0.0f, 0.0f);
    source->emplace_back(0.0f, 1.0f, 0.0f);
    source->emplace_back(0.0f, 0.0f, 1.0f);
    auto target = std::make_shared<Cloud>();
    for (const auto& point : source->points)
    {
        target->emplace_back(point.x + 0.25f, point.y - 0.5f, point.z + 0.75f);
    }

    plapoint::Correspondences correspondences;
    for (int index = 0; index < 4; ++index)
    {
        correspondences.emplace_back(index, index, 0.0f);
    }
    plapoint::registration::TransformationEstimationSVD<Point, Point> transformation;
    Eigen::Matrix4f matrix;
    transformation.estimateRigidTransformation(*source, *target, correspondences, matrix);
    EXPECT_NEAR(matrix(0, 3), 0.25f, 1.0e-5f);
    EXPECT_NEAR(matrix(1, 3), -0.5f, 1.0e-5f);
    EXPECT_NEAR(matrix(2, 3), 0.75f, 1.0e-5f);

    plapoint::registration::CorrespondenceEstimation<Point, Point> estimation;
    estimation.setInputSource(source);
    estimation.setInputTarget(target);
    estimation.setNumberOfThreads(0);
    estimation.determineCorrespondences(correspondences, 2.0);
    EXPECT_EQ(correspondences.size(), source->size());

    plapoint::registration::CorrespondenceRejectorDistance rejector;
    rejector.setMaximumDistance(0.5f);
    rejector.setInputCorrespondences(
        std::make_shared<const plapoint::Correspondences>(plapoint::Correspondences{{0, 0, 0.04f}, {1, 1, 1.0f}}));
    rejector.getCorrespondences(correspondences);
    ASSERT_EQ(correspondences.size(), 1u);
    EXPECT_EQ(correspondences[0].index_query, 0);
}

TEST(IcpApiTest, ConvergenceCriteriaAndVisualizerCallbackAreFunctional)
{
    using Point = plapoint::PointXYZ;
    using Cloud = plapoint::PointCloud<Point>;
    auto cloud = std::make_shared<Cloud>();
    cloud->emplace_back(0.0f, 0.0f, 0.0f);
    cloud->emplace_back(1.0f, 0.0f, 0.0f);
    cloud->emplace_back(0.0f, 1.0f, 0.0f);
    cloud->emplace_back(0.0f, 0.0f, 1.0f);

    plapoint::IterativeClosestPoint<Point, Point> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud);
    icp.setNumberOfThreads(0);
    auto criteria = icp.getConvergeCriteria();
    ASSERT_TRUE(criteria);
    criteria->setMaximumIterationsSimilarTransforms(0);
    criteria->setFailureAfterMaximumIterations(false);

    int callback_count = 0;
    std::function<plapoint::Registration<Point, Point>::UpdateVisualizerCallbackSignature> callback =
        [&callback_count](const Cloud&, const plapoint::Indices&, const Cloud&, const plapoint::Indices&)
    { ++callback_count; };
    ASSERT_TRUE(icp.registerVisualizationCallback(callback));
    Cloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_GE(callback_count, 2);
    EXPECT_NE(criteria->getConvergenceState(),
              plapoint::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NOT_CONVERGED);
}
