#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/plamatrix.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#include <plapoint/registration/icp.h>
#undef private

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif

namespace {

template <typename Scalar>
Scalar rotationDeterminant(const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& T)
{
    return T.getValue(0, 0) * (T.getValue(1, 1) * T.getValue(2, 2) - T.getValue(1, 2) * T.getValue(2, 1))
         - T.getValue(0, 1) * (T.getValue(1, 0) * T.getValue(2, 2) - T.getValue(1, 2) * T.getValue(2, 0))
         + T.getValue(0, 2) * (T.getValue(1, 0) * T.getValue(2, 1) - T.getValue(1, 1) * T.getValue(2, 0));
}

}

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
    EXPECT_THROW(icp.setMaxCorrespondenceDistance(-std::numeric_limits<float>::infinity()), std::invalid_argument);
    EXPECT_NO_THROW(icp.setMaxCorrespondenceDistance(std::numeric_limits<float>::max()));
    EXPECT_NO_THROW(icp.setMaxCorrespondenceDistance(std::numeric_limits<float>::infinity()));
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

TEST(ICPTest, RejectsNonFiniteSourcePointsBeforeAlignment)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    const Scalar pts[4][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    for (int i = 0; i < 4; ++i)
    {
        src_mat.setValue(i, 0, pts[i][0]);
        src_mat.setValue(i, 1, pts[i][1]);
        src_mat.setValue(i, 2, pts[i][2]);
        tgt_mat.setValue(i, 0, pts[i][0]);
        tgt_mat.setValue(i, 1, pts[i][1]);
        tgt_mat.setValue(i, 2, pts[i][2]);
    }
    src_mat.setValue(4, 0, std::numeric_limits<Scalar>::infinity());
    src_mat.setValue(4, 1, 0);
    src_mat.setValue(4, 2, 0);
    tgt_mat.setValue(4, 0, 0);
    tgt_mat.setValue(4, 1, 0);
    tgt_mat.setValue(4, 2, 0);

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);

    Cloud output;
    try
    {
        icp.align(output);
        FAIL() << "Expected non-finite source point to be rejected";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("source cloud contains non-finite point"), std::string::npos);
    }
}

TEST(ICPTest, UsesFiniteTargetNeighborWhenNearestCandidateIsNonFinite)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(8, 3);
    const Scalar pts[4][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    for (int i = 0; i < 4; ++i)
    {
        src_mat.setValue(i, 0, pts[i][0]);
        src_mat.setValue(i, 1, pts[i][1]);
        src_mat.setValue(i, 2, pts[i][2]);

        tgt_mat.setValue(i, 0, std::numeric_limits<Scalar>::quiet_NaN());
        tgt_mat.setValue(i, 1, pts[i][1]);
        tgt_mat.setValue(i, 2, pts[i][2]);

        tgt_mat.setValue(i + 4, 0, pts[i][0]);
        tgt_mat.setValue(i + 4, 1, pts[i][1]);
        tgt_mat.setValue(i + 4, 2, pts[i][2]);
    }

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(5);

    Cloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getFitnessScore(), Scalar(1), Scalar(1e-6));
    EXPECT_NEAR(icp.getFinalRmse(), Scalar(0), Scalar(1e-6));
}

TEST(ICPTest, CollectCorrespondencesKeepsFloatDistancesThatOverflowScalarDifference)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    constexpr Scalar base = std::numeric_limits<Scalar>::max() * Scalar(0.75);
    constexpr Scalar spread = Scalar(1e32);

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    const Scalar offsets[4][3] = {
        {0, 0, 0},
        {0, spread, 0},
        {0, 0, spread},
        {0, spread, spread},
    };
    for (int i = 0; i < 4; ++i)
    {
        src_mat.setValue(i, 0, base + offsets[i][0]);
        src_mat.setValue(i, 1, offsets[i][1]);
        src_mat.setValue(i, 2, offsets[i][2]);
        tgt_mat.setValue(i, 0, -base + offsets[i][0]);
        tgt_mat.setValue(i, 1, offsets[i][1]);
        tgt_mat.setValue(i, 2, offsets[i][2]);
    }

    auto target = std::make_shared<Cloud>(std::move(tgt_mat));
    plapoint::search::KdTree<Scalar, plamatrix::Device::CPU> tree;
    tree.setInputCloud(target);
    tree.build();

    std::vector<int> corr(4, -1);
    std::vector<int> active_indices;
    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.collectCorrespondences(src_mat, target->points(), tree, corr, active_indices);

    EXPECT_EQ(active_indices.size(), 4u);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(corr[static_cast<std::size_t>(i)], i);
    }
}

TEST(ICPTest, AlignReportsUnrepresentableFloatStateInsteadOfProducingNonFiniteTransform)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    constexpr Scalar base = std::numeric_limits<Scalar>::max() * Scalar(0.75);
    constexpr Scalar spread = Scalar(1e32);

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    const Scalar offsets[4][3] = {
        {0, 0, 0},
        {0, spread, 0},
        {0, 0, spread},
        {0, spread, spread},
    };
    for (int i = 0; i < 4; ++i)
    {
        src_mat.setValue(i, 0, base + offsets[i][0]);
        src_mat.setValue(i, 1, offsets[i][1]);
        src_mat.setValue(i, 2, offsets[i][2]);
        tgt_mat.setValue(i, 0, -base + offsets[i][0]);
        tgt_mat.setValue(i, 1, offsets[i][1]);
        tgt_mat.setValue(i, 2, offsets[i][2]);
    }

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);

    Cloud output;
    try
    {
        icp.align(output);
        FAIL() << "Expected unrepresentable transform to throw";
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("not representable"), std::string::npos);
    }
}

TEST(ICPTest, UpdateResidualMetricsUsesScaledRmsForHugeFiniteResiduals)
{
    using Scalar = double;
    constexpr Scalar huge = 1.0e200;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    src_mat.setValue(0, 0, 0); src_mat.setValue(0, 1, 0); src_mat.setValue(0, 2, 0);
    src_mat.setValue(1, 0, 0); src_mat.setValue(1, 1, 0); src_mat.setValue(1, 2, 0);
    tgt_mat.setValue(0, 0, huge);  tgt_mat.setValue(0, 1, 0); tgt_mat.setValue(0, 2, 0);
    tgt_mat.setValue(1, 0, -huge); tgt_mat.setValue(1, 1, 0); tgt_mat.setValue(1, 2, 0);

    std::vector<int> corr{0, 1};
    std::vector<int> active_indices{0, 1};
    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.updateResidualMetrics(src_mat, tgt_mat, corr, active_indices, 2);

    ASSERT_TRUE(std::isfinite(icp.getFinalRmse()));
    EXPECT_NEAR(icp.getFinalRmse() / huge, 1.0, 1e-12);
    EXPECT_EQ(icp.getFitnessScore(), 1.0);
}

TEST(ICPTest, Multiply4x4RejectsUnrepresentableAccumulatedTransform)
{
    using Scalar = float;
    using Icp = plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU>;
    constexpr Scalar huge = std::numeric_limits<Scalar>::max() * Scalar(0.75);

    auto a = Icp::identity4x4();
    auto b = Icp::identity4x4();
    a.setValue(0, 3, huge);
    b.setValue(0, 3, huge);

    EXPECT_THROW((void)Icp::multiply4x4(a, b), std::runtime_error);
}

TEST(ICPTest, NonCollinearGeometryHandlesHugeFiniteDoubleScale)
{
    using Scalar = double;
    constexpr Scalar huge = 1.0e80;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(3, 3);
    points.setValue(0, 0, 0);    points.setValue(0, 1, 0);    points.setValue(0, 2, 0);
    points.setValue(1, 0, huge); points.setValue(1, 1, 0);    points.setValue(1, 2, 0);
    points.setValue(2, 0, 0);    points.setValue(2, 1, huge); points.setValue(2, 2, 0);

    const std::vector<int> active_indices{0, 1, 2};
    EXPECT_TRUE((plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU>::
        hasNonCollinearGeometry(points, active_indices)));
}

TEST(ICPTest, NonCollinearGeometryHandlesNearDoubleMaxFiniteScale)
{
    using Scalar = double;
    constexpr Scalar huge = std::numeric_limits<Scalar>::max() * 0.75;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(3, 3);
    points.setValue(0, 0, -huge); points.setValue(0, 1, 0);    points.setValue(0, 2, 0);
    points.setValue(1, 0, huge);  points.setValue(1, 1, 0);    points.setValue(1, 2, 0);
    points.setValue(2, 0, -huge); points.setValue(2, 1, huge); points.setValue(2, 2, 0);

    const std::vector<int> active_indices{0, 1, 2};
    EXPECT_TRUE((plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU>::
        hasNonCollinearGeometry(points, active_indices)));
}

TEST(ICPTest, NonCollinearGeometryNormalizesExtremeLongDoubleScale)
{
    using Scalar = long double;
    constexpr Scalar huge = std::numeric_limits<Scalar>::max() * 0.75L;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(3, 3);
    points.setValue(0, 0, -huge); points.setValue(0, 1, 0);    points.setValue(0, 2, 0);
    points.setValue(1, 0, huge);  points.setValue(1, 1, 0);    points.setValue(1, 2, 0);
    points.setValue(2, 0, -huge); points.setValue(2, 1, huge); points.setValue(2, 2, 0);

    const std::vector<int> active_indices{0, 1, 2};
    EXPECT_TRUE((plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU>::
        hasNonCollinearGeometry(points, active_indices)));
}

#ifdef PLAPOINT_WITH_CUDA
TEST(ICPTest, GpuRejectsNonFiniteSourcePointsBeforeAlignment)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    const Scalar pts[4][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    for (int i = 0; i < 4; ++i)
    {
        src_mat.setValue(i, 0, pts[i][0]);
        src_mat.setValue(i, 1, pts[i][1]);
        src_mat.setValue(i, 2, pts[i][2]);
        tgt_mat.setValue(i, 0, pts[i][0]);
        tgt_mat.setValue(i, 1, pts[i][1]);
        tgt_mat.setValue(i, 2, pts[i][2]);
    }
    src_mat.setValue(4, 0, infinity);
    src_mat.setValue(4, 1, 0);
    src_mat.setValue(4, 2, 0);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(src_mat));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(tgt_mat));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);

    GpuCloud output;
    try
    {
        icp.align(output);
        FAIL() << "Expected non-finite source point to be rejected";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("source cloud contains non-finite point"), std::string::npos);
    }
}
#endif

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

TEST(ICPTest, CanDisableFinalMetricComputationForThroughput)
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
    icp.setComputeFinalMetrics(false);

    Cloud output;
    icp.align(output);

    const auto& T = icp.getFinalTransformation();
    EXPECT_NEAR(T.getValue(0, 3), Scalar(-1), Scalar(1e-4));
    EXPECT_NEAR(T.getValue(1, 3), Scalar(-2), Scalar(1e-4));
    EXPECT_NEAR(T.getValue(2, 3), Scalar(-3), Scalar(1e-4));
    EXPECT_GT(icp.getFinalRmse(), Scalar(0));
    ASSERT_EQ(output.size(), target->size());
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_NEAR(output[i].x(), (*target)[i].x(), Scalar(1e-4));
        EXPECT_NEAR(output[i].y(), (*target)[i].y(), Scalar(1e-4));
        EXPECT_NEAR(output[i].z(), (*target)[i].z(), Scalar(1e-4));
    }
}

TEST(ICPTest, ReflectionCorrectionProducesProperRotation)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    src_mat.setValue(0, 0, 0); src_mat.setValue(0, 1, 0); src_mat.setValue(0, 2, 0);
    src_mat.setValue(1, 0, 1); src_mat.setValue(1, 1, 0); src_mat.setValue(1, 2, 0);
    src_mat.setValue(2, 0, 0); src_mat.setValue(2, 1, 1); src_mat.setValue(2, 2, 0);
    src_mat.setValue(3, 0, 0); src_mat.setValue(3, 1, 0); src_mat.setValue(3, 2, 1);

    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    tgt_mat.setValue(0, 0, 0);  tgt_mat.setValue(0, 1, 0); tgt_mat.setValue(0, 2, 0);
    tgt_mat.setValue(1, 0, -1); tgt_mat.setValue(1, 1, 0); tgt_mat.setValue(1, 2, 0);
    tgt_mat.setValue(2, 0, 0);  tgt_mat.setValue(2, 1, 1); tgt_mat.setValue(2, 2, 0);
    tgt_mat.setValue(3, 0, 0);  tgt_mat.setValue(3, 1, 0); tgt_mat.setValue(3, 2, 1);

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);
    icp.setMaxCorrespondenceDistance(Scalar(10));

    Cloud output;
    icp.align(output);

    EXPECT_GT(rotationDeterminant(icp.getFinalTransformation()), Scalar(0.9));
}

TEST(ICPTest, AcceptsSmallScaleNonCollinearGeometry)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    constexpr Scalar s = Scalar(1e-4);

    auto src_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(4, 3);
    const Scalar pts[4][3] = {
        {0, 0, 0},
        {s, 0, 0},
        {0, s, 0},
        {0, 0, s},
    };
    for (int i = 0; i < 4; ++i)
    {
        src_mat.setValue(i, 0, pts[i][0]);
        src_mat.setValue(i, 1, pts[i][1]);
        src_mat.setValue(i, 2, pts[i][2]);
        tgt_mat.setValue(i, 0, pts[i][0]);
        tgt_mat.setValue(i, 1, pts[i][1]);
        tgt_mat.setValue(i, 2, pts[i][2]);
    }

    auto source = std::make_shared<Cloud>(std::move(src_mat));
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);

    Cloud output;
    EXPECT_NO_THROW(icp.align(output));
    EXPECT_TRUE(icp.hasConverged());
}
