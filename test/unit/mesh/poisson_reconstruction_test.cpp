#include <gtest/gtest.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>
#include <limits>
#include <string>

namespace
{

template <typename Fn>
void expectInvalidArgumentContaining(Fn&& fn, const std::string& expected_message)
{
    try
    {
        fn();
        FAIL() << "Expected invalid_argument containing: " << expected_message;
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find(expected_message), std::string::npos)
            << "Actual exception: " << e.what();
    }
}

} // namespace

TEST(PoissonReconstructionTest, SphereReconstructsMesh)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    int n_pts = 100;
    Matrix pts(n_pts, 3);
    Matrix nrm(n_pts, 3);
    for (int i = 0; i < n_pts; ++i)
    {
        Scalar theta = Scalar(i) * Scalar(2*3.14159) / Scalar(n_pts);
        Scalar phi = Scalar(i) * Scalar(3.14159) / Scalar(n_pts);
        Scalar x = Scalar(2) * std::sin(phi) * std::cos(theta);
        Scalar y = Scalar(2) * std::sin(phi) * std::sin(theta);
        Scalar z = Scalar(2) * std::cos(phi);
        pts.setValue(i, 0, x); pts.setValue(i, 1, y); pts.setValue(i, 2, z);
        Scalar r = std::sqrt(x*x + y*y + z*z);
        nrm.setValue(i, 0, x/r); nrm.setValue(i, 1, y/r); nrm.setValue(i, 2, z/r);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(nrm));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    pr.setDepth(4);
    pr.setSolverIterations(30);

    auto [verts, faces] = pr.reconstruct();

    EXPECT_GT(verts.rows(), 0);
    EXPECT_GT(faces.rows(), 0);
}

TEST(PoissonReconstructionTest, LargeCoordinatesDoNotUseSentinelBounds)
{
    using Scalar = double;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    constexpr int n_pts = 100;
    constexpr Scalar center = 1.0e12;
    constexpr Scalar radius = 2.0;

    Matrix pts(n_pts, 3);
    Matrix nrm(n_pts, 3);
    for (int i = 0; i < n_pts; ++i)
    {
        Scalar theta = Scalar(i) * Scalar(2 * 3.14159265358979323846) / Scalar(n_pts);
        Scalar phi = Scalar(i) * Scalar(3.14159265358979323846) / Scalar(n_pts);
        Scalar nx = std::sin(phi) * std::cos(theta);
        Scalar ny = std::sin(phi) * std::sin(theta);
        Scalar nz = std::cos(phi);
        pts.setValue(i, 0, center + radius * nx);
        pts.setValue(i, 1, center + radius * ny);
        pts.setValue(i, 2, center + radius * nz);
        nrm.setValue(i, 0, nx);
        nrm.setValue(i, 1, ny);
        nrm.setValue(i, 2, nz);
    }

    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(nrm));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    pr.setDepth(4);
    pr.setSolverIterations(30);

    auto [verts, faces] = pr.reconstruct();

    ASSERT_GT(verts.rows(), 0);
    ASSERT_GT(faces.rows(), 0);
    for (plamatrix::Index r = 0; r < verts.rows(); ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            const Scalar value = verts.getValue(r, c);
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_GT(value, center - Scalar(10));
            EXPECT_LT(value, center + Scalar(10));
        }
    }
}

TEST(PoissonReconstructionTest, RejectsInvalidDepthAndSolverIterations)
{
    plapoint::mesh::PoissonReconstruction<float> pr;

    EXPECT_THROW(pr.setDepth(0), std::invalid_argument);
    EXPECT_THROW(pr.setDepth(9), std::invalid_argument);
    EXPECT_THROW(pr.setDepth(31), std::invalid_argument);
    EXPECT_THROW(pr.setSolverIterations(0), std::invalid_argument);
}

TEST(PoissonReconstructionTest, RejectsUnsetInputCloud)
{
    plapoint::mesh::PoissonReconstruction<float> pr;

    EXPECT_THROW((void)pr.reconstruct(), std::runtime_error);
}

TEST(PoissonReconstructionTest, RejectsCloudWithoutNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    points.setValue(0, 0, 0);
    points.setValue(0, 1, 0);
    points.setValue(0, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);

    EXPECT_THROW((void)pr.reconstruct(), std::runtime_error);
}

TEST(PoissonReconstructionTest, RejectsEmptyInputCloud)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(0, 3);
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);

    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);
}

TEST(PoissonReconstructionTest, SinglePointWithNormalProducesEmptyDegenerateMesh)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    points.setValue(0, 0, 0);
    points.setValue(0, 1, 0);
    points.setValue(0, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(1, 3);
    normals.setValue(0, 0, 0);
    normals.setValue(0, 1, 0);
    normals.setValue(0, 2, 1);
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    pr.setDepth(1);
    pr.setSolverIterations(1);

    auto [verts, faces] = pr.reconstruct();

    EXPECT_EQ(verts.rows(), 0);
    EXPECT_EQ(verts.cols(), 3);
    EXPECT_EQ(faces.rows(), 0);
    EXPECT_EQ(faces.cols(), 3);
}

TEST(PoissonReconstructionTest, RejectsNonFinitePointsAndNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    points.fill(0);
    points.setValue(1, 0, std::numeric_limits<Scalar>::quiet_NaN());
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(2, 3);
    normals.fill(0);
    normals.setValue(0, 2, 1);
    normals.setValue(1, 2, 1);
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);

    auto finite_points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    finite_points.fill(0);
    auto cloud_with_bad_normals = std::make_shared<Cloud>(std::move(finite_points));
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> bad_normals(2, 3);
    bad_normals.fill(0);
    bad_normals.setValue(0, 2, 1);
    bad_normals.setValue(1, 2, std::numeric_limits<Scalar>::infinity());
    cloud_with_bad_normals->setNormals(std::move(bad_normals));

    pr.setInputCloud(cloud_with_bad_normals);
    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);
}

TEST(PoissonReconstructionTest, RejectsZeroLengthNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    points.fill(0);
    points.setValue(1, 0, 1);
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(2, 3);
    normals.fill(0);
    normals.setValue(0, 2, 1);
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);

    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);
}

TEST(PoissonReconstructionTest, RejectsNormalsWhoseLengthIsNotFinite)
{
    using Scalar = double;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(2, 3);
    points.fill(0);
    points.setValue(1, 0, 1);
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(2, 3);
    normals.fill(0);
    normals.setValue(0, 0, std::numeric_limits<Scalar>::max());
    normals.setValue(0, 1, std::numeric_limits<Scalar>::max());
    normals.setValue(0, 2, std::numeric_limits<Scalar>::max());
    normals.setValue(1, 2, 1);
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    pr.setDepth(1);
    pr.setSolverIterations(1);

    expectInvalidArgumentContaining(
        [&]() { (void)pr.reconstruct(); },
        "normals");
}
