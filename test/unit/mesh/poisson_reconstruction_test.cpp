#include <gtest/gtest.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>
#include <limits>

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

TEST(PoissonReconstructionTest, RejectsInvalidDepthAndSolverIterations)
{
    plapoint::mesh::PoissonReconstruction<float> pr;

    EXPECT_THROW(pr.setDepth(0), std::invalid_argument);
    EXPECT_THROW(pr.setDepth(9), std::invalid_argument);
    EXPECT_THROW(pr.setDepth(31), std::invalid_argument);
    EXPECT_THROW(pr.setSolverIterations(0), std::invalid_argument);
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
