#include <gtest/gtest.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>

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
