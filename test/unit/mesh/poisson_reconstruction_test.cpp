#include <gtest/gtest.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "quality/mesh_quality_utils.h"

namespace
{

using FloatCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using FloatMatrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

std::shared_ptr<FloatCloud> makeSphereCloud(int count)
{
    FloatMatrix points(count, 3);
    FloatMatrix normals(count, 3);
    constexpr float pi = 3.14159265358979323846f;
    for (int i = 0; i < count; ++i)
    {
        const float theta = static_cast<float>(i) * 2.0f * pi / static_cast<float>(count);
        const float phi = static_cast<float>(i) * pi / static_cast<float>(count);
        const float x = std::sin(phi) * std::cos(theta);
        const float y = std::sin(phi) * std::sin(theta);
        const float z = std::cos(phi);
        points.setValue(i, 0, x); points.setValue(i, 1, y); points.setValue(i, 2, z);
        normals.setValue(i, 0, x); normals.setValue(i, 1, y); normals.setValue(i, 2, z);
    }
    auto cloud = std::make_shared<FloatCloud>(std::move(points));
    cloud->setNormals(std::move(normals));
    return cloud;
}

static_assert(std::is_copy_constructible_v<plapoint::mesh::PoissonReconstruction<float>>);
static_assert(std::is_copy_assignable_v<plapoint::mesh::PoissonReconstruction<float>>);

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

    const auto& system = pr.lastSystem();
    const auto& matrix = system.matrix;
    ASSERT_EQ(matrix.rows(), static_cast<plamatrix::Index>(system.leafNodes.size()));
    ASSERT_EQ(matrix.cols(), matrix.rows());
    ASSERT_EQ(system.rhs.rows(), matrix.rows());
    std::map<std::pair<plamatrix::Index, plamatrix::Index>, Scalar> entries;
    for (plamatrix::Index row = 0; row < matrix.rows(); ++row)
    {
        Scalar diagonal = Scalar(0);
        plamatrix::Index previous_column = -1;
        for (plamatrix::Index offset = matrix.rowOffsets()[row];
             offset < matrix.rowOffsets()[row + 1]; ++offset)
        {
            const auto column = matrix.colIndices()[offset];
            EXPECT_GT(column, previous_column);
            previous_column = column;
            entries[{row, column}] = matrix.values()[offset];
            if (column == row)
            {
                diagonal = matrix.values()[offset];
            }
        }
        EXPECT_GT(diagonal, Scalar(0));
        EXPECT_TRUE(std::isfinite(system.rhs(row, 0)));
    }
    for (const auto& entry : entries)
    {
        const auto transpose = entries.find({entry.first.second, entry.first.first});
        ASSERT_NE(transpose, entries.end());
        EXPECT_NEAR(entry.second, transpose->second, Scalar(1.0e-6));
    }

    long double quadratic = 0.0L;
    for (plamatrix::Index row = 0; row < matrix.rows(); ++row)
    {
        const long double x_row = static_cast<long double>((row % 7) - 3);
        for (plamatrix::Index offset = matrix.rowOffsets()[row];
             offset < matrix.rowOffsets()[row + 1]; ++offset)
        {
            const auto column = matrix.colIndices()[offset];
            const long double x_column = static_cast<long double>((column % 7) - 3);
            quadratic += x_row * static_cast<long double>(matrix.values()[offset]) * x_column;
        }
    }
    EXPECT_GT(quadratic, 0.0L);

    const auto& report = pr.lastReport();
    EXPECT_EQ(report.requestedDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(report.leafCount, system.leafNodes.size());
    EXPECT_TRUE(std::isfinite(report.solver.initialResidual));
    EXPECT_TRUE(std::isfinite(report.solver.finalResidual));
}

TEST(PoissonReconstructionTest, RepeatedAssemblyIsBitwiseDeterministic)
{
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeSphereCloud(100));
    reconstruction.setDepth(4);
    reconstruction.setSolverIterations(30);
    static_cast<void>(reconstruction.reconstruct());
    const auto& first = reconstruction.lastSystem();
    const std::vector<plamatrix::Index> first_offsets(
        first.matrix.rowOffsets(), first.matrix.rowOffsets() + first.matrix.rows() + 1);
    const std::vector<plamatrix::Index> first_columns(
        first.matrix.colIndices(), first.matrix.colIndices() + first.matrix.nnz());
    const std::vector<float> first_values(
        first.matrix.values(), first.matrix.values() + first.matrix.nnz());
    const std::vector<float> first_rhs(
        first.rhs.data(), first.rhs.data() + first.rhs.size());

    static_cast<void>(reconstruction.reconstruct());
    const auto& second = reconstruction.lastSystem();
    EXPECT_EQ(first_offsets, std::vector<plamatrix::Index>(
        second.matrix.rowOffsets(), second.matrix.rowOffsets() + second.matrix.rows() + 1));
    EXPECT_EQ(first_columns, std::vector<plamatrix::Index>(
        second.matrix.colIndices(), second.matrix.colIndices() + second.matrix.nnz()));
    EXPECT_EQ(first_values, std::vector<float>(
        second.matrix.values(), second.matrix.values() + second.matrix.nnz()));
    EXPECT_EQ(first_rhs, std::vector<float>(
        second.rhs.data(), second.rhs.data() + second.rhs.size()));
}

TEST(PoissonReconstructionTest, QualityFixtureConvergesAndProducesSurface)
{
    auto cloud = plapoint::test::mesh_quality::makeSpherePointCloud<float>(2.0f, 12, 24);
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(cloud);
    reconstruction.setDepth(5);
    reconstruction.setSolverIterations(30);
    auto [vertices, faces] = reconstruction.reconstruct();
    const auto& report = reconstruction.lastReport();
    EXPECT_TRUE(report.solver.converged)
        << "leaves=" << report.leafCount
        << " iterations=" << report.solver.iterations
        << " initial=" << report.solver.initialResidual
        << " final=" << report.solver.finalResidual
        << " field=[" << report.fieldMinimum << ',' << report.fieldMaximum << ']'
        << " iso=" << report.isoLevel;
    EXPECT_GT(vertices.rows(), 50)
        << "leaves=" << report.leafCount
        << " iterations=" << report.solver.iterations
        << " initial=" << report.solver.initialResidual
        << " final=" << report.solver.finalResidual
        << " field=[" << report.fieldMinimum << ',' << report.fieldMaximum << ']'
        << " iso=" << report.isoLevel;
    EXPECT_GT(faces.rows(), 50);
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
    EXPECT_THROW(pr.setSolverTolerance(0.0), std::invalid_argument);
    EXPECT_THROW(pr.setSolverTolerance(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(PoissonReconstructionTest, ExplicitGpuUsesPlaMatrixPcg)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;
    constexpr int count = 48;
    Matrix points(count, 3);
    Matrix normals(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const Scalar theta = Scalar(i) * Scalar(2 * 3.14159265358979323846) / Scalar(count);
        const Scalar phi = Scalar(i) * Scalar(3.14159265358979323846) / Scalar(count);
        const Scalar x = std::sin(phi) * std::cos(theta);
        const Scalar y = std::sin(phi) * std::sin(theta);
        const Scalar z = std::cos(phi);
        points.setValue(i, 0, x); points.setValue(i, 1, y); points.setValue(i, 2, z);
        normals.setValue(i, 0, x); normals.setValue(i, 1, y); normals.setValue(i, 2, z);
    }
    auto cloud = std::make_shared<Cloud>(std::move(points));
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> reconstruction;
    reconstruction.setInputCloud(cloud);
    reconstruction.setDepth(3);
    reconstruction.setSolverIterations(100);
    reconstruction.setSolverTolerance(1.0e-5);
    reconstruction.setProcessingDevice(plapoint::ProcessingDevice::GPU);
    auto [vertices, faces] = reconstruction.reconstruct();

    EXPECT_GT(vertices.rows(), 0);
    EXPECT_GT(faces.rows(), 0);
    EXPECT_EQ(reconstruction.lastReport().requestedDevice, plapoint::ProcessingDevice::GPU);
    EXPECT_EQ(reconstruction.lastReport().actualDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(reconstruction.lastReport().solverDevice, plapoint::ProcessingDevice::GPU);
    EXPECT_FALSE(reconstruction.lastReport().usedFallback);
    EXPECT_TRUE(reconstruction.lastReport().solver.converged);
}

TEST(PoissonReconstructionTest, ExplicitGpuReportsNonConvergence)
{
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeSphereCloud(96));
    reconstruction.setDepth(4);
    reconstruction.setSolverIterations(1);
    reconstruction.setSolverTolerance(1.0e-12);
    reconstruction.setProcessingDevice(plapoint::ProcessingDevice::GPU);
    try
    {
        static_cast<void>(reconstruction.reconstruct());
        FAIL() << "Expected explicit GPU non-convergence to throw";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("did not converge"), std::string::npos)
            << error.what();
    }
}

TEST(PoissonReconstructionTest, AutoReportsWhenGpuAndCpuDoNotConverge)
{
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeSphereCloud(4096));
    reconstruction.setDepth(3);
    reconstruction.setSolverIterations(1);
    reconstruction.setSolverTolerance(1.0e-12);
    reconstruction.setProcessingDevice(plapoint::ProcessingDevice::Auto);
    try
    {
        static_cast<void>(reconstruction.reconstruct());
        FAIL() << "Expected Auto mode to report GPU and CPU non-convergence";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("GPU"), std::string::npos) << message;
        EXPECT_NE(message.find("CPU"), std::string::npos) << message;
        EXPECT_NE(message.find("residual"), std::string::npos) << message;
    }

    const auto& report = reconstruction.lastReport();
    EXPECT_TRUE(report.usedFallback);
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(report.solverDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_NE(report.fallbackReason.find("did not converge"), std::string::npos);
    EXPECT_FALSE(report.solver.converged);
}

TEST(PoissonReconstructionTest, CpuAndGpuProduceComparableSphereGeometry)
{
    auto cloud = makeSphereCloud(256);
    auto reconstruct = [&](plapoint::ProcessingDevice device)
    {
        plapoint::mesh::PoissonReconstruction<float> reconstruction;
        reconstruction.setInputCloud(cloud);
        reconstruction.setDepth(4);
        reconstruction.setSolverIterations(200);
        reconstruction.setSolverTolerance(1.0e-5);
        reconstruction.setProcessingDevice(device);
        auto mesh = reconstruction.reconstruct();
        return std::make_pair(std::move(mesh), reconstruction.lastReport());
    };

    auto [cpu_mesh, cpu_report] = reconstruct(plapoint::ProcessingDevice::CPU);
    auto [gpu_mesh, gpu_report] = reconstruct(plapoint::ProcessingDevice::GPU);
    const auto mean_radius_error = [](const auto& vertices)
    {
        double sum = 0.0;
        for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
        {
            const double x = vertices.getValue(row, 0);
            const double y = vertices.getValue(row, 1);
            const double z = vertices.getValue(row, 2);
            sum += std::abs(std::sqrt(x * x + y * y + z * z) - 1.0);
        }
        return sum / static_cast<double>(vertices.rows());
    };

    const auto& cpu_vertices = std::get<0>(cpu_mesh);
    const auto& gpu_vertices = std::get<0>(gpu_mesh);
    const auto& cpu_faces = std::get<1>(cpu_mesh);
    const auto& gpu_faces = std::get<1>(gpu_mesh);
    ASSERT_GT(cpu_vertices.rows(), 0);
    ASSERT_GT(gpu_vertices.rows(), 0);
    EXPECT_TRUE(cpu_report.solver.converged);
    EXPECT_TRUE(gpu_report.solver.converged);
    EXPECT_NEAR(cpu_report.isoLevel, gpu_report.isoLevel, 1.0e-4);
    EXPECT_NEAR(mean_radius_error(cpu_vertices), mean_radius_error(gpu_vertices), 0.02);
    EXPECT_LT(std::abs(cpu_vertices.rows() - gpu_vertices.rows()), cpu_vertices.rows() / 5 + 1);
    EXPECT_LT(std::abs(cpu_faces.rows() - gpu_faces.rows()), cpu_faces.rows() / 5 + 1);
}
#endif

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
