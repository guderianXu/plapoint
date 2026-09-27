#include <gtest/gtest.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/opencl/opencl_runtime.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "quality/mesh_quality_utils.h"

namespace
{

    using FloatCloud = plapoint::GeometryCloud<float>;
    using FloatMatrix = plamatrix::MatrixXf;

    template <typename SparseMatrix> auto sparseEntries(const SparseMatrix& matrix)
    {
        std::vector<std::tuple<plamatrix::Index, plamatrix::Index, typename SparseMatrix::Scalar>> result;
        result.reserve(static_cast<std::size_t>(matrix.nonZeros()));
        for (plamatrix::Index outer = 0; outer < matrix.outerSize(); ++outer)
        {
            for (typename SparseMatrix::InnerIterator entry(matrix, outer); entry; ++entry)
            {
                result.emplace_back(entry.row(), entry.col(), entry.value());
            }
        }
        return result;
    }

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
            points.operator()(i, 0) = x;
            points.operator()(i, 1) = y;
            points.operator()(i, 2) = z;
            normals.operator()(i, 0) = x;
            normals.operator()(i, 1) = y;
            normals.operator()(i, 2) = z;
        }
        auto cloud = std::make_shared<FloatCloud>(std::move(points));
        cloud->setNormals(std::move(normals));
        return cloud;
    }

    std::shared_ptr<FloatCloud> makeRegularSphereCloud(int rings, int segments)
    {
        const int count = rings * segments;
        FloatMatrix points(count, 3);
        FloatMatrix normals(count, 3);
        constexpr float pi = 3.14159265358979323846f;
        int row = 0;
        for (int ring = 0; ring < rings; ++ring)
        {
            const float phi = static_cast<float>(ring + 1) * pi / static_cast<float>(rings + 1);
            for (int segment = 0; segment < segments; ++segment)
            {
                const float theta = static_cast<float>(segment) * 2.0f * pi / static_cast<float>(segments);
                const float x = std::sin(phi) * std::cos(theta);
                const float y = std::sin(phi) * std::sin(theta);
                const float z = std::cos(phi);
                points.operator()(row, 0) = 2.0f * x;
                points.operator()(row, 1) = 2.0f * y;
                points.operator()(row, 2) = 2.0f * z;
                normals.operator()(row, 0) = x;
                normals.operator()(row, 1) = y;
                normals.operator()(row, 2) = z;
                ++row;
            }
        }
        auto cloud = std::make_shared<FloatCloud>(std::move(points));
        cloud->setNormals(std::move(normals));
        return cloud;
    }

    static_assert(std::is_copy_constructible_v<plapoint::mesh::PoissonReconstruction<float>>);
    static_assert(std::is_copy_assignable_v<plapoint::mesh::PoissonReconstruction<float>>);

    template <typename Fn> void expectInvalidArgumentContaining(Fn&& fn, const std::string& expected_message)
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
    using Cloud = plapoint::GeometryCloud<Scalar>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

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
        pts.operator()(i, 0) = x; pts.operator()(i, 1) = y; pts.operator()(i, 2) = z;
        Scalar r = std::sqrt(x*x + y*y + z*z);
        nrm.operator()(i, 0) = x/r; nrm.operator()(i, 1) = y/r; nrm.operator()(i, 2) = z/r;
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
    using Sparse = std::decay_t<decltype(matrix)>;
    ASSERT_EQ(matrix.rows(), static_cast<plamatrix::Index>(system.leafNodes.size()));
    ASSERT_EQ(matrix.cols(), matrix.rows());
    ASSERT_EQ(system.rhs.rows(), matrix.rows());
    std::map<std::pair<plamatrix::Index, plamatrix::Index>, Scalar> entries;
    for (plamatrix::Index row = 0; row < matrix.rows(); ++row)
    {
        Scalar diagonal = Scalar(0);
        plamatrix::Index previous_column = -1;
        for (Sparse::InnerIterator entry(matrix, row); entry; ++entry)
        {
            const auto column = entry.col();
            EXPECT_GT(column, previous_column);
            previous_column = column;
            entries[{row, column}] = entry.value();
            if (column == row)
            {
                diagonal = entry.value();
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
        for (Sparse::InnerIterator entry(matrix, row); entry; ++entry)
        {
            const auto column = entry.col();
            const long double x_column = static_cast<long double>((column % 7) - 3);
            quadratic += x_row * static_cast<long double>(entry.value()) * x_column;
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
    auto [first_vertices, first_faces] = reconstruction.reconstruct();
    const auto& first = reconstruction.lastSystem();
    const std::vector<float> first_vertex_values(first_vertices.data(), first_vertices.data() + first_vertices.size());
    const std::vector<float> first_face_values(first_faces.data(), first_faces.data() + first_faces.size());
    const auto first_entries = sparseEntries(first.matrix);
    const std::vector<float> first_rhs(first.rhs.data(), first.rhs.data() + first.rhs.size());

    auto [second_vertices, second_faces] = reconstruction.reconstruct();
    const auto& second = reconstruction.lastSystem();
    EXPECT_EQ(first_vertex_values,
              std::vector<float>(second_vertices.data(), second_vertices.data() + second_vertices.size()));
    EXPECT_EQ(first_face_values, std::vector<float>(second_faces.data(), second_faces.data() + second_faces.size()));
    EXPECT_EQ(first_entries, sparseEntries(second.matrix));
    EXPECT_EQ(first_rhs, std::vector<float>(second.rhs.data(), second.rhs.data() + second.rhs.size()));
}

TEST(PoissonReconstructionTest, SparseOctreeIsCompletedAndTwoToOneBalanced)
{
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeSphereCloud(1));
    reconstruction.setDepth(5);
    reconstruction.setSolverIterations(1);

    static_cast<void>(reconstruction.reconstruct());

    const auto& report = reconstruction.lastReport();
    EXPECT_EQ(report.leafCount, 1u);
    EXPECT_EQ(reconstruction.lastSystem().leafNodes.size(), 1u);
    EXPECT_GT(report.octreeLeafCount, 1u);
    EXPECT_LT(report.octreeLeafCount, 1u << 15);
    EXPECT_LE(report.maximumLeafNeighborDepthDifference, 1);
}

TEST(PoissonReconstructionTest, SamplesExtractionGridOnceAndBuildsOneOrientationIndex)
{
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeSphereCloud(96));
    reconstruction.setDepth(3);
    reconstruction.setSolverIterations(30);

    const auto [vertices, faces] = reconstruction.reconstruct();

    ASSERT_GT(vertices.rows(), 0);
    ASSERT_GT(faces.rows(), 0);
    const auto& report = reconstruction.lastReport();
    EXPECT_EQ(report.fieldGridSampleCount, 9u * 9u * 9u);
    EXPECT_EQ(report.orientationIndexBuildCount, 1u);
    EXPECT_GT(report.orientationQueryCount, 0u);
    EXPECT_LE(report.orientationQueryCount, static_cast<std::size_t>(faces.rows()));
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
    using Cloud = plapoint::GeometryCloud<Scalar>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

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
        pts.operator()(i, 0) = center + radius * nx;
        pts.operator()(i, 1) = center + radius * ny;
        pts.operator()(i, 2) = center + radius * nz;
        nrm.operator()(i, 0) = nx;
        nrm.operator()(i, 1) = ny;
        nrm.operator()(i, 2) = nz;
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
            const Scalar value = verts.operator()(r, c);
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

#ifdef PLAPOINT_WITH_OPENCL
TEST(PoissonReconstructionTest, ExplicitOpenClUsesPlaMatrixPcgWithoutFallback)
{
    if (!plapoint::opencl::hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU";
    }
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeRegularSphereCloud(12, 12));
    reconstruction.setDepth(3);
    reconstruction.setSolverIterations(200);
    reconstruction.setSolverTolerance(1.0e-5);
    reconstruction.setProcessingDevice(plapoint::ProcessingDevice::OpenCL);
    auto [vertices, faces] = reconstruction.reconstruct();

    EXPECT_GT(vertices.rows(), 0);
    EXPECT_GT(faces.rows(), 0);
    EXPECT_EQ(reconstruction.lastReport().requestedDevice, plapoint::ProcessingDevice::OpenCL);
    EXPECT_EQ(reconstruction.lastReport().solverDevice, plapoint::ProcessingDevice::OpenCL);
    EXPECT_FALSE(reconstruction.lastReport().usedFallback);
    EXPECT_TRUE(reconstruction.lastReport().solver.converged);
}
#else
TEST(PoissonReconstructionTest, ExplicitOpenClRejectsDisabledBackendWithoutFallback)
{
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeSphereCloud(48));
    reconstruction.setDepth(3);
    reconstruction.setProcessingDevice(plapoint::ProcessingDevice::OpenCL);
    EXPECT_THROW(static_cast<void>(reconstruction.reconstruct()), std::runtime_error);
}
#endif

#ifdef PLAPOINT_WITH_CUDA
TEST(PoissonReconstructionTest, ExplicitGpuUsesPlaMatrixPcg)
{
    using Scalar = float;
    using Cloud = plapoint::GeometryCloud<Scalar>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;
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
        points.operator()(i, 0) = x; points.operator()(i, 1) = y; points.operator()(i, 2) = z;
        normals.operator()(i, 0) = x; normals.operator()(i, 1) = y; normals.operator()(i, 2) = z;
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

TEST(PoissonReconstructionTest, AutoContinuesCpuFromCudaApproximation)
{
    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(makeRegularSphereCloud(16, 16));
    reconstruction.setDepth(4);
    reconstruction.setSolverIterations(30);
    reconstruction.setSolverTolerance(1.0e-6);
    reconstruction.setProcessingDevice(plapoint::ProcessingDevice::Auto);

    auto [vertices, faces] = reconstruction.reconstruct();
    const auto& report = reconstruction.lastReport();

    EXPECT_GT(vertices.rows(), 0);
    EXPECT_GT(faces.rows(), 0);
    EXPECT_TRUE(report.usedFallback);
    EXPECT_EQ(report.solverDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_TRUE(report.solver.converged);
    EXPECT_NE(report.fallbackReason.find("did not converge"), std::string::npos);
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
            const double x = vertices.operator()(row, 0);
            const double y = vertices.operator()(row, 1);
            const double z = vertices.operator()(row, 2);
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
    using Cloud = plapoint::GeometryCloud<Scalar>;

    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    points.operator()(0, 0) = 0;
    points.operator()(0, 1) = 0;
    points.operator()(0, 2) = 0;
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);

    EXPECT_THROW((void)pr.reconstruct(), std::runtime_error);
}

TEST(PoissonReconstructionTest, RejectsEmptyInputCloud)
{
    using Scalar = float;
    using Cloud = plapoint::GeometryCloud<Scalar>;

    auto cloud = std::make_shared<Cloud>(0);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(0, 3);
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);

    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);
}

TEST(PoissonReconstructionTest, SinglePointWithNormalProducesEmptyDegenerateMesh)
{
    using Scalar = float;
    using Cloud = plapoint::GeometryCloud<Scalar>;

    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);
    points.operator()(0, 0) = 0;
    points.operator()(0, 1) = 0;
    points.operator()(0, 2) = 0;
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(1, 3);
    normals.operator()(0, 0) = 0;
    normals.operator()(0, 1) = 0;
    normals.operator()(0, 2) = 1;
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
    using Cloud = plapoint::GeometryCloud<Scalar>;

    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    points.setConstant(0);
    points.operator()(1, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(2, 3);
    normals.setConstant(0);
    normals.operator()(0, 2) = 1;
    normals.operator()(1, 2) = 1;
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);

    auto finite_points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    finite_points.setConstant(0);
    auto cloud_with_bad_normals = std::make_shared<Cloud>(std::move(finite_points));
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> bad_normals(2, 3);
    bad_normals.setConstant(0);
    bad_normals.operator()(0, 2) = 1;
    bad_normals.operator()(1, 2) = std::numeric_limits<Scalar>::infinity();
    cloud_with_bad_normals->setNormals(std::move(bad_normals));

    pr.setInputCloud(cloud_with_bad_normals);
    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);
}

TEST(PoissonReconstructionTest, RejectsZeroLengthNormals)
{
    using Scalar = float;
    using Cloud = plapoint::GeometryCloud<Scalar>;

    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    points.setConstant(0);
    points.operator()(1, 0) = 1;
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(2, 3);
    normals.setConstant(0);
    normals.operator()(0, 2) = 1;
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);

    EXPECT_THROW((void)pr.reconstruct(), std::invalid_argument);
}

TEST(PoissonReconstructionTest, RejectsNormalsWhoseLengthIsNotFinite)
{
    using Scalar = double;
    using Cloud = plapoint::GeometryCloud<Scalar>;

    auto points = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>(2, 3);
    points.setConstant(0);
    points.operator()(1, 0) = 1;
    auto cloud = std::make_shared<Cloud>(std::move(points));

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(2, 3);
    normals.setConstant(0);
    normals.operator()(0, 0) = std::numeric_limits<Scalar>::max();
    normals.operator()(0, 1) = std::numeric_limits<Scalar>::max();
    normals.operator()(0, 2) = std::numeric_limits<Scalar>::max();
    normals.operator()(1, 2) = 1;
    cloud->setNormals(std::move(normals));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    pr.setDepth(1);
    pr.setSolverIterations(1);

    expectInvalidArgumentContaining(
        [&]() { (void)pr.reconstruct(); },
        "normals");
}
