#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <plamatrix/plamatrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plapoint/mesh/poisson_reconstruction.h>

namespace plapoint::test::mesh_quality
{

/// Return a constexpr pi value cast to the requested scalar type.
template <typename Scalar>
constexpr Scalar pi()
{
    return Scalar(3.14159265358979323846264338327950288L);
}

/// Triangle mesh represented as vertex and face matrices returned by mesh algorithms.
template <typename Scalar>
struct Mesh
{
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix vertices;
    Matrix faces;
};

/// Aggregate geometry quality metrics for a mesh expected to approximate a sphere.
struct SphereMeshMetrics
{
    int vertex_count = 0;
    int face_count = 0;
    int invalid_face_count = 0;
    int degenerate_face_count = 0;
    double degenerate_face_ratio = 1.0;
    double max_radius_error = 0.0;
    double mean_radius_error = 0.0;
    double max_abs_coordinate = 0.0;
    double dominant_orientation_ratio = 0.0;
    double outward_orientation_ratio = 0.0;
};

namespace detail
{

template <typename Scalar>
void accumulateVertexMetrics(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& vertices,
    plamatrix::Index row,
    Scalar radius,
    SphereMeshMetrics& metrics,
    double& radius_error_sum)
{
    const double x = static_cast<double>(vertices.getValue(row, 0));
    const double y = static_cast<double>(vertices.getValue(row, 1));
    const double z = static_cast<double>(vertices.getValue(row, 2));
    metrics.max_abs_coordinate = std::max(metrics.max_abs_coordinate, std::abs(x));
    metrics.max_abs_coordinate = std::max(metrics.max_abs_coordinate, std::abs(y));
    metrics.max_abs_coordinate = std::max(metrics.max_abs_coordinate, std::abs(z));

    const double r = std::sqrt(x * x + y * y + z * z);
    const double error = std::abs(r - static_cast<double>(radius));
    metrics.max_radius_error = std::max(metrics.max_radius_error, error);
    radius_error_sum += error;
}

template <typename Scalar>
bool readFaceIndices(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& faces,
    plamatrix::Index row,
    int vertex_count,
    int (&idx)[3])
{
    for (int c = 0; c < 3; ++c)
    {
        const double raw = static_cast<double>(faces.getValue(row, c));
        const double rounded = std::round(raw);
        if (!std::isfinite(raw) || std::abs(raw - rounded) > 1e-4)
        {
            return false;
        }
        idx[c] = static_cast<int>(rounded);
        if (idx[c] < 0 || idx[c] >= vertex_count)
        {
            return false;
        }
    }
    return true;
}

template <typename Scalar>
void accumulateFaceMetrics(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& vertices,
    const int (&idx)[3],
    SphereMeshMetrics& metrics,
    int& outward_count,
    int& inward_count)
{
    constexpr double kDegenerateArea = 1e-10;
    constexpr double kOrientationEpsilon = 1e-12;

    const double ax = static_cast<double>(vertices.getValue(idx[0], 0));
    const double ay = static_cast<double>(vertices.getValue(idx[0], 1));
    const double az = static_cast<double>(vertices.getValue(idx[0], 2));
    const double bx = static_cast<double>(vertices.getValue(idx[1], 0));
    const double by = static_cast<double>(vertices.getValue(idx[1], 1));
    const double bz = static_cast<double>(vertices.getValue(idx[1], 2));
    const double cx = static_cast<double>(vertices.getValue(idx[2], 0));
    const double cy = static_cast<double>(vertices.getValue(idx[2], 1));
    const double cz = static_cast<double>(vertices.getValue(idx[2], 2));

    const double ux = bx - ax;
    const double uy = by - ay;
    const double uz = bz - az;
    const double vx = cx - ax;
    const double vy = cy - ay;
    const double vz = cz - az;
    const double nx = uy * vz - uz * vy;
    const double ny = uz * vx - ux * vz;
    const double nz = ux * vy - uy * vx;
    const double double_area = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (double_area <= kDegenerateArea)
    {
        ++metrics.degenerate_face_count;
        return;
    }

    const double center_x = (ax + bx + cx) / 3.0;
    const double center_y = (ay + by + cy) / 3.0;
    const double center_z = (az + bz + cz) / 3.0;
    const double orientation = nx * center_x + ny * center_y + nz * center_z;
    if (orientation > kOrientationEpsilon)
    {
        ++outward_count;
    }
    else if (orientation < -kOrientationEpsilon)
    {
        ++inward_count;
    }
}

} // namespace detail

/// Generate a sphere mesh from an implicit signed field with Marching Cubes.
template <typename Scalar>
Mesh<Scalar> generateMarchingCubesSphere(Scalar radius, int resolution)
{
    plapoint::mesh::MarchingCubes<Scalar> mc;
    const Scalar half_extent = radius * Scalar(1.5);
    mc.setBounds({-half_extent, -half_extent, -half_extent},
                 { half_extent,  half_extent,  half_extent});
    mc.setResolution(resolution, resolution, resolution);
    mc.setIsoLevel(Scalar(0));

    auto [vertices, faces] = mc.extract([radius](Scalar x, Scalar y, Scalar z)
    {
        return x * x + y * y + z * z - radius * radius;
    });

    return {std::move(vertices), std::move(faces)};
}

/// Generate a point cloud sampled on a sphere with outward normals for reconstruction tests.
template <typename Scalar>
std::shared_ptr<plapoint::PointCloud<Scalar, plamatrix::Device::CPU>>
makeSpherePointCloud(Scalar radius, int rings, int segments)
{
    if (rings < 3 || segments < 3)
    {
        throw std::invalid_argument("sphere point cloud requires at least 3 rings and 3 segments");
    }

    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;
    const int point_count = 2 + (rings - 1) * segments;
    Matrix points(point_count, 3);
    Matrix normals(point_count, 3);

    int row = 0;
    const auto write_point = [&](Scalar x, Scalar y, Scalar z)
    {
        points.setValue(row, 0, x);
        points.setValue(row, 1, y);
        points.setValue(row, 2, z);

        const Scalar inv_radius = Scalar(1) / radius;
        normals.setValue(row, 0, x * inv_radius);
        normals.setValue(row, 1, y * inv_radius);
        normals.setValue(row, 2, z * inv_radius);
        ++row;
    };

    write_point(Scalar(0), Scalar(0), radius);
    for (int r = 1; r < rings; ++r)
    {
        const Scalar theta = pi<Scalar>() * Scalar(r) / Scalar(rings);
        const Scalar sin_theta = std::sin(theta);
        const Scalar cos_theta = std::cos(theta);
        for (int s = 0; s < segments; ++s)
        {
            const Scalar phi = Scalar(2) * pi<Scalar>() * Scalar(s) / Scalar(segments);
            write_point(radius * sin_theta * std::cos(phi),
                        radius * sin_theta * std::sin(phi),
                        radius * cos_theta);
        }
    }
    write_point(Scalar(0), Scalar(0), -radius);

    auto cloud = std::make_shared<plapoint::PointCloud<Scalar, plamatrix::Device::CPU>>(
        std::move(points));
    cloud->setNormals(std::move(normals));
    return cloud;
}

/// Reconstruct a sphere-like mesh from sampled points and normals with Poisson reconstruction.
template <typename Scalar>
Mesh<Scalar> generatePoissonSphere(Scalar radius,
                                   int rings,
                                   int segments,
                                   int depth,
                                   int solver_iterations)
{
    auto cloud = makeSpherePointCloud(radius, rings, segments);

    plapoint::mesh::PoissonReconstruction<Scalar> reconstruction;
    reconstruction.setInputCloud(cloud);
    reconstruction.setDepth(depth);
    reconstruction.setSolverIterations(solver_iterations);

    auto [vertices, faces] = reconstruction.reconstruct();
    return {std::move(vertices), std::move(faces)};
}

/// Measure vertex errors, index validity, degenerate faces, and face orientation consistency.
template <typename Scalar>
SphereMeshMetrics measureSphereMesh(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& vertices,
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& faces,
    Scalar radius)
{
    SphereMeshMetrics metrics;
    metrics.vertex_count = static_cast<int>(vertices.rows());
    metrics.face_count = static_cast<int>(faces.rows());

    double radius_error_sum = 0.0;
    for (plamatrix::Index i = 0; i < vertices.rows(); ++i)
    {
        detail::accumulateVertexMetrics(vertices, i, radius, metrics, radius_error_sum);
    }
    if (vertices.rows() > 0)
    {
        metrics.mean_radius_error = radius_error_sum / static_cast<double>(vertices.rows());
    }

    int valid_face_count = 0;
    int outward_count = 0;
    int inward_count = 0;

    for (plamatrix::Index f = 0; f < faces.rows(); ++f)
    {
        int idx[3] = {0, 0, 0};
        if (!detail::readFaceIndices(faces, f, metrics.vertex_count, idx))
        {
            ++metrics.invalid_face_count;
            continue;
        }
        ++valid_face_count;
        detail::accumulateFaceMetrics(vertices, idx, metrics, outward_count, inward_count);
    }

    if (valid_face_count > 0)
    {
        metrics.degenerate_face_ratio =
            static_cast<double>(metrics.degenerate_face_count) /
            static_cast<double>(valid_face_count);
    }

    const int oriented_count = outward_count + inward_count;
    if (oriented_count > 0)
    {
        metrics.dominant_orientation_ratio =
            static_cast<double>(std::max(outward_count, inward_count)) /
            static_cast<double>(oriented_count);
        metrics.outward_orientation_ratio =
            static_cast<double>(outward_count) / static_cast<double>(oriented_count);
    }

    return metrics;
}

} // namespace plapoint::test::mesh_quality
