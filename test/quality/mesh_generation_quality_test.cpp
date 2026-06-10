#include <gtest/gtest.h>

#include "quality/mesh_quality_utils.h"

TEST(MeshGenerationQualityTest, MarchingCubesSphereMeetsGeometricQualityBounds)
{
    using Scalar = float;

    const auto mesh = plapoint::test::mesh_quality::generateMarchingCubesSphere<Scalar>(
        Scalar(2), 40);
    const auto metrics = plapoint::test::mesh_quality::measureSphereMesh(
        mesh.vertices, mesh.faces, Scalar(2));

    EXPECT_GT(metrics.vertex_count, 1000);
    EXPECT_GT(metrics.face_count, 1500);
    EXPECT_EQ(metrics.invalid_face_count, 0);
    EXPECT_EQ(metrics.degenerate_face_count, 0);
    EXPECT_LT(metrics.max_radius_error, 0.08);
    EXPECT_LT(metrics.mean_radius_error, 0.02);
    EXPECT_GT(metrics.dominant_orientation_ratio, 0.95);
}

TEST(MeshGenerationQualityTest, PoissonSphereProducesUsableMeshForInspection)
{
    using Scalar = float;

    const auto mesh = plapoint::test::mesh_quality::generatePoissonSphere<Scalar>(
        Scalar(2), 12, 24, 5, 30);
    const auto metrics = plapoint::test::mesh_quality::measureSphereMesh(
        mesh.vertices, mesh.faces, Scalar(2));

    EXPECT_GT(metrics.vertex_count, 50);
    EXPECT_GT(metrics.face_count, 50);
    EXPECT_EQ(metrics.invalid_face_count, 0);
    EXPECT_LT(metrics.degenerate_face_ratio, 0.05);
    EXPECT_LT(metrics.mean_radius_error, 1.50);
    EXPECT_LT(metrics.max_radius_error, 2.50);
    EXPECT_LT(metrics.max_abs_coordinate, 3.5);
}
