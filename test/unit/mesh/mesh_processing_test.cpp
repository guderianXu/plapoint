#include <gtest/gtest.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/mesh_processing.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{

    using CpuFloatCloud = plapoint::GeometryCloud<float>;
    using FloatMatrix = plamatrix::MatrixXf;
    using IntMatrix = plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>;

    CpuFloatCloud makeCloud(FloatMatrix&& points, IntMatrix&& faces)
    {
        CpuFloatCloud cloud(std::move(points));
        cloud.setFaces(std::move(faces));
        return cloud;
    }

} // namespace

TEST(MeshProcessingTest, RemoveDegenerateFacesDropsZeroAreaFaces)
{
    FloatMatrix points(4, 3);
    points.operator()(0, 0) = 0.0f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = 1.0f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 0.0f; points.operator()(2, 1) = 1.0f; points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = 2.0f; points.operator()(3, 1) = 2.0f; points.operator()(3, 2) = 0.0f;

    IntMatrix faces(2, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    faces.operator()(1, 0) = 0;
    faces.operator()(1, 1) = 0;
    faces.operator()(1, 2) = 3;

    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(4, 3);
    colors.setConstant(7);

    auto mesh = makeCloud(std::move(points), std::move(faces));
    mesh.setColors(std::move(colors));

    auto cleaned = plapoint::mesh::removeDegenerateFaces(mesh);

    ASSERT_TRUE(cleaned.hasFaces());
    ASSERT_EQ(cleaned.faces()->rows(), 1);
    EXPECT_EQ(cleaned.faces()->operator()(0, 0), 0);
    EXPECT_EQ(cleaned.faces()->operator()(0, 1), 1);
    EXPECT_EQ(cleaned.faces()->operator()(0, 2), 2);
    ASSERT_TRUE(cleaned.hasColors());
    EXPECT_EQ(cleaned.colors()->rows(), 4);
}

TEST(MeshProcessingTest, RemoveSmallConnectedComponentsCompactsRemainingVertices)
{
    FloatMatrix points(7, 3);
    points.operator()(0, 0) = 0.0f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = 1.0f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 0.0f; points.operator()(2, 1) = 1.0f; points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = 1.0f; points.operator()(3, 1) = 1.0f; points.operator()(3, 2) = 0.0f;
    points.operator()(4, 0) = 10.0f; points.operator()(4, 1) = 0.0f; points.operator()(4, 2) = 0.0f;
    points.operator()(5, 0) = 11.0f; points.operator()(5, 1) = 0.0f; points.operator()(5, 2) = 0.0f;
    points.operator()(6, 0) = 10.0f; points.operator()(6, 1) = 1.0f; points.operator()(6, 2) = 0.0f;

    IntMatrix faces(3, 3);
    faces.operator()(0, 0) = 0; faces.operator()(0, 1) = 1; faces.operator()(0, 2) = 2;
    faces.operator()(1, 0) = 1; faces.operator()(1, 1) = 3; faces.operator()(1, 2) = 2;
    faces.operator()(2, 0) = 4; faces.operator()(2, 1) = 5; faces.operator()(2, 2) = 6;

    auto mesh = makeCloud(std::move(points), std::move(faces));

    auto cleaned = plapoint::mesh::removeSmallConnectedComponents(mesh, 2);

    ASSERT_TRUE(cleaned.hasFaces());
    EXPECT_EQ(cleaned.size(), 4u);
    ASSERT_EQ(cleaned.faces()->rows(), 2);
    for (plamatrix::Index r = 0; r < cleaned.faces()->rows(); ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_GE(cleaned.faces()->operator()(r, c), 0);
            EXPECT_LT(cleaned.faces()->operator()(r, c), 4);
        }
    }
}

TEST(MeshProcessingTest, RemoveSmallConnectedComponentsPreservesScalarFields)
{
    FloatMatrix points(7, 3);
    points.operator()(0, 0) = 0.0f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = 1.0f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 0.0f; points.operator()(2, 1) = 1.0f; points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = 1.0f; points.operator()(3, 1) = 1.0f; points.operator()(3, 2) = 0.0f;
    points.operator()(4, 0) = 10.0f; points.operator()(4, 1) = 0.0f; points.operator()(4, 2) = 0.0f;
    points.operator()(5, 0) = 11.0f; points.operator()(5, 1) = 0.0f; points.operator()(5, 2) = 0.0f;
    points.operator()(6, 0) = 10.0f; points.operator()(6, 1) = 1.0f; points.operator()(6, 2) = 0.0f;

    IntMatrix faces(3, 3);
    faces.operator()(0, 0) = 0; faces.operator()(0, 1) = 1; faces.operator()(0, 2) = 2;
    faces.operator()(1, 0) = 1; faces.operator()(1, 1) = 3; faces.operator()(1, 2) = 2;
    faces.operator()(2, 0) = 4; faces.operator()(2, 1) = 5; faces.operator()(2, 2) = 6;

    FloatMatrix scalar_fields(7, 1);
    for (int i = 0; i < 7; ++i)
    {
        scalar_fields.operator()(i, 0) = static_cast<float>(100 + i);
    }

    auto mesh = makeCloud(std::move(points), std::move(faces));
    mesh.setScalarFields({"error"}, std::move(scalar_fields));

    auto cleaned = plapoint::mesh::removeSmallConnectedComponents(mesh, 2);

    ASSERT_EQ(cleaned.size(), 4u);
    ASSERT_TRUE(cleaned.hasScalarField("error"));
    EXPECT_EQ(cleaned.scalarFieldNames(), (std::vector<std::string>{"error"}));
    EXPECT_FLOAT_EQ(cleaned.scalarFields()->operator()(0, 0), 100.0f);
    EXPECT_FLOAT_EQ(cleaned.scalarFields()->operator()(1, 0), 101.0f);
    EXPECT_FLOAT_EQ(cleaned.scalarFields()->operator()(2, 0), 102.0f);
    EXPECT_FLOAT_EQ(cleaned.scalarFields()->operator()(3, 0), 103.0f);
}

TEST(MeshProcessingTest, ConnectedComponentsRequireSharedEdges)
{
    FloatMatrix points(5, 3);
    points.operator()(0, 0) = 0.0f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = 1.0f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 0.0f; points.operator()(2, 1) = 1.0f; points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = -1.0f; points.operator()(3, 1) = 0.0f; points.operator()(3, 2) = 0.0f;
    points.operator()(4, 0) = 0.0f; points.operator()(4, 1) = -1.0f; points.operator()(4, 2) = 0.0f;

    IntMatrix faces(2, 3);
    faces.operator()(0, 0) = 0; faces.operator()(0, 1) = 1; faces.operator()(0, 2) = 2;
    faces.operator()(1, 0) = 0; faces.operator()(1, 1) = 3; faces.operator()(1, 2) = 4;

    auto mesh = makeCloud(std::move(points), std::move(faces));

    auto cleaned = plapoint::mesh::removeSmallConnectedComponents(mesh, 2);

    ASSERT_TRUE(cleaned.hasFaces());
    EXPECT_EQ(cleaned.faces()->rows(), 0);
    EXPECT_EQ(cleaned.size(), 0u);
}

TEST(MeshProcessingTest, RecomputeVertexNormalsAveragesIncidentFaceNormals)
{
    FloatMatrix points(4, 3);
    points.operator()(0, 0) = 0.0f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = 1.0f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 0.0f; points.operator()(2, 1) = 1.0f; points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = 1.0f; points.operator()(3, 1) = 1.0f; points.operator()(3, 2) = 0.0f;

    IntMatrix faces(2, 3);
    faces.operator()(0, 0) = 0; faces.operator()(0, 1) = 1; faces.operator()(0, 2) = 2;
    faces.operator()(1, 0) = 1; faces.operator()(1, 1) = 3; faces.operator()(1, 2) = 2;

    auto mesh = makeCloud(std::move(points), std::move(faces));

    auto with_normals = plapoint::mesh::recomputeVertexNormals(mesh);

    ASSERT_TRUE(with_normals.hasNormals());
    ASSERT_EQ(with_normals.normals()->rows(), 4);
    for (plamatrix::Index i = 0; i < with_normals.normals()->rows(); ++i)
    {
        EXPECT_NEAR(with_normals.normals()->operator()(i, 0), 0.0f, 1.0e-6f);
        EXPECT_NEAR(with_normals.normals()->operator()(i, 1), 0.0f, 1.0e-6f);
        EXPECT_NEAR(with_normals.normals()->operator()(i, 2), 1.0f, 1.0e-6f);
    }
}

TEST(MeshProcessingTest, OrientNormalsOutwardFromCentroidFlipsInwardNormals)
{
    FloatMatrix points(2, 3);
    points.operator()(0, 0) = 0.0f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 1.0f;
    points.operator()(1, 0) = 0.0f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = -1.0f;

    FloatMatrix normals(2, 3);
    normals.operator()(0, 0) = 0.0f; normals.operator()(0, 1) = 0.0f; normals.operator()(0, 2) = -1.0f;
    normals.operator()(1, 0) = 0.0f; normals.operator()(1, 1) = 0.0f; normals.operator()(1, 2) = 1.0f;

    IntMatrix faces(0, 3);
    CpuFloatCloud mesh(std::move(points));
    mesh.setNormals(std::move(normals));
    mesh.setFaces(std::move(faces));

    auto oriented = plapoint::mesh::orientNormalsOutwardFromCentroid(mesh);

    ASSERT_TRUE(oriented.hasNormals());
    EXPECT_FLOAT_EQ(oriented.normals()->operator()(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(oriented.normals()->operator()(1, 2), -1.0f);
}

TEST(MeshProcessingTest, VoxelClusterSimplifyMergesNearbyVerticesAndDropsCollapsedFaces)
{
    FloatMatrix points(4, 3);
    points.operator()(0, 0) = 0.00f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = 0.10f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 1.00f;
    points.operator()(2, 1) = 0.0f;
    points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = 0.00f;
    points.operator()(3, 1) = 1.0f;
    points.operator()(3, 2) = 0.0f;

    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(4, 3);
    colors.operator()(0, 0) = 10;
    colors.operator()(0, 1) = 20;
    colors.operator()(0, 2) = 30;
    colors.operator()(1, 0) = 14;
    colors.operator()(1, 1) = 24;
    colors.operator()(1, 2) = 34;
    colors.operator()(2, 0) = 100;
    colors.operator()(2, 1) = 110;
    colors.operator()(2, 2) = 120;
    colors.operator()(3, 0) = 200; colors.operator()(3, 1) = 210; colors.operator()(3, 2) = 220;

    IntMatrix faces(2, 3);
    faces.operator()(0, 0) = 0; faces.operator()(0, 1) = 2; faces.operator()(0, 2) = 3;
    faces.operator()(1, 0) = 0; faces.operator()(1, 1) = 1; faces.operator()(1, 2) = 2;

    auto mesh = makeCloud(std::move(points), std::move(faces));
    mesh.setColors(std::move(colors));

    auto simplified = plapoint::mesh::voxelClusterSimplify(mesh, 0.25f);

    ASSERT_TRUE(simplified.hasFaces());
    EXPECT_EQ(simplified.size(), 3u);
    ASSERT_EQ(simplified.faces()->rows(), 1);
    ASSERT_TRUE(simplified.hasColors());
    EXPECT_NEAR(simplified.points().operator()(0, 0), 0.05f, 1.0e-6f);
    EXPECT_EQ(simplified.colors()->operator()(0, 0), 12);
}

TEST(MeshProcessingTest, TaubinSmoothMovesInteriorVertexAndPreservesFaces)
{
    FloatMatrix points(5, 3);
    points.operator()(0, 0) = 0.0f; points.operator()(0, 1) = 0.0f; points.operator()(0, 2) = 1.0f;
    points.operator()(1, 0) = -1.0f; points.operator()(1, 1) = 0.0f; points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 0.0f; points.operator()(2, 1) = -1.0f; points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = 1.0f; points.operator()(3, 1) = 0.0f; points.operator()(3, 2) = 0.0f;
    points.operator()(4, 0) = 0.0f; points.operator()(4, 1) = 1.0f; points.operator()(4, 2) = 0.0f;

    IntMatrix faces(4, 3);
    faces.operator()(0, 0) = 0; faces.operator()(0, 1) = 1; faces.operator()(0, 2) = 2;
    faces.operator()(1, 0) = 0; faces.operator()(1, 1) = 2; faces.operator()(1, 2) = 3;
    faces.operator()(2, 0) = 0; faces.operator()(2, 1) = 3; faces.operator()(2, 2) = 4;
    faces.operator()(3, 0) = 0; faces.operator()(3, 1) = 4; faces.operator()(3, 2) = 1;

    auto mesh = makeCloud(std::move(points), std::move(faces));

    auto smoothed = plapoint::mesh::taubinSmooth(mesh, 1, 0.5f, 0.0f);

    ASSERT_TRUE(smoothed.hasFaces());
    ASSERT_EQ(smoothed.faces()->rows(), 4);
    EXPECT_LT(smoothed.points().operator()(0, 2), 1.0f);
    EXPECT_GT(smoothed.points().operator()(0, 2), 0.0f);
}
