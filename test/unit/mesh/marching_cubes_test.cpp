#include <gtest/gtest.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plamatrix/plamatrix.h>
#include <cmath>
#include <functional>

TEST(MarchingCubesTest, SphereIsosurface)
{
    using Scalar = float;

    auto sphere_fn = [](Scalar x, Scalar y, Scalar z) {
        return x*x + y*y + z*z - Scalar(4);
    };

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setBounds({-3,-3,-3}, {3,3,3});
    mc.setResolution(20, 20, 20);
    mc.setIsoLevel(Scalar(0));

    auto [verts, faces] = mc.extract(sphere_fn);

    EXPECT_GT(verts.rows(), 0);
    EXPECT_GT(faces.rows(), 0);
    EXPECT_EQ(verts.cols(), 3);
    EXPECT_EQ(faces.cols(), 3);
}

TEST(MarchingCubesTest, ConstantFieldAboveIsoReturnsEmptyMesh)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setBounds({-1,-1,-1}, {1,1,1});
    mc.setResolution(4, 4, 4);
    mc.setIsoLevel(Scalar(0));

    auto [verts, faces] = mc.extract([](Scalar, Scalar, Scalar)
    {
        return Scalar(1);
    });

    EXPECT_EQ(verts.rows(), 0);
    EXPECT_EQ(verts.cols(), 3);
    EXPECT_EQ(faces.rows(), 0);
    EXPECT_EQ(faces.cols(), 3);
}

TEST(MarchingCubesTest, ConstantFieldEqualToIsoReturnsEmptyMesh)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setBounds({-1,-1,-1}, {1,1,1});
    mc.setResolution(4, 4, 4);
    mc.setIsoLevel(Scalar(0));

    auto [verts, faces] = mc.extract([](Scalar, Scalar, Scalar)
    {
        return Scalar(0);
    });

    EXPECT_EQ(verts.rows(), 0);
    EXPECT_EQ(verts.cols(), 3);
    EXPECT_EQ(faces.rows(), 0);
    EXPECT_EQ(faces.cols(), 3);
}

TEST(MarchingCubesTest, EmptyScalarFunctionIsRejected)
{
    using Scalar = float;
    using MarchingCubes = plapoint::mesh::MarchingCubes<Scalar>;

    MarchingCubes mc;
    mc.setResolution(1, 1, 1);

    EXPECT_THROW(
        (void)mc.extract(typename MarchingCubes::ScalarFunction{}),
        std::invalid_argument);
}

TEST(MarchingCubesTest, RejectsNonPositiveResolution)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;

    EXPECT_THROW(mc.setResolution(0, 4, 4), std::invalid_argument);
    EXPECT_THROW(mc.setResolution(4, -1, 4), std::invalid_argument);
    EXPECT_THROW(mc.setResolution(4, 4, 0), std::invalid_argument);
}
