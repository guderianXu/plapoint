#include <gtest/gtest.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plamatrix/plamatrix.h>
#include <cmath>

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
