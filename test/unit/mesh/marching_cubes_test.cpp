#include <gtest/gtest.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plamatrix/plamatrix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace
{

using QuantizedTriangle = std::array<std::array<int, 3>, 3>;

int cornerIndexForUnitCube(float x, float y, float z)
{
    const bool high_x = x > 0.5f;
    const bool high_y = y > 0.5f;
    const bool high_z = z > 0.5f;
    if (!high_z)
    {
        return high_y ? (high_x ? 2 : 3) : (high_x ? 1 : 0);
    }
    return high_y ? (high_x ? 6 : 7) : (high_x ? 5 : 4);
}

std::vector<QuantizedTriangle> normalizedSingleCubeGeometry(int cube_case)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setBounds({0, 0, 0}, {1, 1, 1});
    mc.setResolution(1, 1, 1);
    mc.setIsoLevel(Scalar(0));

    auto [verts, faces] = mc.extract([cube_case](Scalar x, Scalar y, Scalar z)
    {
        const int corner = cornerIndexForUnitCube(x, y, z);
        return (cube_case & (1 << corner)) != 0 ? Scalar(-1) : Scalar(1);
    });

    std::vector<QuantizedTriangle> triangles;
    triangles.reserve(static_cast<std::size_t>(faces.rows()));
    for (plamatrix::Index face = 0; face < faces.rows(); ++face)
    {
        QuantizedTriangle tri{};
        for (plamatrix::Index col = 0; col < 3; ++col)
        {
            const auto vertex = static_cast<plamatrix::Index>(faces.getValue(face, col));
            tri[static_cast<std::size_t>(col)] = {
                static_cast<int>(std::lround(verts.getValue(vertex, 0) * 2.0f)),
                static_cast<int>(std::lround(verts.getValue(vertex, 1) * 2.0f)),
                static_cast<int>(std::lround(verts.getValue(vertex, 2) * 2.0f))
            };
        }
        std::sort(tri.begin(), tri.end());
        triangles.push_back(tri);
    }
    std::sort(triangles.begin(), triangles.end());
    return triangles;
}

double triangleArea(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& verts,
    plamatrix::Index face_row,
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& faces)
{
    const auto ia = static_cast<plamatrix::Index>(faces.getValue(face_row, 0));
    const auto ib = static_cast<plamatrix::Index>(faces.getValue(face_row, 1));
    const auto ic = static_cast<plamatrix::Index>(faces.getValue(face_row, 2));

    const double ax = verts.getValue(ia, 0);
    const double ay = verts.getValue(ia, 1);
    const double az = verts.getValue(ia, 2);
    const double bx = verts.getValue(ib, 0);
    const double by = verts.getValue(ib, 1);
    const double bz = verts.getValue(ib, 2);
    const double cx = verts.getValue(ic, 0);
    const double cy = verts.getValue(ic, 1);
    const double cz = verts.getValue(ic, 2);

    const double ux = bx - ax;
    const double uy = by - ay;
    const double uz = bz - az;
    const double vx = cx - ax;
    const double vy = cy - ay;
    const double vz = cz - az;
    const double cross_x = uy * vz - uz * vy;
    const double cross_y = uz * vx - ux * vz;
    const double cross_z = ux * vy - uy * vx;
    return 0.5 * std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
}

} // namespace

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

TEST(MarchingCubesTest, RejectsResolutionWhoseSampleCountOverflows)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setResolution(std::numeric_limits<int>::max(), 2, 2);

    EXPECT_THROW(
        (void)mc.extract([](Scalar, Scalar, Scalar)
        {
            return Scalar(1);
        }),
        std::invalid_argument);
}

TEST(MarchingCubesTest, RejectsDegenerateOrNonFiniteBounds)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;

    EXPECT_THROW(mc.setBounds({0, -1, -1}, {0, 1, 1}), std::invalid_argument);
    EXPECT_THROW(mc.setBounds({-1, std::numeric_limits<Scalar>::quiet_NaN(), -1}, {1, 1, 1}),
                 std::invalid_argument);
    EXPECT_THROW(mc.setBounds({-1, -1, -1}, {1, std::numeric_limits<Scalar>::infinity(), 1}),
                 std::invalid_argument);
}

TEST(MarchingCubesTest, RejectsNonFiniteIsoLevel)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;

    EXPECT_THROW(mc.setIsoLevel(std::numeric_limits<Scalar>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(mc.setIsoLevel(std::numeric_limits<Scalar>::infinity()), std::invalid_argument);
}

TEST(MarchingCubesTest, RejectsNonFiniteScalarFunctionSamples)
{
    using Scalar = float;

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setBounds({-1, -1, -1}, {1, 1, 1});
    mc.setResolution(2, 2, 2);
    mc.setIsoLevel(Scalar(0));

    EXPECT_THROW(
        (void)mc.extract([](Scalar, Scalar, Scalar)
        {
            return std::numeric_limits<Scalar>::quiet_NaN();
        }),
        std::invalid_argument);
}

TEST(MarchingCubesTest, KnownNonEmptyCubeCasesProduceTriangles)
{
    using Scalar = float;
    const int cube_cases[] = {44, 46, 47, 57, 59, 196, 198, 208, 209, 211};

    for (int cube_case : cube_cases)
    {
        SCOPED_TRACE(cube_case);
        plapoint::mesh::MarchingCubes<Scalar> mc;
        mc.setBounds({0, 0, 0}, {1, 1, 1});
        mc.setResolution(1, 1, 1);
        mc.setIsoLevel(Scalar(0));

        auto [verts, faces] = mc.extract([cube_case](Scalar x, Scalar y, Scalar z)
        {
            const bool high_x = x > Scalar(0.5);
            const bool high_y = y > Scalar(0.5);
            const bool high_z = z > Scalar(0.5);
            int corner = 0;
            if (!high_z)
            {
                corner = high_y ? (high_x ? 2 : 3) : (high_x ? 1 : 0);
            }
            else
            {
                corner = high_y ? (high_x ? 6 : 7) : (high_x ? 5 : 4);
            }
            return (cube_case & (1 << corner)) != 0 ? Scalar(-1) : Scalar(1);
        });

        EXPECT_GT(verts.rows(), 0) << "cube case " << cube_case;
        EXPECT_GT(faces.rows(), 0) << "cube case " << cube_case;
    }
}

TEST(MarchingCubesTest, ComplementaryCubeCasesProduceSameGeometry)
{
    for (int cube_case = 0; cube_case < 128; ++cube_case)
    {
        SCOPED_TRACE(cube_case);
        EXPECT_EQ(normalizedSingleCubeGeometry(cube_case),
                  normalizedSingleCubeGeometry(cube_case ^ 255));
    }
}

TEST(MarchingCubesTest, AllSingleCubeCasesUseValidNonDegenerateTriangles)
{
    using Scalar = float;

    for (int cube_case = 0; cube_case < 256; ++cube_case)
    {
        SCOPED_TRACE(cube_case);
        plapoint::mesh::MarchingCubes<Scalar> mc;
        mc.setBounds({0, 0, 0}, {1, 1, 1});
        mc.setResolution(1, 1, 1);
        mc.setIsoLevel(Scalar(0));

        auto [verts, faces] = mc.extract([cube_case](Scalar x, Scalar y, Scalar z)
        {
            const int corner = cornerIndexForUnitCube(x, y, z);
            return (cube_case & (1 << corner)) != 0 ? Scalar(-1) : Scalar(1);
        });

        ASSERT_EQ(verts.cols(), 3);
        ASSERT_EQ(faces.cols(), 3);
        for (plamatrix::Index r = 0; r < verts.rows(); ++r)
        {
            EXPECT_TRUE(std::isfinite(verts.getValue(r, 0)));
            EXPECT_TRUE(std::isfinite(verts.getValue(r, 1)));
            EXPECT_TRUE(std::isfinite(verts.getValue(r, 2)));
        }
        for (plamatrix::Index f = 0; f < faces.rows(); ++f)
        {
            for (plamatrix::Index c = 0; c < faces.cols(); ++c)
            {
                const int index = faces.getValue(f, c);
                ASSERT_GE(index, 0);
                ASSERT_LT(index, verts.rows());
            }
            EXPECT_GT(triangleArea(verts, f, faces), 1e-12);
        }
    }
}

TEST(MarchingCubesTest, InterpolatesLowAmplitudeCrossings)
{
    using Scalar = double;

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setBounds({0, 0, 0}, {1, 1, 1});
    mc.setResolution(1, 1, 1);
    mc.setIsoLevel(Scalar(0));

    auto [verts, faces] = mc.extract([](Scalar x, Scalar, Scalar)
    {
        return Scalar(1e-14) * (x - Scalar(0.25));
    });

    ASSERT_GT(verts.rows(), 0);
    ASSERT_GT(faces.rows(), 0);
    for (plamatrix::Index r = 0; r < verts.rows(); ++r)
    {
        EXPECT_NEAR(verts.getValue(r, 0), 0.25, 1e-12);
    }
}
