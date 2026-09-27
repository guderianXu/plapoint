#include <gtest/gtest.h>
#include <plapoint/io/ply_io.h>

#include <filesystem>
#include <fstream>
#include <sstream>

TEST(PlyApiTest, PointXYZRGBUsesPclNamedRoundTripFunctions)
{
    const std::filesystem::path temp_dir(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(temp_dir);
    const auto path = temp_dir / "point_xyzrgb_roundtrip.ply";

    plapoint::PointCloud<plapoint::PointXYZRGB> input;
    input.push_back(plapoint::PointXYZRGB(1.0f, 2.0f, 3.0f, 10, 20, 30));
    input.push_back(plapoint::PointXYZRGB(4.0f, 5.0f, 6.0f, 40, 50, 60));

    ASSERT_EQ(plapoint::io::savePLYFileBinary(path.string(), input), 0);
    plapoint::PointCloud<plapoint::PointXYZRGB> output;
    EXPECT_EQ(plapoint::io::loadPLYFile(path.string(), output), 0);
    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output[1].z, 6.0f);
    EXPECT_EQ(output[1].r, 40);
    EXPECT_EQ(output[1].g, 50);
    EXPECT_EQ(output[1].b, 60);

    std::filesystem::remove(path);
}

TEST(PlyApiTest, PointXYZdPreservesLargeCoordinates)
{
    const std::filesystem::path temp_dir(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(temp_dir);
    const auto path = temp_dir / "point_xyzd_roundtrip.ply";

    plapoint::PointCloud<plapoint::PointXYZd> input;
    input.emplace_back(1000000000.125, -2000000000.375, 3000000000.625);

    ASSERT_EQ(plapoint::io::savePLYFileBinary(path.string(), input), 0);
    plapoint::PointCloud<plapoint::PointXYZd> output;
    ASSERT_EQ(plapoint::io::loadPLYFile(path.string(), output), 0);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_DOUBLE_EQ(output[0].x, input[0].x);
    EXPECT_DOUBLE_EQ(output[0].y, input[0].y);
    EXPECT_DOUBLE_EQ(output[0].z, input[0].z);

    std::filesystem::remove(path);
}

TEST(PlyApiTest, SavesSelectedIndicesWithPclOverload)
{
    const std::filesystem::path temp_dir(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(temp_dir);
    const auto path = temp_dir / "selected_xyz_roundtrip.ply";

    plapoint::PointCloud<plapoint::PointXYZ> input;
    input.emplace_back(1.0f, 0.0f, 0.0f);
    input.emplace_back(2.0f, 0.0f, 0.0f);
    ASSERT_EQ(plapoint::io::savePLYFile(path.string(), input, plapoint::Indices{1}, true), 0);

    plapoint::PointCloud<plapoint::PointXYZ> output;
    ASSERT_EQ(plapoint::io::loadPLYFile(path.string(), output), 0);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_FLOAT_EQ(output[0].x, 2.0f);
    std::filesystem::remove(path);
}

TEST(PlyApiTest, PointXYZIAndSensorPoseRoundTrip)
{
    const std::filesystem::path temp_dir(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(temp_dir);
    const auto path = temp_dir / "point_xyzi_roundtrip.ply";

    plapoint::PointCloud<plapoint::PointXYZI> input;
    input.sensor_origin_ = Eigen::Vector4f(1.0f, 2.0f, 3.0f, 0.0f);
    input.sensor_orientation_ = Eigen::Quaternionf(Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitY()));
    input.emplace_back(1.0f, 2.0f, 3.0f, 0.75f);

    ASSERT_EQ(plapoint::io::savePLYFileBinary(path.string(), input), 0);
    plapoint::PointCloud<plapoint::PointXYZI> output;
    ASSERT_EQ(plapoint::io::loadPLYFile(path.string(), output), 0);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_FLOAT_EQ(output[0].intensity, 0.75f);
    EXPECT_TRUE(output.sensor_origin_.isApprox(input.sensor_origin_));
    EXPECT_TRUE(output.sensor_orientation_.isApprox(input.sensor_orientation_, 1.0e-5f));
    std::filesystem::remove(path);
}

TEST(PlyApiTest, PolygonMeshRoundTripPreservesFaces)
{
    const std::filesystem::path temp_dir(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(temp_dir);
    const auto path = temp_dir / "triangle_mesh_roundtrip.ply";

    plapoint::PointCloud<plapoint::PointXYZ> vertices;
    vertices.emplace_back(0.0f, 0.0f, 0.0f);
    vertices.emplace_back(1.0f, 0.0f, 0.0f);
    vertices.emplace_back(0.0f, 1.0f, 0.0f);
    plapoint::PolygonMesh input;
    plapoint::toPCLPointCloud2(vertices, input.cloud);
    input.polygons.push_back(plapoint::Vertices{{0, 1, 2}});

    ASSERT_EQ(plapoint::io::savePLYFileBinary(path.string(), input), 0);
    plapoint::PolygonMesh output;
    ASSERT_EQ(plapoint::io::loadPLYFile(path.string(), output), 0);
    ASSERT_EQ(output.polygons.size(), 1u);
    EXPECT_EQ(output.polygons[0].vertices, (plapoint::Indices{0, 1, 2}));

    std::ostringstream stream(std::ios::out | std::ios::binary);
    EXPECT_EQ(plapoint::PLYWriter().writeBinary(stream, input.cloud), 0);
    EXPECT_EQ(stream.str().substr(0, 3), "ply");
    std::filesystem::remove(path);
}

TEST(PlyApiTest, WriterGeneratesPublicAsciiAndBinaryHeaders)
{
    plapoint::PointCloud<plapoint::PointXYZRGBA> input;
    input.emplace_back(1.0f, 2.0f, 3.0f, 10, 20, 30, 40);
    plapoint::PCLPointCloud2 blob;
    plapoint::toPCLPointCloud2(input, blob);

    plapoint::PLYWriter writer;
    const auto ascii = writer.generateHeaderASCII(blob, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), 1);
    const auto binary =
        writer.generateHeaderBinary(blob, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), 1, false);

    EXPECT_NE(ascii.find("format ascii 1.0"), std::string::npos);
    EXPECT_NE(ascii.find("property uchar alpha"), std::string::npos);
    EXPECT_NE(binary.find("format binary_"), std::string::npos);
    EXPECT_NE(binary.find("obj_info num_cols 1"), std::string::npos);
}
