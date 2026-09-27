#include <gtest/gtest.h>

#include <plapoint/memory.h>
#include <plapoint/point_cloud_blob.h>
#include <plapoint/point_types.h>
#include <plapoint/register_point_struct.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace
{

    struct EIGEN_ALIGN16 RegisteredPoint
    {
        PCL_ADD_POINT4D;
        PCL_ADD_INTENSITY;
        PCL_ADD_RGB;
        PCL_MAKE_ALIGNED_OPERATOR_NEW
    };

} // namespace

POINT_CLOUD_REGISTER_POINT_STRUCT(RegisteredPoint,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, rgb, rgb))

TEST(PointTypesApiTest, CommonPointTypesExposePclFieldsAndMaps)
{
    plapoint::PointXYZI intensity(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(intensity.data[3], 1.0f);
    EXPECT_FLOAT_EQ(intensity.intensity, 4.0f);

    plapoint::PointXYZRGBA color(1.0f, 2.0f, 3.0f, 10, 20, 30, 40);
    EXPECT_EQ(color.r, 10);
    EXPECT_EQ(color.g, 20);
    EXPECT_EQ(color.b, 30);
    EXPECT_EQ(color.a, 40);
    color.getBGRAVector4cMap() = Eigen::Matrix<std::uint8_t, 4, 1>(1, 2, 3, 4);
    EXPECT_EQ(color.b, 1);
    EXPECT_EQ(color.a, 4);
}

TEST(PointTypesApiTest, CustomRegistrationDrivesBlobReflection)
{
    static_assert(plapoint::traits::has_xyz<RegisteredPoint>::value);
    static_assert(plapoint::traits::has_field_v<RegisteredPoint, plapoint::fields::intensity>);
    static_assert(plapoint::traits::has_field_v<RegisteredPoint, plapoint::fields::rgb>);

    auto cloud = plapoint::make_shared<plapoint::PointCloud<RegisteredPoint>>();
    RegisteredPoint point{};
    point.x = 1.0f;
    point.y = 2.0f;
    point.z = 3.0f;
    point.data[3] = 1.0f;
    point.intensity = 4.0f;
    point.r = 10;
    point.g = 20;
    point.b = 30;
    point.a = 255;
    cloud->push_back(point);

    plapoint::PCLPointCloud2 blob;
    plapoint::toPCLPointCloud2(*cloud, blob);
    ASSERT_EQ(blob.fields.size(), 5u);
    EXPECT_EQ(blob.fields[3].name, "intensity");
    EXPECT_EQ(blob.fields[4].name, "rgb");

    plapoint::PointCloud<RegisteredPoint> restored;
    plapoint::fromPCLPointCloud2(blob, restored);
    ASSERT_EQ(restored.size(), 1u);
    EXPECT_FLOAT_EQ(restored[0].x, 1.0f);
    EXPECT_FLOAT_EQ(restored[0].intensity, 4.0f);
    EXPECT_EQ(restored[0].r, 10);
}

TEST(PointTypesApiTest, BlobConversionConvertsNumericFieldTypes)
{
    plapoint::PCLPointCloud2 blob;
    blob.width = 1;
    blob.height = 1;
    blob.fields = {{"x", 0, plapoint::PCLPointField::FLOAT64, 1},
                   {"y", 8, plapoint::PCLPointField::FLOAT64, 1},
                   {"z", 16, plapoint::PCLPointField::FLOAT64, 1}};
    blob.point_step = 24;
    blob.row_step = 24;
    blob.data.resize(24);
    blob.is_dense = 1;
    const double values[] = {1.25, 2.5, 3.75};
    std::memcpy(blob.data.data(), values, sizeof(values));

    plapoint::PointCloud<plapoint::PointXYZ> cloud;
    plapoint::fromPCLPointCloud2(blob, cloud);
    ASSERT_EQ(cloud.size(), 1u);
    EXPECT_FLOAT_EQ(cloud[0].x, 1.25f);
    EXPECT_FLOAT_EQ(cloud[0].y, 2.5f);
    EXPECT_FLOAT_EQ(cloud[0].z, 3.75f);
}
