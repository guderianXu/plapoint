#include <gtest/gtest.h>
#include "temp_file.h"
#include <plapoint/io/xyz_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

TEST(XyzIOTest, Roundtrip)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i*2));
        pts.setValue(i, 2, Scalar(i*3));
    }
    Cloud cloud(std::move(pts));

    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    plapoint::io::writeXyz(path, cloud);

    auto loaded = plapoint::io::readXyz<Scalar>(path);
    EXPECT_EQ(loaded->size(), 4u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(2, 1), 4.0f);

    std::remove(path.c_str());
}

TEST(XyzIOTest, RoundtripPreservesDoubleLargeCoordinateDeltas)
{
    using Scalar = double;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(2, 3);
    pts.setValue(0, 0, 100000000.01);
    pts.setValue(0, 1, -100000000.02);
    pts.setValue(0, 2, 123456789.125);
    pts.setValue(1, 0, 100000000.02);
    pts.setValue(1, 1, -100000000.03);
    pts.setValue(1, 2, 123456789.25);
    Cloud cloud(std::move(pts));

    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    plapoint::io::writeXyz(path, cloud);

    auto loaded = plapoint::io::readXyz<Scalar>(path);

    ASSERT_EQ(loaded->size(), 2u);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(0, 0), 100000000.01);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(0, 1), -100000000.02);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(0, 2), 123456789.125);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(1, 0), 100000000.02);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(1, 1), -100000000.03);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(1, 2), 123456789.25);

    std::remove(path.c_str());
}

TEST(XyzIOTest, RoundtripPreservesOptionalRgbColumns)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(2, 3);
    pts.setValue(0, 0, 1.0f);
    pts.setValue(0, 1, 2.0f);
    pts.setValue(0, 2, 3.0f);
    pts.setValue(1, 0, 4.0f);
    pts.setValue(1, 1, 5.0f);
    pts.setValue(1, 2, 6.0f);
    Cloud cloud(std::move(pts));

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(2, 3);
    colors.setValue(0, 0, 10);
    colors.setValue(0, 1, 20);
    colors.setValue(0, 2, 30);
    colors.setValue(1, 0, 40);
    colors.setValue(1, 1, 50);
    colors.setValue(1, 2, 60);
    cloud.setColors(std::move(colors));

    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    plapoint::io::writeXyz(path, cloud);

    auto loaded = plapoint::io::readXyz<Scalar>(path);

    ASSERT_EQ(loaded->size(), 2u);
    ASSERT_TRUE(loaded->hasColors());
    EXPECT_EQ(loaded->colors()->getValue(0, 0), 10);
    EXPECT_EQ(loaded->colors()->getValue(0, 1), 20);
    EXPECT_EQ(loaded->colors()->getValue(0, 2), 30);
    EXPECT_EQ(loaded->colors()->getValue(1, 0), 40);
    EXPECT_EQ(loaded->colors()->getValue(1, 1), 50);
    EXPECT_EQ(loaded->colors()->getValue(1, 2), 60);

    std::remove(path.c_str());
}

TEST(XyzIOTest, EmptyFileProducesEmptyCloud)
{
    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
    }

    auto loaded = plapoint::io::readXyz<float>(path);

    EXPECT_EQ(loaded->size(), 0u);

    std::filesystem::remove(path);
}

TEST(XyzIOTest, StrictReadRejectsMalformedRowsWithLineNumber)
{
    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "# comment\n";
        f << "\n";
        f << "1 2 3\n";
        f << "1 2 3 extra columns are rejected\n";
        f << "not a point\n";
        f << "4 5\n";
        f << "6.5 7.5 8.5\n";
        f << "   # indented comment is ignored as an invalid numeric row\n";
    }

    try
    {
        (void)plapoint::io::readXyz<float>(path);
        FAIL() << "Expected strict XYZ parsing to reject malformed rows";
    }
    catch (const std::runtime_error& ex)
    {
        const std::string message = ex.what();
        EXPECT_NE(message.find(path), std::string::npos);
        EXPECT_NE(message.find("line 4"), std::string::npos);
    }

    std::filesystem::remove(path);
}

TEST(XyzIOTest, StrictReadRejectsMalformedRgbColumnsWithLineNumber)
{
    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "1 2 3 10 20 bad\n";
    }

    try
    {
        (void)plapoint::io::readXyz<float>(path);
        FAIL() << "Expected strict XYZ parsing to reject malformed RGB columns";
    }
    catch (const std::runtime_error& ex)
    {
        const std::string message = ex.what();
        EXPECT_NE(message.find(path), std::string::npos);
        EXPECT_NE(message.find("line 1"), std::string::npos);
    }

    std::filesystem::remove(path);
}

TEST(XyzIOTest, PermissiveReadIgnoresMalformedRows)
{
    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "# comment\n";
        f << "\n";
        f << "1 2 3 extra columns are ignored\n";
        f << "not a point\n";
        f << "4 5\n";
        f << "6.5 7.5 8.5\n";
        f << "   # indented comment is ignored\n";
    }

    auto loaded = plapoint::io::readXyz<float>(path, plapoint::io::XyzReadMode::Permissive);

    ASSERT_EQ(loaded->size(), 2u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(loaded->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(loaded->points().getValue(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(loaded->points().getValue(1, 0), 6.5f);
    EXPECT_FLOAT_EQ(loaded->points().getValue(1, 1), 7.5f);
    EXPECT_FLOAT_EQ(loaded->points().getValue(1, 2), 8.5f);

    std::filesystem::remove(path);
}

TEST(XyzIOTest, ReadNonExistentFileThrows)
{
    const plapoint::test::TempFile temp_file(".xyz");
    const auto path = temp_file.string();
    std::filesystem::remove(path);

    EXPECT_THROW((void)plapoint::io::readXyz<float>(path), std::runtime_error);
}
