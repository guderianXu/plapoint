#include <gtest/gtest.h>
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

    std::string path = "/tmp/plapoint_test.xyz";
    plapoint::io::writeXyz(path, cloud);

    auto loaded = plapoint::io::readXyz<Scalar>(path);
    EXPECT_EQ(loaded->size(), 4u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(2, 1), 4.0f);

    std::remove(path.c_str());
}

TEST(XyzIOTest, EmptyFileProducesEmptyCloud)
{
    const auto path = std::filesystem::temp_directory_path() / "plapoint_test_empty.xyz";
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
    }

    auto loaded = plapoint::io::readXyz<float>(path.string());

    EXPECT_EQ(loaded->size(), 0u);

    std::filesystem::remove(path);
}

TEST(XyzIOTest, IgnoresCommentsPartialInvalidAndExtraColumns)
{
    const auto path = std::filesystem::temp_directory_path() / "plapoint_test_mixed_lines.xyz";
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
        f << "   # indented comment is ignored as an invalid numeric row\n";
    }

    auto loaded = plapoint::io::readXyz<float>(path.string());

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
    const auto path = std::filesystem::temp_directory_path() / "plapoint_missing_test.xyz";
    std::filesystem::remove(path);

    EXPECT_THROW((void)plapoint::io::readXyz<float>(path.string()), std::runtime_error);
}
