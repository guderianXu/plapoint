#include <gtest/gtest.h>

#include <plapoint/io/ply_io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef PLAPOINT_TEST_DATA_DIR
#error "PLAPOINT_TEST_DATA_DIR must point to the PlaPoint testData directory"
#endif

namespace {

struct PlyTable
{
    std::vector<std::string> properties;
    std::vector<std::vector<double>> rows;
};

struct ReferenceCloud
{
    std::string relative_path;
    std::size_t points;
    bool has_intensity;
    std::uint64_t intensity_sum;
};

std::string testDataPath(const std::string& relative_path)
{
    return std::string(PLAPOINT_TEST_DATA_DIR) + "/" + relative_path;
}

PlyTable readAsciiPlyTable(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Cannot open PLY test data: " + path);
    }

    std::string line;
    if (!std::getline(stream, line) || line != "ply")
    {
        throw std::runtime_error("Invalid PLY magic in " + path);
    }

    PlyTable table;
    std::size_t vertex_count = 0;
    bool in_vertex = false;
    bool is_ascii = false;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line == "end_header")
        {
            break;
        }

        std::istringstream line_stream(line);
        std::string token;
        line_stream >> token;
        if (token == "format")
        {
            std::string format;
            line_stream >> format;
            is_ascii = format == "ascii";
        }
        else if (token == "element")
        {
            std::string name;
            line_stream >> name >> vertex_count;
            in_vertex = name == "vertex";
        }
        else if (token == "property" && in_vertex)
        {
            std::string type;
            std::string name;
            line_stream >> type >> name;
            table.properties.push_back(name);
        }
    }

    if (!is_ascii)
    {
        throw std::runtime_error("Reference PLY must be ASCII: " + path);
    }

    table.rows.reserve(vertex_count);
    for (std::size_t row = 0; row < vertex_count; ++row)
    {
        if (!std::getline(stream, line))
        {
            throw std::runtime_error("Truncated PLY rows in " + path);
        }

        std::istringstream row_stream(line);
        std::vector<double> values;
        values.reserve(table.properties.size());
        for (std::size_t col = 0; col < table.properties.size(); ++col)
        {
            double value = 0.0;
            if (!(row_stream >> value))
            {
                throw std::runtime_error("Malformed PLY row in " + path);
            }
            values.push_back(value);
        }
        table.rows.push_back(std::move(values));
    }

    return table;
}

std::size_t propertyIndex(const PlyTable& table, const std::string& property)
{
    const auto it = std::find(table.properties.begin(), table.properties.end(), property);
    if (it == table.properties.end())
    {
        throw std::runtime_error("Missing PLY property: " + property);
    }
    return static_cast<std::size_t>(std::distance(table.properties.begin(), it));
}

const std::vector<ReferenceCloud>& referenceClouds()
{
    static const std::vector<ReferenceCloud> clouds = {
        {"real_reconstruction/merged/merged_dense_filtered.ply", 252224, false, 0},
        {"real_reconstruction/merged/merged_dense_gray.ply", 252224, true, 35486521},
        {"real_reconstruction/pairs/002_001/cloud.ply", 61161, false, 0},
        {"real_reconstruction/pairs/002_001/cloud_gray.ply", 61161, true, 8350423},
        {"real_reconstruction/pairs/003_002/cloud.ply", 64011, false, 0},
        {"real_reconstruction/pairs/003_002/cloud_gray.ply", 64011, true, 8787220},
        {"real_reconstruction/pairs/004_003/cloud.ply", 64999, false, 0},
        {"real_reconstruction/pairs/004_003/cloud_gray.ply", 64999, true, 9215797},
        {"real_reconstruction/pairs/005_004/cloud.ply", 62053, false, 0},
        {"real_reconstruction/pairs/005_004/cloud_gray.ply", 62053, true, 9133081},
    };
    return clouds;
}

void expectProperties(const PlyTable& table, bool has_intensity)
{
    std::vector<std::string> expected = {"x", "y", "z", "error"};
    if (has_intensity)
    {
        expected.push_back("intensity");
    }
    EXPECT_EQ(table.properties, expected);
}

} // namespace

TEST(RealReconstructionDataTest, ReferencePlyInventoryHasExpectedCountsAndSchema)
{
    for (const auto& reference : referenceClouds())
    {
        const PlyTable table = readAsciiPlyTable(testDataPath(reference.relative_path));
        SCOPED_TRACE(reference.relative_path);
        EXPECT_EQ(table.rows.size(), reference.points);
        expectProperties(table, reference.has_intensity);
    }
}

TEST(RealReconstructionDataTest, GrayPlyFilesLoadIntensityAsPointCloudColors)
{
    for (const auto& reference : referenceClouds())
    {
        if (!reference.has_intensity)
        {
            continue;
        }

        const auto cloud = plapoint::io::readPly<float>(testDataPath(reference.relative_path));
        ASSERT_EQ(cloud->size(), reference.points) << reference.relative_path;
        ASSERT_TRUE(cloud->hasColors()) << reference.relative_path;

        std::uint8_t min_intensity = std::numeric_limits<std::uint8_t>::max();
        std::uint8_t max_intensity = 0;
        std::uint64_t sum = 0;
        for (std::size_t row = 0; row < cloud->size(); ++row)
        {
            const auto index = static_cast<plamatrix::Index>(row);
            const auto r = cloud->colors()->getValue(index, 0);
            const auto g = cloud->colors()->getValue(index, 1);
            const auto b = cloud->colors()->getValue(index, 2);
            EXPECT_EQ(g, r) << reference.relative_path << " row=" << row;
            EXPECT_EQ(b, r) << reference.relative_path << " row=" << row;
            min_intensity = std::min(min_intensity, r);
            max_intensity = std::max(max_intensity, r);
            sum += r;
        }

        EXPECT_EQ(min_intensity, 6) << reference.relative_path;
        EXPECT_EQ(max_intensity, 255) << reference.relative_path;
        EXPECT_EQ(sum, reference.intensity_sum) << reference.relative_path;
    }
}

TEST(RealReconstructionDataTest, GrayPlyGeometryMatchesNonGrayReference)
{
    const std::vector<std::pair<std::string, std::string>> paths = {
        {
            "real_reconstruction/merged/merged_dense_filtered.ply",
            "real_reconstruction/merged/merged_dense_gray.ply",
        },
        {"real_reconstruction/pairs/002_001/cloud.ply", "real_reconstruction/pairs/002_001/cloud_gray.ply"},
        {"real_reconstruction/pairs/003_002/cloud.ply", "real_reconstruction/pairs/003_002/cloud_gray.ply"},
        {"real_reconstruction/pairs/004_003/cloud.ply", "real_reconstruction/pairs/004_003/cloud_gray.ply"},
        {"real_reconstruction/pairs/005_004/cloud.ply", "real_reconstruction/pairs/005_004/cloud_gray.ply"},
    };

    for (const auto& path_pair : paths)
    {
        const PlyTable plain = readAsciiPlyTable(testDataPath(path_pair.first));
        const PlyTable gray = readAsciiPlyTable(testDataPath(path_pair.second));
        SCOPED_TRACE(path_pair.first);
        ASSERT_EQ(plain.rows.size(), gray.rows.size());
        ASSERT_GE(plain.properties.size(), 4u);
        ASSERT_GE(gray.properties.size(), 5u);

        for (std::size_t row = 0; row < plain.rows.size(); ++row)
        {
            for (std::size_t col = 0; col < 4; ++col)
            {
                EXPECT_DOUBLE_EQ(plain.rows[row][col], gray.rows[row][col])
                    << "row=" << row << " col=" << col;
            }
        }
    }
}

TEST(RealReconstructionDataTest, MergedCloudHasStableGeometryQualityStats)
{
    const PlyTable table = readAsciiPlyTable(
        testDataPath("real_reconstruction/merged/merged_dense_gray.ply"));
    ASSERT_EQ(table.rows.size(), 252224u);

    const auto x_col = propertyIndex(table, "x");
    const auto y_col = propertyIndex(table, "y");
    const auto z_col = propertyIndex(table, "z");
    const auto error_col = propertyIndex(table, "error");

    std::array<double, 3> min_xyz = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    std::array<double, 3> max_xyz = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    std::vector<double> errors;
    errors.reserve(table.rows.size());

    for (const auto& row : table.rows)
    {
        const std::array<double, 3> point = {row[x_col], row[y_col], row[z_col]};
        ASSERT_TRUE(std::isfinite(point[0]));
        ASSERT_TRUE(std::isfinite(point[1]));
        ASSERT_TRUE(std::isfinite(point[2]));
        ASSERT_TRUE(std::isfinite(row[error_col]));
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            min_xyz[axis] = std::min(min_xyz[axis], point[axis]);
            max_xyz[axis] = std::max(max_xyz[axis], point[axis]);
        }
        errors.push_back(row[error_col]);
    }

    std::sort(errors.begin(), errors.end());
    EXPECT_NEAR(min_xyz[0], -1450.555647, 1e-6);
    EXPECT_NEAR(min_xyz[1], -343.288674, 1e-6);
    EXPECT_NEAR(min_xyz[2], 338.133909, 1e-6);
    EXPECT_NEAR(max_xyz[0], 1094.690391, 1e-6);
    EXPECT_NEAR(max_xyz[1], 363.932627, 1e-6);
    EXPECT_NEAR(max_xyz[2], 2838.979987, 1e-6);
    EXPECT_NEAR(errors[errors.size() / 2], 0.001415, 1e-9);
    EXPECT_LE(errors[static_cast<std::size_t>(errors.size() * 99 / 100)], 0.001760);
}
