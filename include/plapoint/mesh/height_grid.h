#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <plapoint/core/point_cloud.h>

namespace plapoint::mesh
{

template <typename Scalar>
struct HeightGridOptions
{
    int width = 0;
    int height = 0;
    int resolution = 128;
    Scalar padding = Scalar(0);
    Scalar maxHeightJump = Scalar(0);
    Scalar minAbsNormalZ = Scalar(0.08);
    int maxFillPassForFaces = 8;
};

template <typename Scalar>
struct HeightGrid
{
    int width = 0;
    int height = 0;
    Scalar minX = Scalar(0);
    Scalar minY = Scalar(0);
    Scalar stepX = Scalar(1);
    Scalar stepY = Scalar(1);
    std::vector<Scalar> heights;
    std::vector<Scalar> weights;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint16_t> fillPass;

    Scalar& at(int ix, int iy)
    {
        return heights[static_cast<std::size_t>(iy * width + ix)];
    }

    Scalar at(int ix, int iy) const
    {
        return heights[static_cast<std::size_t>(iy * width + ix)];
    }

    void setValid(int ix, int iy, bool value)
    {
        valid[static_cast<std::size_t>(iy * width + ix)] = value ? 1 : 0;
    }

    bool isValid(int ix, int iy) const
    {
        return valid[static_cast<std::size_t>(iy * width + ix)] != 0;
    }

    std::uint16_t fillPassAt(int ix, int iy) const
    {
        return fillPass[static_cast<std::size_t>(iy * width + ix)];
    }
};

namespace detail
{

template <typename Scalar>
Scalar clamp01(Scalar value)
{
    return std::clamp(value, Scalar(0), Scalar(1));
}

template <typename Scalar>
Scalar gridHeightJumpFallback(const HeightGrid<Scalar>& grid)
{
    const Scalar step = std::sqrt(grid.stepX * grid.stepX + grid.stepY * grid.stepY);
    return std::max(Scalar(1.0e-6), step * Scalar(1.5));
}

template <typename Scalar>
Scalar estimateHeightJump(const HeightGrid<Scalar>& grid, Scalar configured)
{
    if (std::isfinite(configured) && configured > Scalar(0))
    {
        return configured;
    }

    std::vector<Scalar> diffs;
    for (int y = 0; y < grid.height; ++y)
    {
        for (int x = 0; x < grid.width; ++x)
        {
            if (!grid.isValid(x, y))
            {
                continue;
            }
            if (x + 1 < grid.width && grid.isValid(x + 1, y))
            {
                diffs.push_back(std::abs(grid.at(x, y) - grid.at(x + 1, y)));
            }
            if (y + 1 < grid.height && grid.isValid(x, y + 1))
            {
                diffs.push_back(std::abs(grid.at(x, y) - grid.at(x, y + 1)));
            }
        }
    }
    if (diffs.empty())
    {
        return gridHeightJumpFallback(grid);
    }
    const auto middle = diffs.begin() + static_cast<std::ptrdiff_t>(diffs.size() / 2);
    std::nth_element(diffs.begin(), middle, diffs.end());
    return std::max(gridHeightJumpFallback(grid), (*middle) * Scalar(4));
}

template <typename Scalar>
bool triangleReliable(
    const HeightGrid<Scalar>& grid,
    int ax, int ay,
    int bx, int by,
    int cx, int cy,
    Scalar max_height_jump,
    Scalar min_abs_normal_z,
    int max_fill_pass)
{
    if (!grid.isValid(ax, ay) || !grid.isValid(bx, by) || !grid.isValid(cx, cy))
    {
        return false;
    }
    if (std::max({
            static_cast<int>(grid.fillPassAt(ax, ay)),
            static_cast<int>(grid.fillPassAt(bx, by)),
            static_cast<int>(grid.fillPassAt(cx, cy))}) > max_fill_pass)
    {
        return false;
    }

    const Scalar za = grid.at(ax, ay);
    const Scalar zb = grid.at(bx, by);
    const Scalar zc = grid.at(cx, cy);
    if (std::abs(za - zb) > max_height_jump ||
        std::abs(za - zc) > max_height_jump ||
        std::abs(zb - zc) > max_height_jump)
    {
        return false;
    }

    const long double x0 = static_cast<long double>(grid.minX + Scalar(ax) * grid.stepX);
    const long double y0 = static_cast<long double>(grid.minY + Scalar(ay) * grid.stepY);
    const long double x1 = static_cast<long double>(grid.minX + Scalar(bx) * grid.stepX);
    const long double y1 = static_cast<long double>(grid.minY + Scalar(by) * grid.stepY);
    const long double x2 = static_cast<long double>(grid.minX + Scalar(cx) * grid.stepX);
    const long double y2 = static_cast<long double>(grid.minY + Scalar(cy) * grid.stepY);
    const long double ux = x1 - x0;
    const long double uy = y1 - y0;
    const long double uz = static_cast<long double>(zb - za);
    const long double vx = x2 - x0;
    const long double vy = y2 - y0;
    const long double vz = static_cast<long double>(zc - za);
    const long double nx = uy * vz - uz * vy;
    const long double ny = uz * vx - ux * vz;
    const long double nz = ux * vy - uy * vx;
    const long double length = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (length <= std::numeric_limits<long double>::epsilon())
    {
        return false;
    }
    return std::abs(nz / length) >= static_cast<long double>(min_abs_normal_z);
}

template <typename Scalar>
int nearestSourcePointIndex(
    const PointCloud<Scalar, plamatrix::Device::CPU>& source,
    Scalar x,
    Scalar y,
    Scalar z)
{
    if (source.size() == 0)
    {
        return -1;
    }
    long double best = std::numeric_limits<long double>::max();
    int best_index = 0;
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        const long double dx = static_cast<long double>(source.points().getValue(static_cast<plamatrix::Index>(i), 0) - x);
        const long double dy = static_cast<long double>(source.points().getValue(static_cast<plamatrix::Index>(i), 1) - y);
        const long double dz = static_cast<long double>(source.points().getValue(static_cast<plamatrix::Index>(i), 2) - z);
        const long double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best)
        {
            best = d2;
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}

} // namespace detail

template <typename Scalar>
HeightGrid<Scalar> buildHeightGrid(
    const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
    const HeightGridOptions<Scalar>& options = {})
{
    HeightGrid<Scalar> grid;
    if (cloud.size() == 0)
    {
        return grid;
    }
    if (!std::isfinite(options.padding) || options.padding < Scalar(0))
    {
        throw std::invalid_argument("buildHeightGrid: padding must be finite and non-negative");
    }

    Scalar min_x = cloud.points().getValue(0, 0);
    Scalar max_x = min_x;
    Scalar min_y = cloud.points().getValue(0, 1);
    Scalar max_y = min_y;
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        const Scalar x = cloud.points().getValue(row, 0);
        const Scalar y = cloud.points().getValue(row, 1);
        const Scalar z = cloud.points().getValue(row, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            throw std::invalid_argument("buildHeightGrid: points must be finite");
        }
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    Scalar span_x = std::max(max_x - min_x, Scalar(1.0e-6));
    Scalar span_y = std::max(max_y - min_y, Scalar(1.0e-6));
    min_x -= span_x * options.padding;
    max_x += span_x * options.padding;
    min_y -= span_y * options.padding;
    max_y += span_y * options.padding;
    span_x = std::max(max_x - min_x, Scalar(1.0e-6));
    span_y = std::max(max_y - min_y, Scalar(1.0e-6));

    const int requested_width = options.width > 0 ? options.width : options.resolution;
    const int requested_height = options.height > 0 ? options.height : options.resolution;
    if (requested_width < 2 || requested_height < 2)
    {
        throw std::invalid_argument("buildHeightGrid: grid dimensions must be at least 2");
    }

    grid.width = requested_width;
    grid.height = requested_height;
    grid.minX = min_x;
    grid.minY = min_y;
    grid.stepX = span_x / Scalar(grid.width - 1);
    grid.stepY = span_y / Scalar(grid.height - 1);
    const std::size_t cell_count = static_cast<std::size_t>(grid.width) * static_cast<std::size_t>(grid.height);
    grid.heights.assign(cell_count, Scalar(0));
    grid.weights.assign(cell_count, Scalar(0));
    grid.valid.assign(cell_count, 0);
    grid.fillPass.assign(cell_count, 0);

    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        const Scalar x = cloud.points().getValue(row, 0);
        const Scalar y = cloud.points().getValue(row, 1);
        const Scalar z = cloud.points().getValue(row, 2);
        const Scalar gx = (x - grid.minX) / grid.stepX;
        const Scalar gy = (y - grid.minY) / grid.stepY;
        const int ix = std::clamp(static_cast<int>(std::floor(gx)), 0, grid.width - 2);
        const int iy = std::clamp(static_cast<int>(std::floor(gy)), 0, grid.height - 2);
        const Scalar tx = detail::clamp01(gx - Scalar(ix));
        const Scalar ty = detail::clamp01(gy - Scalar(iy));
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const Scalar wx = dx ? tx : Scalar(1) - tx;
                const Scalar wy = dy ? ty : Scalar(1) - ty;
                const Scalar w = wx * wy;
                const std::size_t cell = static_cast<std::size_t>((iy + dy) * grid.width + (ix + dx));
                grid.heights[cell] += z * w;
                grid.weights[cell] += w;
            }
        }
    }

    for (std::size_t cell = 0; cell < cell_count; ++cell)
    {
        if (grid.weights[cell] > Scalar(1.0e-9))
        {
            grid.heights[cell] /= grid.weights[cell];
            grid.valid[cell] = 1;
        }
    }
    return grid;
}

template <typename Scalar>
void fillHoles(HeightGrid<Scalar>& grid, int max_passes = 8)
{
    if (max_passes <= 0 || grid.width <= 0 || grid.height <= 0)
    {
        return;
    }

    std::vector<Scalar> next_heights = grid.heights;
    std::vector<std::uint8_t> next_valid = grid.valid;
    std::vector<std::uint16_t> next_fill_pass = grid.fillPass;
    for (int pass = 0; pass < max_passes; ++pass)
    {
        bool changed = false;
        for (int y = 0; y < grid.height; ++y)
        {
            for (int x = 0; x < grid.width; ++x)
            {
                const std::size_t cell = static_cast<std::size_t>(y * grid.width + x);
                if (grid.valid[cell])
                {
                    next_heights[cell] = grid.heights[cell];
                    next_valid[cell] = 1;
                    next_fill_pass[cell] = grid.fillPass[cell];
                    continue;
                }

                Scalar sum = Scalar(0);
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height)
                        {
                            continue;
                        }
                        if (grid.isValid(nx, ny))
                        {
                            sum += grid.at(nx, ny);
                            ++count;
                        }
                    }
                }
                if (count > 0)
                {
                    next_heights[cell] = sum / Scalar(count);
                    next_valid[cell] = 1;
                    next_fill_pass[cell] = static_cast<std::uint16_t>(pass + 1);
                    changed = true;
                }
            }
        }

        grid.heights.swap(next_heights);
        grid.valid.swap(next_valid);
        grid.fillPass.swap(next_fill_pass);
        next_heights = grid.heights;
        next_valid = grid.valid;
        next_fill_pass = grid.fillPass;
        if (!changed)
        {
            break;
        }
    }
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> heightGridToMesh(
    const HeightGrid<Scalar>& grid,
    const PointCloud<Scalar, plamatrix::Device::CPU>& source_cloud,
    const HeightGridOptions<Scalar>& options = {})
{
    if (grid.width < 2 || grid.height < 2)
    {
        PointCloud<Scalar, plamatrix::Device::CPU> empty(0);
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(0, 3);
        empty.setFaces(std::move(faces));
        return empty;
    }

    std::vector<int> grid_to_vertex(static_cast<std::size_t>(grid.width) * static_cast<std::size_t>(grid.height), -1);
    std::vector<std::pair<int, int>> vertex_cells;
    vertex_cells.reserve(grid_to_vertex.size());
    for (int y = 0; y < grid.height; ++y)
    {
        for (int x = 0; x < grid.width; ++x)
        {
            if (!grid.isValid(x, y))
            {
                continue;
            }
            grid_to_vertex[static_cast<std::size_t>(y * grid.width + x)] = static_cast<int>(vertex_cells.size());
            vertex_cells.emplace_back(x, y);
        }
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(vertex_cells.size()), 3);
    std::unique_ptr<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>> colors;
    if (source_cloud.hasColors())
    {
        colors = std::make_unique<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>>(
            static_cast<plamatrix::Index>(vertex_cells.size()), 3);
    }
    std::unique_ptr<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>> intensities;
    if (source_cloud.hasIntensities())
    {
        intensities = std::make_unique<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>>(
            static_cast<plamatrix::Index>(vertex_cells.size()), 1);
    }

    for (std::size_t i = 0; i < vertex_cells.size(); ++i)
    {
        const auto [x, y] = vertex_cells[i];
        const Scalar px = grid.minX + Scalar(x) * grid.stepX;
        const Scalar py = grid.minY + Scalar(y) * grid.stepY;
        const Scalar pz = grid.at(x, y);
        points.setValue(static_cast<plamatrix::Index>(i), 0, px);
        points.setValue(static_cast<plamatrix::Index>(i), 1, py);
        points.setValue(static_cast<plamatrix::Index>(i), 2, pz);
        const int nearest = (colors || intensities)
            ? detail::nearestSourcePointIndex(source_cloud, px, py, pz)
            : -1;
        if (colors)
        {
            if (nearest >= 0)
            {
                colors->setValue(static_cast<plamatrix::Index>(i), 0, source_cloud.colors()->getValue(nearest, 0));
                colors->setValue(static_cast<plamatrix::Index>(i), 1, source_cloud.colors()->getValue(nearest, 1));
                colors->setValue(static_cast<plamatrix::Index>(i), 2, source_cloud.colors()->getValue(nearest, 2));
            }
        }
        if (intensities)
        {
            if (nearest >= 0)
            {
                intensities->setValue(
                    static_cast<plamatrix::Index>(i),
                    0,
                    source_cloud.intensities()->getValue(nearest, 0));
            }
        }
    }

    const Scalar max_height_jump = detail::estimateHeightJump(grid, options.maxHeightJump);
    const int max_fill_pass = std::max(0, options.maxFillPassForFaces);
    std::vector<std::array<int, 3>> faces;
    for (int y = 0; y < grid.height - 1; ++y)
    {
        for (int x = 0; x < grid.width - 1; ++x)
        {
            const int i00 = grid_to_vertex[static_cast<std::size_t>(y * grid.width + x)];
            const int i10 = grid_to_vertex[static_cast<std::size_t>(y * grid.width + x + 1)];
            const int i01 = grid_to_vertex[static_cast<std::size_t>((y + 1) * grid.width + x)];
            const int i11 = grid_to_vertex[static_cast<std::size_t>((y + 1) * grid.width + x + 1)];
            if (i00 < 0 || i10 < 0 || i01 < 0 || i11 < 0)
            {
                continue;
            }

            const Scalar diag_main = std::abs(grid.at(x, y) - grid.at(x + 1, y + 1));
            const Scalar diag_alt = std::abs(grid.at(x + 1, y) - grid.at(x, y + 1));
            if (diag_main <= diag_alt)
            {
                if (detail::triangleReliable(grid, x, y, x + 1, y, x + 1, y + 1,
                                             max_height_jump, options.minAbsNormalZ, max_fill_pass))
                {
                    faces.push_back({i00, i10, i11});
                }
                if (detail::triangleReliable(grid, x, y, x + 1, y + 1, x, y + 1,
                                             max_height_jump, options.minAbsNormalZ, max_fill_pass))
                {
                    faces.push_back({i00, i11, i01});
                }
            }
            else
            {
                if (detail::triangleReliable(grid, x, y, x + 1, y, x, y + 1,
                                             max_height_jump, options.minAbsNormalZ, max_fill_pass))
                {
                    faces.push_back({i00, i10, i01});
                }
                if (detail::triangleReliable(grid, x + 1, y, x + 1, y + 1, x, y + 1,
                                             max_height_jump, options.minAbsNormalZ, max_fill_pass))
                {
                    faces.push_back({i10, i11, i01});
                }
            }
        }
    }

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_matrix(
        static_cast<plamatrix::Index>(faces.size()), 3);
    for (std::size_t r = 0; r < faces.size(); ++r)
    {
        face_matrix.setValue(static_cast<plamatrix::Index>(r), 0, faces[r][0]);
        face_matrix.setValue(static_cast<plamatrix::Index>(r), 1, faces[r][1]);
        face_matrix.setValue(static_cast<plamatrix::Index>(r), 2, faces[r][2]);
    }

    PointCloud<Scalar, plamatrix::Device::CPU> mesh(std::move(points));
    if (colors)
    {
        mesh.setColors(std::move(*colors));
    }
    if (intensities)
    {
        mesh.setIntensities(std::move(*intensities));
    }
    mesh.setFaces(std::move(face_matrix));
    return mesh;
}

} // namespace plapoint::mesh
