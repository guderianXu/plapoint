#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>

#include <plapoint/core/point_cloud.h>

namespace plapoint {
namespace io {

namespace detail {

struct ObjVertexIndices
{
    int v = 0;
    int t = 0;
    int n = 0;
    bool has_t = false;
    bool has_n = false;
};

inline int parseObjIndexComponent(std::string_view s)
{
    if (s.empty())
    {
        throw std::invalid_argument("empty OBJ index component");
    }

    int value = 0;
    const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
    if (parsed.ec == std::errc::result_out_of_range)
    {
        throw std::out_of_range(
            "OBJ index component out of range '" + std::string(s) + "'");
    }
    if (parsed.ec != std::errc() || parsed.ptr != s.data() + s.size())
    {
        throw std::invalid_argument(
            "invalid OBJ index component '" + std::string(s) + "'");
    }
    if (value == 0)
    {
        throw std::out_of_range("OBJ indices are 1-based and must not be zero");
    }
    return value > 0 ? value - 1 : value;
}

inline int resolveObjIndex(int idx, size_t count, const char* label)
{
    const int resolved = idx < 0
        ? static_cast<int>(count) + idx
        : idx;
    if (resolved < 0 || static_cast<size_t>(resolved) >= count)
    {
        throw std::out_of_range(std::string(label) + " index out of range");
    }
    return resolved;
}

inline ObjVertexIndices parseFaceVertex(std::string_view s)
{
    ObjVertexIndices idx;
    size_t slash1 = s.find('/');
    if (slash1 == std::string_view::npos)
    {
        idx.v = parseObjIndexComponent(s);
        return idx;
    }
    idx.v = parseObjIndexComponent(s.substr(0, slash1));
    size_t slash2 = s.find('/', slash1 + 1);
    if (slash2 == std::string_view::npos)
    {
        if (slash1 + 1 < s.size())
        {
            idx.t = parseObjIndexComponent(s.substr(slash1 + 1));
            idx.has_t = true;
        }
        return idx;
    }
    if (slash2 == slash1 + 1)
    {
        if (slash2 + 1 < s.size())
        {
            idx.n = parseObjIndexComponent(s.substr(slash2 + 1));
            idx.has_n = true;
        }
        return idx;
    }
    idx.t = parseObjIndexComponent(s.substr(slash1 + 1, slash2 - slash1 - 1));
    idx.has_t = true;
    if (slash2 + 1 < s.size())
    {
        idx.n = parseObjIndexComponent(s.substr(slash2 + 1));
        idx.has_n = true;
    }
    return idx;
}

inline std::string_view nextObjToken(const char*& cursor, const char* end)
{
    while (cursor < end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r'))
    {
        ++cursor;
    }
    const char* begin = cursor;
    while (cursor < end && *cursor != ' ' && *cursor != '\t' && *cursor != '\r')
    {
        ++cursor;
    }
    return std::string_view(begin, static_cast<std::size_t>(cursor - begin));
}

template <typename Scalar>
inline Scalar parseObjScalar(std::string_view token,
                             const std::string& context,
                             const char* label)
{
    Scalar value = Scalar(0);
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), value, std::chars_format::general);
    if (token.empty() || parsed.ec != std::errc() ||
        parsed.ptr != token.data() + token.size() || !std::isfinite(value))
    {
        throw std::invalid_argument(context + label + " must be a finite number");
    }
    return value;
}

inline std::string objLineContext(const std::string& path, std::size_t lineNumber)
{
    return "OBJ parse error in " + path + " line " + std::to_string(lineNumber) + ": ";
}

template <typename Scalar>
inline Scalar parseObjVertexColorAttribute(
    const std::string& token,
    const std::string& context)
{
    errno = 0;
    char* end = nullptr;
    const long double value = std::strtold(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || errno == ERANGE ||
        !std::isfinite(value) ||
        value < static_cast<long double>(std::numeric_limits<Scalar>::lowest()) ||
        value > static_cast<long double>(std::numeric_limits<Scalar>::max()))
    {
        throw std::invalid_argument(context + "vertex color attribute must be a finite number");
    }
    return static_cast<Scalar>(value);
}

template <typename Scalar>
inline std::uint8_t objColorByte(Scalar value, bool normalized)
{
    double color = static_cast<double>(value);
    if (!std::isfinite(color))
    {
        throw std::invalid_argument("OBJ vertex color must be finite");
    }
    if (normalized)
    {
        color *= 255.0;
    }
    color = std::round(color);
    color = std::clamp(color, 0.0, 255.0);
    return static_cast<std::uint8_t>(color);
}

} // namespace detail

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readObj(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error("Cannot open OBJ file: " + path);
    }

    std::vector<Scalar> vx, vy, vz;
    std::vector<std::uint8_t> vr, vg, vb;
    std::vector<bool> has_vertex_color;
    std::vector<Scalar> nx, ny, nz;
    std::vector<Scalar> tx, ty;
    std::vector<std::array<int, 3>> face_verts;
    std::vector<std::array<int, 3>> face_tex;
    std::vector<std::array<int, 3>> face_normals;
    std::string mtlLib;

    f.seekg(0, std::ios::end);
    const std::streamoff file_size = f.tellg();
    if (file_size < 0)
    {
        throw std::runtime_error("Cannot determine OBJ file size: " + path);
    }
    f.seekg(0, std::ios::beg);
    std::string file_data(static_cast<std::size_t>(file_size), '\0');
    if (file_size > 0 && !f.read(file_data.data(), file_size))
    {
        throw std::runtime_error("Cannot read OBJ file: " + path);
    }

    std::size_t vertex_line_count = 0;
    std::size_t normal_line_count = 0;
    std::size_t texture_line_count = 0;
    std::size_t face_line_count = 0;
    const char* count_cursor = file_data.data();
    const char* file_end = file_data.data() + file_data.size();
    while (count_cursor < file_end)
    {
        const char* line_end = static_cast<const char*>(
            std::memchr(count_cursor, '\n', static_cast<std::size_t>(file_end - count_cursor)));
        if (!line_end)
        {
            line_end = file_end;
        }
        const char* token_cursor = count_cursor;
        const std::string_view token = detail::nextObjToken(token_cursor, line_end);
        if (token == "v") ++vertex_line_count;
        else if (token == "vn") ++normal_line_count;
        else if (token == "vt") ++texture_line_count;
        else if (token == "f") ++face_line_count;
        count_cursor = line_end < file_end ? line_end + 1 : file_end;
    }
    vx.reserve(vertex_line_count);
    vy.reserve(vertex_line_count);
    vz.reserve(vertex_line_count);
    vr.reserve(vertex_line_count);
    vg.reserve(vertex_line_count);
    vb.reserve(vertex_line_count);
    has_vertex_color.reserve(vertex_line_count);
    nx.reserve(normal_line_count);
    ny.reserve(normal_line_count);
    nz.reserve(normal_line_count);
    tx.reserve(texture_line_count);
    ty.reserve(texture_line_count);
    face_verts.reserve(face_line_count);
    face_tex.reserve(face_line_count);
    face_normals.reserve(face_line_count);

    const char* line_cursor = file_data.data();
    std::size_t line_number = 0;
    while (line_cursor < file_end)
    {
        ++line_number;
        const char* line_end = static_cast<const char*>(
            std::memchr(line_cursor, '\n', static_cast<std::size_t>(file_end - line_cursor)));
        if (!line_end)
        {
            line_end = file_end;
        }
        const char* token_cursor = line_cursor;
        const std::string_view token = detail::nextObjToken(token_cursor, line_end);
        line_cursor = line_end < file_end ? line_end + 1 : file_end;
        if (token.empty() || token.front() == '#')
        {
            continue;
        }
        const std::string context = detail::objLineContext(path, line_number);

        if (token == "v")
        {
            const Scalar x = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "vertex x");
            const Scalar y = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "vertex y");
            const Scalar z = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "vertex z");
            vx.push_back(x);
            vy.push_back(y);
            vz.push_back(z);

            std::array<Scalar, 3> last_extra{{Scalar(0), Scalar(0), Scalar(0)}};
            std::size_t extra_count = 0;
            while (true)
            {
                const std::string_view extra_token = detail::nextObjToken(token_cursor, line_end);
                if (extra_token.empty() || extra_token.front() == '#')
                {
                    break;
                }
                last_extra[extra_count % 3] = detail::parseObjScalar<Scalar>(
                    extra_token, context, "vertex color attribute");
                ++extra_count;
            }
            if (extra_count >= 3)
            {
                const Scalar r = last_extra[(extra_count - 3) % 3];
                const Scalar g = last_extra[(extra_count - 2) % 3];
                const Scalar b = last_extra[(extra_count - 1) % 3];
                const bool normalized =
                    r >= Scalar(0) && r <= Scalar(1) &&
                    g >= Scalar(0) && g <= Scalar(1) &&
                    b >= Scalar(0) && b <= Scalar(1);
                vr.push_back(detail::objColorByte(r, normalized));
                vg.push_back(detail::objColorByte(g, normalized));
                vb.push_back(detail::objColorByte(b, normalized));
                has_vertex_color.push_back(true);
            }
            else
            {
                vr.push_back(0);
                vg.push_back(0);
                vb.push_back(0);
                has_vertex_color.push_back(false);
            }
        }
        else if (token == "vn")
        {
            const Scalar x = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "normal x");
            const Scalar y = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "normal y");
            const Scalar z = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "normal z");
            nx.push_back(x);
            ny.push_back(y);
            nz.push_back(z);
        }
        else if (token == "vt")
        {
            const Scalar u = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "texture u");
            const Scalar v = detail::parseObjScalar<Scalar>(
                detail::nextObjToken(token_cursor, line_end), context, "texture v");
            tx.push_back(u);
            ty.push_back(v);
        }
        else if (token == "f")
        {
            std::array<int, 3> first{{-1, -1, -1}};
            std::array<int, 3> previous{{-1, -1, -1}};
            std::size_t corner_count = 0;
            while (true)
            {
                const std::string_view face_token = detail::nextObjToken(token_cursor, line_end);
                if (face_token.empty() || face_token.front() == '#')
                {
                    break;
                }
                try
                {
                    const auto idx = detail::parseFaceVertex(face_token);
                    std::array<int, 3> current{{
                        detail::resolveObjIndex(idx.v, vx.size(), "OBJ face vertex"), -1, -1}};
                    if (idx.has_t)
                    {
                        current[1] = detail::resolveObjIndex(
                            idx.t, tx.size(), "OBJ face texture");
                    }
                    if (idx.has_n)
                    {
                        current[2] = detail::resolveObjIndex(
                            idx.n, nx.size(), "OBJ face normal");
                    }
                    if (corner_count == 0)
                    {
                        first = current;
                    }
                    else if (corner_count >= 2)
                    {
                        face_verts.push_back({first[0], previous[0], current[0]});
                        face_tex.push_back({first[1], previous[1], current[1]});
                        face_normals.push_back({first[2], previous[2], current[2]});
                    }
                    previous = current;
                    ++corner_count;
                }
                catch (const std::invalid_argument& e)
                {
                    throw std::invalid_argument(
                        context + "invalid face index token '" + std::string(face_token) +
                        "': " + e.what());
                }
                catch (const std::out_of_range& e)
                {
                    throw std::out_of_range(
                        context + "invalid face index token '" + std::string(face_token) +
                        "': " + e.what());
                }
            }
            if (corner_count < 3)
            {
                throw std::invalid_argument(context + "face must contain at least 3 vertices");
            }
        }
        else if (token == "mtllib")
        {
            const std::string_view material_token = detail::nextObjToken(token_cursor, line_end);
            if (material_token.empty())
            {
                throw std::invalid_argument(context + "mtllib line must contain a file name");
            }
            mtlLib.assign(material_token.data(), material_token.size());
        }
    }

    std::string().swap(file_data);

    size_t n = vx.size();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
        static_cast<plamatrix::Index>(n), 3);
    for (size_t i = 0; i < n; ++i)
    {
        pts(static_cast<plamatrix::Index>(i), 0) = vx[i];
        pts(static_cast<plamatrix::Index>(i), 1) = vy[i];
        pts(static_cast<plamatrix::Index>(i), 2) = vz[i];
    }
    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(
        std::move(pts));

    const bool can_assign_colors = n > 0 &&
        has_vertex_color.size() == n &&
        std::all_of(has_vertex_color.begin(), has_vertex_color.end(), [](bool has_color) {
            return has_color;
        });
    if (can_assign_colors)
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
            static_cast<plamatrix::Index>(n), 3);
        for (size_t i = 0; i < n; ++i)
        {
            colors(static_cast<plamatrix::Index>(i), 0) = vr[i];
            colors(static_cast<plamatrix::Index>(i), 1) = vg[i];
            colors(static_cast<plamatrix::Index>(i), 2) = vb[i];
        }
        cloud->setColors(std::move(colors));
    }

    std::vector<int> vertex_normal_indices(n, -1);
    std::vector<bool> vertex_referenced_by_face(n, false);
    const bool has_face_normal_indices = std::any_of(
        face_normals.begin(), face_normals.end(),
        [](const std::array<int, 3>& row) {
            return std::any_of(row.begin(), row.end(), [](int idx) { return idx >= 0; });
        });
    bool face_normal_indices_consistent = !nx.empty() && has_face_normal_indices;
    if (!nx.empty() && has_face_normal_indices)
    {
        for (size_t fi = 0; fi < face_verts.size(); ++fi)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                const int vi = face_verts[fi][c];
                if (vi >= 0 && static_cast<size_t>(vi) < n)
                {
                    vertex_referenced_by_face[static_cast<size_t>(vi)] = true;
                }
                const int ni = face_normals[fi][c];
                if (ni < 0)
                {
                    continue;
                }
                if (vi < 0 || static_cast<size_t>(vi) >= n ||
                    ni < 0 || static_cast<size_t>(ni) >= nx.size())
                {
                    throw std::out_of_range("OBJ face normal index out of range");
                }
                int& assigned = vertex_normal_indices[static_cast<size_t>(vi)];
                if (assigned >= 0 && assigned != ni)
                {
                    face_normal_indices_consistent = false;
                    continue;
                }
                if (assigned < 0)
                {
                    assigned = ni;
                }
            }
        }
    }

    const bool use_ordered_normals = !nx.empty() && nx.size() == n;
    const bool all_vertices_have_face_normals =
        std::all_of(vertex_normal_indices.begin(), vertex_normal_indices.end(),
                    [](int idx) { return idx >= 0; });
    bool ordered_normals_only_fill_unreferenced_vertices = use_ordered_normals;
    for (size_t i = 0; i < n && ordered_normals_only_fill_unreferenced_vertices; ++i)
    {
        if (vertex_normal_indices[i] < 0 && vertex_referenced_by_face[i])
        {
            ordered_normals_only_fill_unreferenced_vertices = false;
        }
    }
    const bool can_assign_normals =
        (face_normal_indices_consistent &&
         (all_vertices_have_face_normals || ordered_normals_only_fill_unreferenced_vertices)) ||
        (!face_normal_indices_consistent && ordered_normals_only_fill_unreferenced_vertices) ||
        (!has_face_normal_indices && use_ordered_normals);

    if (can_assign_normals)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(
            static_cast<plamatrix::Index>(n), 3);
        for (size_t i = 0; i < n; ++i)
        {
            const size_t ni =
                (face_normal_indices_consistent && vertex_normal_indices[i] >= 0)
                    ? static_cast<size_t>(vertex_normal_indices[i])
                    : i;
            nrm(static_cast<plamatrix::Index>(i), 0) = nx[ni];
            nrm(static_cast<plamatrix::Index>(i), 1) = ny[ni];
            nrm(static_cast<plamatrix::Index>(i), 2) = nz[ni];
        }
        cloud->setNormals(std::move(nrm));
    }

    if (!tx.empty())
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> tex(
            static_cast<plamatrix::Index>(tx.size()), 2);
        for (size_t i = 0; i < tx.size(); ++i)
        {
            tex(static_cast<plamatrix::Index>(i), 0) = tx[i];
            tex(static_cast<plamatrix::Index>(i), 1) = ty[i];
        }
        cloud->setTextureCoords(std::move(tex));
    }

    if (!face_verts.empty())
    {
        plamatrix::Index nf = static_cast<plamatrix::Index>(face_verts.size());
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(nf, 3);
        for (plamatrix::Index fi = 0; fi < nf; ++fi)
        {
            for (int c = 0; c < 3 && c < static_cast<int>(face_verts[fi].size()); ++c)
            {
                faces(fi, c) = face_verts[fi][c];
            }
        }
        cloud->setFaces(std::move(faces));

        const bool has_complete_face_texture_indices = !face_tex.empty() &&
            face_tex.size() == face_verts.size() &&
            std::all_of(face_tex.begin(), face_tex.end(), [](const std::array<int, 3>& row) {
                return std::all_of(row.begin(), row.end(), [](int idx) { return idx >= 0; });
            });

        if (has_complete_face_texture_indices)
        {
            plamatrix::DenseMatrix<int, plamatrix::Device::CPU> ft(nf, 3);
            for (plamatrix::Index fi = 0; fi < nf; ++fi)
            {
                for (int c = 0; c < 3; ++c)
                {
                    ft(fi, c) = face_tex[fi][c];
                }
            }
            cloud->setFaceTextureIndices(std::move(ft));
        }
    }

    if (!mtlLib.empty())
    {
        cloud->setMaterialLibraryFile(mtlLib);
    }

    return cloud;
}

template <typename Scalar>
void writeObj(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud)
{
    std::ofstream f(path);
    if (!f)
    {
        throw std::runtime_error("Cannot write OBJ file: " + path);
    }
    f << std::setprecision(std::numeric_limits<Scalar>::max_digits10);

    const auto& mtlLib = cloud.materialLibraryFile();
    if (!mtlLib.empty())
    {
        f << "mtllib " << mtlLib << "\n";
    }

    for (size_t i = 0; i < cloud.size(); ++i)
    {
        f << "v " << cloud.points()(static_cast<plamatrix::Index>(i), 0)
          << " " << cloud.points()(static_cast<plamatrix::Index>(i), 1)
          << " " << cloud.points()(static_cast<plamatrix::Index>(i), 2);
        if (cloud.hasColors())
        {
            f << " " << static_cast<int>(cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 0))
              << " " << static_cast<int>(cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 1))
              << " " << static_cast<int>(cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 2));
        }
        f << "\n";
    }

    if (cloud.hasNormals())
    {
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            f << "vn " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0)
              << " " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1)
              << " " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2) << "\n";
        }
    }

    if (cloud.hasTextureCoords())
    {
        for (plamatrix::Index i = 0; i < cloud.textureCoords()->rows(); ++i)
        {
            f << "vt " << cloud.textureCoords()->getValue(i, 0)
              << " " << cloud.textureCoords()->getValue(i, 1) << "\n";
        }
    }

    if (cloud.hasFaces())
    {
        bool use_tex = cloud.hasFaceTextureIndices();
        bool use_nrm = cloud.hasNormals();
        if (!cloud.materialLibraryFile().empty() && !cloud.textureImageFile().empty())
        {
            f << "usemtl material0\n";
        }
        for (plamatrix::Index fi = 0; fi < cloud.faces()->rows(); ++fi)
        {
            f << "f";
            for (int c = 0; c < 3; ++c)
            {
                int vi = cloud.faces()->getValue(fi, c) + 1; // OBJ 1-indexed
                f << " " << vi;
                if (use_tex || use_nrm)
                {
                    f << "/";
                    if (use_tex)
                    {
                        f << cloud.faceTextureIndices()->getValue(fi, c) + 1;
                    }
                    if (use_nrm)
                    {
                        f << "/" << vi;
                    }
                }
            }
            f << "\n";
        }
    }
}

} // namespace io
} // namespace plapoint
