#pragma once

#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/core/types.h>
#include <plamatrix/ops/point_cloud.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

namespace plapoint {
namespace mesh {

// Edge table: which cube edges are intersected by the isosurface
// Indexed by 8-bit cube corner mask (1 bit per corner)
namespace detail {
inline int edgeTable(int idx)
{
    static const int table[256] = {
        0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc , 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
    };
    return table[idx & 255];
}

// Triangle table: each case has a vector of edge indices terminated by -1
// Returns pointer to static data for the given case
inline const std::vector<int>& triTable(int cubeIndex)
{
    static const std::vector<int> table[256] = {
        {}, // 0
        {0,8,3},{0,1,9},{1,8,3,9,8,1},{1,2,10},{0,8,3,1,2,10},
        {9,2,10,0,2,9},{2,8,3,2,10,8,10,9,8},{3,11,2},{0,11,2,8,11,0},
        {1,9,0,2,3,11},{1,11,2,1,9,11,9,8,11},{3,10,1,11,10,3},
        {0,10,1,0,8,10,8,11,10},{3,9,0,3,11,9,11,10,9},{9,8,10,10,8,11},
        {4,7,8},{4,3,0,7,3,4},{0,1,9,8,4,7},{4,1,9,4,7,1,7,3,1},
        {1,2,10,8,4,7},{3,4,7,3,0,4,1,2,10},{9,2,10,9,0,2,8,4,7},
        {2,10,9,2,9,7,2,7,3,7,9,4},{8,4,7,3,11,2},{11,4,7,11,2,4,2,0,4},
        {9,0,1,8,4,7,2,3,11},{4,7,11,9,4,11,9,11,2,9,2,1},{3,10,1,3,11,10,7,8,4},
        {1,11,10,1,4,11,1,0,4,7,11,4},{4,7,8,9,0,11,9,11,10,11,0,3},
        {4,7,11,4,11,9,9,11,10}, // 31
        {9,5,4},{9,5,4,0,8,3},{0,5,4,1,5,0},{8,5,4,8,3,5,3,1,5},
        {1,2,10,9,5,4},{3,0,8,1,2,10,4,9,5},{5,2,10,5,4,2,4,0,2},
        {2,10,5,3,2,5,3,5,4,3,4,8},{9,5,4,2,3,11},{11,0,8,11,2,0,9,5,4},
        {5,0,1,5,4,0,3,11,2},{11,2,1,11,1,5,11,5,8,11,8,4},{10,1,2,9,5,4,3,10,1,3,11,10},
        {0,8,11,0,11,10,10,11,1,4,9,5},{5,4,9,3,11,0,3,0,10,10,0,2},
        {5,4,8,5,8,11,11,8,3,10,5,11,10,11,2}, // 47
        {9,7,8,5,7,9},{9,3,0,9,5,3,5,7,3},{0,7,8,0,1,7,1,5,7},{1,5,3,3,5,7},
        {9,7,8,9,5,7,10,1,2},{10,1,2,9,5,0,5,3,0,5,7,3},{8,0,2,8,2,5,8,5,7,10,2,5},
        {2,10,5,2,5,3,3,5,7},{7,9,5,7,8,9,3,11,2},{9,5,7,9,7,2,9,2,0,2,7,3},
        {2,3,11,0,1,8,1,7,8,1,5,7},{11,2,1,11,1,5,11,5,7,11,7,8},
        {9,7,8,9,5,7,10,1,3,10,11,3},{5,7,0,5,0,9,7,11,0,1,0,10,11,10,0},
        {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0},{11,10,5,7,11,5}, // 63
        {10,6,5},{0,8,3,5,10,6},{9,0,1,5,10,6},{1,8,3,1,9,8,5,10,6},
        {1,6,5,2,6,1},{1,6,5,1,2,6,3,0,8},{9,6,5,9,0,6,0,2,6},{5,9,8,5,8,2,5,2,6,3,2,8},
        {2,3,11,10,6,5},{11,0,8,11,2,0,10,6,5},{0,1,9,2,3,11,5,10,6},
        {5,10,6,1,9,2,9,11,2,9,8,11},{6,3,11,6,5,3,5,1,3},{0,8,11,0,11,5,0,5,1,5,11,6},
        {3,11,6,0,3,6,0,6,5,0,5,9},{6,5,9,6,9,11,11,9,8}, // 79
        {5,10,6,4,7,8},{4,3,0,4,7,3,6,5,10},{1,9,0,5,10,6,8,4,7},
        {10,6,5,1,9,7,1,7,3,7,9,4},{6,1,2,6,5,1,4,7,8},{1,2,5,5,2,6,3,0,4,3,4,7},
        {8,4,7,9,0,5,0,6,5,0,2,6},{7,3,9,7,9,4,3,2,9,5,9,6,2,6,9},{3,11,2,7,8,4,10,6,5},
        {5,10,6,4,7,2,4,2,0,2,7,11},{0,1,9,4,7,8,2,3,11,5,10,6},
        {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6},{8,4,7,3,11,5,3,5,1,5,11,6},
        {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11},{0,5,9,0,6,5,0,3,6,11,6,3,8,4,7},
        {6,5,9,6,9,11,4,7,9,7,11,9}, // 95
        {10,4,9,6,4,10},{4,10,6,4,9,10,0,8,3},{10,0,1,10,6,0,6,4,0},
        {8,3,1,8,1,6,8,6,4,6,1,10},{1,4,9,1,2,4,2,6,4},{3,0,8,1,2,9,2,4,9,2,6,4},
        {0,2,4,4,2,6},{8,3,2,8,2,4,4,2,6},{10,4,9,10,6,4,11,2,3},
        {0,8,2,2,8,11,4,9,10,4,10,6},{3,11,2,0,1,6,0,6,4,6,1,10},
        {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1},{9,6,4,9,3,6,9,1,3,11,6,3},
        {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1},{3,11,6,3,6,0,0,6,4},{6,4,8,11,6,8}, // 111
        {7,10,6,7,8,10,8,9,10},{0,7,3,0,10,7,0,9,10,6,7,10},{10,6,7,1,10,7,1,7,8,1,8,0},
        {10,6,7,10,7,1,1,7,3},{1,2,6,1,6,8,1,8,9,8,6,7},{2,6,7,2,7,3,6,7,2,9,0,1},
        {7,8,0,7,0,6,6,0,2},{7,3,2,6,7,2},{2,3,11,10,6,8,10,8,9,8,6,7},
        {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7},{1,8,0,1,7,8,1,10,7,6,7,10,2,3,11},
        {11,2,1,11,1,7,10,6,1,6,7,1},{8,9,6,8,6,7,9,1,6,11,6,3,1,3,6},
        {0,9,1,11,6,7},{7,8,0,7,0,6,3,11,0,11,6,0},{7,11,6}, // 127
        {7,6,11},{3,0,8,11,7,6},{0,1,9,11,7,6},{8,1,9,8,3,1,11,7,6},{10,1,2,6,11,7},
        {1,2,10,3,0,8,6,11,7},{2,9,0,2,10,9,6,11,7},{6,11,7,2,10,3,10,8,3,10,9,8},
        {7,2,3,6,2,7},{7,0,8,7,6,0,6,2,0},{2,7,6,2,3,7,0,1,9},{1,6,2,1,8,6,1,9,8,8,7,6},
        {10,7,6,10,1,7,1,3,7},{10,7,6,1,7,10,1,8,7,1,0,8},{0,3,7,0,7,6,0,6,9,6,7,10},
        {7,6,10,7,10,8,8,10,9}, // 143
        {6,8,4,11,8,6},{3,6,11,3,0,6,0,4,6},{8,6,11,8,4,6,9,0,1},{9,4,6,9,6,3,9,3,1,11,3,6},
        {6,8,4,6,11,8,2,10,1},{1,2,10,3,0,11,0,6,11,0,4,6},{4,11,8,4,6,11,0,2,9,2,10,9},
        {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3},{8,2,3,8,6,2,8,4,6,6,2,7},
        {0,2,3,0,2,7,0,7,4,2,7,6},{8,4,6,8,6,2,8,2,3,2,6,7,0,1,9},{9,4,6,9,6,3,9,3,1,11,3,6},
        {6,8,4,6,11,8,10,1,3,10,3,7},{1,3,10,1,3,7,3,7,6,8,4,6,0,4,3},{0,3,8,6,8,10,7,6,9,1,9,4},
        {10,7,6,9,4,3,9,3,1,10,9,3,10,3,7}, // 159
        {9,4,8,9,8,3,11,3,6,4,6,8},{3,4,9,0,4,3},{0,1,9,8,4,11,11,3,8},
        {1,9,4,1,4,11,1,11,2,11,4,6},{3,8,4,3,4,11,8,4,6,1,2,10},{1,2,10,3,0,11,0,4,11,0,2,7,0,7,4},
        {0,2,9,0,9,4,0,4,8,2,10,9,4,6,8,11,3,8,6,11,8},{10,9,2,10,2,6,6,2,11,6,11,4},
        {10,1,2,6,8,7,6,3,7},{10,1,2,6,0,7,6,4,7,0,8,7},{9,0,1,10,6,2,6,3,2,6,8,3},
        {1,9,4,1,4,6,1,6,10,2,1,10,4,8,6,3,1,8},{2,3,10,10,3,7,10,7,6,3,7,10},
        {2,3,7,2,7,10,3,0,7,6,10,7,4,7,6,0,8,7},{9,0,1,10,6,7,10,7,3,10,3,2},
        {9,4,8,9,8,1,1,8,3,2,1,3,6,7,3,2,3,7},{6,10,7,6,1,7,6,1,3,10,1,7,1,3,7},
        {6,10,1,6,1,7,10,1,2,0,8,1,8,7,1},{0,3,8,9,0,8,6,10,7},{6,10,7,9,4,8,9,8,1,9,1,2},
        {4,10,7,4,9,10,9,10,1,10,7,6,1,10,2,3,11,2},{0,8,3,2,0,3,4,9,7,9,10,7,7,6,10},
        {0,1,9,8,4,7,3,11,2,10,6,5},{5,10,6,1,9,2,9,11,2,9,8,11},{9,6,10,9,4,6,2,7,3,7,6,2},
        {2,3,7,2,7,9,2,9,1,7,6,9,0,8,3,4,6,7,9,4,7},{7,11,2,7,6,9,7,9,4,6,10,9},
        {4,6,8,4,8,9,6,10,8,2,8,3,10,2,8},{10,7,6,1,7,10,1,5,7,1,9,5,8,4,7},
        {}, // 189
    };
    static const std::vector<int> empty;
    if (cubeIndex < 0 || cubeIndex >= 256) return empty;
    return table[cubeIndex];
}

template <typename Scalar>
Scalar interp(Scalar iso, Scalar v0, Scalar v1, Scalar p0, Scalar p1)
{
    if (std::abs(v1 - v0) < Scalar(1e-12)) return p0;
    return p0 + (p1 - p0) * (iso - v0) / (v1 - v0);
}

} // namespace detail

template <typename Scalar>
class MarchingCubes
{
public:
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;
    using Vec3 = plamatrix::Vec3<Scalar>;
    using ScalarFunction = std::function<Scalar(Scalar, Scalar, Scalar)>;

    void setBounds(const Vec3& min_corner, const Vec3& max_corner)
    {
        _min = min_corner; _max = max_corner;
    }

    void setResolution(int nx, int ny, int nz) { _nx = nx; _ny = ny; _nz = nz; }
    void setIsoLevel(Scalar iso) { _iso = iso; }

    std::tuple<Matrix, Matrix> extract(const ScalarFunction& fn) const
    {
        Scalar dx = (_max.x - _min.x) / Scalar(_nx);
        Scalar dy = (_max.y - _min.y) / Scalar(_ny);
        Scalar dz = (_max.z - _min.z) / Scalar(_nz);

        int vx_r = _nx + 1, vy_r = _ny + 1, vz_r = _nz + 1;
        std::vector<Scalar> field(static_cast<std::size_t>(vz_r * vy_r * vx_r));
        for (int iz = 0; iz < vz_r; ++iz)
            for (int iy = 0; iy < vy_r; ++iy)
                for (int ix = 0; ix < vx_r; ++ix)
                {
                    Scalar x = _min.x + Scalar(ix) * dx;
                    Scalar y = _min.y + Scalar(iy) * dy;
                    Scalar z = _min.z + Scalar(iz) * dz;
                    field[static_cast<std::size_t>(iz*vy_r*vx_r + iy*vx_r + ix)] = fn(x, y, z);
                }

        std::vector<Scalar> vx, vy, vz;
        std::vector<int> tri_idx;

        int corners[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
        int edge_pairs[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};

        for (int iz = 0; iz < _nz; ++iz)
            for (int iy = 0; iy < _ny; ++iy)
                for (int ix = 0; ix < _nx; ++ix)
                {
                    Scalar vals[8];
                    Vec3 pts[8];
                    for (int c = 0; c < 8; ++c)
                    {
                        int cx = ix + corners[c][0], cy = iy + corners[c][1], cz = iz + corners[c][2];
                        pts[c] = {_min.x+Scalar(cx)*dx, _min.y+Scalar(cy)*dy, _min.z+Scalar(cz)*dz};
                        vals[c] = field[static_cast<std::size_t>(cz*vy_r*vx_r + cy*vx_r + cx)];
                    }

                    int cube_idx = 0;
                    for (int c = 0; c < 8; ++c)
                        if (vals[c] < _iso) cube_idx |= (1 << c);

                    int ef = detail::edgeTable(cube_idx);
                    if (ef == 0) continue;

                    Vec3 ev[12] = {};
                    for (int e = 0; e < 12; ++e)
                    {
                        if (ef & (1 << e))
                        {
                            int i0 = edge_pairs[e][0], i1 = edge_pairs[e][1];
                            Scalar t = detail::interp(_iso, vals[i0], vals[i1], Scalar(0), Scalar(1));
                            ev[e] = {pts[i0].x+t*(pts[i1].x-pts[i0].x),
                                     pts[i0].y+t*(pts[i1].y-pts[i0].y),
                                     pts[i0].z+t*(pts[i1].z-pts[i0].z)};
                        }
                    }

                    const auto& tris = detail::triTable(cube_idx);
                    for (std::size_t t = 0; t + 2 < tris.size(); t += 3)
                    {
                        int e0 = tris[t], e1 = tris[t+1], e2 = tris[t+2];
                        vx.push_back(ev[e0].x); vy.push_back(ev[e0].y); vz.push_back(ev[e0].z);
                        tri_idx.push_back(static_cast<int>(vx.size()) - 1);
                        vx.push_back(ev[e1].x); vy.push_back(ev[e1].y); vz.push_back(ev[e1].z);
                        tri_idx.push_back(static_cast<int>(vx.size()) - 1);
                        vx.push_back(ev[e2].x); vy.push_back(ev[e2].y); vz.push_back(ev[e2].z);
                        tri_idx.push_back(static_cast<int>(vx.size()) - 1);
                    }
                }

        Matrix verts(static_cast<plamatrix::Index>(vx.size()), 3);
        for (std::size_t i = 0; i < vx.size(); ++i)
        {
            verts(static_cast<plamatrix::Index>(i), 0) = vx[i];
            verts(static_cast<plamatrix::Index>(i), 1) = vy[i];
            verts(static_cast<plamatrix::Index>(i), 2) = vz[i];
        }

        plamatrix::Index nf = static_cast<plamatrix::Index>(tri_idx.size()) / 3;
        Matrix faces(nf, 3);
        for (plamatrix::Index f = 0; f < nf; ++f)
        {
            faces(f, 0) = Scalar(tri_idx[static_cast<std::size_t>(f*3)]);
            faces(f, 1) = Scalar(tri_idx[static_cast<std::size_t>(f*3+1)]);
            faces(f, 2) = Scalar(tri_idx[static_cast<std::size_t>(f*3+2)]);
        }

        return {std::move(verts), std::move(faces)};
    }

private:
    Vec3 _min{-1,-1,-1}, _max{1,1,1};
    int _nx = 10, _ny = 10, _nz = 10;
    Scalar _iso = 0;
};

} // namespace mesh
} // namespace plapoint
