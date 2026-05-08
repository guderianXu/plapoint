#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace plapoint {
namespace mesh {

template <typename Scalar>
class PoissonReconstruction
{
public:
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;
    using PointCloudType = PointCloud<Scalar, plamatrix::Device::CPU>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud) { _cloud = cloud; }
    void setDepth(int d) { _depth = d; }
    void setSolverIterations(int n) { _solver_iters = n; }

    std::tuple<Matrix, Matrix> reconstruct() const
    {
        if (!_cloud) throw std::runtime_error("Poisson: input cloud not set");
        if (!_cloud->hasNormals()) throw std::runtime_error("Poisson: cloud must have normals");

        // Compute bounding box
        Scalar min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10, min_z = 1e10, max_z = -1e10;
        int n = static_cast<int>(_cloud->size());
        for (int i = 0; i < n; ++i)
        {
            Scalar x = _cloud->points()(i, 0), y = _cloud->points()(i, 1), z = _cloud->points()(i, 2);
            if (x < min_x) min_x = x; if (x > max_x) max_x = x;
            if (y < min_y) min_y = y; if (y > max_y) max_y = y;
            if (z < min_z) min_z = z; if (z > max_z) max_z = z;
        }
        Scalar pad = Scalar(0.2) * std::max({max_x-min_x, max_y-min_y, max_z-min_z, Scalar(1e-6)});
        min_x -= pad; max_x += pad; min_y -= pad; max_y += pad; min_z -= pad; max_z += pad;

        int res = 1 << _depth;
        int vr = res + 1;
        Scalar dx = (max_x - min_x) / Scalar(res);

        // Splat normals to compute divergence on grid vertices
        std::vector<Scalar> div(static_cast<std::size_t>(vr*vr*vr), 0);
        std::vector<Scalar> wsum(static_cast<std::size_t>(vr*vr*vr), 0);

        for (int pi = 0; pi < n; ++pi)
        {
            Scalar px = _cloud->points()(pi, 0), py = _cloud->points()(pi, 1), pz = _cloud->points()(pi, 2);
            Scalar nx = _cloud->normals()->getValue(pi, 0);
            Scalar ny = _cloud->normals()->getValue(pi, 1);
            Scalar nz = _cloud->normals()->getValue(pi, 2);

            int ix = std::clamp(static_cast<int>(std::round((px - min_x) / dx)), 0, vr-1);
            int iy = std::clamp(static_cast<int>(std::round((py - min_x) / dx)), 0, vr-1);
            int iz = std::clamp(static_cast<int>(std::round((pz - min_x) / dx)), 0, vr-1);

            for (int sx = -1; sx <= 1; ++sx)
                for (int sy = -1; sy <= 1; ++sy)
                    for (int sz = -1; sz <= 1; ++sz)
                    {
                        int gx = ix + sx, gy = iy + sy, gz = iz + sz;
                        if (gx < 0 || gx >= vr || gy < 0 || gy >= vr || gz < 0 || gz >= vr) continue;
                        Scalar w = (Scalar(1)-std::abs(Scalar(sx))) * (Scalar(1)-std::abs(Scalar(sy))) * (Scalar(1)-std::abs(Scalar(sz)));
                        std::size_t idx = static_cast<std::size_t>(gz*vr*vr + gy*vr + gx);
                        div[idx] += w * (nx + ny + nz);
                        wsum[idx] += w;
                    }
        }

        // Gauss-Seidel solver
        std::vector<Scalar> chi(static_cast<std::size_t>(vr*vr*vr), 0);
        for (int iter = 0; iter < _solver_iters; ++iter)
        {
            for (int iz = 0; iz < vr; ++iz)
                for (int iy = 0; iy < vr; ++iy)
                    for (int ix = 0; ix < vr; ++ix)
                    {
                        std::size_t idx = static_cast<std::size_t>(iz*vr*vr + iy*vr + ix);
                        Scalar sum = 0;
                        int count = 0;
                        if (ix > 0)    { sum += chi[static_cast<std::size_t>(iz*vr*vr+iy*vr+ix-1)]; ++count; }
                        if (ix < vr-1) { sum += chi[static_cast<std::size_t>(iz*vr*vr+iy*vr+ix+1)]; ++count; }
                        if (iy > 0)    { sum += chi[static_cast<std::size_t>(iz*vr*vr+(iy-1)*vr+ix)]; ++count; }
                        if (iy < vr-1) { sum += chi[static_cast<std::size_t>(iz*vr*vr+(iy+1)*vr+ix)]; ++count; }
                        if (iz > 0)    { sum += chi[static_cast<std::size_t>((iz-1)*vr*vr+iy*vr+ix)]; ++count; }
                        if (iz < vr-1) { sum += chi[static_cast<std::size_t>((iz+1)*vr*vr+iy*vr+ix)]; ++count; }
                        chi[idx] = (sum - dx*dx * div[idx]) / Scalar(count);
                    }
        }

        // Extract isosurface
        auto chi_fn = [&](Scalar x, Scalar y, Scalar z) -> Scalar {
            int ix = std::clamp(static_cast<int>((x - min_x) / dx), 0, vr-1);
            int iy = std::clamp(static_cast<int>((y - min_x) / dx), 0, vr-1);
            int iz = std::clamp(static_cast<int>((z - min_x) / dx), 0, vr-1);
            return chi[static_cast<std::size_t>(iz*vr*vr + iy*vr + ix)];
        };

        MarchingCubes<Scalar> mc;
        mc.setBounds({min_x, min_x, min_x}, {max_x, max_x, max_x});
        mc.setResolution(res, res, res);
        mc.setIsoLevel(Scalar(0));
        return mc.extract(chi_fn);
    }

private:
    std::shared_ptr<const PointCloudType> _cloud;
    int _depth = 6;
    int _solver_iters = 20;
};

} // namespace mesh
} // namespace plapoint
