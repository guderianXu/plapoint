# Plapoint Phase 3 — Filters, Mesh, I/O

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add RadiusOutlierRemoval, UniformDownsample, NormalRefinement, MarchingCubes, PoissonReconstruction, and PLY I/O — completing the feature set needed by plascan.

**Architecture:** All classes follow `<Scalar, Dev>` template pattern. Filters inherit from `Filter<Scalar, Dev>`. Mesh algorithms operate on CPU dense matrices. PLY I/O extends PointCloud with optional normals field.

**Tech Stack:** C++17, plamatrix (DenseMatrix, solve), KdTree (radius search)

---

### Task 1: Extend PointCloud with Optional Normals

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`

- [ ] **Step 1: Add optional normals support**

Add `setNormals`, `hasNormals`, and `normals()` accessor:

```cpp
// After existing public section in point_cloud.h, before private:

    /// Set optional normals (Nx3 matrix)
    void setNormals(const MatrixType& n)
    {
        if (n.rows() != _points.rows() || n.cols() != 3)
            throw std::runtime_error("Normals must match point count and be Nx3");
        _normals = std::make_unique<MatrixType>();
        // Copy element by element (DenseMatrix is move-only)
        *_normals = MatrixType(n.rows(), n.cols());
        for (plamatrix::Index r = 0; r < n.rows(); ++r)
            for (int c = 0; c < 3; ++c)
                _normals->setValue(r, c, pointGet(n, r, c));
    }

    void setNormals(MatrixType&& n)
    {
        if (n.rows() != _points.rows() || n.cols() != 3)
            throw std::runtime_error("Normals must match point count and be Nx3");
        _normals = std::make_unique<MatrixType>(std::move(n));
    }

    bool hasNormals() const { return _normals != nullptr; }

    const MatrixType* normals() const { return _normals.get(); }

    MatrixType* normals() { return _normals.get(); }
```

Add `#include <memory>` and private field `std::unique_ptr<MatrixType> _normals;`.

Also add a helper `pointGet` static function used for copying:
```cpp
private:
    static Scalar pointGet(const MatrixType& m, plamatrix::Index r, int c)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return m(r, c);
        else
            return m.getValue(r, c);
    }
```

- [ ] **Step 2: Write test**

```cpp
// test/unit/core/point_cloud_normals_test.cpp
#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointCloudTest, OptionalNormals)
{
    using Scalar = float;
    plapoint::PointCloud<Scalar, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasNormals());

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> n(10, 3);
    n.fill(1.0f);
    cloud.setNormals(std::move(n));
    EXPECT_TRUE(cloud.hasNormals());
    EXPECT_EQ(cloud.normals()->rows(), 10);
    EXPECT_FLOAT_EQ(cloud.normals()->getValue(0, 0), 1.0f);
}

TEST(PointCloudTest, NormalsWrongSize)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(5);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> bad(3, 3);
    bad.fill(0);
    EXPECT_THROW(cloud.setNormals(bad), std::runtime_error);
}
```

- [ ] **Step 3: Verify test fails on missing fields, then update header**

Run: `cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=*Normals*`
Expected: Tests fail (no hasNormals etc.), implement, then PASS.

- [ ] **Step 4: Commit**

```bash
git add include/plapoint/core/point_cloud.h test/unit/core/point_cloud_normals_test.cpp
git commit -m "feat: add optional normals to PointCloud"
git push origin master
```

---

### Task 2: RadiusOutlierRemoval Filter

**Files:**
- Create: `include/plapoint/filters/radius_outlier_removal.h`
- Create: `test/unit/filters/radius_outlier_removal_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/filters/radius_outlier_removal_test.cpp
#include <gtest/gtest.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(RadiusOutlierRemovalTest, RemovesIsolatedPoint)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(11, 3);
    for (int i = 0; i < 10; ++i) { pts.setValue(i, 0, Scalar(i)*0.01f); pts.setValue(i, 1, 0); pts.setValue(i, 2, 0); }
    pts.setValue(10, 0, 100); pts.setValue(10, 1, 0); pts.setValue(10, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> ror;
    ror.setInputCloud(cloud);
    ror.setRadius(Scalar(1.0));
    ror.setMinNeighbors(2);

    Cloud output;
    ror.filter(output);
    EXPECT_EQ(output.size(), 10u);
}
```

- [ ] **Step 2: Verify test fails**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation error (header not found).

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/filters/radius_outlier_removal.h
#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class RadiusOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setRadius(Scalar r) { _radius = r; }
    void setMinNeighbors(int n) { _min_pts = n; }

protected:
    void applyFilter(PointCloudType& output) override
    {
        auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
        tree->setInputCloud(this->_input);
        tree->build();

        std::size_t n = this->_input->size();
        std::vector<int> inliers;

        for (std::size_t i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(static_cast<int>(i));
            auto neighbors = tree->radiusSearch(pt, _radius);
            if (static_cast<int>(neighbors.size()) >= _min_pts)
                inliers.push_back(static_cast<int>(i));
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(inliers.size()), 3);
        for (std::size_t i = 0; i < inliers.size(); ++i)
        {
            int src = inliers[i];
            pts(static_cast<plamatrix::Index>(i), 0) = pointCoord(src, 0);
            pts(static_cast<plamatrix::Index>(i), 1) = pointCoord(src, 1);
            pts(static_cast<plamatrix::Index>(i), 2) = pointCoord(src, 2);
        }
        output = PointCloudType(std::move(pts));
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return this->_input->points()(idx, dim);
        else
            return this->_input->points().getValue(idx, dim);
    }
    plamatrix::Vec3<Scalar> pointVec(int idx) const
        { return {pointCoord(idx,0), pointCoord(idx,1), pointCoord(idx,2)}; }

    Scalar _radius = 0.1;
    int _min_pts = 2;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=RadiusOutlierRemoval*`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/filters/radius_outlier_removal.h test/unit/filters/radius_outlier_removal_test.cpp
git commit -m "feat: add RadiusOutlierRemoval filter"
git push origin master
```

---

### Task 3: UniformDownsample Filter

**Files:**
- Create: `include/plapoint/filters/uniform_downsample.h`
- Create: `test/unit/filters/uniform_downsample_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/filters/uniform_downsample_test.cpp
#include <gtest/gtest.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(UniformDownsampleTest, KeepsEveryNthPoint)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(10, 3);
    for (int i = 0; i < 10; ++i) { pts.setValue(i, 0, Scalar(i)); pts.setValue(i, 1, 0); pts.setValue(i, 2, 0); }
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(3);

    Cloud output;
    ud.filter(output);
    // Every 3rd: indices 0, 3, 6, 9 = 4 points
    EXPECT_EQ(output.size(), 4u);
}
```

- [ ] **Step 2: Verify test fails (compilation error)**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation error.

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/filters/uniform_downsample.h
#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class UniformDownsample : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setStep(int s) { _step = std::max(1, s); }

protected:
    void applyFilter(PointCloudType& output) override
    {
        std::size_t n = this->_input->size();
        std::size_t out_n = (n + static_cast<std::size_t>(_step) - 1) / static_cast<std::size_t>(_step);

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(out_n), 3);
        std::size_t out_idx = 0;
        for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(_step))
        {
            pts(static_cast<plamatrix::Index>(out_idx), 0) = pointCoord(static_cast<int>(i), 0);
            pts(static_cast<plamatrix::Index>(out_idx), 1) = pointCoord(static_cast<int>(i), 1);
            pts(static_cast<plamatrix::Index>(out_idx), 2) = pointCoord(static_cast<int>(i), 2);
            ++out_idx;
        }
        output = PointCloudType(std::move(pts));
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return this->_input->points()(idx, dim);
        else
            return this->_input->points().getValue(idx, dim);
    }

    int _step = 2;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=UniformDownsample*`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/filters/uniform_downsample.h test/unit/filters/uniform_downsample_test.cpp
git commit -m "feat: add UniformDownsample filter"
git push origin master
```

---

### Task 4: NormalRefinement

**Files:**
- Create: `include/plapoint/features/normal_refinement.h`
- Create: `test/unit/features/normal_refinement_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/features/normal_refinement_test.cpp
#include <gtest/gtest.h>
#include <plapoint/features/normal_refinement.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>

TEST(NormalRefinementTest, SmoothNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    // 4 points on XY plane, normals = (0,0,1) but with noise
    Matrix pts(4, 3);
    pts.setValue(0,0,0); pts.setValue(0,1,0); pts.setValue(0,2,0);
    pts.setValue(1,0,1); pts.setValue(1,1,0); pts.setValue(1,2,0);
    pts.setValue(2,0,0); pts.setValue(2,1,1); pts.setValue(2,2,0);
    pts.setValue(3,0,1); pts.setValue(3,1,1); pts.setValue(3,2,0);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    Matrix normals(4, 3);
    // Noisy: mostly (0,0,1) with one flipped
    normals.setValue(0,0,0.1f); normals.setValue(0,1,0.1f); normals.setValue(0,2,1.0f);
    normals.setValue(1,0,0.1f); normals.setValue(1,1,0.0f); normals.setValue(1,2,0.9f);
    normals.setValue(2,0,0.0f); normals.setValue(2,1,0.1f); normals.setValue(2,2,1.1f);
    normals.setValue(3,0,0.0f); normals.setValue(3,1,0.0f); normals.setValue(3,2,-1.0f); // flipped
    cloud->setNormals(std::move(normals));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalRefinement<Scalar, plamatrix::Device::CPU> nr;
    nr.setInputCloud(cloud);
    nr.setSearchMethod(tree);
    nr.smooth(4);   // smooth with 4 neighbors

    auto* ns = cloud->normals();
    // After smoothing, all should be approximately (0,0,1)
    for (int i = 0; i < 4; ++i)
    {
        Scalar nx = ns->getValue(i, 0), ny = ns->getValue(i, 1), nz = ns->getValue(i, 2);
        Scalar len = std::sqrt(nx*nx + ny*ny + nz*nz);
        EXPECT_NEAR(len, Scalar(1), Scalar(1e-4));
        EXPECT_GT(nz / len, Scalar(0));
    }
}
```

- [ ] **Step 2: Verify test fails**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation error.

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/features/normal_refinement.h
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/ops/point_cloud.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class NormalRefinement
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<PointCloudType>& cloud) { _cloud = cloud; }
    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree) { _tree = tree; }

    void smooth(int k)
    {
        if (!_cloud) throw std::runtime_error("NormalRefinement: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalRefinement: search method not set");
        if (!_cloud->hasNormals()) throw std::runtime_error("NormalRefinement: cloud has no normals");

        auto* normals = _cloud->normals();
        int n = static_cast<int>(_cloud->size());

        // Copy current normals to temp
        std::vector<plamatrix::Vec3<Scalar>> temp(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            temp[static_cast<std::size_t>(i)] = {normals->getValue(i,0), normals->getValue(i,1), normals->getValue(i,2)};

        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(i);
            auto neighbors = _tree->nearestKSearch(pt, k);
            Scalar sx = 0, sy = 0, sz = 0;
            for (int nb : neighbors)
            {
                sx += temp[static_cast<std::size_t>(nb)].x;
                sy += temp[static_cast<std::size_t>(nb)].y;
                sz += temp[static_cast<std::size_t>(nb)].z;
            }
            Scalar len = std::sqrt(sx*sx + sy*sy + sz*sz);
            if (len > Scalar(1e-10))
            {
                normals->setValue(i, 0, sx / len);
                normals->setValue(i, 1, sy / len);
                normals->setValue(i, 2, sz / len);
            }
        }
    }

    void orientConsistently(const plamatrix::Vec3<Scalar>& viewpoint)
    {
        if (!_cloud || !_cloud->hasNormals()) return;
        auto* normals = _cloud->normals();
        int n = static_cast<int>(_cloud->size());
        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(i);
            Scalar dx = viewpoint.x - pt.x, dy = viewpoint.y - pt.y, dz = viewpoint.z - pt.z;
            Scalar nx = normals->getValue(i, 0), ny = normals->getValue(i, 1), nz = normals->getValue(i, 2);
            if (dx * nx + dy * ny + dz * nz < 0)
            {
                normals->setValue(i, 0, -nx);
                normals->setValue(i, 1, -ny);
                normals->setValue(i, 2, -nz);
            }
        }
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return _cloud->points()(idx, dim);
        else
            return _cloud->points().getValue(idx, dim);
    }
    plamatrix::Vec3<Scalar> pointVec(int idx) const
        { return {pointCoord(idx,0), pointCoord(idx,1), pointCoord(idx,2)}; }

    std::shared_ptr<PointCloudType> _cloud;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=NormalRefinement*`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/features/normal_refinement.h test/unit/features/normal_refinement_test.cpp
git commit -m "feat: add NormalRefinement with smoothing and orientation"
git push origin master
```

---

### Task 5: MarchingCubes

**Files:**
- Create: `include/plapoint/mesh/marching_cubes.h`
- Create: `test/unit/mesh/marching_cubes_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/mesh/marching_cubes_test.cpp
#include <gtest/gtest.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plamatrix/plamatrix.h>
#include <cmath>

TEST(MarchingCubesTest, SphereIsosurface)
{
    using Scalar = float;

    // Define a sphere implicit function: f(x,y,z) = x^2 + y^2 + z^2 - 4
    auto sphere_fn = [](Scalar x, Scalar y, Scalar z) {
        return x*x + y*y + z*z - Scalar(4);
    };

    plapoint::mesh::MarchingCubes<Scalar> mc;
    mc.setBounds({-3,-3,-3}, {3,3,3});
    mc.setResolution(20, 20, 20);
    mc.setIsoLevel(Scalar(0));

    auto [verts, faces] = mc.extract(sphere_fn);

    // Should have produced triangles
    EXPECT_GT(verts.rows(), 0);
    EXPECT_GT(faces.rows(), 0);
    EXPECT_EQ(verts.cols(), 3);
    EXPECT_EQ(faces.cols(), 3);

    // All vertices should be approximately on the sphere surface
    for (plamatrix::Index i = 0; i < verts.rows(); ++i)
    {
        Scalar x = verts.getValue(i, 0);
        Scalar y = verts.getValue(i, 1);
        Scalar z = verts.getValue(i, 2);
        Scalar r2 = x*x + y*y + z*z;
        EXPECT_NEAR(r2, Scalar(4), Scalar(0.5));
    }
}
```

- [ ] **Step 2: Verify test fails**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`

- [ ] **Step 3: Write implementation**

Marching cubes with the classic 256-case lookup table. Core algorithm:

```cpp
// include/plapoint/mesh/marching_cubes.h
#pragma once

#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/core/types.h>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

namespace plapoint {
namespace mesh {

// Edge table and triangle table (classic marching cubes 256 cases)
// See: http://paulbourke.net/geometry/polygonise/

namespace detail {

inline int edgeTable(int cubeIndex)
{
    static const int table[256] = {
        0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
    };
    return table[cubeIndex & 255];
}

inline const int* triTable(int cubeIndex)
{
    static const int table[256][16] = {
        {-1},
{0,8,3,-1},{0,1,9,-1},{1,8,3,9,8,1,-1},{1,2,10,-1},{0,8,3,1,2,10,-1},
{9,2,10,0,2,9,-1},{2,8,3,2,10,8,10,9,8,-1},{3,11,2,-1},{0,11,2,8,11,0,-1},
{1,9,0,2,3,11,-1},{1,11,2,1,9,11,9,8,11,-1},{3,10,1,11,10,3,-1},
{0,10,1,0,8,10,8,11,10,-1},{3,9,0,3,11,9,11,10,9,-1},
{9,8,10,10,8,11,-1},{4,7,8,-1},{4,3,0,7,3,4,-1},{0,1,9,8,4,7,-1},
{4,1,9,4,7,1,7,3,1,-1},{1,2,10,8,4,7,-1},{3,4,7,3,0,4,1,2,10,-1},
{9,2,10,9,0,2,8,4,7,-1},{2,10,9,2,9,7,2,7,3,7,9,4,-1},{8,4,7,3,11,2,-1},
{11,4,7,11,2,4,2,0,4,-1},{9,0,1,8,4,7,2,3,11,-1},
{4,7,11,9,4,11,9,11,2,9,2,1,-1},{3,10,1,3,11,10,7,8,4,-1},
{1,11,10,1,4,11,1,0,4,7,11,4,-1},{4,7,8,9,0,11,9,11,10,11,0,3,-1},
{4,7,11,4,11,9,9,11,10,-1},{9,5,4,-1},{9,5,4,0,8,3,-1},{0,5,4,1,5,0,-1},
{8,5,4,8,3,5,3,1,5,-1},{1,2,10,9,5,4,-1},{3,0,8,1,2,10,4,9,5,-1},
{5,2,10,5,4,2,4,0,2,-1},{2,10,5,3,2,5,3,5,4,3,4,8,-1},
{9,5,4,2,3,11,-1},{11,0,8,11,2,0,9,5,4,-1},{5,0,1,5,4,0,3,11,2,-1},
{11,2,1,11,1,5,11,5,8,11,8,4,-1},{10,1,2,9,5,4,3,10,1,3,11,10,-1},
{0,8,11,0,11,10,10,11,1,4,9,5,-1},{5,4,9,3,11,0,3,0,10,10,0,2,-1},
{5,4,8,5,8,11,11,8,3,10,5,11,10,11,2,-1},{9,7,8,5,7,9,-1},
{9,3,0,9,5,3,5,7,3,-1},{0,7,8,0,1,7,1,5,7,-1},{1,5,3,3,5,7,-1},
{9,7,8,9,5,7,10,1,2,-1},{10,1,2,9,5,0,5,3,0,5,7,3,-1},
{8,0,2,8,2,5,8,5,7,10,2,5,-1},{2,10,5,2,5,3,3,5,7,-1},
{7,9,5,7,8,9,3,11,2,-1},{9,5,7,9,7,2,9,2,0,2,7,3,11,2,3,-1},
{2,3,11,0,1,8,1,7,8,1,5,7,-1},{11,2,1,11,1,5,11,5,7,11,7,8,1,5,8,-1},
{9,7,8,9,5,7,10,3,10,1,3,10,11,3,-1},{5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
{11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
{11,10,5,7,11,5,-1},{10,6,5,-1},{0,8,3,5,10,6,-1},{9,0,1,5,10,6,-1},
{1,8,3,1,9,8,5,10,6,-1},{1,6,5,2,6,1,-1},{1,6,5,1,2,6,3,0,8,-1},
{9,6,5,9,0,6,0,2,6,-1},{5,9,8,5,8,2,5,2,6,3,2,8,-1},
{2,3,11,10,6,5,-1},{11,0,8,11,2,0,10,6,5,-1},{0,1,9,2,3,11,5,10,6,-1},
{5,10,6,1,9,2,9,11,2,9,8,11,-1},{6,3,11,6,5,3,5,1,3,-1},
{0,8,11,0,11,5,0,5,1,5,11,6,-1},{3,11,6,0,3,6,0,6,5,0,5,9,-1},
{6,5,9,6,9,11,11,9,8,-1},{5,10,6,4,7,8,-1},{4,3,0,4,7,3,6,5,10,-1},
{1,9,0,5,10,6,8,4,7,-1},{10,6,5,1,9,7,1,7,3,7,9,4,-1},
{6,1,2,6,5,1,4,7,8,-1},{1,2,5,5,2,6,3,0,4,3,4,7,-1},
{8,4,7,9,0,5,0,6,5,0,2,6,-1},{7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
{3,11,2,7,8,4,10,6,5,-1},{5,10,6,4,7,2,4,2,0,2,7,11,-1},
{0,1,9,4,7,8,2,3,11,5,10,6,-1},{9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
{8,4,7,3,11,5,3,5,1,5,11,6,-1},{5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
{0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},{6,5,9,6,9,11,4,7,9,7,11,9,-1},
{10,4,9,6,4,10,-1},{4,10,6,4,9,10,0,8,3,-1},{10,0,1,10,6,0,6,4,0,-1},
{8,3,1,8,1,6,8,6,4,6,1,10,-1},{1,4,9,1,2,4,2,6,4,-1},
{3,0,8,1,2,9,2,4,9,2,6,4,-1},{0,2,4,4,2,6,-1},
{8,3,2,8,2,4,4,2,6,-1},{10,4,9,10,6,4,11,2,3,-1},
{0,8,2,2,8,11,4,9,10,4,10,6,-1},{3,11,2,0,1,6,0,6,4,6,1,10,-1},
{6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
{9,6,4,9,3,6,9,1,3,11,6,3,-1},
{8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
{3,11,6,3,6,0,0,6,4,-1},{6,4,8,11,6,8,-1},{7,10,6,7,8,10,8,9,10,-1},
{0,7,3,0,10,7,0,9,10,6,7,10,-1},{10,6,7,1,10,7,1,7,8,1,8,0,-1},
{10,6,7,10,7,1,1,7,3,-1},{1,2,6,1,6,8,1,8,9,8,6,7,-1},
{2,6,7,2,7,3,6,7,2,9,0,1,-1},{7,8,0,7,0,6,6,0,2,-1},
{7,3,2,6,7,2,-1},{2,3,11,10,6,8,10,8,9,8,6,7,-1},
{2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
{1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
{11,2,1,11,1,7,10,6,1,6,7,1,-1},
{8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
{0,9,1,11,6,7,-1},{7,8,0,7,0,6,3,11,0,11,6,0,-1},
{7,11,6,-1},{7,6,11,-1},{3,0,8,11,7,6,-1},{0,1,9,11,7,6,-1},
{8,1,9,8,3,1,11,7,6,-1},{10,1,2,6,11,7,-1},{1,2,10,3,0,8,6,11,7,-1},
{2,9,0,2,10,9,6,11,7,-1},{6,11,7,2,10,3,10,8,3,10,9,8,-1},
{7,2,3,6,2,7,-1},{7,0,8,7,6,0,6,2,0,-1},{2,7,6,2,3,7,0,1,9,-1},
{1,6,2,1,8,6,1,9,8,8,7,6,-1},{10,7,6,10,1,7,1,3,7,-1},
{10,7,6,1,7,10,1,8,7,1,0,8,-1},{0,3,7,0,7,6,0,6,9,6,7,10,-1},
{7,6,10,7,10,8,8,10,9,-1},{6,8,4,11,8,6,-1},{3,6,11,3,0,6,0,4,6,-1},
{8,6,11,8,4,6,9,0,1,-1},{9,4,6,9,6,3,9,3,1,11,3,6,-1},
{6,8,4,6,11,8,2,10,1,-1},{1,2,10,3,0,11,0,6,11,0,4,6,-1},
{4,11,8,4,6,11,0,2,9,2,10,9,-1},{10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
{8,2,3,8,6,2,8,4,6,6,2,7,-1},{0,2,3,0,2,7,0,7,4,2,7,6,-1},
{8,4,6,8,6,2,8,2,3,2,6,7,0,1,9,-1},
{9,4,6,9,6,3,9,3,1,8,4,9,6,7,3,2,1,3,-1},
{6,8,4,6,11,8,10,1,3,10,3,7,-1},{1,3,10,1,3,7,3,7,6,8,4,6,0,4,3,-1},
{0,3,8,0,6,8,0,9,6,0,1,9,4,6,8,10,7,6,9,10,6,-1},
{10,7,6,9,4,3,9,3,1,10,9,3,10,3,7,-1},
{9,4,8,9,8,3,8,4,6,3,8,11,-1},{3,4,9,3,4,9,4,6,9,0,4,3,-1},
{0,1,9,8,4,11,8,11,3,8,4,11,-1},{1,9,4,1,4,11,1,11,2,11,4,6,-1},
{3,8,4,3,4,11,8,4,6,1,2,10,-1},{1,2,10,3,0,11,0,4,11,0,2,7,0,7,4,-1},
{0,2,9,0,9,4,0,4,8,2,10,9,4,6,8,11,3,8,6,11,8,-1},
{10,9,2,10,2,6,2,9,4,6,9,4,6,11,2,6,4,11,-1},
{10,1,2,6,8,7,6,3,7,-1},{10,1,2,6,0,7,6,4,7,0,8,7,-1},
{9,0,1,10,6,2,6,3,2,6,8,3,-1},{1,9,4,1,4,6,1,6,10,2,1,10,4,8,6,3,1,8,-1},
{2,3,10,10,3,7,10,7,6,3,7,10,-1},{2,3,7,2,7,10,3,0,7,6,10,7,4,7,6,0,8,7,-1},
{9,0,1,10,6,7,10,7,3,10,3,2,-1},
{9,4,8,9,8,1,8,4,6,1,8,3,2,1,3,6,7,3,2,3,7,-1},
{6,10,7,6,1,7,6,1,3,10,1,7,1,3,7,-1},
{6,10,1,6,1,7,10,1,2,0,8,1,8,7,1,-1},{0,3,8,9,0,8,6,10,7,-1},
{6,10,7,9,4,8,9,8,1,9,1,2,-1},
{4,10,7,4,9,10,9,10,1,10,7,6,1,10,2,3,11,2,-1},
{0,8,3,2,0,3,4,9,7,9,10,7,7,6,10,-1},
{0,1,9,8,4,7,3,11,2,10,6,5,-1},{5,10,6,1,9,2,9,11,2,9,8,11,-1},
{9,6,10,9,4,6,2,7,3,7,6,2,-1},{2,3,7,2,7,9,2,9,1,7,6,9,0,8,3,4,6,7,9,4,7,-1},
{7,11,2,7,6,9,7,9,4,6,10,9,-1},
{4,6,8,4,8,9,6,10,8,2,8,3,10,2,8,-1},
{10,7,6,1,7,10,1,5,7,1,9,5,8,4,7,-1}
    };
    return table[cubeIndex & 255];
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

    void setBounds(const Vec3& min_corner, const Vec3& max_corner)
    {
        _min = min_corner; _max = max_corner;
    }

    void setResolution(int nx, int ny, int nz)
    {
        _nx = nx; _ny = ny; _nz = nz;
    }

    void setIsoLevel(Scalar iso) { _iso = iso; }

    using ScalarFunction = std::function<Scalar(Scalar, Scalar, Scalar)>;

    std::tuple<Matrix, Matrix> extract(const ScalarFunction& fn) const
    {
        Scalar dx = (_max.x - _min.x) / Scalar(_nx);
        Scalar dy = (_max.y - _min.y) / Scalar(_ny);
        Scalar dz = (_max.z - _min.z) / Scalar(_nz);

        // Evaluate scalar field at grid vertices
        std::vector<Scalar> field(static_cast<std::size_t>((_nx+1)*(_ny+1)*(_nz+1)));
        for (int iz = 0; iz <= _nz; ++iz)
            for (int iy = 0; iy <= _ny; ++iy)
                for (int ix = 0; ix <= _nx; ++ix)
                {
                    Scalar x = _min.x + Scalar(ix) * dx;
                    Scalar y = _min.y + Scalar(iy) * dy;
                    Scalar z = _min.z + Scalar(iz) * dz;
                    std::size_t idx = static_cast<std::size_t>(iz*(_ny+1)*(_nx+1) + iy*(_nx+1) + ix);
                    field[idx] = fn(x, y, z);
                }

        std::vector<Scalar> vx, vy, vz;
        std::vector<int> tri_indices;

        for (int iz = 0; iz < _nz; ++iz)
            for (int iy = 0; iy < _ny; ++iy)
                for (int ix = 0; ix < _nx; ++ix)
                {
                    // 8 corners of the cell
                    Scalar vals[8];
                    Vec3 pts[8];
                    int corners[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
                    for (int c = 0; c < 8; ++c)
                    {
                        int cx = ix + corners[c][0], cy = iy + corners[c][1], cz = iz + corners[c][2];
                        pts[c] = {_min.x + Scalar(cx)*dx, _min.y + Scalar(cy)*dy, _min.z + Scalar(cz)*dz};
                        vals[c] = field[static_cast<std::size_t>(cz*(_ny+1)*(_nx+1) + cy*(_nx+1) + cx)];
                    }

                    int cube_idx = 0;
                    for (int c = 0; c < 8; ++c)
                        if (vals[c] < _iso) cube_idx |= (1 << c);

                    int edge_flags = detail::edgeTable(cube_idx);
                    if (edge_flags == 0) continue;

                    Vec3 edge_verts[12];
                    int edge_pairs[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
                    for (int e = 0; e < 12; ++e)
                    {
                        if (edge_flags & (1 << e))
                        {
                            int i0 = edge_pairs[e][0], i1 = edge_pairs[e][1];
                            Scalar t = detail::interp(_iso, vals[i0], vals[i1], Scalar(0), Scalar(1));
                            edge_verts[e] = {
                                pts[i0].x + t*(pts[i1].x - pts[i0].x),
                                pts[i0].y + t*(pts[i1].y - pts[i0].y),
                                pts[i0].z + t*(pts[i1].z - pts[i0].z)
                            };
                        }
                    }

                    const int* tris = detail::triTable(cube_idx);
                    for (int t = 0; tris[t] != -1; t += 3)
                    {
                        for (int v = 0; v < 3; ++v)
                        {
                            vx.push_back(edge_verts[tris[t+v]].x);
                            vy.push_back(edge_verts[tris[t+v]].y);
                            vz.push_back(edge_verts[tris[t+v]].z);
                            tri_indices.push_back(static_cast<int>(vx.size()) - 1);
                        }
                    }
                }

        Matrix verts(static_cast<plamatrix::Index>(vx.size()), 3);
        for (std::size_t i = 0; i < vx.size(); ++i)
        {
            verts(static_cast<plamatrix::Index>(i), 0) = vx[i];
            verts(static_cast<plamatrix::Index>(i), 1) = vy[i];
            verts(static_cast<plamatrix::Index>(i), 2) = vz[i];
        }

        plamatrix::Index nf = static_cast<plamatrix::Index>(tri_indices.size()) / 3;
        Matrix faces(nf, 3);
        for (plamatrix::Index f = 0; f < nf; ++f)
        {
            faces(f, 0) = Scalar(tri_indices[static_cast<std::size_t>(f*3)]);
            faces(f, 1) = Scalar(tri_indices[static_cast<std::size_t>(f*3 + 1)]);
            faces(f, 2) = Scalar(tri_indices[static_cast<std::size_t>(f*3 + 2)]);
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
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=MarchingCubes*`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/mesh/marching_cubes.h test/unit/mesh/marching_cubes_test.cpp
git commit -m "feat: add MarchingCubes isosurface extraction"
git push origin master
```

---

### Task 6: Poisson Surface Reconstruction

**Files:**
- Create: `include/plapoint/mesh/poisson_reconstruction.h`
- Create: `test/unit/mesh/poisson_reconstruction_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/mesh/poisson_reconstruction_test.cpp
#include <gtest/gtest.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>

TEST(PoissonReconstructionTest, SphereReconstructsMesh)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    // Sample points on a sphere of radius 2
    int n_pts = 100;
    Matrix pts(n_pts, 3);
    Matrix nrm(n_pts, 3);
    for (int i = 0; i < n_pts; ++i)
    {
        Scalar theta = Scalar(i) * Scalar(2*3.14159) / Scalar(n_pts);
        Scalar phi = Scalar(i) * Scalar(3.14159) / Scalar(n_pts);
        Scalar x = Scalar(2) * std::sin(phi) * std::cos(theta);
        Scalar y = Scalar(2) * std::sin(phi) * std::sin(theta);
        Scalar z = Scalar(2) * std::cos(phi);
        pts.setValue(i, 0, x); pts.setValue(i, 1, y); pts.setValue(i, 2, z);
        Scalar r = std::sqrt(x*x + y*y + z*z);
        nrm.setValue(i, 0, x/r); nrm.setValue(i, 1, y/r); nrm.setValue(i, 2, z/r);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(nrm));

    plapoint::mesh::PoissonReconstruction<Scalar> pr;
    pr.setInputCloud(cloud);
    pr.setDepth(5);  // small depth for fast test

    auto [verts, faces] = pr.reconstruct();

    EXPECT_GT(verts.rows(), 0);
    EXPECT_GT(faces.rows(), 0);
}
```

- [ ] **Step 2: Verify test fails**

Run: compilation error.

- [ ] **Step 3: Write implementation**

Poisson surface reconstruction using a regular grid + Gauss-Seidel solver:

```cpp
// include/plapoint/mesh/poisson_reconstruction.h
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
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

        // Compute bounding box and expand slightly
        Scalar min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10, min_z = 1e10, max_z = -1e10;
        int n = static_cast<int>(_cloud->size());
        for (int i = 0; i < n; ++i)
        {
            Scalar x = pointCoord(i, 0), y = pointCoord(i, 1), z = pointCoord(i, 2);
            if (x < min_x) min_x = x; if (x > max_x) max_x = x;
            if (y < min_y) min_y = y; if (y > max_y) max_y = y;
            if (z < min_z) min_z = z; if (z > max_z) max_z = z;
        }
        Scalar pad = Scalar(0.1) * std::max({max_x-min_x, max_y-min_y, max_z-min_z});
        min_x -= pad; max_x += pad; min_y -= pad; max_y += pad; min_z -= pad; max_z += pad;

        int res = 1 << _depth;  // 2^depth cells per dimension
        int vr = res + 1;       // vertices per dimension
        Scalar dx = (max_x - min_x) / Scalar(res);
        Scalar dy = (max_y - min_y) / Scalar(res);
        Scalar dz = (max_z - min_z) / Scalar(res);

        // Splat normals into divergence field (on grid vertices)
        std::vector<Scalar> div(static_cast<std::size_t>(vr*vr*vr), 0);
        std::vector<Scalar> weight(static_cast<std::size_t>(vr*vr*vr), 0);

        for (int pi = 0; pi < n; ++pi)
        {
            Scalar px = pointCoord(pi, 0), py = pointCoord(pi, 1), pz = pointCoord(pi, 2);
            Scalar nx = normalCoord(pi, 0), ny = normalCoord(pi, 1), nz = normalCoord(pi, 2);

            // Find nearest grid vertex
            int ix = static_cast<int>(std::round((px - min_x) / dx));
            int iy = static_cast<int>(std::round((py - min_y) / dy));
            int iz = static_cast<int>(std::round((pz - min_z) / dz));
            ix = std::clamp(ix, 0, vr-1);
            iy = std::clamp(iy, 0, vr-1);
            iz = std::clamp(iz, 0, vr-1);

            // Trilinear splat to neighboring vertices
            for (int sx = -1; sx <= 1; ++sx)
                for (int sy = -1; sy <= 1; ++sy)
                    for (int sz = -1; sz <= 1; ++sz)
                    {
                        int gx = ix + sx, gy = iy + sy, gz = iz + sz;
                        if (gx < 0 || gx >= vr || gy < 0 || gy >= vr || gz < 0 || gz >= vr) continue;
                        Scalar wx = Scalar(1) - std::abs(Scalar(sx));
                        Scalar wy = Scalar(1) - std::abs(Scalar(sy));
                        Scalar wz = Scalar(1) - std::abs(Scalar(sz));
                        Scalar w = wx * wy * wz;
                        std::size_t idx = static_cast<std::size_t>(gz*vr*vr + gy*vr + gx);
                        div[idx] += w * nx + w * ny + w * nz; // Approximate: sum of normal components
                        weight[idx] += w;
                    }
        }

        // Gauss-Seidel solver for the Poisson equation
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
                        if (ix > 0)    { sum += chi[static_cast<std::size_t>(iz*vr*vr + iy*vr + ix-1)]; ++count; }
                        if (ix < vr-1) { sum += chi[static_cast<std::size_t>(iz*vr*vr + iy*vr + ix+1)]; ++count; }
                        if (iy > 0)    { sum += chi[static_cast<std::size_t>(iz*vr*vr + (iy-1)*vr + ix)]; ++count; }
                        if (iy < vr-1) { sum += chi[static_cast<std::size_t>(iz*vr*vr + (iy+1)*vr + ix)]; ++count; }
                        if (iz > 0)    { sum += chi[static_cast<std::size_t>((iz-1)*vr*vr + iy*vr + ix)]; ++count; }
                        if (iz < vr-1) { sum += chi[static_cast<std::size_t>((iz+1)*vr*vr + iy*vr + ix)]; ++count; }
                        chi[idx] = (sum - dx*dx * div[idx]) / Scalar(count);
                    }
        }

        // Extract isosurface
        auto chi_fn = [&](Scalar x, Scalar y, Scalar z) -> Scalar {
            int ix = std::clamp(static_cast<int>((x - min_x) / dx), 0, vr-1);
            int iy = std::clamp(static_cast<int>((y - min_y) / dy), 0, vr-1);
            int iz = std::clamp(static_cast<int>((z - min_z) / dz), 0, vr-1);
            return chi[static_cast<std::size_t>(iz*vr*vr + iy*vr + ix)];
        };

        MarchingCubes<Scalar> mc;
        mc.setBounds({min_x, min_y, min_z}, {max_x, max_y, max_z});
        mc.setResolution(res, res, res);
        mc.setIsoLevel(Scalar(0));

        return mc.extract(chi_fn);
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (plamatrix::Device::CPU == plamatrix::Device::CPU)
            return _cloud->points()(idx, dim);
        else
            return _cloud->points().getValue(idx, dim);
    }
    Scalar normalCoord(int idx, int dim) const
    {
        auto* n = _cloud->normals();
        return n->getValue(idx, dim);
    }

    std::shared_ptr<const PointCloudType> _cloud;
    int _depth = 6;
    int _solver_iters = 20;
};

} // namespace mesh
} // namespace plapoint
```

Note: PointCloud only supports CPU. The `if constexpr` guard is fine but we can simplify since Poisson is CPU-only anyway.

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=Poisson*`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/mesh/poisson_reconstruction.h test/unit/mesh/poisson_reconstruction_test.cpp
git commit -m "feat: add Poisson surface reconstruction"
git push origin master
```

---

### Task 7: PLY I/O

**Files:**
- Create: `include/plapoint/io/ply_io.h`
- Create: `test/unit/io/ply_io_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/io/ply_io_test.cpp
#include <gtest/gtest.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdio>
#include <fstream>
#include <string>

TEST(PlyIOTest, RoundtripASCII)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(5, 3);
    Matrix nrm(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i*2));
        pts.setValue(i, 2, Scalar(i*3));
        nrm.setValue(i, 0, 0); nrm.setValue(i, 1, 0); nrm.setValue(i, 2, 1);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(nrm));

    std::string path = "/tmp/plapoint_test_ascii.ply";
    EXPECT_NO_THROW(plapoint::io::writePly(path, cloud));

    auto loaded = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(loaded->size(), 5u);
    EXPECT_TRUE(loaded->hasNormals());
    EXPECT_FLOAT_EQ(loaded->normals()->getValue(0, 2), 1.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadOnlyPositions)
{
    using Scalar = float;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    // Write a minimal PLY with just positions
    std::string path = "/tmp/plapoint_test_minimal.ply";
    {
        std::ofstream f(path);
        f << "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        f << "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(cloud->size(), 3u);
    EXPECT_FALSE(cloud->hasNormals());
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(2, 2), 9.0f);

    std::remove(path.c_str());
}
```

- [ ] **Step 2: Verify test fails**

Run: compilation error.

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/io/ply_io.h
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plapoint {
namespace io {

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readPly(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open PLY file: " + path);

    std::string line;
    std::getline(f, line); // "ply"
    if (line != "ply") throw std::runtime_error("Not a PLY file");

    std::getline(f, line); // "format ..."
    bool binary = (line.find("binary") != std::string::npos);

    int n_verts = 0;
    std::vector<std::string> props;
    bool has_nx = false, has_ny = false, has_nz = false;

    while (std::getline(f, line))
    {
        if (line == "end_header") break;
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element" && iss >> token >> n_verts && token == "vertex") {}
        else if (token == "property")
        {
            std::string type, name;
            iss >> type >> name;
            props.push_back(name);
            if (name == "nx") has_nx = true;
            if (name == "ny") has_ny = true;
            if (name == "nz") has_nz = true;
        }
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(n_verts, 3);
    bool have_normals = has_nx && has_ny && has_nz;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(n_verts, 3);

    for (int i = 0; i < n_verts; ++i)
    {
        std::getline(f, line);
        std::istringstream iss(line);
        Scalar val;
        for (std::size_t j = 0; j < props.size(); ++j)
        {
            iss >> val;
            if (props[j] == "x") pts(i, 0) = val;
            else if (props[j] == "y") pts(i, 1) = val;
            else if (props[j] == "z") pts(i, 2) = val;
            else if (props[j] == "nx") nrm(i, 0) = val;
            else if (props[j] == "ny") nrm(i, 1) = val;
            else if (props[j] == "nz") nrm(i, 2) = val;
        }
    }

    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
    if (have_normals) cloud->setNormals(std::move(nrm));
    return cloud;
}

template <typename Scalar>
void writePly(const std::string& path,
              const std::shared_ptr<const PointCloud<Scalar, plamatrix::Device::CPU>>& cloud)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write PLY file: " + path);

    bool with_normals = cloud->hasNormals();

    f << "ply\nformat ascii 1.0\nelement vertex " << cloud->size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    if (with_normals)
        f << "property float nx\nproperty float ny\nproperty float nz\n";
    f << "end_header\n";

    for (std::size_t i = 0; i < cloud->size(); ++i)
    {
        f << cloud->points().getValue(static_cast<plamatrix::Index>(i), 0) << " "
          << cloud->points().getValue(static_cast<plamatrix::Index>(i), 1) << " "
          << cloud->points().getValue(static_cast<plamatrix::Index>(i), 2);
        if (with_normals)
            f << " " << cloud->normals()->getValue(static_cast<plamatrix::Index>(i), 0)
              << " " << cloud->normals()->getValue(static_cast<plamatrix::Index>(i), 1)
              << " " << cloud->normals()->getValue(static_cast<plamatrix::Index>(i), 2);
        f << "\n";
    }
}

} // namespace io
} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: all tests including PLY.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/io/ply_io.h test/unit/io/ply_io_test.cpp
git commit -m "feat: add PLY file I/O (ASCII read/write)"
git push origin master
```

---

### Final Verification

- [ ] **Run all tests**

```bash
cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests
```
Expected: All ~26 tests PASS.
