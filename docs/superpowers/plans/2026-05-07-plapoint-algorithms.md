# Plapoint Algorithms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 core point cloud algorithms (KdTree, VoxelGrid, StatisticalOutlierRemoval, NormalEstimation, ICP) following the dependency chain from spatial indexing up to registration.

**Architecture:** CPU-first template classes `<Scalar, Dev>` matching plamatrix patterns. Each algorithm is independently testable following TDD. KdTree is the foundational spatial index; filters (VoxelGrid, SOR) inherit from `Filter<Scalar, Dev>`; NormalEstimation and ICP are standalone algorithm classes using KdTree + plamatrix decompositions.

**Tech Stack:** C++17, plamatrix (DenseMatrix, covarianceMatrix, SVD, Vec3), Google Test, CMake

---

### Task 1: KdTree — 3D Spatial Index

**Files:**
- Create: `include/plapoint/search/kdtree.h`
- Create: `test/unit/search/kdtree_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/search/kdtree_test.cpp
#include <gtest/gtest.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

class KdTreeTest : public ::testing::Test
{
protected:
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using KdTree = plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>;

    void SetUp() override
    {
        auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(6, 3);
        // 6 points: two clusters at (0,0,0) and (10,10,10)
        mat.setValue(0, 0, 0); mat.setValue(0, 1, 0); mat.setValue(0, 2, 0);
        mat.setValue(1, 0, 1); mat.setValue(1, 1, 0); mat.setValue(1, 2, 0);
        mat.setValue(2, 0, 0); mat.setValue(2, 1, 1); mat.setValue(2, 2, 0);
        mat.setValue(3, 0, 10); mat.setValue(3, 1, 10); mat.setValue(3, 2, 10);
        mat.setValue(4, 0, 11); mat.setValue(4, 1, 10); mat.setValue(4, 2, 10);
        mat.setValue(5, 0, 10); mat.setValue(5, 1, 11); mat.setValue(5, 2, 10);
        cloud = std::make_shared<Cloud>(std::move(mat));
    }

    std::shared_ptr<Cloud> cloud;
};

TEST_F(KdTreeTest, BuildAndSize)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();
    // tree built without exception
}

TEST_F(KdTreeTest, ThrowsIfNoInput)
{
    KdTree tree;
    EXPECT_THROW(tree.build(), std::runtime_error);
}

TEST_F(KdTreeTest, NearestKSearchSinglePoint)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    auto results = tree.nearestKSearch(query, 3);
    ASSERT_EQ(results.size(), 3u);
    // points 0,1,2 are closest to origin
}

TEST_F(KdTreeTest, RadiusSearch)
{
    KdTree tree;
    tree.setInputCloud(cloud);
    tree.build();

    plamatrix::Vec3<Scalar> query{0, 0, 0};
    auto results = tree.radiusSearch(query, Scalar(2.0));
    // Should find points 0,1,2 within radius 2 of origin
    ASSERT_EQ(results.size(), 3u);
}
```

- [ ] **Step 2: Verify test fails**

Run: `cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation failure (kdtree.h not found).

- [ ] **Step 3: Write minimal KdTree implementation**

```cpp
// include/plapoint/search/kdtree.h
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/ops/point_cloud.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

namespace plapoint {
namespace search {

template <typename Scalar>
struct KdTreeNode
{
    int point_idx;
    int left;
    int right;
    int split_dim;
    Scalar split_val;
};

template <typename Scalar, plamatrix::Device Dev>
class KdTree
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud)
    {
        _cloud = cloud;
    }

    void build()
    {
        if (!_cloud)
        {
            throw std::runtime_error("KdTree: input cloud not set");
        }
        _nodes.clear();
        std::vector<int> indices(static_cast<std::size_t>(_cloud->size()));
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = static_cast<int>(i);
        }
        _nodes.reserve(indices.size());
        buildRecursive(indices, 0, static_cast<int>(indices.size()) - 1, 0);
    }

    std::vector<int> nearestKSearch(const plamatrix::Vec3<Scalar>& query, int k) const
    {
        std::vector<int> result;
        if (_nodes.empty() || k <= 0) return result;

        using DistIndex = std::pair<Scalar, int>;
        auto cmp = [](const DistIndex& a, const DistIndex& b) { return a.first < b.first; };
        std::priority_queue<DistIndex, std::vector<DistIndex>, decltype(cmp)> pq(cmp);

        nearestKSearchRecursive(query, k, 0, pq);

        result.resize(pq.size());
        for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i)
        {
            result[static_cast<std::size_t>(i)] = pq.top().second;
            pq.pop();
        }
        return result;
    }

    std::vector<int> radiusSearch(const plamatrix::Vec3<Scalar>& query, Scalar radius) const
    {
        std::vector<int> result;
        if (_nodes.empty()) return result;
        radiusSearchRecursive(query, radius * radius, 0, result);
        return result;
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return _cloud->points()(idx, dim);
        }
        else
        {
            return _cloud->points().getValue(idx, dim);
        }
    }

    plamatrix::Vec3<Scalar> pointVec(int idx) const
    {
        return {pointCoord(idx, 0), pointCoord(idx, 1), pointCoord(idx, 2)};
    }

    Scalar distSq(const plamatrix::Vec3<Scalar>& a, const plamatrix::Vec3<Scalar>& b) const
    {
        Scalar dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    int buildRecursive(std::vector<int>& indices, int start, int end, int depth)
    {
        if (start > end) return -1;

        int dim = depth % 3;
        int mid = start + (end - start) / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end + 1,
                         [&](int a, int b) { return pointCoord(a, dim) < pointCoord(b, dim); });

        int node_idx = static_cast<int>(_nodes.size());
        KdTreeNode<Scalar> node{};
        node.point_idx = indices[static_cast<std::size_t>(mid)];
        node.split_dim = dim;
        node.split_val = pointCoord(node.point_idx, dim);
        _nodes.push_back(node);

        _nodes[static_cast<std::size_t>(node_idx)].left  = buildRecursive(indices, start, mid - 1, depth + 1);
        _nodes[static_cast<std::size_t>(node_idx)].right = buildRecursive(indices, mid + 1, end, depth + 1);
        return node_idx;
    }

    void nearestKSearchRecursive(const plamatrix::Vec3<Scalar>& query, int k,
                                 int node_idx,
                                 std::priority_queue<std::pair<Scalar, int>,
                                     std::vector<std::pair<Scalar, int>>,
                                     decltype([](const std::pair<Scalar, int>& a,
                                                 const std::pair<Scalar, int>& b) { return a.first < b.first; })>& pq) const
    {
        if (node_idx < 0) return;
        const auto& node = _nodes[static_cast<std::size_t>(node_idx)];

        auto pt = pointVec(node.point_idx);
        Scalar d = distSq(query, pt);

        if (static_cast<int>(pq.size()) < k)
        {
            pq.push({d, node.point_idx});
        }
        else if (d < pq.top().first)
        {
            pq.pop();
            pq.push({d, node.point_idx});
        }

        int dim = node.split_dim;
        Scalar diff = (dim == 0 ? query.x : (dim == 1 ? query.y : query.z)) - node.split_val;
        int near = diff <= 0 ? node.left : node.right;
        int far  = diff <= 0 ? node.right : node.left;

        nearestKSearchRecursive(query, k, near, pq);

        Scalar max_dist = pq.empty() ? std::numeric_limits<Scalar>::max() : pq.top().first;
        if (diff * diff < max_dist || static_cast<int>(pq.size()) < k)
        {
            nearestKSearchRecursive(query, k, far, pq);
        }
    }

    void radiusSearchRecursive(const plamatrix::Vec3<Scalar>& query, Scalar r2,
                               int node_idx, std::vector<int>& result) const
    {
        if (node_idx < 0) return;
        const auto& node = _nodes[static_cast<std::size_t>(node_idx)];

        auto pt = pointVec(node.point_idx);
        Scalar d = distSq(query, pt);
        if (d <= r2)
        {
            result.push_back(node.point_idx);
        }

        int dim = node.split_dim;
        Scalar diff = (dim == 0 ? query.x : (dim == 1 ? query.y : query.z)) - node.split_val;

        if (diff <= 0)
        {
            radiusSearchRecursive(query, r2, node.left, result);
            if (diff * diff <= r2) radiusSearchRecursive(query, r2, node.right, result);
        }
        else
        {
            radiusSearchRecursive(query, r2, node.right, result);
            if (diff * diff <= r2) radiusSearchRecursive(query, r2, node.left, result);
        }
    }

    std::shared_ptr<const PointCloudType> _cloud;
    std::vector<KdTreeNode<Scalar>> _nodes;
};

} // namespace search
} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=KdTreeTest.*`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/search/kdtree.h test/unit/search/kdtree_test.cpp
git commit -m "feat: add 3D KdTree with KNN and radius search"
git push origin master
```

---

### Task 2: VoxelGrid Downsampling Filter

**Files:**
- Create: `include/plapoint/filters/voxel_grid.h`
- Create: `test/unit/filters/voxel_grid_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/filters/voxel_grid_test.cpp
#include <gtest/gtest.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(VoxelGridTest, DownsamplesUniformGrid)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // Create 8 points forming a 2x2x2 cube (corners of a unit cube)
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(8, 3);
    int idx = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
            {
                mat.setValue(idx, 0, Scalar(x));
                mat.setValue(idx, 1, Scalar(y));
                mat.setValue(idx, 2, Scalar(z));
                ++idx;
            }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(2.0), Scalar(2.0), Scalar(2.0));

    Cloud output;
    vg.filter(output);
    // All 8 points fall into one voxel => 1 centroid
    EXPECT_EQ(output.size(), 1u);
}

TEST(VoxelGridTest, PreservesSinglePoint)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    mat.setValue(0, 0, 1.0f);
    mat.setValue(0, 1, 2.0f);
    mat.setValue(0, 2, 3.0f);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(0.1), Scalar(0.1), Scalar(0.1));

    Cloud output;
    vg.filter(output);
    EXPECT_EQ(output.size(), 1u);
}

TEST(VoxelGridTest, ThrowsOnZeroLeafSize)
{
    plapoint::VoxelGrid<float, plamatrix::Device::CPU> vg;
    EXPECT_THROW(vg.setLeafSize(0, 1, 1), std::invalid_argument);
}
```

- [ ] **Step 2: Verify test fails (compilation error)**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation failure.

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/filters/voxel_grid.h
#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class VoxelGrid : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setLeafSize(Scalar lx, Scalar ly, Scalar lz)
    {
        if (lx <= 0 || ly <= 0 || lz <= 0)
        {
            throw std::invalid_argument("VoxelGrid: leaf size must be positive");
        }
        _leaf_x = lx;
        _leaf_y = ly;
        _leaf_z = lz;
    }

protected:
    void applyFilter(PointCloudType& output) override
    {
        if (!this->_input) return;

        using Key = std::tuple<int, int, int>;
        struct Accum { Scalar sum_x = 0, sum_y = 0, sum_z = 0; int count = 0; };
        std::map<Key, Accum> voxels;

        for (std::size_t i = 0; i < this->_input->size(); ++i)
        {
            Key key{
                static_cast<int>(std::floor(pointCoord(i, 0) / _leaf_x)),
                static_cast<int>(std::floor(pointCoord(i, 1) / _leaf_y)),
                static_cast<int>(std::floor(pointCoord(i, 2) / _leaf_z))
            };
            auto& acc = voxels[key];
            acc.sum_x += pointCoord(i, 0);
            acc.sum_y += pointCoord(i, 1);
            acc.sum_z += pointCoord(i, 2);
            acc.count += 1;
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(voxels.size(), 3);
        int out_idx = 0;
        for (const auto& [key, acc] : voxels)
        {
            pts(out_idx, 0) = acc.sum_x / static_cast<Scalar>(acc.count);
            pts(out_idx, 1) = acc.sum_y / static_cast<Scalar>(acc.count);
            pts(out_idx, 2) = acc.sum_z / static_cast<Scalar>(acc.count);
            ++out_idx;
        }
        output = PointCloudType(std::move(pts));
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return this->_input->points()(idx, dim);
        }
        else
        {
            return this->_input->points().getValue(idx, dim);
        }
    }

    Scalar _leaf_x = 1;
    Scalar _leaf_y = 1;
    Scalar _leaf_z = 1;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=VoxelGridTest.*`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/filters/voxel_grid.h test/unit/filters/voxel_grid_test.cpp
git commit -m "feat: add VoxelGrid downsampling filter"
git push origin master
```

---

### Task 3: StatisticalOutlierRemoval Filter

**Files:**
- Create: `include/plapoint/filters/statistical_outlier_removal.h`
- Create: `test/unit/filters/statistical_outlier_removal_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/filters/statistical_outlier_removal_test.cpp
#include <gtest/gtest.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(SORTest, RemovesSingleOutlier)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // 10 points clustered at origin + 1 far outlier
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(11, 3);
    for (int i = 0; i < 10; ++i)
    {
        mat.setValue(i, 0, Scalar(i) * 0.01f);
        mat.setValue(i, 1, 0);
        mat.setValue(i, 2, 0);
    }
    mat.setValue(10, 0, 100); mat.setValue(10, 1, 0); mat.setValue(10, 2, 0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    sor.setSearchMethod(tree);
    sor.setMeanK(4);
    sor.setStddevMulThresh(Scalar(1.0));

    Cloud output;
    sor.filter(output);
    EXPECT_EQ(output.size(), 10u);
}

TEST(SORTest, ThrowsIfNoSearchMethod)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(5, 3);
    mat.fill(0);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> sor;
    sor.setInputCloud(cloud);
    // No search method set

    Cloud output;
    EXPECT_THROW(sor.filter(output), std::runtime_error);
}
```

- [ ] **Step 2: Verify test fails**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation failure.

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/filters/statistical_outlier_removal.h
#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class StatisticalOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setMeanK(int k) { _mean_k = k; }

    void setStddevMulThresh(Scalar m) { _stddev_mul = m; }

    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree)
    {
        _tree = tree;
    }

protected:
    void applyFilter(PointCloudType& output) override
    {
        if (!_tree)
        {
            throw std::runtime_error("StatisticalOutlierRemoval: search method not set");
        }

        std::size_t n = this->_input->size();
        std::vector<Scalar> mean_dists(n, 0);

        for (std::size_t i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(static_cast<int>(i));
            auto neighbors = _tree->nearestKSearch(pt, _mean_k + 1);
            Scalar sum = 0;
            int count = 0;
            for (int nb : neighbors)
            {
                if (nb != static_cast<int>(i))
                {
                    auto pt_nb = pointVec(nb);
                    Scalar dx = pt.x - pt_nb.x, dy = pt.y - pt_nb.y, dz = pt.z - pt_nb.z;
                    sum += std::sqrt(dx * dx + dy * dy + dz * dz);
                    ++count;
                }
            }
            mean_dists[i] = (count > 0) ? sum / static_cast<Scalar>(count) : 0;
        }

        Scalar global_mean = 0;
        for (auto d : mean_dists) global_mean += d;
        global_mean /= static_cast<Scalar>(n);

        Scalar global_var = 0;
        for (auto d : mean_dists) { Scalar diff = d - global_mean; global_var += diff * diff; }
        global_var /= static_cast<Scalar>(n);
        Scalar global_stddev = std::sqrt(global_var);

        Scalar threshold = global_mean + _stddev_mul * global_stddev;

        std::vector<int> inliers;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (mean_dists[i] <= threshold)
            {
                inliers.push_back(static_cast<int>(i));
            }
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(inliers.size(), 3);
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
        {
            return this->_input->points()(idx, dim);
        }
        else
        {
            return this->_input->points().getValue(idx, dim);
        }
    }

    plamatrix::Vec3<Scalar> pointVec(int idx) const
    {
        return {pointCoord(idx, 0), pointCoord(idx, 1), pointCoord(idx, 2)};
    }

    int _mean_k = 8;
    Scalar _stddev_mul = 1;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=SORTest.*`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/filters/statistical_outlier_removal.h test/unit/filters/statistical_outlier_removal_test.cpp
git commit -m "feat: add StatisticalOutlierRemoval filter"
git push origin master
```

---

### Task 4: NormalEstimation Feature

**Files:**
- Create: `include/plapoint/features/normal_estimation.h`
- Create: `test/unit/features/normal_estimation_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/features/normal_estimation_test.cpp
#include <gtest/gtest.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(NormalEstimationTest, PlaneNormals)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // Points on the XY plane (z=0) => normals should be (0,0,1) or (0,0,-1)
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(9, 3);
    int idx = 0;
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
        {
            mat.setValue(idx, 0, Scalar(x));
            mat.setValue(idx, 1, Scalar(y));
            mat.setValue(idx, 2, 0);
            ++idx;
        }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    plapoint::NormalEstimation<Scalar, plamatrix::Device::CPU> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(8);

    auto normals = ne.compute();
    EXPECT_EQ(normals.rows(), 9);
    EXPECT_EQ(normals.cols(), 3);

    // Center point normal should be approximately (0,0,1) or (0,0,-1)
    Scalar z = normals.getValue(4, 2);
    EXPECT_GT(std::abs(z), Scalar(0.9));
}

TEST(NormalEstimationTest, ThrowsIfNoInput)
{
    plapoint::NormalEstimation<float, plamatrix::Device::CPU> ne;
    EXPECT_THROW(ne.compute(), std::runtime_error);
}
```

- [ ] **Step 2: Verify test fails**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation failure.

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/features/normal_estimation.h
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <memory>
#include <stdexcept>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class NormalEstimation
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud)
    {
        _cloud = cloud;
    }

    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree)
    {
        _tree = tree;
    }

    void setKSearch(int k) { _k = k; }

    plamatrix::DenseMatrix<Scalar, Dev> compute() const
    {
        if (!_cloud) throw std::runtime_error("NormalEstimation: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalEstimation: search method not set");

        int n = static_cast<int>(_cloud->size());
        plamatrix::DenseMatrix<Scalar, Dev> normals(n, 3);

        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(i);
            auto neighbors = _tree->nearestKSearch(pt, _k);
            if (neighbors.size() < 3) continue; // need at least 3 points for a plane

            // Build neighbor matrix
            int nn = static_cast<int>(neighbors.size());
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nb(nn, 3);
            for (int j = 0; j < nn; ++j)
            {
                auto npt = pointVec(neighbors[static_cast<std::size_t>(j)]);
                nb(j, 0) = npt.x;
                nb(j, 1) = npt.y;
                nb(j, 2) = npt.z;
            }

            auto cov = plamatrix::covarianceMatrix(nb);
            auto [U, S, Vt] = plamatrix::svd(cov);

            // Smallest singular value => last column of V = last row of Vt
            plamatrix::Vec3<Scalar> normal{Vt.getValue(2, 0), Vt.getValue(2, 1), Vt.getValue(2, 2)};

            normals.setValue(i, 0, normal.x);
            normals.setValue(i, 1, normal.y);
            normals.setValue(i, 2, normal.z);
        }
        return normals;
    }

private:
    plamatrix::Vec3<Scalar> pointVec(int idx) const
    {
        return {
            pointCoord(idx, 0),
            pointCoord(idx, 1),
            pointCoord(idx, 2)
        };
    }

    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return _cloud->points()(idx, dim);
        }
        else
        {
            return _cloud->points().getValue(idx, dim);
        }
    }

    std::shared_ptr<const PointCloudType> _cloud;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
    int _k = 10;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=NormalEstimationTest.*`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/features/normal_estimation.h test/unit/features/normal_estimation_test.cpp
git commit -m "feat: add NormalEstimation using PCA on local neighborhoods"
git push origin master
```

---

### Task 5: IterativeClosestPoint Registration

**Files:**
- Create: `include/plapoint/registration/icp.h`
- Create: `test/unit/registration/icp_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/registration/icp_test.cpp
#include <gtest/gtest.h>
#include <plapoint/registration/icp.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(ICPTest, IdentityAlignment)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // Same cloud as source and target => identity transform
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(10, 3);
    for (int i = 0; i < 10; ++i)
    {
        mat.setValue(i, 0, Scalar(i));
        mat.setValue(i, 1, Scalar(i % 3));
        mat.setValue(i, 2, Scalar(i % 2));
    }
    auto source = std::make_shared<Cloud>(mat);  // copy source
    // Target: same points in a separate matrix
    auto tgt_mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(10, 3);
    for (int i = 0; i < 10; ++i)
    {
        tgt_mat.setValue(i, 0, Scalar(i));
        tgt_mat.setValue(i, 1, Scalar(i % 3));
        tgt_mat.setValue(i, 2, Scalar(i % 2));
    }
    auto target = std::make_shared<Cloud>(std::move(tgt_mat));

    plapoint::IterativeClosestPoint<Scalar, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(10);

    Cloud output;
    icp.align(output);

    // Check convergence
    EXPECT_TRUE(icp.hasConverged());

    // Final transform should be near-identity
    auto T = icp.getFinalTransformation();
    EXPECT_NEAR(T.getValue(0, 0), Scalar(1), Scalar(1e-3));
    EXPECT_NEAR(T.getValue(0, 3), Scalar(0), Scalar(1e-3));
}

TEST(ICPTest, ThrowsIfNoInput)
{
    plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
    plapoint::PointCloud<float, plamatrix::Device::CPU> output;
    EXPECT_THROW(icp.align(output), std::runtime_error);
}
```

- [ ] **Step 2: Verify test fails**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc)`
Expected: Compilation failure.

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/registration/icp.h
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class IterativeClosestPoint
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;
    using Matrix4 = plamatrix::DenseMatrix<Scalar, Dev>;

    void setInputSource(const std::shared_ptr<const PointCloudType>& cloud) { _source = cloud; }
    void setInputTarget(const std::shared_ptr<const PointCloudType>& cloud) { _target = cloud; }
    void setMaxIterations(int n) { _max_iter = n; }
    void setTransformationEpsilon(Scalar eps) { _eps = eps; }

    void align(PointCloudType& output)
    {
        if (!_source) throw std::runtime_error("ICP: source cloud not set");
        if (!_target) throw std::runtime_error("ICP: target cloud not set");

        // Build KD-tree on target
        auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
        tree->setInputCloud(_target);
        tree->build();

        // Initialize transform to identity
        Matrix4 T_accum = identity4x4();
        int n = static_cast<int>(_source->size());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> src_cpu = toCpuCopy(_source->points());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> tgt_cpu = toCpuCopy(_target->points());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cur = src_cpu; // Nx3

        _converged = false;

        for (int iter = 0; iter < _max_iter; ++iter)
        {
            // Find correspondences
            std::vector<int> corr(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                plamatrix::Vec3<Scalar> pt{cur(i, 0), cur(i, 1), cur(i, 2)};
                auto nn = tree->nearestKSearch(pt, 1);
                corr[static_cast<std::size_t>(i)] = nn.empty() ? 0 : nn[0];
            }

            // Compute centroids
            plamatrix::Vec3<Scalar> src_centroid{0, 0, 0};
            plamatrix::Vec3<Scalar> tgt_centroid{0, 0, 0};
            for (int i = 0; i < n; ++i)
            {
                int j = corr[static_cast<std::size_t>(i)];
                src_centroid.x += cur(i, 0); src_centroid.y += cur(i, 1); src_centroid.z += cur(i, 2);
                tgt_centroid.x += tgt_cpu(j, 0); tgt_centroid.y += tgt_cpu(j, 1); tgt_centroid.z += tgt_cpu(j, 2);
            }
            src_centroid.x /= Scalar(n); src_centroid.y /= Scalar(n); src_centroid.z /= Scalar(n);
            tgt_centroid.x /= Scalar(n); tgt_centroid.y /= Scalar(n); tgt_centroid.z /= Scalar(n);

            // Build cross-covariance
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> H(3, 3);
            H.fill(0);
            for (int i = 0; i < n; ++i)
            {
                int j = corr[static_cast<std::size_t>(i)];
                Scalar sx = cur(i, 0) - src_centroid.x;
                Scalar sy = cur(i, 1) - src_centroid.y;
                Scalar sz = cur(i, 2) - src_centroid.z;
                Scalar tx = tgt_cpu(j, 0) - tgt_centroid.x;
                Scalar ty = tgt_cpu(j, 1) - tgt_centroid.y;
                Scalar tz = tgt_cpu(j, 2) - tgt_centroid.z;
                H(0, 0) += sx * tx; H(0, 1) += sx * ty; H(0, 2) += sx * tz;
                H(1, 0) += sy * tx; H(1, 1) += sy * ty; H(1, 2) += sy * tz;
                H(2, 0) += sz * tx; H(2, 1) += sz * ty; H(2, 2) += sz * tz;
            }

            auto [U, S, Vt] = plamatrix::svd(H);
            // R = V * U^T
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> R(3, 3);
            for (int ri = 0; ri < 3; ++ri)
                for (int rj = 0; rj < 3; ++rj)
                {
                    Scalar sum = 0;
                    for (int rk = 0; rk < 3; ++rk)
                        sum += Vt.getValue(rk, ri) * U.getValue(rk, rj);
                    R(ri, rj) = sum;
                }

            // Ensure det(R) = 1 (handle reflection)
            Scalar det = R(0,0)*(R(1,1)*R(2,2)-R(1,2)*R(2,1))
                        -R(0,1)*(R(1,0)*R(2,2)-R(1,2)*R(2,0))
                        +R(0,2)*(R(1,0)*R(2,1)-R(1,1)*R(2,0));
            if (det < 0)
            {
                // Flip last column of V
                for (int ri = 0; ri < 3; ++ri)
                    for (int rj = 0; rj < 3; ++rj)
                    {
                        Scalar sum = 0;
                        for (int rk = 0; rk < 3; ++rk)
                        {
                            Scalar vk = (rk == 2) ? -Vt.getValue(rk, ri) : Vt.getValue(rk, ri);
                            sum += vk * U.getValue(rk, rj);
                        }
                        R(ri, rj) = sum;
                    }
            }

            plamatrix::Vec3<Scalar> t{
                tgt_centroid.x - (R(0,0)*src_centroid.x + R(0,1)*src_centroid.y + R(0,2)*src_centroid.z),
                tgt_centroid.y - (R(1,0)*src_centroid.x + R(1,1)*src_centroid.y + R(1,2)*src_centroid.z),
                tgt_centroid.z - (R(2,0)*src_centroid.x + R(2,1)*src_centroid.y + R(2,2)*src_centroid.z)
            };

            Matrix4 T_step = identity4x4();
            T_step.setValue(0, 0, R(0,0)); T_step.setValue(0, 1, R(0,1)); T_step.setValue(0, 2, R(0,2)); T_step.setValue(0, 3, t.x);
            T_step.setValue(1, 0, R(1,0)); T_step.setValue(1, 1, R(1,1)); T_step.setValue(1, 2, R(1,2)); T_step.setValue(1, 3, t.y);
            T_step.setValue(2, 0, R(2,0)); T_step.setValue(2, 1, R(2,1)); T_step.setValue(2, 2, R(2,2)); T_step.setValue(2, 3, t.z);

            T_accum = multiply4x4(T_step, T_accum);

            // Apply transform to source for next iteration
            cur = plamatrix::transformPoints(T_step, cur);

            // Check convergence
            Scalar delta = std::abs(R(0,0)-1) + std::abs(R(1,1)-1) + std::abs(R(2,2)-1)
                         + std::abs(R(0,1)) + std::abs(R(0,2)) + std::abs(R(1,0))
                         + std::abs(R(1,2)) + std::abs(R(2,0)) + std::abs(R(2,1))
                         + std::abs(t.x) + std::abs(t.y) + std::abs(t.z);
            if (delta < _eps)
            {
                _converged = true;
                break;
            }
        }

        _final_T = T_accum;
        output = PointCloudType(plamatrix::transformPoints(T_accum, src_cpu));
    }

    Matrix4 getFinalTransformation() const { return _final_T; }
    bool hasConverged() const { return _converged; }

private:
    static Matrix4 identity4x4()
    {
        Matrix4 I(4, 4);
        I.fill(0);
        I.setValue(0, 0, 1); I.setValue(1, 1, 1); I.setValue(2, 2, 1); I.setValue(3, 3, 1);
        return I;
    }

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> toCpuCopy(
        const plamatrix::DenseMatrix<Scalar, Dev>& m)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> copy(m.rows(), m.cols());
            for (plamatrix::Index i = 0; i < m.rows(); ++i)
                for (plamatrix::Index j = 0; j < m.cols(); ++j)
                    copy(i, j) = m(i, j);
            return copy;
        }
        else
        {
            return m.toCpu();
        }
    }

    static Matrix4 multiply4x4(const Matrix4& A, const Matrix4& B)
    {
        Matrix4 C(4, 4);
        C.fill(0);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    C.setValue(i, j, C.getValue(i, j) + A.getValue(i, k) * B.getValue(k, j));
        return C;
    }

    std::shared_ptr<const PointCloudType> _source;
    std::shared_ptr<const PointCloudType> _target;
    int _max_iter = 50;
    Scalar _eps = Scalar(1e-6);
    Matrix4 _final_T;
    bool _converged = false;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**

Run: `cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=ICPTest.*`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/plapoint/registration/icp.h test/unit/registration/icp_test.cpp
git commit -m "feat: add IterativeClosestPoint registration"
git push origin master
```

---

### Final Verification

- [ ] **Run all tests**

Run: `cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/install && cmake --build . -j$(nproc) && ./test/plapoint_tests`
Expected: All tests from all 5 tasks PASS (4 + 4 + 3 + 2 + 2 + 2 = 17 tests).
