# Plapoint Algorithms Design

## 1. Overview

This is Phase 2 of `plapoint`, implementing core point cloud algorithms on top of Phase 1's `PointCloud<Scalar, Dev>` and `Filter<Scalar, Dev>` base classes. All modules follow the template `<Scalar, Dev>` pattern from `plamatrix`. CPU-first strategy; GPU kernels are deferred.

## 2. KdTree (Spatial Indexing)

### `plapoint::search::KdTree<Scalar, Dev>`

CPU 3D kd-tree providing KNN and radius search.

```cpp
template <typename Scalar, Dev Device>
class KdTree {
public:
    using PointCloudType = PointCloud<Scalar, Device>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud);
    void build();                                              // throws if no input

    // K-nearest neighbors. Returns indices sorted by distance.
    std::vector<std::vector<int>> nearestKSearch(int k) const;
    std::vector<int> nearestKSearch(const Vec3<Scalar>& point, int k) const;

    // Radius search. Returns indices within radius.
    std::vector<std::vector<int>> radiusSearch(Scalar radius) const;
    std::vector<int> radiusSearch(const Vec3<Scalar>& point, Scalar radius) const;
};
```

**Internal**: Recursive median-split 3D tree on CPU. Nodes store point index and split axis. Leaf size: 16.

**Error handling**: `build()` without input → `std::runtime_error`. Empty k/radius → empty result.

## 3. VoxelGrid (Downsampling Filter)

### `plapoint::VoxelGrid<Scalar, Dev>`

Downsamples cloud using 3D voxel grid. Each voxel cell is approximated by the centroid of all points inside it.

```cpp
template <typename Scalar, Dev Device>
class VoxelGrid : public Filter<Scalar, Device> {
public:
    void setLeafSize(Scalar lx, Scalar ly, Scalar lz);
    // filter(output) inherited from Filter — call after setInputCloud

protected:
    void applyFilter(PointCloudType& output) override;
};
```

**Algorithm**: Hash points to voxel indices, accumulate centroids per voxel, output centroid for each occupied voxel.

**Error handling**: Zero/negative leaf size → `std::invalid_argument`.

## 4. StatisticalOutlierRemoval (Denoising Filter)

### `plapoint::StatisticalOutlierRemoval<Scalar, Dev>`

Removes points whose mean distance to their K neighbors exceeds a threshold.

```cpp
template <typename Scalar, Dev Device>
class StatisticalOutlierRemoval : public Filter<Scalar, Device> {
public:
    void setMeanK(int k);                  // neighbors to analyze (default 8)
    void setStddevMulThresh(Scalar m);     // stddev multiplier (default 1.0)
    // filter(output) inherited

protected:
    void applyFilter(PointCloudType& output) override;
};
```

**Algorithm**: KNN via KdTree → mean distances → threshold = mean + stddevMulThresh * stddev → keep inliers.

**Dependency**: Requires `plapoint::search::KdTree`.

## 5. NormalEstimation (Feature Estimation)

### `plapoint::NormalEstimation<Scalar, Dev>`

Estimates surface normals per point using PCA on local neighborhoods.

```cpp
template <typename Scalar, Dev Device>
class NormalEstimation {
public:
    using PointCloudType = PointCloud<Scalar, Device>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud);
    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Device>> tree);
    void setKSearch(int k);

    // Output: Nx3 matrix of normal vectors (one per input point)
    DenseMatrix<Scalar, Device> compute() const;
};
```

**Algorithm**: For each point, compute covariance of K neighbors, SVD/eigh to get smallest eigenvector as normal.

**Dependency**: Requires `KdTree` + `plamatrix::covarianceMatrix` + SVD.

## 6. IterativeClosestPoint (Registration)

### `plapoint::IterativeClosestPoint<Scalar, Dev>`

Aligns source cloud to target using ICP.

```cpp
template <typename Scalar, Dev Device>
class IterativeClosestPoint {
public:
    using PointCloudType = PointCloud<Scalar, Device>;
    using Matrix4 = DenseMatrix<Scalar, Device>;  // 4x4

    void setInputSource(const std::shared_ptr<const PointCloudType>& cloud);
    void setInputTarget(const std::shared_ptr<const PointCloudType>& cloud);
    void setMaxIterations(int n);
    void setTransformationEpsilon(Scalar eps);

    void align(PointCloudType& output);
    Matrix4 getFinalTransformation() const;
    bool hasConverged() const;
};
```

**Algorithm**: Standard point-to-point ICP loop — KdTree on target → nearest neighbor correspondences → SVD for rigid transform → update source → repeat until convergence.

**Dependency**: Requires `KdTree` + `plamatrix::rigidTransform` + SVD.

## 7. Testing Strategy

Each module gets TDD treatment:
- **KdTree**: build, 1-NN on trivial set, radius search with known distances
- **VoxelGrid**: uniform grid (all leaf=1), empty input, single-point
- **StatisticalOutlierRemoval**: clean cluster + 1 outlier, pure noise
- **NormalEstimation**: plane normals (known), sphere normals (point outward)
- **ICP**: identity alignment (source==target), known translation, convergence flag

All tests `float/Device::CPU` only; no GPU tests yet.

## 8. File Layout

```
include/plapoint/
  search/kdtree.h
  filters/voxel_grid.h
  filters/statistical_outlier_removal.h
  features/normal_estimation.h
  registration/icp.h

test/unit/
  search/kdtree_test.cpp
  filters/voxel_grid_test.cpp
  filters/statistical_outlier_removal_test.cpp
  features/normal_estimation_test.cpp
  registration/icp_test.cpp
```
