# PlaPoint

GPU-accelerated point cloud processing library built on [PlaMatrix](https://github.com/guderianXu/plamatrix).

## Features

### Core
- **PointCloud\<Scalar, Dev\>** — Nx3 point cloud with optional normals, GPU/CPU transfer via `toGpu()`/`toCpu()`

### Spatial Indexing
- **KdTree\<Scalar, Dev\>** — 3D kd-tree with KNN search (priority queue) and radius search

### Filters
- **VoxelGrid** — centroid-based voxel downsampling
- **StatisticalOutlierRemoval** — KNN distance statistics outlier filtering
- **RadiusOutlierRemoval** — radius-based neighbor count filtering
- **UniformDownsample** — keep every Nth point
- **Filter\<Scalar, Dev\>** — abstract base class for all filters

### Features
- **NormalEstimation** — PCA-based surface normal estimation via covariance + SVD
- **NormalRefinement** — normal smoothing (KNN averaging) and viewpoint-based orientation

### Registration
- **IterativeClosestPoint** (ICP) — point-to-point ICP with SVD-based rigid transform

### Mesh
- **MarchingCubes** — isosurface extraction from implicit scalar fields
- **PoissonReconstruction** — Poisson surface reconstruction (Gauss-Seidel solver + MC extraction)

### I/O
- **PLY** — ASCII read/write with positions and optional normals

## GPU Acceleration

When `PLAPOINT_WITH_CUDA=ON` and CUDA Toolkit is available:

- **Brute-force KNN** (`src/knn_gpu.cu`) — batched K-nearest neighbor search with one CUDA block per query point, shared memory top-K reduction
- **Voxel Grid** (`src/voxel_grid_gpu.cu`) — GPU voxel hashing with atomic centroid accumulation
- Explicit template instantiations in `src/plapoint.cpp` reduce downstream compile times

## Requirements

- C++17
- CMake ≥ 3.18
- [PlaMatrix](https://github.com/guderianXu/plamatrix) (math backend)
- CUDA Toolkit (optional, for GPU kernels)
- Google Test (for tests)

## Build

```bash
# Build and install plamatrix first
cd plamatrix && mkdir build && cd build
cmake .. -DPLAMATRIX_WITH_CUDA=ON
cmake --build . -j$(nproc)
cmake --install . --prefix ../install

# Build plapoint
cd plapoint && mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/path/to/plamatrix/install
cmake --build . -j$(nproc)
./test/plapoint_tests
```

## API Overview

```cpp
#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/features/normal_refinement.h>
#include <plapoint/registration/icp.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/io/ply_io.h>

using namespace plapoint;

// Create a point cloud on CPU
PointCloud<float, plamatrix::Device::CPU> cloud(1000);
cloud.points().fill(1.0f);

// Transfer to GPU
auto gpu_cloud = cloud.toGpu();

// Build kd-tree and search
auto tree = std::make_shared<search::KdTree<float, plamatrix::Device::CPU>>();
tree->setInputCloud(std::make_shared<PointCloud<...>>(std::move(cloud)));
tree->build();
auto neighbors = tree->nearestKSearch({0, 0, 0}, 10);

// Filter
VoxelGrid<float, plamatrix::Device::CPU> vg;
vg.setInputCloud(...);
vg.setLeafSize(0.1, 0.1, 0.1);
PointCloud<float, plamatrix::Device::CPU> output;
vg.filter(output);

// Estimate normals
NormalEstimation<float, plamatrix::Device::CPU> ne;
ne.setInputCloud(...);
ne.setSearchMethod(tree);
auto normals = ne.compute();

// ICP registration
IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
icp.setInputSource(source);
icp.setInputTarget(target);
icp.align(aligned);

// Reconstruct surface
PoissonReconstruction<float> pr;
pr.setInputCloud(point_cloud_with_normals);
pr.setDepth(6);
auto [verts, faces] = pr.reconstruct();

// PLY I/O
auto cloud = io::readPly<float>("input.ply");
io::writePly("output.ply", *cloud);
```

## License

MIT
