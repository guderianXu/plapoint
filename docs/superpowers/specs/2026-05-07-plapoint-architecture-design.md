# Plapoint Library Architecture Design

## 1. Overview
`plapoint` is a point cloud processing library built on top of the `plamatrix` math backend. It focuses on photogrammetry needs, providing GPU-accelerated algorithms for filtering, registration, feature extraction, and spatial indexing. It follows a PCL-like algorithm class design while maintaining the explicit device template paradigms (`Device::CPU`, `Device::GPU`) established by `plamatrix`.

## 2. Core Data Structures

### `plapoint::PointCloud<Scalar, Dev>`
The foundational data structure representing a point cloud.
- **Internal Storage**: Uses `plamatrix::DenseMatrix<Scalar, Dev>` for underlying data. Points are stored as an Nx3 matrix.
- **Optional Fields**: Can optionally store normals (Nx3) or intensities (Nx1) as additional dense matrices.
- **Device Management**: Implements `.toGpu()` and `.toCpu()` methods that return a new point cloud on the respective device, matching the semantics of `plamatrix`.

## 3. Spatial Indexing

### `plapoint::search::KdTree<Scalar, Dev>`
Provides nearest-neighbor search capabilities, which are fundamental to algorithms like normal estimation and ICP.
- **Input**: Takes a `PointCloud` as input.
- **Operations**: Supports $K$-nearest neighbor search and radius search.
- **Device Specifics**:
    - For `Device::CPU`, standard tree construction and search.
    - For `Device::GPU`, uses optimized CUDA kernels for batched construction and querying.

## 4. Algorithm Classes

Following the PCL design philosophy, algorithms are encapsulated in classes. This allows for configuring parameters and optionally injecting shared state (like a pre-computed KD-Tree) before execution.

### Filters
- **Base Concept**: Classes that take an input cloud and produce a downsampled or cleaned output cloud.
- **`plapoint::VoxelGrid<Scalar, Dev>`**: Downsamples the cloud using a 3D voxel grid.
    - Methods: `.setLeafSize(x, y, z)`, `.setInputCloud(cloud)`, `.filter(output_cloud)`.
- **`plapoint::StatisticalOutlierRemoval<Scalar, Dev>`**: Removes noisy points based on local neighborhood statistics.

### Registration
- **`plapoint::IterativeClosestPoint<Scalar, Dev>`** (ICP)
    - Methods: `.setInputSource(source)`, `.setInputTarget(target)`, `.setMaxIterations(n)`, `.align(output_cloud)`.
    - Retrieves the final 4x4 transformation matrix (using `plamatrix::rigidTransform` formats).
    - Can optionally accept a pre-built KD-Tree for the target cloud to avoid recomputing it every iteration.

### Features
- **`plapoint::NormalEstimation<Scalar, Dev>`**
    - Estimates surface normals using PCA on the local neighborhood (leveraging `plamatrix::svd` or `plamatrix::eigh` on covariance matrices).
    - Requires a spatial search method (KD-Tree) to find neighbors.

## 5. Integration with PlaMatrix
- `plapoint` strictly uses `plamatrix` for all dense matrix operations, decompositions, and basic 3D transformations.
- The repository structure will place `plapoint` headers in `include/plapoint/` and sources in `src/`.
- The build system will link `plapoint` against `plamatrix`.