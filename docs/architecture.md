# PlaPoint Architecture

Public point-type include entry points under `include/plapoint/` forward to the
cloud, search, filter, feature, registration, and I/O implementations. The supported
surface follows PCL 1.15.1 signatures and semantics; unsupported PCL modules remain outside
the library boundary.

## Ownership and devices

The root namespace exposes the common point records (`PointXYZ`, `PointXYZI`,
`PointXYZRGB`, `PointXYZRGBA`, `Normal`, `PointNormal`) and `PointCloud<PointT>` with
public `points`, cloud shape, metadata, Eigen maps, and point indexing. Registered custom
records use the same compile-time field traits. `PointXYZd` keeps planetary XYZ coordinates
in double precision.
These point-type APIs materialize a PlaMatrix Eigen-style coordinate matrix at the
boundary where needed and currently run search, voxel filtering, normal estimation, and ICP on
CPU. `GeometryCloud<Scalar>` owns CPU geometry, attributes, topology, and material strings.
The internal scalar/device classes use Eigen-style PlaMatrix host matrices and explicit
resident CUDA storage. The point-type ICP returns PCL mean squared distance from
`getFitnessScore()`; its PlaPoint inlier ratio has a separate getter.
`search::Search`, `Filter<PointT>`, `FilterIndices<PointT>`,
`Feature<PointInT, PointOutT>`, and `Registration<PointSource, PointTarget, Scalar>`
provide the polymorphic point-type operations. `PCLPointCloud2` and `PolygonMesh` form the
dynamic I/O boundary for PCD/PLY and registration rejectors. PlaPoint keeps its own
implementation files and has no PCL build dependency.

`internal::DeviceCloud<Scalar, Device>` owns column-major
`plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>` host storage
or `plamatrix::internal::ResidentMatrix<Scalar>` CUDA storage, plus point-aligned attributes.
A GPU cloud retains its shared `ExecutionContext`; resident results that can outlive their
producer retain ownership too. The ordinary constructors and CPU-to-GPU cloud transfer create
the context internally. Custom device code can explicitly provide one for reuse.
GPU backend algorithms accept GPU clouds directly and return resident matrices or clouds; public
`GeometryCloud` convenience APIs perform transfers and return CPU-owned results. Context-sensitive buffers are
released before the owning context, including during cross-context move assignment.
Synchronous operations such as `align()` complete their output writes before returning.
For `Async` calls, synchronize the supplied producer stream before downloading a resident
result or accessing it on a different stream; `toHostMatrix()` waits only for its own context.

Same-device copy setters delegate to PlaMatrix bulk copies. CPU coefficients use Eigen-style
`operator()`/`coeff()`, while custom CUDA kernels borrow checked `MatrixView`/`ConstMatrixView`
from resident storage. View-based statistics, grouping, indexing, and eigensolve calls reuse
PlaMatrix kernels without constructing legacy matrix containers. Move setters remain the zero-copy ownership
transfer path. Large-file readers can use the public PLY/OBJ streaming visitors to keep
input memory bounded and expose cancellation/progress at chunk or record boundaries.

Scoped `editPoints()` access advances `pointsRevision()` at entry and exit and blocks cache
reuse while the editor is alive. Legacy mutable `points()` marks an untracked alias because a
retained matrix reference can write without another accessor call. CPU KdTree compares such
clouds with its exact snapshot; GPU indexes and ICP caches rebuild conservatively. Attribute-only
changes do not invalidate the point-position revision.

## Neighbor search

The point-type `KdTreeFLANN<PointT, Dist>` builds its index in `setInputCloud()` and accepts
point representations, optional subset indices, and squared-distance output. The polymorphic
`search::KdTree<PointT>` wraps that interface for feature and registration pipelines. Internal CPU
`DeviceKdTree` builds on first search unless `build()` is called. Internal GPU `DeviceKdTree`
selects one of three backends:

- `UniformGrid`: cached `GpuSpatialIndex`, deterministic bounded search, `k <= 32`.
- `BruteForce`: CUDA tiled/shared-memory search for small or unsuitable grid work.
- `CpuBruteForce`: compatibility path for `k > 32`.

The uniform grid stores finite points sorted by cell key and source index. Radius
queries inspect intersecting cells. KNN expands cell shells until the next shell cannot
improve the current kth result. Unsafe coordinate quantization, high occupancy, sparse
cell volumes, or queries requiring more than eight empty shells choose brute force.
`GpuSpatialIndex` owns its prepared point snapshot and accepts caller-owned query workspaces.
GPU ICP keeps its specialized target grid in the ICP workspace and can populate it explicitly
with `prepareGpuTargetSpatialIndex()` before latency-sensitive alignment.

PlaMatrix supplies the reusable CUDA primitives for three-component signed cell keys: stable
lexicographic sorting, run-length encoding, and offset scans. PlaPoint supplies the spatial
meaning of those keys and keeps voxel centroid/remapping logic and ICP lookup, neighborhood,
correspondence, and pose logic inside the point-cloud layer.

## Feature and filter pipelines

GPU normal estimation performs indexed KNN, covariance accumulation, PlaMatrix batched
symmetric eigensolve, normalization, and deterministic sign selection on device.
Normal smoothing and viewpoint orientation also use device kernels.

CPU normal estimation accumulates a fixed 3x3 covariance from the neighbor list and uses
`SelfAdjointEigenSolver<Matrix<Scalar, 3, 3>>` to select the smallest eigenvector. CPU
point-to-point ICP keeps the general fixed-size `svd3x3()` numerical primitive for its
cross-covariance; point matching and pose-update semantics remain in PlaPoint.

CUDA ICP white-box counters and their accessors live in dedicated test-only implementation
headers and are compiled only with `PLAPOINT_ENABLE_TESTING`. Production and benchmark-only
objects therefore contain no `ForTesting` accessors or `g_icp_*` counter storage; test code
uses the guarded `plapoint/gpu/icp_testing.h` declarations instead of duplicating them.

GPU RadiusOR consumes saturated radius counts. GPU SOR computes indexed neighbor means,
normalizes distance statistics to avoid overflow, reduces through PlaMatrix, creates a
keep mask, and stably compacts points and all point-aligned attributes. Only requested
removed-index diagnostics are copied to host.

CUDA VoxelGrid and voxel clustering quantize points in PlaPoint, group the resulting cell keys
through PlaMatrix, and compute overflow-safe centroids in point-cloud-specific kernels. Voxel
clustering also obtains finite coordinate bounds through PlaMatrix statistics before grouping.

## High-level selection

`ProcessingPolicy` centralizes tunable dispatch thresholds. CPU-owned `Auto` calls use
a point-count boundary for transfer-bound linear work and a point-count-by-neighbor
work estimate for KNN-style algorithms. An explicit accelerator request does not
silently change devices: unsupported parameters, unavailable CUDA/OpenCL, allocation
failure, or kernel failure are rethrown. `Auto` records ordinary policy choices in
`selectionReason`; only failed backend attempts set `usedFallback`/`fallbackReason`.
Attributed voxel clouds do not select the partial OpenCL implementation in Auto mode.

`ProcessingReport` records requested and actual devices, the neighbor backend, whether
a runtime fallback occurred, and its reason. A normal Auto policy choice is not a
failure fallback.

See [GPU search](gpu-search.md) for API limits and workspace lifetime contracts.

## Mesh and terrain

CUDA Marching Cubes consumes an x-fastest device scalar field, counts triangles,
exclusive-scans offsets through PlaMatrix, and emits deterministic GPU vertex/face storage.

Device HeightGrid uses explicit XY bounds for its asynchronous entry point. Aggregation,
color weighting, validity masks, fill-pass tracking, and ping-pong hole fill stay on the
producer stream. Automatic bounds belong to the synchronous convenience wrapper because
they require a host-visible reduction result.

Poisson completes only missing siblings below existing internal nodes, then locally refines
coarse leaves until face-, edge-, and corner-adjacent leaves satisfy 2:1 balance. Empty
structural leaves are excluded from the linear system. Assembly orders populated leaves and
undirected edges deterministically, anchors one row per connected graph component, and solves
the resulting SPD CSR with PlaMatrix Jacobi-PCG. The extraction field is sampled once and
reused for extrema and Marching Cubes, while face orientation builds one CPU kd-tree. Only the
sparse solve can run on GPU today; field sampling and extraction remain CPU.
`PoissonProcessingReport` separates full-chain `actualDevice` from `solverDevice` and records
balance, sampling, orientation-index, and non-convergence diagnostics.

See [GPU mesh processing](gpu-mesh.md) for stream and status contracts.
