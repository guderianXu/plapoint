# PlaPoint Architecture

## Ownership and devices

`PointCloud<Scalar, Device>` owns PlaMatrix column-major point storage and optional
point-aligned attributes. GPU algorithms accept GPU clouds directly and return GPU
matrices or clouds; CPU-owned convenience APIs perform explicit transfers and return
CPU-owned results.

Same-device copy setters delegate to PlaMatrix bulk copies rather than issuing one
`getValue()`/`setValue()` pair per scalar. Move setters remain the zero-copy ownership
transfer path. Large-file readers can use the public PLY/OBJ streaming visitors to keep
input memory bounded and expose cancellation/progress at chunk or record boundaries.

Mutable point access advances `pointsRevision()` and marks untracked aliases so cached
spatial structures cannot silently reuse stale coordinates. Attribute-only changes do
not invalidate the point-position revision.

## Neighbor search

CPU `KdTree` uses a built tree and rejects searches before `build()`. GPU `KdTree`
selects one of three backends:

- `UniformGrid`: cached `GpuSpatialIndex`, deterministic bounded search, `k <= 32`.
- `BruteForce`: CUDA tiled/shared-memory search for small or unsuitable grid work.
- `CpuBruteForce`: compatibility path for `k > 32`.

The uniform grid stores finite points sorted by cell key and source index. Radius
queries inspect intersecting cells. KNN expands cell shells until the next shell cannot
improve the current kth result. Unsafe coordinate quantization, high occupancy, sparse
cell volumes, or queries requiring more than eight empty shells choose brute force.

## Feature and filter pipelines

GPU normal estimation performs indexed KNN, covariance accumulation, PlaMatrix batched
symmetric eigensolve, normalization, and deterministic sign selection on device.
Normal smoothing and viewpoint orientation also use device kernels.

CPU normal estimation accumulates centroid and compact 3x3 covariance directly from the
neighbor list, then calls PlaMatrix's allocation-free `symmetricEigh3x3()` and selects the
smallest eigenvector. CPU point-to-point ICP similarly sends its stack 3x3 cross-covariance
to `svd3x3()`. Neither hot path constructs a dynamic neighbor matrix or invokes general SVD.

CUDA ICP white-box counters and their accessors live in dedicated test-only implementation
headers and are compiled only with `PLAPOINT_ENABLE_TESTING`. Production and benchmark-only
objects therefore contain no `ForTesting` accessors or `g_icp_*` counter storage; test code
uses the guarded `plapoint/gpu/icp_testing.h` declarations instead of duplicating them.

GPU RadiusOR consumes saturated radius counts. GPU SOR computes indexed neighbor means,
normalizes distance statistics to avoid overflow, reduces through PlaMatrix, creates a
keep mask, and stably compacts points and all point-aligned attributes. Only requested
removed-index diagnostics are copied to host.

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
