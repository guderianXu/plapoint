# PlaPoint Architecture

## Ownership and devices

`PointCloud<Scalar, Device>` owns PlaMatrix column-major point storage and optional
point-aligned attributes. GPU algorithms accept GPU clouds directly and return GPU
matrices or clouds; CPU-owned convenience APIs perform explicit transfers and return
CPU-owned results.

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

GPU RadiusOR consumes saturated radius counts. GPU SOR computes indexed neighbor means,
normalizes distance statistics to avoid overflow, reduces through PlaMatrix, creates a
keep mask, and stably compacts points and all point-aligned attributes. Only requested
removed-index diagnostics are copied to host.

## High-level selection

`ProcessingPolicy` centralizes benchmark-derived thresholds. CPU-owned `Auto` calls use
CPU below 4096 points and try GPU at or above that boundary. An explicit GPU request
does not silently change devices: unsupported parameters, unavailable CUDA, allocation
failure, or kernel failure are rethrown. `Auto` records an attempted-GPU failure and
retries CPU.

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

Poisson assembly orders leaves and undirected edges deterministically, anchors one row per
connected graph component, and solves the resulting SPD CSR with PlaMatrix Jacobi-PCG.
Only the sparse solve can run on GPU today; field sampling and Marching Cubes extraction
remain CPU. `PoissonProcessingReport` separates full-chain `actualDevice` from
`solverDevice` and records non-convergence fallback details.

See [GPU mesh processing](gpu-mesh.md) for stream and status contracts.
