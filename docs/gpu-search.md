# Accelerator Search and Selection

## Uniform-grid index

`gpu::GpuSpatialIndex<Scalar>` indexes finite points from one GPU cloud revision.
Construction uses PlaMatrix to compute the finite-row mask once and reuse it for column bounds,
compact finite source indices, sort `(key, sourceIndex)`, run-length encode cells, reduce maximum
occupancy, and build offsets. `buildAdaptive()` reuses those bounds when it selects a cell size
instead of repeating the statistics pass. Expected construction cost is `O(n log n)` from sorting;
radius work depends on intersected cells, and KNN work depends on expanded shells and occupancy.

An index match requires the same point-storage identity and address, point count,
`pointsRevision()`, and cell size. `editPoints()` invalidates caches at both entry and exit,
then allows reuse again. Requesting legacy mutable `points()` exposes an unbounded alias and
therefore disables GPU cache reuse for the lifetime of that cloud, even if the caller does not
subsequently change a coordinate. CPU KdTree compatibility checks such aliases against its
exact point snapshot and rebuilds only when the bytes differ.

`GpuSpatialIndex` is the prepared-search object: `build()` or `buildAdaptive()` copies the
finite indexed points into index-owned storage, and query calls do not read the source cloud.
The source may therefore change after construction without changing the existing snapshot;
use `matches()` when a cache needs to decide whether to rebuild. Reuse one immutable index
with a separate `GpuSpatialQueryWorkspace` per overlapping stream.

Finite-radius GPU ICP has a separate prepared target path because its kernels use a specialized
column-major grid. `IterativeClosestPoint::prepareGpuTargetSpatialIndex()` builds that grid
synchronously and returns `true` when the grid backend applies. A later `align()` on the same
ICP object reuses it while the target and radius remain current.

CUDA voxel grouping and ICP target-grid construction share PlaMatrix's full-range signed
three-component key primitives: stable lexicographic sort, run-length encoding, and exclusive
scan. Voxel-cluster bounds also use PlaMatrix finite-column statistics. PlaPoint still owns point
quantization, invalid-point policy, centroid and attribute aggregation, ICP cell lookup,
neighborhood traversal, correspondence selection, and pose estimation.

## Query contracts

- Indexed KNN supports `1 <= k <= 32`.
- Radius equality is inclusive.
- Results use stable `(distance, sourceIndex)` ordering.
- Non-finite source points are not indexed; non-finite queries produce no neighbors.
- Missing KNN entries use index `-1`; the index is authoritative when a finite distance
  cannot be represented in the requested scalar type.

Grid KNN is not universally faster. PlaPoint selects brute force for small work,
pathological occupancy/sparsity, unsafe key quantization, or queries more than eight
empty cells outside an indexed axis. The last case prevents large shell scans for
queries far outside a thin cloud.

## Processing policy and reports

CPU-owned high-level APIs first apply a transfer-amortization policy. Linear work below
`ProcessingPolicy::autoGpuPointThreshold` stays on CPU; KNN-style work uses
`autoPrefersNeighborhoodGpu(pointCount, neighbors)`. Work above those boundaries tries
CUDA, OpenCL, then CPU. `indexedKnnWorkThreshold` also remains the query-by-point
work-product threshold used by CUDA KNN implementation selection.

`ProcessingReport` exposes:

- `requestedDevice`
- `actualDevice` (`usedDevice` is the compatibility alias)
- `neighborBackend`
- `usedFallback`
- `fallbackReason`
- `selectionReason`

Explicit `CUDA` or `OpenCL` requests never silently retry another backend. `GPU` is an
alias for `CUDA`. Auto records each unavailable, unsupported, or failed backend in
`fallbackReason`; a successful OpenCL call after CUDA failure therefore reports OpenCL
as `actualDevice` with `usedFallback == true`. A deliberate small-work CPU choice is not
a fallback and is explained by `selectionReason`.

## OpenCL device selection

PlaPoint's OpenCL path uses PlaMatrix's OpenCL runtime, queues, program cache, and
device-buffer resources. Public inputs and outputs for the current high-level
algorithms remain CPU-owned, and point-cloud-specific kernels stay in PlaPoint.
The current scope is voxel downsampling,
statistical/radius outlier removal, normal-estimation KNN, and height-grid aggregation.

`opencl::enumerateOpenClGpuDevices()` returns stable platform/device-order GPU indices
and device metadata. `PLAMATRIX_OPENCL_DEVICE_INDEX` is read on first OpenCL runtime
use; set it to one of those indices, or leave it unset/use `-1` for automatic selection.
`PLAPOINT_OPENCL_DEVICE_INDEX` remains supported when the PlaMatrix variable is unset.
Automatic selection prefers discrete GPUs and then higher compute-unit counts. A bad
explicit index, an unavailable device, or a missing compiler produces a detailed error
for explicit OpenCL calls and an Auto fallback reason. Kernels target OpenCL C 1.2.
Uniform-grid candidate work and serial per-voxel/per-cell reductions have conservative
host-side limits to avoid long-running kernels and Windows driver-watchdog resets; Auto
falls through to CPU when a pathological distribution exceeds one of these limits.
The current adaptive KNN cell-size heuristic is tuned for volumetric distributions.
Very large surface-like or line-like photogrammetry clouds can hit occupancy or global
span guards and fall back to CPU; this release should not be presented as a guaranteed
OpenCL acceleration path for multi-million-point SOR or normal estimation.
OpenCL processing is also in-memory rather than out-of-core: algorithms retain several
O(N) host/device buffers, and bilinear height gridding materializes and sorts as many as
four contributions per point. Very large inputs can therefore fail allocation and let
Auto retry CPU; attributed voxel processing additionally runs CPU attribute aggregation.

## Workspace and stream lifetime

Async query inputs, index storage, outputs, workspace, and CUDA stream must outlive all
queued work. Do not overlap operations that share a `GpuSpatialQueryWorkspace` or an
algorithm workspace. Returned ordinary PlaMatrix allocations may outlive a workspace
after the producing stream has synchronized. Destroy result allocations before
destroying a stream that may still reference them.

For asynchronous writes performed through a `PointCloud::PointEdit`, synchronize the writing
stream before the editor is destroyed. Its destructor marks the end of the mutation window;
ending that window while writes are still queued would make subsequent cache decisions race the
producer stream.

## Local calibration

Use the dedicated benchmark mode:

```powershell
build-cuda\benchmarks\Release\plapoint_benchmarks.exe `
  --points 1000 --iterations 3 --search-features-only
build-cuda\benchmarks\Release\plapoint_benchmarks.exe `
  --points 100000 --iterations 3 --search-features-only
```

The suite reports adaptive spatial-index construction, separate brute-force and forced-index
KNN rows, radius count, normal estimation, normal smoothing, SOR, and RadiusOR. Forced-index
rows intentionally bypass Auto geometry checks so pathological distributions remain measurable.

On the RTX 5080 / CUDA 13.1 validation machine, the final three-iteration baseline was:

| Points | Auto KNN | Brute KNN | Forced grid KNN | GPU SOR | GPU RadiusOR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.06 ms | 0.86 ms | 55.71 ms | 1.52 ms | 0.34 ms |
| 100,000 | 14.56 ms | 12.82 ms | 5950.74 ms | 11.27 ms | 0.56 ms |

The synthetic queries extend outside a thin source cloud, so this baseline specifically
validates the empty-shell fallback. It is not a claim that brute force wins for compact,
well-populated grids; the forced row remains available to calibrate other datasets.
