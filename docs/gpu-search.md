# Accelerator Search and Selection

## Uniform-grid index

`gpu::GpuSpatialIndex<Scalar>` indexes finite points from one GPU cloud revision.
Construction compacts finite source indices, generates checked 64-bit cell keys,
radix-sorts `(key, sourceIndex)`, run-length encodes cells, and builds offsets. Expected
construction cost is `O(n log n)` from sorting; radius work depends on intersected cells,
and KNN work depends on expanded shells and occupancy.

An index match requires the same point-storage identity and address, point count,
`pointsRevision()`, and cell size. Requesting mutable `points()` invalidates cache reuse,
even if the caller does not subsequently change a coordinate.

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

CPU-owned high-level APIs use the strict `ProcessingDevice::Auto` order CUDA, OpenCL,
then CPU. Point count does not change this order. `ProcessingPolicy::autoGpuPointThreshold`
is retained only as a source-compatibility performance hint and is not consulted by
high-level Auto dispatch. `indexedKnnWorkThreshold` remains a query-by-point work-product
threshold internal to CUDA KNN implementation selection.

`ProcessingReport` exposes:

- `requestedDevice`
- `actualDevice` (`usedDevice` is the compatibility alias)
- `neighborBackend`
- `usedFallback`
- `fallbackReason`

Explicit `CUDA` or `OpenCL` requests never silently retry another backend. `GPU` is an
alias for `CUDA`. Auto records each unavailable, unsupported, or failed backend in
`fallbackReason`; a successful OpenCL call after CUDA failure therefore reports OpenCL
as `actualDevice` with `usedFallback == true`.

## OpenCL device selection

PlaPoint's OpenCL path is independent of PlaMatrix device storage: public inputs and
outputs remain CPU-owned while kernels use private OpenCL buffers. The current scope is
voxel downsampling, statistical/radius outlier removal, normal-estimation KNN, and
height-grid aggregation.

`opencl::enumerateOpenClGpuDevices()` returns stable platform/device-order GPU indices
and device metadata. `PLAPOINT_OPENCL_DEVICE_INDEX` is read on first OpenCL runtime use;
set it to one of those indices, or leave it unset/use `-1` for automatic selection.
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

## Local calibration

Use the dedicated benchmark mode:

```powershell
build-cuda\benchmarks\Release\plapoint_benchmarks.exe `
  --points 1000 --iterations 3 --search-features-only
build-cuda\benchmarks\Release\plapoint_benchmarks.exe `
  --points 100000 --iterations 3 --search-features-only
```

The suite reports separate brute-force and forced-index KNN rows plus radius count,
normal estimation, normal smoothing, SOR, and RadiusOR. Forced-index rows intentionally
bypass Auto geometry checks so pathological distributions remain measurable.

On the RTX 5080 / CUDA 13.1 validation machine, the final three-iteration baseline was:

| Points | Auto KNN | Brute KNN | Forced grid KNN | GPU SOR | GPU RadiusOR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.06 ms | 0.86 ms | 55.71 ms | 1.52 ms | 0.34 ms |
| 100,000 | 14.56 ms | 12.82 ms | 5950.74 ms | 11.27 ms | 0.56 ms |

The synthetic queries extend outside a thin source cloud, so this baseline specifically
validates the empty-shell fallback. It is not a claim that brute force wins for compact,
well-populated grids; the forced row remains available to calibrate other datasets.
