# GPU Mesh Processing

This document describes the CUDA Marching Cubes, HeightGrid, and Poisson solver boundaries.

## Marching Cubes

`gpu::marchingCubes()` accepts a GPU column vector containing
`(nx + 1) * (ny + 1) * (nz + 1)` scalar samples in x-fastest order. It counts triangles,
computes deterministic offsets, and returns a GPU `PointCloud` containing vertices and faces.
The current public call synchronizes the supplied stream before returning because the exact
output size is needed for allocation.

`MarchingCubesGpuWorkspace` is move-only and bound to its first stream. Before destroying a
non-default stream, synchronize it, call `closeAsyncAllocation()`, and synchronize once more.

## Device HeightGrid

The asynchronous build requires explicit finite bounds:

```cpp
mesh::HeightGridOptions<float> options;
options.width = 1024;
options.height = 1024;
options.useExplicitBounds = true;
options.minX = min_x;
options.maxX = max_x;
options.minY = min_y;
options.maxY = max_y;

gpu::HeightGridGpuWorkspace<float> workspace;
auto grid = gpu::buildHeightGridDeviceAsync(cloud_gpu, options, workspace, stream);
gpu::fillHolesAsync(grid, 8, 1, 1, workspace, stream);
auto host_grid = gpu::downloadHeightGrid(grid, stream);
```

Mean, Min, and Max elevation aggregation run on device. Mean uses weights; colors use weighted
aggregation. Validity and `fillPass` remain device-resident through multi-pass ping-pong fill.
The convenience `buildHeightGrid()` wrapper may compute automatic bounds synchronously before
launching aggregation.

Strict non-finite input checking writes device status and reports the error at
`downloadHeightGrid()`. `skipNonFinite=true` drops those points instead. A grid with pending work
must be filled, synchronized, or downloaded on its producer stream. The workspace also remains
stream-bound until `resetStream()` synchronizes and rebinds it.

## Poisson PCG

`PoissonReconstruction` constructs sorted symmetric CSR deterministically. Each connected graph
component receives a deterministic, degree-scaled anchor, which removes the constant Laplacian
null mode and makes the matrix SPD without a float-sized fixed epsilon. The iso level is derived
from the solved field and clamped to the actual extraction-grid range.

```cpp
mesh::PoissonReconstruction<float> reconstruction;
reconstruction.setInputCloud(cloud_with_normals);
reconstruction.setSolverIterations(200);
reconstruction.setSolverTolerance(1.0e-5);
reconstruction.setProcessingDevice(ProcessingDevice::Auto);
auto [vertices, faces] = reconstruction.reconstruct();
const auto& report = reconstruction.lastReport();
```

CPU, CUDA, and OpenCL selections call PlaMatrix Jacobi-PCG on the same CSR. Explicit accelerator
non-convergence throws with iteration and residual context. `Auto` tries CUDA, OpenCL, then CPU,
continues from a non-converged accelerator iterate when available, and fills `fallbackReason`.
`solverDevice` identifies the PCG device; `actualDevice` remains CPU because octree field
evaluation and final Marching Cubes extraction are not yet device-resident. `lastSystem()` exposes
the latest CSR/RHS for diagnostics and benchmark reuse; its reference is invalidated by the next
`reconstruct()` on that object.

## Benchmarks

```bash
plapoint_benchmarks --points 4096 --poisson-points 8192 --poisson-depth 6 \
  --iterations 3 --mesh-only
```

The Poisson input uses a uniform Fibonacci sphere. Its point count and octree depth are
independent from the generic mesh point count; defaults are 8192 points and depth 6.

The mode emits:

- `marching_cubes_field`: device scalar field to GPU triangle soup.
- `height_grid_fill`: device aggregation plus multi-pass fill.
- `poisson_solve`: GPU PCG on a preassembled Poisson CSR.
- `poisson_end_to_end`: complete Poisson reconstruction with GPU PCG and CPU extraction.
