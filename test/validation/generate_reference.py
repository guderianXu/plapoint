#!/usr/bin/env python3
"""Generate reference data for cross-validation of plapoint algorithms.
Uses numpy/scipy for mathematically correct reference results.
Writes PLY and NPY files that C++ tests load and compare."""

import numpy as np
import struct
import sys
from pathlib import Path
from scipy.spatial import KDTree as ScipyKDTree

OUT = Path(__file__).parent / "reference_data"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

def write_ply(path, points, normals=None):
    """Write ASCII PLY file."""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if normals is not None:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for i, pt in enumerate(points):
            f.write(f"{pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f}")
            if normals is not None:
                f.write(f" {normals[i][0]:.8f} {normals[i][1]:.8f} {normals[i][2]:.8f}")
            f.write("\n")

def write_npy(path, arr):
    np.save(path, arr)

# ============================================================
# Test 1: KdTree KNN — exact results via scipy.spatial.KDTree
# ============================================================
print("Generating KdTree KNN reference...")
N = 200
points = rng.uniform(-10, 10, (N, 3)).astype(np.float32)
queries = rng.uniform(-10, 10, (50, 3)).astype(np.float32)

tree = ScipyKDTree(points)
k_vals = [1, 4, 8, 16]
for k in k_vals:
    distances, indices = tree.query(queries, k=k)
    if k == 1:
        indices = indices[:, np.newaxis]
        distances = distances[:, np.newaxis]
    write_npy(OUT / f"knn_points.npy", points)
    write_npy(OUT / f"knn_queries.npy", queries)
    write_npy(OUT / f"knn_indices_k{k}.npy", indices.astype(np.int32))
    write_npy(OUT / f"knn_dists_k{k}.npy", distances.astype(np.float32))

# ============================================================
# Test 2: VoxelGrid — reference centroid per voxel
# ============================================================
print("Generating VoxelGrid reference...")
N = 500
points = rng.uniform(-5, 5, (N, 3)).astype(np.float32)
leaf = 0.5

# Compute voxel centroids (same algorithm as plapoint)
voxel_dict = {}
for i, pt in enumerate(points):
    vx = int(np.floor(pt[0] / leaf))
    vy = int(np.floor(pt[1] / leaf))
    vz = int(np.floor(pt[2] / leaf))
    key = (vx, vy, vz)
    if key not in voxel_dict:
        voxel_dict[key] = {'sum': np.zeros(3), 'count': 0}
    voxel_dict[key]['sum'] += pt
    voxel_dict[key]['count'] += 1

centroids = np.array([v['sum'] / v['count'] for v in voxel_dict.values()], dtype=np.float32)

write_npy(OUT / "voxelgrid_input.npy", points)
write_npy(OUT / "voxelgrid_leaf.npy", np.array([leaf], dtype=np.float32))
write_npy(OUT / "voxelgrid_centroids.npy", centroids)

# ============================================================
# Test 3: RadiusOutlierRemoval
# ============================================================
print("Generating RadiusOutlierRemoval reference...")
# 100 clustered points + 3 isolated outliers
cluster = rng.normal(0, 0.1, (100, 3)).astype(np.float32)
outliers = np.array([[10, 0, 0], [-8, 0, 0], [0, 0, 9]], dtype=np.float32)
points = np.vstack([cluster, outliers])

tree = ScipyKDTree(points)
radius = 1.0
min_neighbors = 5
inlier_mask = np.zeros(len(points), dtype=np.int32)
for i in range(len(points)):
    neighbors = tree.query_ball_point(points[i], radius)
    if len(neighbors) >= min_neighbors:
        inlier_mask[i] = 1

write_npy(OUT / "radius_filter_input.npy", points)
write_npy(OUT / "radius_filter_inliers.npy", inlier_mask)

# ============================================================
# Test 4: NormalEstimation — sphere normals
# ============================================================
print("Generating NormalEstimation reference...")
N = 200
# Points on unit sphere
phi = rng.uniform(0, 2*np.pi, N)
theta = np.arccos(rng.uniform(-1, 1, N))
r = 1.0
points = np.column_stack([
    r * np.sin(theta) * np.cos(phi),
    r * np.sin(theta) * np.sin(phi),
    r * np.cos(theta)
]).astype(np.float32)

# Ground truth normals: outward pointing = normalized position
normals = points.copy()  # since r=1, position IS the normal

tree = ScipyKDTree(points)
K = 16
# Compute PCA normals
estimated = np.zeros_like(points)
for i in range(N):
    dists, idxs = tree.query(points[i], k=K)
    nb = points[idxs]
    centroid = nb.mean(axis=0)
    centered = nb - centroid
    cov = centered.T @ centered / K
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    n = eigenvectors[:, 0]  # smallest eigenvalue
    # Orient toward outward (positive dot with position)
    if np.dot(n, points[i]) < 0:
        n = -n
    estimated[i] = n

write_npy(OUT / "normal_est_input.npy", points)
write_npy(OUT / "normal_est_normals.npy", estimated)
write_npy(OUT / "normal_est_k.npy", np.array([K], dtype=np.int32))

# ============================================================
# Test 5: ICP — known rigid transform
# ============================================================
print("Generating ICP reference...")
N = 100
source = rng.normal(0, 0.5, (N, 3)).astype(np.float64)

# Apply known rotation + translation
angle = 0.5  # rad
axis = np.array([0, 0, 1], dtype=np.float64)
axis = axis / np.linalg.norm(axis)
K_mat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
R = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)
t = np.array([0.5, 0.3, -0.2], dtype=np.float64)

target = (source @ R.T + t).astype(np.float64)

write_npy(OUT / "icp_source.npy", source.astype(np.float64))
write_npy(OUT / "icp_target.npy", target.astype(np.float64))
write_npy(OUT / "icp_R.npy", R)
write_npy(OUT / "icp_t.npy", t)

# ============================================================
# Test 6: MarchingCubes — sphere surface
# ============================================================
print("Generating MarchingCubes reference...")
# For a sphere of radius 2, all vertices should satisfy x²+y²+z² ≈ 4
# We test by evaluating the implicit function
sphere_r = 2.0
write_npy(OUT / "mc_sphere_r.npy", np.array([sphere_r], dtype=np.float32))

# ============================================================
# Test 7: PoissonReconstruction — sphere
# ============================================================
print("Generating Poisson reference...")
# Same sphere points as NormalEstimation test, already have normals
write_ply(OUT / "poisson_input.ply", points, normals)
write_npy(OUT / "poisson_input.npy", points)
write_npy(OUT / "poisson_normals.npy", normals)

# ============================================================
# Test 8: UniformDownsample
# ============================================================
print("Generating UniformDownsample reference...")
N = 20
points = rng.uniform(-5, 5, (N, 3)).astype(np.float32)
step = 3
indices = np.arange(0, N, step, dtype=np.int32)
downsampled = points[indices]

write_npy(OUT / "uniform_down_input.npy", points)
write_npy(OUT / "uniform_down_step.npy", np.array([step], dtype=np.int32))
write_npy(OUT / "uniform_down_output.npy", downsampled)

print(f"\nAll reference data written to {OUT}/")
print("Files:", sorted([f.name for f in OUT.iterdir()]))
