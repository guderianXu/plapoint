#pragma once

namespace plapoint
{
namespace opencl
{
namespace detail
{

inline constexpr const char* voxelKernelSource = R"CLC(
#ifdef PLAPOINT_REAL_DOUBLE
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
typedef double real;
#else
typedef float real;
#endif

__kernel void voxelCentroids(
    __global const real* points,
    __global const int* sorted_indices,
    __global const int* offsets,
    __global const int* counts,
    int voxel_count,
    __global real* centroids)
{
    const int voxel = (int)get_global_id(0);
    if (voxel >= voxel_count) return;
    real mean_x = (real)0;
    real mean_y = (real)0;
    real mean_z = (real)0;
    const int count = counts[voxel];
    for (int item = 0; item < count; ++item)
    {
        const int point = sorted_indices[offsets[voxel] + item];
        const real weight = (real)1 / (real)(item + 1);
        const size_t point_offset = (size_t)point * (size_t)3;
        mean_x += (points[point_offset] - mean_x) * weight;
        mean_y += (points[point_offset + 1] - mean_y) * weight;
        mean_z += (points[point_offset + 2] - mean_z) * weight;
    }
    const size_t output_offset = (size_t)voxel * (size_t)3;
    centroids[output_offset] = mean_x;
    centroids[output_offset + 1] = mean_y;
    centroids[output_offset + 2] = mean_z;
}
)CLC";

inline constexpr const char* neighborKernelSource = R"CLC(
#ifdef PLAPOINT_REAL_DOUBLE
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
typedef double real;
#else
typedef float real;
#endif

#define MAX_K 32
#define MAX_SHELL 4096
#define MAX_CELL_VISITS 262144
#define MAX_POINT_VISITS 1048576

int compare_cell(__global const long* cells, int cell, long x, long y, long z)
{
    const size_t offset = (size_t)cell * (size_t)3;
    const long cx = cells[offset];
    const long cy = cells[offset + 1];
    const long cz = cells[offset + 2];
    if (cx != x) return cx < x ? -1 : 1;
    if (cy != y) return cy < y ? -1 : 1;
    if (cz != z) return cz < z ? -1 : 1;
    return 0;
}

int find_cell(__global const long* cells, int cell_count, long x, long y, long z)
{
    int first = 0;
    int last = cell_count;
    while (first < last)
    {
        const int middle = first + (last - first) / 2;
        if (compare_cell(cells, middle, x, y, z) < 0) first = middle + 1;
        else last = middle;
    }
    return first < cell_count && compare_cell(cells, first, x, y, z) == 0 ? first : -1;
}

real point_distance(__global const real* points, int lhs, int rhs)
{
    const size_t lhs_offset = (size_t)lhs * (size_t)3;
    const size_t rhs_offset = (size_t)rhs * (size_t)3;
    const real dx = points[lhs_offset] - points[rhs_offset];
    const real dy = points[lhs_offset + 1] - points[rhs_offset + 1];
    const real dz = points[lhs_offset + 2] - points[rhs_offset + 2];
    return hypot(hypot(dx, dy), dz);
}

void add_neighbor(int point, real distance, int k, __private int* found,
    __private int indices[MAX_K], __private real distances[MAX_K])
{
    int position = *found;
    if (position >= k)
    {
        position = k - 1;
        if (distance > distances[position]
            || (distance == distances[position] && point >= indices[position])) return;
    }
    else ++(*found);
    while (position > 0 && (distance < distances[position - 1]
        || (distance == distances[position - 1] && point < indices[position - 1])))
    {
        distances[position] = distances[position - 1];
        indices[position] = indices[position - 1];
        --position;
    }
    distances[position] = distance;
    indices[position] = point;
}

void visit_knn_cell(__global const real* points, __global const int* sorted_indices,
    __global const int* offsets, __global const int* counts, int query, int cell, int k,
    __private int* found, __private int indices[MAX_K], __private real distances[MAX_K],
    __private int* point_visits)
{
    if (cell < 0) return;
    for (int item = 0; item < counts[cell] && *point_visits < MAX_POINT_VISITS; ++item)
    {
        ++(*point_visits);
        const int point = sorted_indices[offsets[cell] + item];
        const real distance = point_distance(points, query, point);
        if (isfinite(distance)) add_neighbor(point, distance, k, found, indices, distances);
    }
}

void visit_knn_coordinate(__global const real* points, __global const long* cells,
    __global const int* sorted_indices, __global const int* offsets,
    __global const int* counts, int cell_count, long span_x, long span_y, long span_z,
    int query, long x, long y, long z, int k, __private int* found,
    __private int indices[MAX_K], __private real distances[MAX_K],
    __private int* cell_visits, __private int* point_visits)
{
    if (x < 0 || x > span_x || y < 0 || y > span_y || z < 0 || z > span_z) return;
    if (*cell_visits >= MAX_CELL_VISITS || *point_visits >= MAX_POINT_VISITS) return;
    ++(*cell_visits);
    visit_knn_cell(points, sorted_indices, offsets, counts, query,
        find_cell(cells, cell_count, x, y, z), k, found, indices, distances, point_visits);
}

__kernel void knnMeanDistances(__global const real* points, __global const uchar* finite_mask,
    __global const long* query_cells, __global const long* cells,
    __global const int* sorted_indices, __global const int* offsets,
    __global const int* counts, int point_count, int cell_count,
    long span_x, long span_y, long span_z, real cell_size, int k,
    __global real* mean_distances, __global uchar* valid_distances,
    int write_neighbors, __global int* neighbor_indices)
{
    const int query = (int)get_global_id(0);
    if (query >= point_count) return;
    mean_distances[query] = (real)0;
    valid_distances[query] = (uchar)0;
    if (write_neighbors)
        for (int item = 0; item < k; ++item)
            neighbor_indices[(size_t)query * (size_t)k + (size_t)item] = -1;
    if (!finite_mask[query]) return;
    int indices[MAX_K];
    real distances[MAX_K];
    for (int item = 0; item < k; ++item)
    {
        indices[item] = -1;
        distances[item] = (real)INFINITY;
    }
    int found = 0;
    int cell_visits = 0;
    int point_visits = 0;
    const size_t query_offset = (size_t)query * (size_t)3;
    const long center_x = query_cells[query_offset];
    const long center_y = query_cells[query_offset + 1];
    const long center_z = query_cells[query_offset + 2];
    const long last_shell = max(max(center_x, span_x - center_x),
        max(max(center_y, span_y - center_y), max(center_z, span_z - center_z)));
    if (last_shell > (long)MAX_SHELL) return;

    for (long shell = 0; shell <= last_shell; ++shell)
    {
        const long first_x = max((long)0, center_x - shell);
        const long last_x = min(span_x, center_x + shell);
        const long first_y = max((long)0, center_y - shell);
        const long last_y = min(span_y, center_y + shell);
        const long first_z = max((long)0, center_z - shell);
        const long last_z = min(span_z, center_z + shell);
        const long negative_z = center_z - shell;
        const long positive_z = center_z + shell;
        for (long x = first_x; x <= last_x; ++x)
            for (long y = first_y; y <= last_y; ++y)
            {
                visit_knn_coordinate(points, cells, sorted_indices, offsets, counts, cell_count,
                    span_x, span_y, span_z, query, x, y, negative_z, k, &found, indices,
                    distances, &cell_visits, &point_visits);
                if (positive_z != negative_z)
                    visit_knn_coordinate(points, cells, sorted_indices, offsets, counts, cell_count,
                        span_x, span_y, span_z, query, x, y, positive_z, k, &found, indices,
                        distances, &cell_visits, &point_visits);
            }
        const long interior_first_z = max(first_z, center_z - shell + 1);
        const long interior_last_z = min(last_z, center_z + shell - 1);
        const long negative_x = center_x - shell;
        const long positive_x = center_x + shell;
        for (long y = first_y; y <= last_y; ++y)
            for (long z = interior_first_z; z <= interior_last_z; ++z)
            {
                visit_knn_coordinate(points, cells, sorted_indices, offsets, counts, cell_count,
                    span_x, span_y, span_z, query, negative_x, y, z, k, &found, indices,
                    distances, &cell_visits, &point_visits);
                if (positive_x != negative_x)
                    visit_knn_coordinate(points, cells, sorted_indices, offsets, counts, cell_count,
                        span_x, span_y, span_z, query, positive_x, y, z, k, &found, indices,
                        distances, &cell_visits, &point_visits);
            }
        const long interior_first_x = max(first_x, center_x - shell + 1);
        const long interior_last_x = min(last_x, center_x + shell - 1);
        const long negative_y = center_y - shell;
        const long positive_y = center_y + shell;
        for (long x = interior_first_x; x <= interior_last_x; ++x)
            for (long z = interior_first_z; z <= interior_last_z; ++z)
            {
                visit_knn_coordinate(points, cells, sorted_indices, offsets, counts, cell_count,
                    span_x, span_y, span_z, query, x, negative_y, z, k, &found, indices,
                    distances, &cell_visits, &point_visits);
                if (positive_y != negative_y)
                    visit_knn_coordinate(points, cells, sorted_indices, offsets, counts, cell_count,
                        span_x, span_y, span_z, query, x, positive_y, z, k, &found, indices,
                        distances, &cell_visits, &point_visits);
            }
        if (found == k && distances[k - 1] < cell_size * (real)shell) break;
        if (cell_visits >= MAX_CELL_VISITS || point_visits >= MAX_POINT_VISITS) break;
    }
    if (cell_visits >= MAX_CELL_VISITS || point_visits >= MAX_POINT_VISITS) return;
    real mean = (real)0;
    int count = 0;
    for (int item = 0; item < min(found, k); ++item)
        if (indices[item] != query)
        {
            ++count;
            mean += (distances[item] - mean) / (real)count;
        }
    mean_distances[query] = count > 0 ? mean : (real)0;
    valid_distances[query] = found == k ? (uchar)1 : (uchar)0;
    if (write_neighbors)
        for (int item = 0; item < min(found, k); ++item)
            neighbor_indices[(size_t)query * (size_t)k + (size_t)item] = indices[item];
}

__kernel void radiusKeepMask(__global const real* points, __global const uchar* finite_mask,
    __global const long* query_cells, __global const long* cells,
    __global const int* sorted_indices, __global const int* offsets,
    __global const int* counts, int point_count, int cell_count,
    long span_x, long span_y, long span_z, real radius, int min_neighbors,
    __global uchar* keep_mask)
{
    const int query = (int)get_global_id(0);
    if (query >= point_count) return;
    keep_mask[query] = (uchar)0;
    if (!finite_mask[query]) return;
    const size_t query_offset = (size_t)query * (size_t)3;
    const long center_x = query_cells[query_offset];
    const long center_y = query_cells[query_offset + 1];
    const long center_z = query_cells[query_offset + 2];
    int found = 0;
    for (long dx = -1; dx <= 1 && found < min_neighbors; ++dx)
    {
        const long x = center_x + dx;
        if (x < 0 || x > span_x) continue;
        for (long dy = -1; dy <= 1 && found < min_neighbors; ++dy)
        {
            const long y = center_y + dy;
            if (y < 0 || y > span_y) continue;
            for (long dz = -1; dz <= 1 && found < min_neighbors; ++dz)
            {
                const long z = center_z + dz;
                if (z < 0 || z > span_z) continue;
                const int cell = find_cell(cells, cell_count, x, y, z);
                if (cell < 0) continue;
                for (int item = 0; item < counts[cell]; ++item)
                {
                    const int point = sorted_indices[offsets[cell] + item];
                    const real distance = point_distance(points, query, point);
                    if (isfinite(distance) && distance <= radius && ++found >= min_neighbors) break;
                }
            }
        }
    }
    keep_mask[query] = found >= min_neighbors ? (uchar)1 : (uchar)0;
}
)CLC";

} // namespace detail
} // namespace opencl
} // namespace plapoint
