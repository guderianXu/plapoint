#pragma once

namespace plapoint
{
namespace opencl
{
namespace detail
{

inline constexpr const char* heightGridKernelSource = R"CLC(
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

__kernel void aggregateHeightCells(
    __global const real* point_z,
    __global const uchar* point_colors,
    __global const int* source_indices,
    __global const real* contribution_weights,
    __global const int* cell_ids,
    __global const int* offsets,
    __global const int* counts,
    int occupied_count,
    int aggregation,
    int has_colors,
    __global real* heights,
    __global real* weights,
    __global uchar* valid,
    __global uchar* colors)
{
    const int occupied = (int)get_global_id(0);
    if (occupied >= occupied_count) return;
    const int cell = cell_ids[occupied];
    real height = (real)0;
    real total_weight = (real)0;
    real color_r = (real)0;
    real color_g = (real)0;
    real color_b = (real)0;
    for (int item = 0; item < counts[occupied]; ++item)
    {
        const int contribution = offsets[occupied] + item;
        const int point = source_indices[contribution];
        const real weight = contribution_weights[contribution];
        const real z = point_z[point];
        if (aggregation == 1)
            height = total_weight <= (real)0 || z < height ? z : height;
        else if (aggregation == 2)
            height = total_weight <= (real)0 || z > height ? z : height;
        else
            height += z * weight;
        total_weight += weight;
        if (has_colors)
        {
            const size_t color_offset = (size_t)point * (size_t)3;
            color_r += (real)point_colors[color_offset] * weight;
            color_g += (real)point_colors[color_offset + 1] * weight;
            color_b += (real)point_colors[color_offset + 2] * weight;
        }
    }
    weights[cell] = total_weight;
    if (total_weight <= (real)1.0e-9) return;
    heights[cell] = aggregation == 0 ? height / total_weight : height;
    valid[cell] = (uchar)1;
    if (has_colors)
    {
        const size_t output = (size_t)cell * (size_t)3;
        colors[output] = (uchar)clamp(floor(color_r / total_weight + (real)0.5), (real)0, (real)255);
        colors[output + 1] = (uchar)clamp(floor(color_g / total_weight + (real)0.5), (real)0, (real)255);
        colors[output + 2] = (uchar)clamp(floor(color_b / total_weight + (real)0.5), (real)0, (real)255);
    }
}
)CLC";

} // namespace detail
} // namespace opencl
} // namespace plapoint
