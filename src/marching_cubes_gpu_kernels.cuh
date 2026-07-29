constexpr int kBlockSize = 256;
constexpr int kTriangleTableWidth = 16;

__constant__ int triangle_table[256 * kTriangleTableWidth];
__constant__ int corner_offsets[24] = {
    0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
    0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1
};
__constant__ int edge_corners[24] = {
    0, 1,  1, 2,  2, 3,  3, 0,
    4, 5,  5, 6,  6, 7,  7, 4,
    0, 4,  1, 5,  2, 6,  3, 7
};

void initializeTriangleTable()
{
    static std::mutex mutex;
    static std::set<int> initialized_devices;
    int device = 0;
    PLAPOINT_CHECK_CUDA(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized_devices.count(device) != 0)
    {
        return;
    }

    std::array<int, 256 * kTriangleTableWidth> host_table{};
    host_table.fill(-1);
    for (int cube_case = 0; cube_case < 256; ++cube_case)
    {
        const auto& triangles = mesh::detail::triTable(cube_case);
        for (std::size_t index = 0;
             index < triangles.size() && index < kTriangleTableWidth;
             ++index)
        {
            host_table[static_cast<std::size_t>(cube_case * kTriangleTableWidth) + index] =
                triangles[index];
        }
    }
    PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(
        triangle_table, host_table.data(), host_table.size() * sizeof(int)));
    initialized_devices.insert(device);
}

template <typename Scalar>
__device__ void loadCube(
    const Scalar* field,
    int ix,
    int iy,
    int iz,
    int sample_x,
    int sample_y,
    Scalar* values)
{
    for (int corner = 0; corner < 8; ++corner)
    {
        const int x = ix + corner_offsets[corner * 3];
        const int y = iy + corner_offsets[corner * 3 + 1];
        const int z = iz + corner_offsets[corner * 3 + 2];
        values[corner] = field[z * sample_y * sample_x + y * sample_x + x];
    }
}

template <typename Scalar>
__global__ void classifyCubesKernel(
    const Scalar* field,
    int nx,
    int ny,
    int nz,
    Scalar iso,
    plamatrix::Index* triangle_counts,
    int* status)
{
    const int cube = blockIdx.x * blockDim.x + threadIdx.x;
    const int cube_count = nx * ny * nz;
    if (cube >= cube_count)
    {
        return;
    }

    const int ix = cube % nx;
    const int iy = (cube / nx) % ny;
    const int iz = cube / (nx * ny);
    Scalar values[8];
    loadCube(field, ix, iy, iz, nx + 1, ny + 1, values);
    int cube_case = 0;
    for (int corner = 0; corner < 8; ++corner)
    {
        if (!isfinite(static_cast<double>(values[corner])))
        {
            atomicCAS(status, 0, 1);
        }
        if (values[corner] < iso)
        {
            cube_case |= 1 << corner;
        }
    }

    int edge_count = 0;
    while (edge_count < 15
           && triangle_table[cube_case * kTriangleTableWidth + edge_count] >= 0)
    {
        ++edge_count;
    }
    triangle_counts[cube] = static_cast<plamatrix::Index>(edge_count / 3);
}

template <typename Scalar>
__device__ Scalar interpolationFactor(Scalar iso, Scalar first, Scalar second)
{
    const Scalar denominator = second - first;
    return denominator == Scalar{0}
        ? Scalar{0.5} : (iso - first) / denominator;
}

template <typename Scalar>
__device__ void edgePoint(
    int edge,
    const Scalar* values,
    int ix,
    int iy,
    int iz,
    Scalar dx,
    Scalar dy,
    Scalar dz,
    Scalar min_x,
    Scalar min_y,
    Scalar min_z,
    Scalar iso,
    Scalar* point)
{
    const int first = edge_corners[edge * 2];
    const int second = edge_corners[edge * 2 + 1];
    const Scalar t = interpolationFactor(iso, values[first], values[second]);
    const Scalar first_x = Scalar(ix + corner_offsets[first * 3]);
    const Scalar first_y = Scalar(iy + corner_offsets[first * 3 + 1]);
    const Scalar first_z = Scalar(iz + corner_offsets[first * 3 + 2]);
    const Scalar second_x = Scalar(ix + corner_offsets[second * 3]);
    const Scalar second_y = Scalar(iy + corner_offsets[second * 3 + 1]);
    const Scalar second_z = Scalar(iz + corner_offsets[second * 3 + 2]);
    point[0] = min_x + (first_x + t * (second_x - first_x)) * dx;
    point[1] = min_y + (first_y + t * (second_y - first_y)) * dy;
    point[2] = min_z + (first_z + t * (second_z - first_z)) * dz;
}

template <typename Scalar>
__global__ void emitTrianglesKernel(
    const Scalar* field,
    int nx,
    int ny,
    int nz,
    Scalar min_x,
    Scalar min_y,
    Scalar min_z,
    Scalar dx,
    Scalar dy,
    Scalar dz,
    Scalar iso,
    const plamatrix::Index* triangle_offsets,
    Scalar* points,
    int* faces,
    int vertex_count,
    int face_count)
{
    const int cube = blockIdx.x * blockDim.x + threadIdx.x;
    const int cube_count = nx * ny * nz;
    if (cube >= cube_count)
    {
        return;
    }

    const int ix = cube % nx;
    const int iy = (cube / nx) % ny;
    const int iz = cube / (nx * ny);
    Scalar values[8];
    loadCube(field, ix, iy, iz, nx + 1, ny + 1, values);
    int cube_case = 0;
    for (int corner = 0; corner < 8; ++corner)
    {
        if (values[corner] < iso)
        {
            cube_case |= 1 << corner;
        }
    }

    const Scalar gx = ((values[1] + values[2] + values[5] + values[6])
        - (values[0] + values[3] + values[4] + values[7])) / (Scalar{4} * dx);
    const Scalar gy = ((values[2] + values[3] + values[6] + values[7])
        - (values[0] + values[1] + values[4] + values[5])) / (Scalar{4} * dy);
    const Scalar gz = ((values[4] + values[5] + values[6] + values[7])
        - (values[0] + values[1] + values[2] + values[3])) / (Scalar{4} * dz);
    const int first_face = static_cast<int>(triangle_offsets[cube]);
    for (int local_face = 0; local_face < 5; ++local_face)
    {
        const int table_position = cube_case * kTriangleTableWidth + local_face * 3;
        const int edge0 = triangle_table[table_position];
        if (edge0 < 0)
        {
            break;
        }
        const int edge1 = triangle_table[table_position + 1];
        const int edge2 = triangle_table[table_position + 2];
        Scalar point0[3];
        Scalar point1[3];
        Scalar point2[3];
        edgePoint(edge0, values, ix, iy, iz, dx, dy, dz,
                  min_x, min_y, min_z, iso, point0);
        edgePoint(edge1, values, ix, iy, iz, dx, dy, dz,
                  min_x, min_y, min_z, iso, point1);
        edgePoint(edge2, values, ix, iy, iz, dx, dy, dz,
                  min_x, min_y, min_z, iso, point2);

        const Scalar ux = point1[0] - point0[0];
        const Scalar uy = point1[1] - point0[1];
        const Scalar uz = point1[2] - point0[2];
        const Scalar vx = point2[0] - point0[0];
        const Scalar vy = point2[1] - point0[1];
        const Scalar vz = point2[2] - point0[2];
        const Scalar normal_x = uy * vz - uz * vy;
        const Scalar normal_y = uz * vx - ux * vz;
        const Scalar normal_z = ux * vy - uy * vx;
        if (normal_x * gx + normal_y * gy + normal_z * gz < Scalar{0})
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const Scalar temporary = point1[axis];
                point1[axis] = point2[axis];
                point2[axis] = temporary;
            }
        }

        const int face = first_face + local_face;
        const int first_vertex = face * 3;
        for (int axis = 0; axis < 3; ++axis)
        {
            points[first_vertex + axis * vertex_count] = point0[axis];
            points[first_vertex + 1 + axis * vertex_count] = point1[axis];
            points[first_vertex + 2 + axis * vertex_count] = point2[axis];
            faces[face + axis * face_count] = first_vertex + axis;
        }
    }
}
