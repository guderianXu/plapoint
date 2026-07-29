#pragma once

template <typename Scalar>
struct HeightGridBounds
{
    Scalar minX;
    Scalar maxX;
    Scalar minY;
    Scalar maxY;
    int finiteCount;
};

template <typename Scalar>
struct HeightGridBoundsTransform
{
    const Scalar* points;
    int pointCount;
    Scalar maxValue;

    __host__ __device__ HeightGridBounds<Scalar> operator()(int index) const
    {
        const Scalar x = points[index];
        const Scalar y = points[pointCount + index];
        const Scalar z = points[2 * pointCount + index];
        if (!isfinite(static_cast<double>(x)) ||
            !isfinite(static_cast<double>(y)) ||
            !isfinite(static_cast<double>(z)))
        {
            return {maxValue, -maxValue, maxValue, -maxValue, 0};
        }
        return {x, x, y, y, 1};
    }
};

template <typename Scalar>
struct HeightGridBoundsReduce
{
    __host__ __device__ HeightGridBounds<Scalar> operator()(
        const HeightGridBounds<Scalar>& lhs,
        const HeightGridBounds<Scalar>& rhs) const
    {
        return {
            lhs.minX < rhs.minX ? lhs.minX : rhs.minX,
            lhs.maxX > rhs.maxX ? lhs.maxX : rhs.maxX,
            lhs.minY < rhs.minY ? lhs.minY : rhs.minY,
            lhs.maxY > rhs.maxY ? lhs.maxY : rhs.maxY,
            lhs.finiteCount + rhs.finiteCount};
    }
};

template <typename Scalar>
__global__ void validateHeightGridPointsKernel(
    const Scalar* points,
    int point_count,
    int* status)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count)
    {
        return;
    }
    const Scalar x = points[index];
    const Scalar y = points[point_count + index];
    const Scalar z = points[2 * point_count + index];
    if (!isfinite(static_cast<double>(x)) || !isfinite(static_cast<double>(y)) ||
        !isfinite(static_cast<double>(z)))
    {
        atomicExch(status, 1);
    }
}

template <typename Scalar>
__device__ Scalar clamp01Device(Scalar value)
{
    return value < Scalar(0) ? Scalar(0) : (value > Scalar(1) ? Scalar(1) : value);
}

__device__ int clampGridIndexDevice(int value, int upper_inclusive)
{
    return value < 0 ? 0 : (value > upper_inclusive ? upper_inclusive : value);
}

__device__ void atomicAddScalar(float* address, float value)
{
    atomicAdd(address, value);
}

__device__ void atomicAddScalar(double* address, double value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    unsigned long long assumed = 0;
    do
    {
        assumed = old;
        old = atomicCAS(bits, assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#else
    atomicAdd(address, value);
#endif
}

__device__ void atomicMinScalar(float* address, float value)
{
    auto* bits = reinterpret_cast<int*>(address);
    int old = *bits;
    while (value < __int_as_float(old))
    {
        const int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed)
        {
            break;
        }
    }
}

__device__ void atomicMinScalar(double* address, double value)
{
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (value < __longlong_as_double(old))
    {
        const unsigned long long assumed = old;
        old = atomicCAS(bits, assumed, __double_as_longlong(value));
        if (old == assumed)
        {
            break;
        }
    }
}

__device__ void atomicMaxScalar(float* address, float value)
{
    auto* bits = reinterpret_cast<int*>(address);
    int old = *bits;
    while (value > __int_as_float(old))
    {
        const int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed)
        {
            break;
        }
    }
}

__device__ void atomicMaxScalar(double* address, double value)
{
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (value > __longlong_as_double(old))
    {
        const unsigned long long assumed = old;
        old = atomicCAS(bits, assumed, __double_as_longlong(value));
        if (old == assumed)
        {
            break;
        }
    }
}

template <typename Scalar>
__global__ void initializeHeightGridKernel(
    int cell_count,
    int aggregation,
    bool has_colors,
    Scalar extreme,
    Scalar* heights,
    Scalar* weights,
    std::uint8_t* valid,
    std::uint16_t* fill_pass,
    Scalar* color_sums,
    Scalar* color_weights,
    std::uint8_t* colors)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cell_count)
    {
        return;
    }
    heights[cell] = aggregation == 1 ? extreme : (aggregation == 2 ? -extreme : Scalar(0));
    weights[cell] = Scalar(0);
    valid[cell] = 0;
    fill_pass[cell] = 0;
    if (has_colors)
    {
        color_weights[cell] = Scalar(0);
        for (int channel = 0; channel < 3; ++channel)
        {
            color_sums[cell * 3 + channel] = Scalar(0);
            colors[cell + channel * cell_count] = 0;
        }
    }
}

template <typename Scalar>
__global__ void splatHeightGridKernel(
    const Scalar* points,
    const std::uint8_t* colors,
    int point_count,
    int width,
    int height,
    bool has_colors,
    bool skip_non_finite,
    bool use_bilinear_splat,
    int aggregation,
    Scalar min_x,
    Scalar min_y,
    Scalar step_x,
    Scalar step_y,
    Scalar* heights,
    Scalar* weights,
    Scalar* color_sums,
    Scalar* color_weights)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count)
    {
        return;
    }
    const Scalar x = points[index];
    const Scalar y = points[point_count + index];
    const Scalar z = points[2 * point_count + index];
    if (!isfinite(static_cast<double>(x)) || !isfinite(static_cast<double>(y)) ||
        !isfinite(static_cast<double>(z)))
    {
        static_cast<void>(skip_non_finite);
        return;
    }
    const Scalar gx = (x - min_x) / step_x;
    const Scalar gy = (y - min_y) / step_y;
    if (gx < Scalar(0) || gx > Scalar(width - 1) ||
        gy < Scalar(0) || gy > Scalar(height - 1))
    {
        return;
    }

    const auto accumulate_cell = [&](int cell, Scalar weight) {
        if (weight <= Scalar(0))
        {
            return;
        }
        if (aggregation == 1)
        {
            atomicMinScalar(heights + cell, z);
        }
        else if (aggregation == 2)
        {
            atomicMaxScalar(heights + cell, z);
        }
        else
        {
            atomicAddScalar(heights + cell, z * weight);
        }
        atomicAddScalar(weights + cell, weight);
        if (has_colors)
        {
            atomicAddScalar(color_sums + cell * 3, static_cast<Scalar>(colors[index]) * weight);
            atomicAddScalar(
                color_sums + cell * 3 + 1,
                static_cast<Scalar>(colors[point_count + index]) * weight);
            atomicAddScalar(
                color_sums + cell * 3 + 2,
                static_cast<Scalar>(colors[2 * point_count + index]) * weight);
            atomicAddScalar(color_weights + cell, weight);
        }
    };

    if (!use_bilinear_splat)
    {
        const int ix = clampGridIndexDevice(
            static_cast<int>(floor(static_cast<double>(gx) + 0.5)), width - 1);
        const int iy = clampGridIndexDevice(
            static_cast<int>(floor(static_cast<double>(gy) + 0.5)), height - 1);
        accumulate_cell(iy * width + ix, Scalar(1));
        return;
    }

    const int ix = clampGridIndexDevice(
        static_cast<int>(floor(static_cast<double>(gx))), width - 2);
    const int iy = clampGridIndexDevice(
        static_cast<int>(floor(static_cast<double>(gy))), height - 2);
    const Scalar tx = clamp01Device(gx - Scalar(ix));
    const Scalar ty = clamp01Device(gy - Scalar(iy));
    for (int dy = 0; dy <= 1; ++dy)
    {
        for (int dx = 0; dx <= 1; ++dx)
        {
            const Scalar wx = dx != 0 ? tx : Scalar(1) - tx;
            const Scalar wy = dy != 0 ? ty : Scalar(1) - ty;
            accumulate_cell((iy + dy) * width + ix + dx, wx * wy);
        }
    }
}

template <typename Scalar>
__device__ std::uint8_t colorByteDevice(Scalar weighted_sum, Scalar weight)
{
    if (weight <= Scalar(0))
    {
        return 0;
    }
    const Scalar value = weighted_sum / weight;
    return value <= Scalar(0) ? 0
        : (value >= Scalar(255) ? 255
                                : static_cast<std::uint8_t>(floor(static_cast<double>(value) + 0.5)));
}

template <typename Scalar>
__global__ void normalizeHeightGridKernel(
    int cell_count,
    int aggregation,
    bool has_colors,
    Scalar* heights,
    const Scalar* weights,
    const Scalar* color_sums,
    const Scalar* color_weights,
    std::uint8_t* valid,
    std::uint8_t* colors)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cell_count)
    {
        return;
    }
    if (weights[cell] <= Scalar(1.0e-9))
    {
        heights[cell] = Scalar(0);
        valid[cell] = 0;
        return;
    }
    if (aggregation == 0)
    {
        heights[cell] /= weights[cell];
    }
    valid[cell] = 1;
    if (has_colors && color_weights[cell] > Scalar(0))
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            colors[cell + channel * cell_count] =
                colorByteDevice(color_sums[cell * 3 + channel], color_weights[cell]);
        }
    }
}

template <typename Scalar>
__global__ void fillHeightGridHolesKernel(
    int width,
    int height,
    int pass,
    int min_neighbors,
    int search_radius,
    bool has_colors,
    const Scalar* input_heights,
    const std::uint8_t* input_valid,
    const std::uint8_t* input_colors,
    const std::uint16_t* input_fill_pass,
    Scalar* output_heights,
    std::uint8_t* output_valid,
    std::uint8_t* output_colors,
    std::uint16_t* output_fill_pass)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = width * height;
    if (cell >= cell_count)
    {
        return;
    }
    if (input_valid[cell] != 0)
    {
        output_heights[cell] = input_heights[cell];
        output_valid[cell] = 1;
        output_fill_pass[cell] = input_fill_pass[cell];
        if (has_colors)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                output_colors[cell + channel * cell_count] =
                    input_colors[cell + channel * cell_count];
            }
        }
        return;
    }

    const int x = cell % width;
    const int y = cell / width;
    Scalar height_sum = Scalar(0);
    Scalar weight_sum = Scalar(0);
    double color_sums[3] = {0.0, 0.0, 0.0};
    double color_weight = 0.0;
    int count = 0;
    for (int dy = -search_radius; dy <= search_radius; ++dy)
    {
        for (int dx = -search_radius; dx <= search_radius; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            {
                continue;
            }
            const int neighbor = ny * width + nx;
            if (input_valid[neighbor] == 0)
            {
                continue;
            }
            const Scalar distance = static_cast<Scalar>(
                sqrt(static_cast<double>(dx) * dx + static_cast<double>(dy) * dy));
            const Scalar weight = Scalar(1) / (distance > Scalar(1.0e-6)
                                                    ? distance
                                                    : Scalar(1.0e-6));
            height_sum += input_heights[neighbor] * weight;
            weight_sum += weight;
            if (has_colors)
            {
                for (int channel = 0; channel < 3; ++channel)
                {
                    color_sums[channel] += static_cast<double>(
                        input_colors[neighbor + channel * cell_count]) * weight;
                }
                color_weight += weight;
            }
            ++count;
        }
    }

    output_heights[cell] = input_heights[cell];
    output_valid[cell] = 0;
    output_fill_pass[cell] = input_fill_pass[cell];
    if (has_colors)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            output_colors[cell + channel * cell_count] =
                input_colors[cell + channel * cell_count];
        }
    }
    if (count < min_neighbors)
    {
        return;
    }

    output_heights[cell] = height_sum / (weight_sum > Scalar(1.0e-6)
                                              ? weight_sum
                                              : Scalar(1.0e-6));
    output_valid[cell] = 1;
    output_fill_pass[cell] = static_cast<std::uint16_t>(pass + 1);
    if (has_colors && color_weight > 0.0)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            const double value = color_sums[channel] / color_weight;
            output_colors[cell + channel * cell_count] = value <= 0.0 ? 0
                : (value >= 255.0 ? 255
                                  : static_cast<std::uint8_t>(floor(value + 0.5)));
        }
    }
}
