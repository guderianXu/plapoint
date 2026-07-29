#pragma once

#include <cfloat>
#include <climits>

#include <cuda_runtime.h>

namespace plapoint
{
namespace gpu
{
namespace detail
{

struct DistanceKey
{
    int exponent;
    double mantissa;
};

inline __device__ DistanceKey absoluteDifferenceKey(double lhs, double rhs)
{
    if (lhs == rhs)
    {
        return {INT_MIN, 0.0};
    }
    if (signbit(lhs) == signbit(rhs) || lhs == 0.0 || rhs == 0.0)
    {
        int exponent = 0;
        const double mantissa = frexp(fabs(lhs - rhs), &exponent);
        return {exponent, mantissa};
    }

    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const double lhs_mantissa = frexp(fabs(lhs), &lhs_exponent);
    const double rhs_mantissa = frexp(fabs(rhs), &rhs_exponent);
    const int exponent = max(lhs_exponent, rhs_exponent);
    const double sum = ldexp(lhs_mantissa, lhs_exponent - exponent)
        + ldexp(rhs_mantissa, rhs_exponent - exponent);
    int adjustment = 0;
    const double mantissa = frexp(sum, &adjustment);
    return {exponent + adjustment, mantissa};
}

inline __device__ DistanceKey finiteDistanceKey(
    double qx, double qy, double qz,
    double px, double py, double pz)
{
    if (!isfinite(qx) || !isfinite(qy) || !isfinite(qz)
        || !isfinite(px) || !isfinite(py) || !isfinite(pz))
    {
        return {INT_MAX, DBL_MAX};
    }
    const DistanceKey components[3] = {
        absoluteDifferenceKey(qx, px),
        absoluteDifferenceKey(qy, py),
        absoluteDifferenceKey(qz, pz)};
    const int exponent = max(
        components[0].exponent, max(components[1].exponent, components[2].exponent));
    if (exponent == INT_MIN)
    {
        return {INT_MIN, 0.0};
    }
    const double distance = norm3d(
        components[0].mantissa == 0.0 ? 0.0
                                      : ldexp(components[0].mantissa, components[0].exponent - exponent),
        components[1].mantissa == 0.0 ? 0.0
                                      : ldexp(components[1].mantissa, components[1].exponent - exponent),
        components[2].mantissa == 0.0 ? 0.0
                                      : ldexp(components[2].mantissa, components[2].exponent - exponent));
    int adjustment = 0;
    const double mantissa = frexp(distance, &adjustment);
    return {exponent + adjustment, mantissa};
}

inline __device__ bool distanceLess(const DistanceKey& lhs, const DistanceKey& rhs)
{
    return lhs.exponent < rhs.exponent
        || (lhs.exponent == rhs.exponent && lhs.mantissa < rhs.mantissa);
}

template <typename Scalar>
inline __device__ Scalar squaredOutputDistance(const DistanceKey& distance)
{
    const double maximum = sizeof(Scalar) == sizeof(float) ? FLT_MAX : DBL_MAX;
    const int maximum_exponent = sizeof(Scalar) == sizeof(float) ? FLT_MAX_EXP : DBL_MAX_EXP;
    if (distance.exponent == INT_MIN)
    {
        return Scalar(0);
    }
    int squared_mantissa_exponent = 0;
    const double squared_mantissa = frexp(
        distance.mantissa * distance.mantissa, &squared_mantissa_exponent);
    const int squared_exponent = 2 * distance.exponent + squared_mantissa_exponent;
    if (squared_exponent > maximum_exponent)
    {
        return static_cast<Scalar>(maximum);
    }
    const double squared_distance = ldexp(squared_mantissa, squared_exponent);
    return static_cast<Scalar>(squared_distance >= maximum ? maximum : squared_distance);
}

} // namespace detail
} // namespace gpu
} // namespace plapoint
