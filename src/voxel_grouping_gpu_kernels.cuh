#pragma once

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/ops/grouping.h>
#include <utility>
#include <plamatrix/internal/ops/indexing.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>

#include <plapoint/gpu/cuda_check.h>

namespace plapoint
{
    namespace gpu
    {
        namespace detail
        {

            constexpr int kVoxelGroupingBlockSize = 256;

            struct VoxelGroupingMetadata
            {
                plamatrix::Index groupCount;
                int keyError;
            };

            struct WeightedMeanAccumulator
            {
                double mean;
                int count;
            };

            __host__ __device__ inline WeightedMeanAccumulator
            combineWeightedMeans(const WeightedMeanAccumulator& left, const WeightedMeanAccumulator& right)
            {
                if (left.count == 0)
                {
                    return right;
                }
                if (right.count == 0)
                {
                    return left;
                }

                const int total_count = left.count + right.count;
                const double total = static_cast<double>(total_count);
                const double left_weight = static_cast<double>(left.count) / total;
                const double right_weight = static_cast<double>(right.count) / total;
                return {
                    left.mean * left_weight + right.mean * right_weight,
                    total_count,
                };
            }

            template <typename Scalar>
            __device__ int
            checkedVoxelCoordinate(Scalar coordinate, Scalar origin, Scalar cell_size, std::int32_t& result)
            {
                const double value = static_cast<double>(coordinate);
                if (!isfinite(value))
                {
                    return 1;
                }

                const double relative = value - static_cast<double>(origin);
                const double scaled = floor(relative / static_cast<double>(cell_size));
                if (!isfinite(relative) || !isfinite(scaled) || scaled < static_cast<double>(INT_MIN) ||
                    scaled > static_cast<double>(INT_MAX))
                {
                    return 2;
                }

                result = static_cast<std::int32_t>(scaled);
                return 0;
            }

            template <typename Scalar>
            __global__ void makeVoxelKeysAndIndicesKernel(const Scalar* points,
                                                          int point_count,
                                                          Scalar origin_x,
                                                          Scalar origin_y,
                                                          Scalar origin_z,
                                                          Scalar cell_x,
                                                          Scalar cell_y,
                                                          Scalar cell_z,
                                                          plamatrix::internal::Int32Key3* keys,
                                                          plamatrix::Index* indices,
                                                          int* error_code)
            {
                const int row = blockIdx.x * blockDim.x + threadIdx.x;
                if (row >= point_count)
                {
                    return;
                }

                plamatrix::internal::Int32Key3 key{};
                int error = checkedVoxelCoordinate(points[row], origin_x, cell_x, key.x);
                if (error == 0)
                {
                    error = checkedVoxelCoordinate(points[point_count + row], origin_y, cell_y, key.y);
                }
                if (error == 0)
                {
                    error = checkedVoxelCoordinate(points[2 * point_count + row], origin_z, cell_z, key.z);
                }
                if (error != 0)
                {
                    atomicCAS(error_code, 0, error);
                    key = {};
                }

                keys[row] = key;
                indices[row] = static_cast<plamatrix::Index>(row);
            }

            template <typename Scalar>
            __global__ void writeVoxelCentroidsAndRemapKernel(const Scalar* points,
                                                              int point_count,
                                                              const plamatrix::Index* sorted_indices,
                                                              const plamatrix::Index* group_offsets,
                                                              const plamatrix::Index* group_counts,
                                                              int group_count,
                                                              Scalar* output,
                                                              int* point_remap)
            {
                const int group = static_cast<int>(blockIdx.x);
                if (group >= group_count)
                {
                    return;
                }

                const plamatrix::Index start = group_offsets[group];
                const plamatrix::Index count = group_counts[group];
                WeightedMeanAccumulator local_x{0.0, 0};
                WeightedMeanAccumulator local_y{0.0, 0};
                WeightedMeanAccumulator local_z{0.0, 0};
                for (plamatrix::Index offset = threadIdx.x; offset < count; offset += blockDim.x)
                {
                    const auto source = sorted_indices[start + offset];
                    if (point_remap)
                    {
                        point_remap[source] = group;
                    }
                    local_x = combineWeightedMeans(local_x, {static_cast<double>(points[source]), 1});
                    local_y = combineWeightedMeans(local_y, {static_cast<double>(points[point_count + source]), 1});
                    local_z = combineWeightedMeans(local_z, {static_cast<double>(points[2 * point_count + source]), 1});
                }

                __shared__ WeightedMeanAccumulator x_partials[kVoxelGroupingBlockSize];
                __shared__ WeightedMeanAccumulator y_partials[kVoxelGroupingBlockSize];
                __shared__ WeightedMeanAccumulator z_partials[kVoxelGroupingBlockSize];
                x_partials[threadIdx.x] = local_x;
                y_partials[threadIdx.x] = local_y;
                z_partials[threadIdx.x] = local_z;
                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
                {
                    if (threadIdx.x < stride)
                    {
                        x_partials[threadIdx.x] =
                            combineWeightedMeans(x_partials[threadIdx.x], x_partials[threadIdx.x + stride]);
                        y_partials[threadIdx.x] =
                            combineWeightedMeans(y_partials[threadIdx.x], y_partials[threadIdx.x + stride]);
                        z_partials[threadIdx.x] =
                            combineWeightedMeans(z_partials[threadIdx.x], z_partials[threadIdx.x + stride]);
                    }
                    __syncthreads();
                }

                if (threadIdx.x == 0)
                {
                    output[group] = static_cast<Scalar>(x_partials[0].mean);
                    output[group_count + group] = static_cast<Scalar>(y_partials[0].mean);
                    output[2 * group_count + group] = static_cast<Scalar>(z_partials[0].mean);
                }
            }

            template <typename Scalar>
            int groupVoxelCentroidsColumnMajor(const Scalar* points,
                                               int point_count,
                                               Scalar origin_x,
                                               Scalar origin_y,
                                               Scalar origin_z,
                                               Scalar cell_x,
                                               Scalar cell_y,
                                               Scalar cell_z,
                                               Scalar* output,
                                               int* point_remap,
                                               const char* operation,
                                               plamatrix::internal::ExecutionContext& context,
                                               cudaStream_t stream)
            {
                if (point_count <= 0)
                {
                    return 0;
                }

                const auto matrix_count = static_cast<plamatrix::Index>(point_count);
                auto keys = plamatrix::internal::ResidentMatrix<plamatrix::internal::Int32Key3>(matrix_count, 1, context);
                auto indices = plamatrix::internal::ResidentMatrix<plamatrix::Index>(matrix_count, 1, context);
                auto sorted_keys = plamatrix::internal::ResidentMatrix<plamatrix::internal::Int32Key3>(matrix_count, 1, context);
                auto sorted_indices = plamatrix::internal::ResidentMatrix<plamatrix::Index>(matrix_count, 1, context);
                auto group_counts = plamatrix::internal::ResidentMatrix<plamatrix::Index>(matrix_count, 1, context);
                auto run_count = plamatrix::internal::ResidentMatrix<plamatrix::Index>(1, 1, context);
                DeviceBuffer<int> key_error(1);
                HostPinnedBuffer<VoxelGroupingMetadata> host_metadata(1);
                auto* metadata = host_metadata.get();

                PLAPOINT_CHECK_CUDA(cudaMemsetAsync(key_error.get(), 0, sizeof(int), stream));
                PLAPOINT_CHECK_CUDA(cudaMemsetAsync(
                    group_counts.data(), 0, static_cast<std::size_t>(point_count) * sizeof(plamatrix::Index), stream));
                const int grid_size =
                    point_count / kVoxelGroupingBlockSize + (point_count % kVoxelGroupingBlockSize != 0 ? 1 : 0);
                makeVoxelKeysAndIndicesKernel<<<grid_size, kVoxelGroupingBlockSize, 0, stream>>>(points,
                                                                                                 point_count,
                                                                                                 origin_x,
                                                                                                 origin_y,
                                                                                                 origin_z,
                                                                                                 cell_x,
                                                                                                 cell_y,
                                                                                                 cell_z,
                                                                                                 keys.data(),
                                                                                                 indices.data(),
                                                                                                 key_error.get());
                PLAPOINT_CHECK_CUDA(cudaGetLastError());

                plamatrix::internal::GroupingWorkspace grouping_workspace;
                plamatrix::internal::sortByKeyAsync(std::as_const(keys).template view<plamatrix::internal::Device::GPU>(),
                                          std::as_const(indices).template view<plamatrix::internal::Device::GPU>(),
                                          sorted_keys.template view<plamatrix::internal::Device::GPU>(),
                                          sorted_indices.template view<plamatrix::internal::Device::GPU>(),
                                          grouping_workspace,
                                          stream);
                // Stream ordering makes the original key/index inputs reusable after sorting: RLE writes
                // unique keys back into keys, and the scan writes group offsets back into indices.
                plamatrix::internal::runLengthEncodeAsync(std::as_const(sorted_keys).template view<plamatrix::internal::Device::GPU>(),
                                                keys.template view<plamatrix::internal::Device::GPU>(),
                                                group_counts.template view<plamatrix::internal::Device::GPU>(),
                                                run_count.template view<plamatrix::internal::Device::GPU>(),
                                                grouping_workspace,
                                                stream);

                plamatrix::internal::IndexingWorkspace scan_workspace;
                plamatrix::internal::exclusiveScanAsync(std::as_const(group_counts).template view<plamatrix::internal::Device::GPU>(),
                                              indices.template view<plamatrix::internal::Device::GPU>(),
                                              scan_workspace,
                                              stream);
                PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                    &metadata->groupCount, run_count.data(), sizeof(plamatrix::Index), cudaMemcpyDeviceToHost, stream));
                PLAPOINT_CHECK_CUDA(
                    cudaMemcpyAsync(&metadata->keyError, key_error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
                grouping_workspace.closeAsyncAllocation();
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
                scan_workspace.checkStatus("voxel group offsets");
                scan_workspace.closeAsyncAllocation();

                if (metadata->keyError == 1)
                {
                    throw std::invalid_argument(std::string(operation) + ": points must be finite");
                }
                if (metadata->keyError == 2)
                {
                    throw std::out_of_range(std::string(operation) + ": voxel index is outside int range");
                }
                if (metadata->keyError != 0)
                {
                    throw std::runtime_error(std::string(operation) + ": invalid voxel key error code");
                }
                if (metadata->groupCount <= 0 || metadata->groupCount > matrix_count ||
                    metadata->groupCount > static_cast<plamatrix::Index>(std::numeric_limits<int>::max()))
                {
                    throw std::runtime_error(std::string(operation) + ": PlaMatrix returned an invalid group count");
                }

                const int group_count = static_cast<int>(metadata->groupCount);
                writeVoxelCentroidsAndRemapKernel<<<group_count, kVoxelGroupingBlockSize, 0, stream>>>(
                    points,
                    point_count,
                    sorted_indices.data(),
                    indices.data(),
                    group_counts.data(),
                    group_count,
                    output,
                    point_remap);
                PLAPOINT_CHECK_CUDA(cudaGetLastError());
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
                return group_count;
            }

        } // namespace detail
    } // namespace gpu
} // namespace plapoint
