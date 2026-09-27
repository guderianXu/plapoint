#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_cluster.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/ops/statistics.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/ops/reduction.h>

#include "voxel_grouping_gpu_kernels.cuh"

namespace plapoint
{
    namespace
    {

        template <typename Scalar>
        std::array<Scalar, 3> finitePointMinimum(const plamatrix::internal::ResidentMatrix<Scalar>& points, cudaStream_t stream)
        {
            auto minimum = plamatrix::internal::ResidentMatrix<Scalar>(1, 3, points.context());
            auto maximum = plamatrix::internal::ResidentMatrix<Scalar>(1, 3, points.context());
            auto valid_count = plamatrix::internal::ResidentMatrix<plamatrix::Index>(1, 1, points.context());
            plamatrix::internal::ReductionWorkspace workspace;
            plamatrix::internal::finiteColumnBoundsAsync(points.template view<plamatrix::internal::Device::GPU>(),
                                               minimum.template view<plamatrix::internal::Device::GPU>(),
                                               maximum.template view<plamatrix::internal::Device::GPU>(),
                                               valid_count.template view<plamatrix::internal::Device::GPU>(),
                                               workspace,
                                               stream);

            workspace.closeAsyncAllocation();
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            const auto host_minimum = minimum.toHostMatrix();
            const auto host_valid_count = valid_count.toHostMatrix();

            if (host_valid_count(0, 0) != points.rows())
            {
                throw std::invalid_argument("voxelClusterSimplify GPU: points must be finite");
            }
            return {
                host_minimum(0, 0),
                host_minimum(0, 1),
                host_minimum(0, 2),
            };
        }

        template <typename Scalar>
        int voxelClusterSimplifyColumnMajor(const Scalar* d_points,
                                            int point_count,
                                            Scalar cluster_size,
                                            Scalar min_x,
                                            Scalar min_y,
                                            Scalar min_z,
                                            Scalar* d_out_points,
                                            int* d_point_remap,
                                            plamatrix::internal::ExecutionContext& context,
                                            cudaStream_t stream)
        {
            if (point_count <= 0)
            {
                return 0;
            }
            if (!d_points || !d_out_points || !d_point_remap)
            {
                throw std::invalid_argument("voxelClusterSimplify GPU: device pointers must not be null");
            }
            if (!std::isfinite(cluster_size) || cluster_size <= Scalar(0))
            {
                throw std::invalid_argument("voxelClusterSimplify GPU: cluster size must be finite and positive");
            }

            return gpu::detail::groupVoxelCentroidsColumnMajor(d_points,
                                                               point_count,
                                                               min_x,
                                                               min_y,
                                                               min_z,
                                                               cluster_size,
                                                               cluster_size,
                                                               cluster_size,
                                                               d_out_points,
                                                               d_point_remap,
                                                               "voxelClusterSimplify GPU",
                                                               context,
                                                               stream);
        }

        plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> facesToMatrix(const std::vector<std::array<int, 3>>& faces)
        {
            plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> matrix(static_cast<plamatrix::Index>(faces.size()), 3);
            for (std::size_t row = 0; row < faces.size(); ++row)
            {
                matrix.operator()(static_cast<plamatrix::Index>(row), 0) = faces[row][0];
                matrix.operator()(static_cast<plamatrix::Index>(row), 1) = faces[row][1];
                matrix.operator()(static_cast<plamatrix::Index>(row), 2) = faces[row][2];
            }
            return matrix;
        }

        std::vector<int> copyPointRemapToCpu(const plamatrix::internal::ResidentMatrix<int>& point_remap_gpu,
                                             std::size_t point_count)
        {
            if (point_remap_gpu.cols() != 1 || point_remap_gpu.rows() < static_cast<plamatrix::Index>(point_count))
            {
                throw std::invalid_argument("voxelClusterSimplify GPU: point remap matrix has invalid shape");
            }
            const auto point_remap_cpu = point_remap_gpu.toHostMatrix();
            std::vector<int> point_remap(point_count);
            for (std::size_t row = 0; row < point_count; ++row)
            {
                point_remap[row] = point_remap_cpu(static_cast<plamatrix::Index>(row), 0);
            }
            return point_remap;
        }

        std::vector<int> computeClusterCounts(const std::vector<int>& point_remap, int cluster_count)
        {
            std::vector<int> cluster_counts(static_cast<std::size_t>(cluster_count), 0);
            for (int cluster_index : point_remap)
            {
                if (cluster_index < 0 || cluster_index >= cluster_count)
                {
                    throw std::runtime_error("voxelClusterSimplify GPU: point remap index out of range");
                }
                ++cluster_counts[static_cast<std::size_t>(cluster_index)];
            }
            return cluster_counts;
        }

        template <typename Attribute> Attribute roundedAttribute(long double value)
        {
            const long rounded = std::lround(value);
            const long lo = static_cast<long>(std::numeric_limits<Attribute>::min());
            const long hi = static_cast<long>(std::numeric_limits<Attribute>::max());
            return static_cast<Attribute>(std::clamp(rounded, lo, hi));
        }

        template <typename Scalar>
        void setAveragedColors(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& mesh,
                               const std::vector<int>& point_remap,
                               const std::vector<int>& cluster_counts,
                               plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& output)
        {
            if (!mesh.hasColors())
            {
                return;
            }

            const auto colors_cpu = mesh.colors()->toHostMatrix();
            plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(static_cast<plamatrix::Index>(cluster_counts.size()), 3);
            std::vector<long double> sums(cluster_counts.size() * 3u, 0.0L);
            for (std::size_t row = 0; row < point_remap.size(); ++row)
            {
                const auto cluster = static_cast<std::size_t>(point_remap[row]);
                for (int col = 0; col < 3; ++col)
                {
                    sums[cluster * 3u + static_cast<std::size_t>(col)] +=
                        static_cast<long double>(colors_cpu.operator()(static_cast<plamatrix::Index>(row), col));
                }
            }

            for (std::size_t row = 0; row < cluster_counts.size(); ++row)
            {
                const long double inv = 1.0L / static_cast<long double>(cluster_counts[row]);
                for (int col = 0; col < 3; ++col)
                {
                    colors.operator()(static_cast<plamatrix::Index>(row), col) =
                        roundedAttribute<std::uint8_t>(sums[row * 3u + static_cast<std::size_t>(col)] * inv);
                }
            }

            output.setColors(plamatrix::internal::ResidentMatrix<std::uint8_t>::copyFrom(colors, output.executionContext()));
        }

        template <typename Scalar>
        void setAveragedIntensities(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& mesh,
                                    const std::vector<int>& point_remap,
                                    const std::vector<int>& cluster_counts,
                                    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& output)
        {
            if (!mesh.hasIntensities())
            {
                return;
            }

            const auto intensities_cpu = mesh.intensities()->toHostMatrix();
            plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(static_cast<plamatrix::Index>(cluster_counts.size()), 1);
            std::vector<long double> sums(cluster_counts.size(), 0.0L);
            for (std::size_t row = 0; row < point_remap.size(); ++row)
            {
                const auto cluster = static_cast<std::size_t>(point_remap[row]);
                sums[cluster] +=
                    static_cast<long double>(intensities_cpu.operator()(static_cast<plamatrix::Index>(row), 0));
            }

            for (std::size_t row = 0; row < cluster_counts.size(); ++row)
            {
                const long double inv = 1.0L / static_cast<long double>(cluster_counts[row]);
                intensities.operator()(static_cast<plamatrix::Index>(row), 0) =
                    roundedAttribute<std::uint16_t>(sums[row] * inv);
            }

            output.setIntensities(
                plamatrix::internal::ResidentMatrix<std::uint16_t>::copyFrom(intensities, output.executionContext()));
        }

        template <typename Scalar>
        void setRemappedFaces(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& mesh,
                              const std::vector<int>& point_remap,
                              plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& output)
        {
            std::vector<std::array<int, 3>> simplified_faces;
            if (mesh.hasFaces())
            {
                const auto faces_cpu = mesh.faces()->toHostMatrix();
                simplified_faces.reserve(static_cast<std::size_t>(faces_cpu.rows()));
                for (plamatrix::Index row = 0; row < faces_cpu.rows(); ++row)
                {
                    const int old_a = faces_cpu.operator()(row, 0);
                    const int old_b = faces_cpu.operator()(row, 1);
                    const int old_c = faces_cpu.operator()(row, 2);
                    if (old_a < 0 || old_b < 0 || old_c < 0 || static_cast<std::size_t>(old_a) >= point_remap.size() ||
                        static_cast<std::size_t>(old_b) >= point_remap.size() ||
                        static_cast<std::size_t>(old_c) >= point_remap.size())
                    {
                        throw std::out_of_range("voxelClusterSimplify GPU: face index out of range");
                    }

                    const int a = point_remap[static_cast<std::size_t>(old_a)];
                    const int b = point_remap[static_cast<std::size_t>(old_b)];
                    const int c = point_remap[static_cast<std::size_t>(old_c)];
                    if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c)
                    {
                        continue;
                    }
                    simplified_faces.push_back({a, b, c});
                }
            }

            output.setFaces(
                plamatrix::internal::ResidentMatrix<int>::copyFrom(facesToMatrix(simplified_faces), output.executionContext()));
        }

        template <typename Scalar>
        plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>
        makeEmptyOutput(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& mesh)
        {
            plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> output(
                plamatrix::internal::ResidentMatrix<Scalar>(0, 3, mesh.executionContext()), mesh.executionContext());
            if (mesh.hasColors())
            {
                output.setColors(plamatrix::internal::ResidentMatrix<std::uint8_t>(0, 3, mesh.executionContext()));
            }
            if (mesh.hasIntensities())
            {
                output.setIntensities(plamatrix::internal::ResidentMatrix<std::uint16_t>(0, 1, mesh.executionContext()));
            }
            output.setFaces(plamatrix::internal::ResidentMatrix<int>(0, 3, mesh.executionContext()));
            return output;
        }

        template <typename Scalar>
        plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>
        voxelClusterSimplifyGpuImpl(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& mesh,
                                    Scalar cluster_size)
        {
            mesh.validate();
            if (!std::isfinite(cluster_size) || cluster_size <= Scalar(0))
            {
                throw std::invalid_argument("voxelClusterSimplify: cluster size must be finite and positive");
            }

            const std::size_t n = mesh.size();
            if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::overflow_error("voxelClusterSimplify GPU: point count exceeds int range");
            }
            if (n == 0)
            {
                return makeEmptyOutput(mesh);
            }

            const auto minimum = finitePointMinimum(mesh.points(), 0);
            plamatrix::internal::ResidentMatrix<Scalar> centroid_storage(
                static_cast<plamatrix::Index>(n), 3, mesh.executionContext());
            plamatrix::internal::ResidentMatrix<int> point_remap_gpu(
                static_cast<plamatrix::Index>(n), 1, mesh.executionContext());
            const int cluster_count = voxelClusterSimplifyColumnMajor(mesh.points().data(),
                                                                      static_cast<int>(n),
                                                                      cluster_size,
                                                                      minimum[0],
                                                                      minimum[1],
                                                                      minimum[2],
                                                                      centroid_storage.data(),
                                                                      point_remap_gpu.data(),
                                                                      *mesh.executionContext(),
                                                                      0);

            plamatrix::internal::ResidentMatrix<Scalar> points(
                static_cast<plamatrix::Index>(cluster_count), 3, mesh.executionContext());
            PLAPOINT_CHECK_CUDA(cudaMemcpy(points.data(),
                                           centroid_storage.data(),
                                           static_cast<std::size_t>(cluster_count) * 3u * sizeof(Scalar),
                                           cudaMemcpyDeviceToDevice));
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));

            plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> output(std::move(points), mesh.executionContext());
            const auto point_remap = copyPointRemapToCpu(point_remap_gpu, n);
            const auto cluster_counts = computeClusterCounts(point_remap, cluster_count);
            setAveragedColors(mesh, point_remap, cluster_counts, output);
            setAveragedIntensities(mesh, point_remap, cluster_counts, output);
            setRemappedFaces(mesh, point_remap, output);
            return output;
        }

    } // namespace

    namespace mesh
    {

        plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>
        voxelClusterSimplify(const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& mesh, float cluster_size)
        {
            return voxelClusterSimplifyGpuImpl(mesh, cluster_size);
        }

        plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>
        voxelClusterSimplify(const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& mesh, double cluster_size)
        {
            return voxelClusterSimplifyGpuImpl(mesh, cluster_size);
        }

    } // namespace mesh
} // namespace plapoint
