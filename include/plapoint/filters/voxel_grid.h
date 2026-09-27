#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/core/point_cloud_bridge.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_grid.h>
#endif
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/core/device.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace plapoint
{

    template <typename Scalar,
              plamatrix::internal::Device Dev = plamatrix::internal::Device::CPU,
              typename Enable = void>
    class VoxelGrid : public Filter<Scalar, Dev>
    {
    public:
        using PointCloudType = plapoint::internal::DeviceCloud<Scalar, Dev>;

        void setLeafSize(Scalar lx, Scalar ly, Scalar lz)
        {
            if (!std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz) || lx <= 0 || ly <= 0 || lz <= 0)
            {
                throw std::invalid_argument("VoxelGrid: leaf size must be positive");
            }
            _leaf_x = lx;
            _leaf_y = ly;
            _leaf_z = lz;
        }

        /// Set XYZ leaf sizes from a PlaMatrix vector matching PCL's four-component overload.
        /// The fourth component is unused because this filter indexes only XYZ coordinates.
        void setLeafSize(const plamatrix::Vector4f& leaf_size)
        {
            setLeafSize(static_cast<Scalar>(leaf_size(0)),
                        static_cast<Scalar>(leaf_size(1)),
                        static_cast<Scalar>(leaf_size(2)));
        }

        /// Return XYZ leaf sizes for the matrix-backed filter.
        plamatrix::Vector3f getLeafSize() const
        {
            return plamatrix::Vector3f(
                static_cast<float>(_leaf_x), static_cast<float>(_leaf_y), static_cast<float>(_leaf_z));
        }

    protected:
        void applyFilter(PointCloudType& output) override
        {
            if (!this->_input)
                return;
            if constexpr (Dev == plamatrix::internal::Device::GPU)
            {
#ifndef PLAPOINT_WITH_CUDA
                throw std::runtime_error("PlaPoint was built without CUDA support");
#else
                applyFilterGpu(output);
                return;
#endif
            }
            else
            {
                const auto& cpu_points = this->_input->pointsCpu();

                struct Accum
                {
                    long double mean_x = 0, mean_y = 0, mean_z = 0;
                    long double mean_nx = 0, mean_ny = 0, mean_nz = 0;
                    long double mean_r = 0, mean_g = 0, mean_b = 0;
                    long double mean_intensity = 0;
                    std::vector<long double> mean_scalar_fields;
                    int count = 0;
                };
                std::unordered_map<VoxelKey, Accum, VoxelKeyHash> voxels;
                voxels.reserve(this->_input->size());
                const bool have_normals = this->_input->hasNormals();
                const bool have_colors = this->_input->hasColors();
                const bool have_intensities = this->_input->hasIntensities();
                const bool have_scalar_fields = this->_input->hasScalarFields();
                const auto* input_normals = this->_input->normals();
                const auto* input_colors = this->_input->colors();
                const auto* input_intensities = this->_input->intensities();
                const auto* input_scalar_fields = this->_input->scalarFields();
                const auto scalar_field_count = static_cast<plamatrix::Index>(this->_input->scalarFieldNames().size());

                for (std::size_t i = 0; i < this->_input->size(); ++i)
                {
                    const auto row = static_cast<plamatrix::Index>(i);
                    const Scalar x = cpu_points(row, 0);
                    const Scalar y = cpu_points(row, 1);
                    const Scalar z = cpu_points(row, 2);
                    VoxelKey key{
                        checkedVoxelIndex(x, _leaf_x), checkedVoxelIndex(y, _leaf_y), checkedVoxelIndex(z, _leaf_z)};
                    auto& acc = voxels[key];
                    acc.count += 1;
                    updateMean(acc.mean_x, static_cast<long double>(x), acc.count);
                    updateMean(acc.mean_y, static_cast<long double>(y), acc.count);
                    updateMean(acc.mean_z, static_cast<long double>(z), acc.count);
                    if (have_normals)
                    {
                        updateMean(acc.mean_nx, static_cast<long double>(input_normals->operator()(row, 0)), acc.count);
                        updateMean(acc.mean_ny, static_cast<long double>(input_normals->operator()(row, 1)), acc.count);
                        updateMean(acc.mean_nz, static_cast<long double>(input_normals->operator()(row, 2)), acc.count);
                    }
                    if (have_colors)
                    {
                        updateMean(acc.mean_r, static_cast<long double>(input_colors->operator()(row, 0)), acc.count);
                        updateMean(acc.mean_g, static_cast<long double>(input_colors->operator()(row, 1)), acc.count);
                        updateMean(acc.mean_b, static_cast<long double>(input_colors->operator()(row, 2)), acc.count);
                    }
                    if (have_intensities)
                    {
                        updateMean(acc.mean_intensity,
                                   static_cast<long double>(input_intensities->operator()(row, 0)),
                                   acc.count);
                    }
                    if (have_scalar_fields)
                    {
                        if (acc.mean_scalar_fields.empty())
                        {
                            acc.mean_scalar_fields.assign(static_cast<std::size_t>(scalar_field_count), 0.0L);
                        }
                        for (plamatrix::Index c = 0; c < scalar_field_count; ++c)
                        {
                            updateMean(acc.mean_scalar_fields[static_cast<std::size_t>(c)],
                                       static_cast<long double>(input_scalar_fields->operator()(row, c)),
                                       acc.count);
                        }
                    }
                }

                plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> pts(
                    static_cast<plamatrix::Index>(voxels.size()), 3);
                std::unique_ptr<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>> normals;
                std::unique_ptr<plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>> colors;
                std::unique_ptr<plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>> intensities;
                std::unique_ptr<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>> scalar_fields;
                if (have_normals)
                {
                    normals = std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                        static_cast<plamatrix::Index>(voxels.size()), 3);
                }
                if (have_colors)
                {
                    colors = std::make_unique<plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>>(
                        static_cast<plamatrix::Index>(voxels.size()), 3);
                }
                if (have_intensities)
                {
                    intensities =
                        std::make_unique<plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>>(
                            static_cast<plamatrix::Index>(voxels.size()), 1);
                }
                if (have_scalar_fields)
                {
                    scalar_fields = std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                        static_cast<plamatrix::Index>(voxels.size()), scalar_field_count);
                }
                std::vector<VoxelKey> ordered_keys;
                ordered_keys.reserve(voxels.size());
                for (const auto& kv : voxels)
                {
                    ordered_keys.push_back(kv.first);
                }
                std::sort(ordered_keys.begin(), ordered_keys.end());

                int out_idx = 0;
                for (const auto& key : ordered_keys)
                {
                    const auto& acc = voxels.at(key);
                    pts(out_idx, 0) = checkedCentroid(acc.mean_x);
                    pts(out_idx, 1) = checkedCentroid(acc.mean_y);
                    pts(out_idx, 2) = checkedCentroid(acc.mean_z);
                    if (normals)
                    {
                        (*normals)(out_idx, 0) = checkedCentroid(acc.mean_nx);
                        (*normals)(out_idx, 1) = checkedCentroid(acc.mean_ny);
                        (*normals)(out_idx, 2) = checkedCentroid(acc.mean_nz);
                    }
                    if (colors)
                    {
                        (*colors)(out_idx, 0) = roundedAttribute<std::uint8_t>(acc.mean_r);
                        (*colors)(out_idx, 1) = roundedAttribute<std::uint8_t>(acc.mean_g);
                        (*colors)(out_idx, 2) = roundedAttribute<std::uint8_t>(acc.mean_b);
                    }
                    if (intensities)
                    {
                        (*intensities)(out_idx, 0) = roundedAttribute<std::uint16_t>(acc.mean_intensity);
                    }
                    if (scalar_fields)
                    {
                        for (plamatrix::Index c = 0; c < scalar_field_count; ++c)
                        {
                            (*scalar_fields)(out_idx, c) =
                                checkedCentroid(acc.mean_scalar_fields[static_cast<std::size_t>(c)]);
                        }
                    }
                    ++out_idx;
                }
                output = this->makeOutputCloud(std::move(pts));
                setOutputAttributes(output, normals.get(), colors.get(), intensities.get(), scalar_fields.get());
            }
        }

    private:
#ifdef PLAPOINT_WITH_CUDA
        template <plamatrix::internal::Device D = Dev>
        std::enable_if_t<D == plamatrix::internal::Device::GPU, void> applyFilterGpu(PointCloudType& output)
        {
            if (this->_input->hasNormals() || this->_input->hasColors() || this->_input->hasIntensities() ||
                this->_input->hasScalarFields())
            {
                auto cpu_input =
                    std::make_shared<plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>>(
                        this->_input->toCpu());
                VoxelGrid<Scalar, plamatrix::internal::Device::CPU> cpu_filter;
                cpu_filter.setInputCloud(cpu_input);
                cpu_filter.setLeafSize(_leaf_x, _leaf_y, _leaf_z);
                plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> cpu_output;
                cpu_filter.filter(cpu_output);
                output = cpu_output.toGpu(this->_input->executionContext());
                return;
            }

            const std::size_t n = this->_input->size();
            if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::overflow_error("VoxelGrid GPU: point count exceeds int range");
            }
            const int point_count = static_cast<int>(n);
            if (point_count == 0)
            {
                output =
                    PointCloudType(plamatrix::internal::ResidentMatrix<Scalar>(0, 3, this->_input->executionContext()),
                                   this->_input->executionContext());
                return;
            }

            plamatrix::internal::ResidentMatrix<Scalar> centroid_storage(
                static_cast<plamatrix::Index>(n), 3, this->_input->executionContext());
            const int centroid_count = gpu::voxelGridDownsampleColumnMajor(
                this->_input->points(), _leaf_x, _leaf_y, _leaf_z, centroid_storage);

            plamatrix::internal::ResidentMatrix<Scalar> points(
                static_cast<plamatrix::Index>(centroid_count), 3, this->_input->executionContext());
            if (centroid_count > 0)
            {
                PLAPOINT_CHECK_CUDA(cudaMemcpy(points.data(),
                                               centroid_storage.data(),
                                               static_cast<std::size_t>(centroid_count) * 3u * sizeof(Scalar),
                                               cudaMemcpyDeviceToDevice));
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
            }
            output = PointCloudType(std::move(points), this->_input->executionContext());
        }
#endif

        struct VoxelKey
        {
            int x;
            int y;
            int z;

            bool operator==(const VoxelKey& other) const
            {
                return x == other.x && y == other.y && z == other.z;
            }

            bool operator<(const VoxelKey& other) const
            {
                if (x != other.x)
                    return x < other.x;
                if (y != other.y)
                    return y < other.y;
                return z < other.z;
            }
        };

        struct VoxelKeyHash
        {
            std::size_t operator()(const VoxelKey& key) const
            {
                std::size_t seed = static_cast<std::size_t>(key.x) + 0x9e3779b97f4a7c15ULL;
                seed ^= static_cast<std::size_t>(key.y) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
                seed ^= static_cast<std::size_t>(key.z) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
                return seed;
            }
        };

        static int checkedVoxelIndex(Scalar coordinate, Scalar leaf)
        {
            if (!std::isfinite(coordinate))
            {
                throw std::invalid_argument("VoxelGrid: points must be finite");
            }
            const double scaled = std::floor(static_cast<double>(coordinate) / static_cast<double>(leaf));
            if (!std::isfinite(scaled) || scaled < static_cast<double>(std::numeric_limits<int>::min()) ||
                scaled > static_cast<double>(std::numeric_limits<int>::max()))
            {
                throw std::out_of_range("VoxelGrid: voxel index is outside int range");
            }
            return static_cast<int>(scaled);
        }

        static Scalar checkedCentroid(long double centroid)
        {
            if (!std::isfinite(centroid) || centroid < -static_cast<long double>(std::numeric_limits<Scalar>::max()) ||
                centroid > static_cast<long double>(std::numeric_limits<Scalar>::max()))
            {
                throw std::out_of_range("VoxelGrid: centroid is outside scalar range");
            }
            return static_cast<Scalar>(centroid);
        }

        static void updateMean(long double& mean, long double value, int count)
        {
            const long double weight = 1.0L / static_cast<long double>(count);
            mean += (value - mean) * weight;
        }

        template <typename Attribute> static Attribute roundedAttribute(long double value)
        {
            if (!std::isfinite(value))
            {
                throw std::out_of_range("VoxelGrid: attribute mean is not finite");
            }
            const long double rounded = std::round(value);
            const long double lo = static_cast<long double>(std::numeric_limits<Attribute>::min());
            const long double hi = static_cast<long double>(std::numeric_limits<Attribute>::max());
            return static_cast<Attribute>(std::clamp(rounded, lo, hi));
        }

        void setOutputAttributes(PointCloudType& output,
                                 plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>* normals,
                                 plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>* colors,
                                 plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>* intensities,
                                 plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>* scalar_fields) const
        {
            if (normals)
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    output.setNormals(std::move(*normals));
                }
                else
                {
                    output.setNormals(
                        plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(*normals, *output.executionContext()));
                }
            }
            if (colors)
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    output.setColors(std::move(*colors));
                }
                else
                {
                    output.setColors(plamatrix::internal::ResidentMatrix<std::uint8_t>::copyFrom(
                        *colors, *output.executionContext()));
                }
            }
            if (intensities)
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    output.setIntensities(std::move(*intensities));
                }
                else
                {
                    output.setIntensities(plamatrix::internal::ResidentMatrix<std::uint16_t>::copyFrom(
                        *intensities, *output.executionContext()));
                }
            }
            if (scalar_fields)
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    output.setScalarFields(this->_input->scalarFieldNames(), std::move(*scalar_fields));
                }
                else
                {
                    output.setScalarFields(this->_input->scalarFieldNames(),
                                           plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(
                                               *scalar_fields, *output.executionContext()));
                }
            }
        }

        Scalar _leaf_x = 1;
        Scalar _leaf_y = 1;
        Scalar _leaf_z = 1;
    };

    namespace detail
    {

        template <typename T, typename = void> struct HasColorMembers : std::false_type
        {
        };

        template <typename T>
        struct HasColorMembers<T, std::void_t<decltype(T::r), decltype(T::g), decltype(T::b), decltype(T::a)>>
            : std::true_type
        {
        };

        template <typename PointT, typename FilterT> class PointVoxelGridAdapter : public Filter<PointT>
        {
        public:
            using Scalar = std::decay_t<decltype(PointT::x)>;
            using PointCloudType = PointCloud<PointT>;
            using PointCloudConstPtr = typename PointCloudType::ConstPtr;
            using Ptr = std::shared_ptr<FilterT>;
            using ConstPtr = std::shared_ptr<const FilterT>;

            PointVoxelGridAdapter()
            {
                this->filter_name_ = "VoxelGrid";
            }

            void setLeafSize(float lx, float ly, float lz)
            {
                if (!std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz) || lx <= 0.0f || ly <= 0.0f ||
                    lz <= 0.0f)
                {
                    throw std::invalid_argument("VoxelGrid: leaf size must be positive and finite");
                }
                _leaf_size = Eigen::Vector4f(lx, ly, lz, 1.0f);
                _inverse_leaf_size = _leaf_size.array().inverse();
            }

            void setLeafSize(const Eigen::Vector4f& leaf_size)
            {
                setLeafSize(leaf_size(0), leaf_size(1), leaf_size(2));
            }

            Eigen::Vector3f getLeafSize() const
            {
                return _leaf_size.head<3>();
            }
            void setDownsampleAllData(bool downsample)
            {
                _downsample_all_data = downsample;
            }
            bool getDownsampleAllData() const
            {
                return _downsample_all_data;
            }
            void setMinimumPointsNumberPerVoxel(unsigned int count)
            {
                _min_points_per_voxel = count;
            }
            unsigned int getMinimumPointsNumberPerVoxel() const
            {
                return _min_points_per_voxel;
            }
            void setSaveLeafLayout(bool save)
            {
                _save_leaf_layout = save;
            }
            bool getSaveLeafLayout() const
            {
                return _save_leaf_layout;
            }
            Eigen::Vector3i getMinBoxCoordinates() const
            {
                return _min_box.head<3>();
            }
            Eigen::Vector3i getMaxBoxCoordinates() const
            {
                return _max_box.head<3>();
            }
            Eigen::Vector3i getNrDivisions() const
            {
                return _divisions.head<3>();
            }
            Eigen::Vector3i getDivisionMultiplier() const
            {
                return _division_multiplier.head<3>();
            }

            int getCentroidIndex(const PointT& point) const
            {
                return getCentroidIndexAt(getGridCoordinates(
                    static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z)));
            }

            std::vector<int> getNeighborCentroidIndices(const PointT& reference_point,
                                                        const Eigen::MatrixXi& relative_coordinates) const
            {
                const Eigen::Vector3i reference = getGridCoordinates(static_cast<float>(reference_point.x),
                                                                     static_cast<float>(reference_point.y),
                                                                     static_cast<float>(reference_point.z));
                std::vector<int> neighbors(static_cast<std::size_t>(relative_coordinates.cols()), -1);
                for (Eigen::Index column = 0; column < relative_coordinates.cols(); ++column)
                {
                    neighbors[static_cast<std::size_t>(column)] =
                        getCentroidIndexAt(reference + relative_coordinates.col(column));
                }
                return neighbors;
            }

            std::vector<int> getLeafLayout() const
            {
                return _leaf_layout;
            }

            Eigen::Vector3i getGridCoordinates(float x, float y, float z) const
            {
                return Eigen::Vector3i(checkedGridCoordinate(x, _inverse_leaf_size(0)),
                                       checkedGridCoordinate(y, _inverse_leaf_size(1)),
                                       checkedGridCoordinate(z, _inverse_leaf_size(2)));
            }

            int getCentroidIndexAt(const Eigen::Vector3i& coordinates) const
            {
                if (_leaf_layout.empty() || (coordinates.array() < _min_box.head<3>().array()).any() ||
                    (coordinates.array() > _max_box.head<3>().array()).any())
                {
                    return -1;
                }
                const Eigen::Vector3i local = coordinates - _min_box.head<3>();
                const std::int64_t index = static_cast<std::int64_t>(local(0)) * _division_multiplier(0) +
                                           static_cast<std::int64_t>(local(1)) * _division_multiplier(1) +
                                           static_cast<std::int64_t>(local(2)) * _division_multiplier(2);
                return index < 0 || static_cast<std::size_t>(index) >= _leaf_layout.size()
                           ? -1
                           : _leaf_layout[static_cast<std::size_t>(index)];
            }

            void setFilterFieldName(const std::string& name)
            {
                _filter_field_name = name;
            }
            std::string getFilterFieldName() const
            {
                return _filter_field_name;
            }
            void setFilterLimits(const double& minimum, const double& maximum)
            {
                _filter_limit_min = minimum;
                _filter_limit_max = maximum;
            }
            void getFilterLimits(double& minimum, double& maximum) const
            {
                minimum = _filter_limit_min;
                maximum = _filter_limit_max;
            }
            void setFilterLimitsNegative(bool negative)
            {
                _filter_limit_negative = negative;
            }
            void getFilterLimitsNegative(bool& negative) const
            {
                negative = _filter_limit_negative;
            }
            bool getFilterLimitsNegative() const
            {
                return _filter_limit_negative;
            }

        protected:
            void applyFilter(PointCloudType& output) override
            {
                using Key = std::array<std::int64_t, 3>;
                std::map<Key, Indices> voxels;
                bool initialized_bounds = false;
                Eigen::Vector3i min_box = Eigen::Vector3i::Zero();
                Eigen::Vector3i max_box = Eigen::Vector3i::Zero();

                for (const int index : *this->indices_)
                {
                    if (index < 0 || static_cast<std::size_t>(index) >= this->input_->size())
                    {
                        throw std::out_of_range("VoxelGrid: input index is outside the cloud");
                    }
                    const auto& point = this->input_->points[static_cast<std::size_t>(index)];
                    if (!finitePoint(point) || !passesFieldFilter(point))
                    {
                        continue;
                    }
                    const Eigen::Vector3i grid(
                        checkedGridCoordinate(static_cast<double>(point.x), _inverse_leaf_size(0)),
                        checkedGridCoordinate(static_cast<double>(point.y), _inverse_leaf_size(1)),
                        checkedGridCoordinate(static_cast<double>(point.z), _inverse_leaf_size(2)));
                    if (!initialized_bounds)
                    {
                        min_box = max_box = grid;
                        initialized_bounds = true;
                    }
                    else
                    {
                        min_box = min_box.cwiseMin(grid);
                        max_box = max_box.cwiseMax(grid);
                    }
                    voxels[{grid(0), grid(1), grid(2)}].push_back(index);
                }

                output.clear();
                output.is_dense = true;
                resetLayout();
                if (!initialized_bounds)
                {
                    return;
                }
                configureLayout(min_box, max_box);
                output.reserve(voxels.size());
                for (const auto& entry : voxels)
                {
                    if (entry.second.size() < _min_points_per_voxel)
                    {
                        continue;
                    }
                    PointT centroid{};
                    AverageFields average{*this->input_, entry.second, centroid, _downsample_all_data};
                    for_each_type<typename traits::fieldList<PointT>::type>(average);
                    if (_downsample_all_data)
                    {
                        averageColor(*this->input_, entry.second, centroid);
                    }
                    const int centroid_index = static_cast<int>(output.size());
                    output.push_back(centroid);
                    if (_save_leaf_layout)
                    {
                        const Eigen::Vector3i coordinates(static_cast<int>(entry.first[0]),
                                                          static_cast<int>(entry.first[1]),
                                                          static_cast<int>(entry.first[2]));
                        const Eigen::Vector3i local = coordinates - _min_box.head<3>();
                        const std::int64_t layout_index =
                            static_cast<std::int64_t>(local(0)) * _division_multiplier(0) +
                            static_cast<std::int64_t>(local(1)) * _division_multiplier(1) +
                            static_cast<std::int64_t>(local(2)) * _division_multiplier(2);
                        _leaf_layout[static_cast<std::size_t>(layout_index)] = centroid_index;
                    }
                }
            }

        private:
            struct AverageFields
            {
                const PointCloudType& cloud;
                const Indices& indices;
                PointT& output;
                bool all_data;

                template <typename Tag> void operator()()
                {
                    constexpr bool coordinate = std::is_same_v<Tag, fields::x> || std::is_same_v<Tag, fields::y> ||
                                                std::is_same_v<Tag, fields::z>;
                    constexpr bool packed_color = std::is_same_v<Tag, fields::rgb> || std::is_same_v<Tag, fields::rgba>;
                    if constexpr (!packed_color)
                    {
                        if (!coordinate && !all_data)
                        {
                            return;
                        }
                        using Field = typename traits::datatype<PointT, Tag>::type;
                        using Value = std::remove_all_extents_t<Field>;
                        constexpr std::size_t count = traits::datatype<PointT, Tag>::size;
                        auto* destination = reinterpret_cast<Value*>(reinterpret_cast<std::uint8_t*>(&output) +
                                                                     traits::offset<PointT, Tag>::value);
                        for (std::size_t component = 0; component < count; ++component)
                        {
                            long double sum = 0.0L;
                            for (const int index : indices)
                            {
                                const auto& point = cloud.points[static_cast<std::size_t>(index)];
                                const auto* source = reinterpret_cast<const Value*>(
                                    reinterpret_cast<const std::uint8_t*>(&point) + traits::offset<PointT, Tag>::value);
                                sum += static_cast<long double>(source[component]);
                            }
                            destination[component] = static_cast<Value>(sum / indices.size());
                        }
                    }
                }
            };

            struct FieldValueReader
            {
                const PointT& point;
                const std::string& field_name;
                bool& found;
                double& value;

                template <typename Tag> void operator()()
                {
                    if (found || field_name != traits::name<PointT, Tag>::value)
                    {
                        return;
                    }
                    using Field = typename traits::datatype<PointT, Tag>::type;
                    using Value = std::remove_all_extents_t<Field>;
                    const auto* source = reinterpret_cast<const Value*>(reinterpret_cast<const std::uint8_t*>(&point) +
                                                                        traits::offset<PointT, Tag>::value);
                    value = static_cast<double>(source[0]);
                    found = true;
                }
            };

            static bool finitePoint(const PointT& point)
            {
                return std::isfinite(static_cast<double>(point.x)) && std::isfinite(static_cast<double>(point.y)) &&
                       std::isfinite(static_cast<double>(point.z));
            }

            bool passesFieldFilter(const PointT& point) const
            {
                if (_filter_field_name.empty())
                {
                    return true;
                }
                bool found = false;
                double value = 0.0;
                FieldValueReader reader{point, _filter_field_name, found, value};
                for_each_type<typename traits::fieldList<PointT>::type>(reader);
                if (!found)
                {
                    throw std::invalid_argument("VoxelGrid: filter field is not registered: " + _filter_field_name);
                }
                const bool inside = value >= _filter_limit_min && value <= _filter_limit_max;
                return _filter_limit_negative ? !inside : inside;
            }

            static void averageColor(const PointCloudType& cloud, const Indices& indices, PointT& output)
            {
                if constexpr (HasColorMembers<PointT>::value)
                {
                    if constexpr (traits::has_field_v<PointT, fields::rgb> || traits::has_field_v<PointT, fields::rgba>)
                    {
                        std::uint64_t red = 0;
                        std::uint64_t green = 0;
                        std::uint64_t blue = 0;
                        std::uint64_t alpha = 0;
                        for (const int index : indices)
                        {
                            const auto& point = cloud.points[static_cast<std::size_t>(index)];
                            red += point.r;
                            green += point.g;
                            blue += point.b;
                            alpha += point.a;
                        }
                        output.r = static_cast<std::uint8_t>(red / indices.size());
                        output.g = static_cast<std::uint8_t>(green / indices.size());
                        output.b = static_cast<std::uint8_t>(blue / indices.size());
                        output.a = static_cast<std::uint8_t>(alpha / indices.size());
                    }
                }
            }

            static int checkedGridCoordinate(double coordinate, double inverse_leaf_size)
            {
                const double grid = std::floor(static_cast<double>(coordinate) * inverse_leaf_size);
                if (grid < std::numeric_limits<int>::lowest() || grid > std::numeric_limits<int>::max())
                {
                    throw std::overflow_error("VoxelGrid: grid coordinate exceeds int range");
                }
                return static_cast<int>(grid);
            }

            void resetLayout()
            {
                _min_box.setZero();
                _max_box.setZero();
                _divisions.setZero();
                _division_multiplier.setZero();
                _leaf_layout.clear();
            }

            void configureLayout(const Eigen::Vector3i& minimum, const Eigen::Vector3i& maximum)
            {
                _min_box.head<3>() = minimum;
                _max_box.head<3>() = maximum;
                const Eigen::Array<std::int64_t, 3, 1> divisions =
                    maximum.cast<std::int64_t>().array() - minimum.cast<std::int64_t>().array() + 1;
                if ((divisions > std::numeric_limits<int>::max()).any())
                {
                    throw std::overflow_error("VoxelGrid: division count exceeds int range");
                }
                _divisions.head<3>() = divisions.cast<int>().matrix();
                _division_multiplier(0) = 1;
                _division_multiplier(1) = _divisions(0);
                const std::int64_t xy = static_cast<std::int64_t>(_divisions(0)) * _divisions(1);
                if (xy > std::numeric_limits<int>::max())
                {
                    throw std::overflow_error("VoxelGrid: leaf layout multiplier exceeds int range");
                }
                _division_multiplier(2) = static_cast<int>(xy);
                if (_save_leaf_layout)
                {
                    const std::uint64_t layout_size = static_cast<std::uint64_t>(xy) * _divisions(2);
                    if (layout_size > std::vector<int>().max_size())
                    {
                        throw std::overflow_error("VoxelGrid: leaf layout exceeds addressable size");
                    }
                    _leaf_layout.assign(static_cast<std::size_t>(layout_size), -1);
                }
            }

            Eigen::Vector4f _leaf_size = Eigen::Vector4f::Ones();
            Eigen::Array4f _inverse_leaf_size = Eigen::Array4f::Ones();
            bool _downsample_all_data = true;
            bool _save_leaf_layout = false;
            std::vector<int> _leaf_layout;
            Eigen::Vector4i _min_box = Eigen::Vector4i::Zero();
            Eigen::Vector4i _max_box = Eigen::Vector4i::Zero();
            Eigen::Vector4i _divisions = Eigen::Vector4i::Zero();
            Eigen::Vector4i _division_multiplier = Eigen::Vector4i::Zero();
            std::string _filter_field_name;
            double _filter_limit_min = std::numeric_limits<float>::lowest();
            double _filter_limit_max = std::numeric_limits<float>::max();
            bool _filter_limit_negative = false;
            unsigned int _min_points_per_voxel = 0;
        };

    } // namespace detail

    template <typename PointT>
    class VoxelGrid<PointT, plamatrix::internal::Device::CPU, std::enable_if_t<!std::is_arithmetic_v<PointT>>>
        : public detail::PointVoxelGridAdapter<PointT, VoxelGrid<PointT>>
    {
    };

} // namespace plapoint
