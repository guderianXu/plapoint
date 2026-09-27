#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plapoint/core/exceptions.h>
#include <plapoint/core/header.h>
#include <plapoint/core/point_types.h>
#include <plapoint/memory.h>

namespace plapoint
{

    using index_t = int;
    using uindex_t = unsigned int;
    using Indices = std::vector<index_t>;
    using IndicesPtr = std::shared_ptr<Indices>;
    using IndicesConstPtr = std::shared_ptr<const Indices>;

    struct PointIndices
    {
        using Ptr = std::shared_ptr<PointIndices>;
        using ConstPtr = std::shared_ptr<const PointIndices>;

        PCLHeader header;
        Indices indices;
    };

    using PointIndicesPtr = PointIndices::Ptr;
    using PointIndicesConstPtr = PointIndices::ConstPtr;

    namespace gpu
    {
        namespace marching_cubes_detail
        {
            struct PointCloudAccess;
        }
    } // namespace gpu

    template <typename Scalar, plamatrix::internal::Device Dev> class MatrixIterativeClosestPoint;

    template <typename PointT> class PointCloud;

    namespace internal
    {

        /// Internal Nx3 matrix cloud with optional attributes and CPU/GPU transfer helpers.
        template <typename Scalar,
                  plamatrix::internal::Device Dev = plamatrix::internal::Device::CPU,
                  typename Enable = void>
        class DeviceCloud
        {
            template <typename, plamatrix::internal::Device, typename> friend class DeviceCloud;
            template <typename, plamatrix::internal::Device> friend class ::plapoint::MatrixIterativeClosestPoint;
            friend struct ::plapoint::gpu::marching_cubes_detail::PointCloudAccess;

        public:
            using MatrixType = std::conditional_t<Dev == plamatrix::internal::Device::CPU,
                                                  plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>,
                                                  plamatrix::internal::ResidentMatrix<Scalar>>;
            template <typename Value>
            using AttributeMatrix = std::conditional_t<Dev == plamatrix::internal::Device::CPU,
                                                       plamatrix::Matrix<Value, plamatrix::Dynamic, plamatrix::Dynamic>,
                                                       plamatrix::internal::ResidentMatrix<Value>>;

            /// Scoped mutable access to point positions. The cloud must outlive the editor and
            /// remain at the same address while the editor is alive.
            /// Point-derived caches are invalidated when editing begins and again when it ends.
            class PointEdit
            {
                friend class DeviceCloud;

            public:
                PointEdit(const PointEdit&) = delete;
                PointEdit& operator=(const PointEdit&) = delete;

                PointEdit(PointEdit&& other) noexcept : _cloud(std::exchange(other._cloud, nullptr))
                {
                }

                PointEdit& operator=(PointEdit&& other) noexcept
                {
                    if (this != &other)
                    {
                        finish();
                        _cloud = std::exchange(other._cloud, nullptr);
                    }
                    return *this;
                }

                ~PointEdit()
                {
                    finish();
                }

                MatrixType& get()
                {
                    if (!_cloud)
                    {
                        throw std::logic_error("DeviceCloud point editor has been moved from");
                    }
                    return _cloud->_points;
                }

                MatrixType& operator*()
                {
                    return get();
                }
                MatrixType* operator->()
                {
                    return &get();
                }

            private:
                explicit PointEdit(DeviceCloud& cloud) : _cloud(&cloud)
                {
                    _cloud->beginPointEdit();
                }

                void finish() noexcept
                {
                    if (_cloud)
                    {
                        _cloud->finishPointEdit();
                        _cloud = nullptr;
                    }
                }

                DeviceCloud* _cloud = nullptr;
            };

            /// Lightweight read-only accessor for one point and its optional attributes.
            class PointView
            {
            public:
                Scalar x() const
                {
                    return _cloud.readValue(_cloud.points(), static_cast<plamatrix::Index>(_idx), 0);
                }
                Scalar y() const
                {
                    return _cloud.readValue(_cloud.points(), static_cast<plamatrix::Index>(_idx), 1);
                }
                Scalar z() const
                {
                    return _cloud.readValue(_cloud.points(), static_cast<plamatrix::Index>(_idx), 2);
                }

                uint8_t r() const
                {
                    requireColors();
                    return _cloud.readValue(*_cloud.colors(), static_cast<plamatrix::Index>(_idx), 0);
                }
                uint8_t g() const
                {
                    requireColors();
                    return _cloud.readValue(*_cloud.colors(), static_cast<plamatrix::Index>(_idx), 1);
                }
                uint8_t b() const
                {
                    requireColors();
                    return _cloud.readValue(*_cloud.colors(), static_cast<plamatrix::Index>(_idx), 2);
                }

                std::uint16_t intensity() const
                {
                    requireIntensities();
                    return _cloud.readValue(*_cloud.intensities(), static_cast<plamatrix::Index>(_idx), 0);
                }

                Scalar nx() const
                {
                    requireNormals();
                    return _cloud.readValue(*_cloud.normals(), static_cast<plamatrix::Index>(_idx), 0);
                }
                Scalar ny() const
                {
                    requireNormals();
                    return _cloud.readValue(*_cloud.normals(), static_cast<plamatrix::Index>(_idx), 1);
                }
                Scalar nz() const
                {
                    requireNormals();
                    return _cloud.readValue(*_cloud.normals(), static_cast<plamatrix::Index>(_idx), 2);
                }

                Scalar u() const
                {
                    requireTextureCoords();
                    return _cloud.readValue(*_cloud.textureCoords(), static_cast<plamatrix::Index>(_idx), 0);
                }
                Scalar v() const
                {
                    requireTextureCoords();
                    return _cloud.readValue(*_cloud.textureCoords(), static_cast<plamatrix::Index>(_idx), 1);
                }

                Scalar scalar(const std::string& name) const
                {
                    const int field_index = _cloud.scalarFieldIndex(name);
                    if (field_index < 0 || !_cloud.scalarFields())
                    {
                        throw std::runtime_error("PointView: cloud has no scalar field named " + name);
                    }
                    return _cloud.readValue(*_cloud.scalarFields(), static_cast<plamatrix::Index>(_idx), field_index);
                }

            private:
                friend class DeviceCloud;
                PointView(const DeviceCloud& cloud, size_t idx) : _cloud(cloud), _idx(idx)
                {
                }

                void requireColors() const
                {
                    if (!_cloud.hasColors())
                    {
                        throw std::runtime_error("PointView: cloud has no colors");
                    }
                }

                void requireIntensities() const
                {
                    if (!_cloud.hasIntensities())
                    {
                        throw std::runtime_error("PointView: cloud has no intensities");
                    }
                }

                void requireNormals() const
                {
                    if (!_cloud.hasNormals())
                    {
                        throw std::runtime_error("PointView: cloud has no normals");
                    }
                }

                void requireTextureCoords() const
                {
                    if (!_cloud.hasTextureCoords())
                    {
                        throw std::runtime_error("PointView: cloud has no texture coordinates");
                    }
                    if (!_cloud.hasPointAlignedTextureCoords())
                    {
                        throw std::runtime_error("PointView: texture coordinates are face-indexed");
                    }
                    if (_idx >= static_cast<size_t>(_cloud.textureCoords()->rows()))
                    {
                        throw std::runtime_error("PointView: point texture coordinate index out of range");
                    }
                }

                const DeviceCloud& _cloud;
                size_t _idx;
            };

            PointView operator[](size_t idx) const
            {
                if (idx >= size())
                {
                    throw std::out_of_range("DeviceCloud: point index out of range");
                }
                return PointView(*this, idx);
            }

            /// Construct an empty Nx3 point cloud.
            DeviceCloud() : _context(makeDefaultContext()), _points(makeMatrix<Scalar>(0, 3, _context))
            {
            }

            /// Construct an Nx3 point cloud with all coordinates value-initialized.
            explicit DeviceCloud(size_t num_points)
                : _context(makeDefaultContext()),
                  _points(makeMatrix<Scalar>(static_cast<plamatrix::Index>(num_points), 3, _context))
            {
            }

            /// Construct from an Nx3 matrix, throwing if the matrix does not have three columns.
            template <plamatrix::internal::Device D = Dev,
                      std::enable_if_t<D == plamatrix::internal::Device::CPU, int> = 0>
            explicit DeviceCloud(MatrixType&& pts) : _context(), _points(std::move(pts))
            {
                if (_points.cols() != 3)
                {
                    throw std::runtime_error("DeviceCloud requires Nx3 matrix");
                }
            }

            /// Adopt a resident point matrix and keep its execution context alive.
            template <plamatrix::internal::Device D = Dev,
                      std::enable_if_t<D == plamatrix::internal::Device::GPU, int> = 0>
            DeviceCloud(MatrixType&& pts, std::shared_ptr<plamatrix::internal::ExecutionContext> context)
                : _context(std::move(context)), _points(std::move(pts))
            {
                if (!_context || _context->backend() != plamatrix::internal::Backend::Cuda || _points.cols() != 3)
                {
                    throw std::invalid_argument("DeviceCloud requires a CUDA execution context and Nx3 points");
                }
                _points.validateContext(*_context);
            }

            DeviceCloud(DeviceCloud&&) noexcept = default;

            DeviceCloud& operator=(DeviceCloud&& other) noexcept
            {
                if (this != &other)
                {
                    DeviceCloud next(std::move(other));
                    swap(next);
                }
                return *this;
            }

            void swap(DeviceCloud& other) noexcept
            {
                using std::swap;
                swap(_points, other._points);
                swap(_normals, other._normals);
                swap(_colors, other._colors);
                swap(_intensities, other._intensities);
                swap(_scalarFields, other._scalarFields);
                swap(_scalarFieldNames, other._scalarFieldNames);
                swap(_textureCoords, other._textureCoords);
                swap(_faces, other._faces);
                swap(_faceTextureIndices, other._faceTextureIndices);
                swap(_materialLibraryFile, other._materialLibraryFile);
                swap(_textureImageFile, other._textureImageFile);
                swap(_points_revision, other._points_revision);
                swap(_mutable_points_alias_issued, other._mutable_points_alias_issued);
                swap(_active_point_edits, other._active_point_edits);
                swap(_points_identity, other._points_identity);
                swap(_points_cpu_cache, other._points_cpu_cache);
                swap(_context, other._context);
            }

            size_t size() const
            {
                return _points.rows();
            }

            const MatrixType& points() const
            {
                return _points;
            }

            const std::shared_ptr<plamatrix::internal::ExecutionContext>& executionContext() const noexcept
            {
                return _context;
            }

            /// Nonzero generation for caches derived from point positions; wraps from UINT64_MAX to one.
            std::uint64_t pointsRevision() const noexcept
            {
                return _points_revision;
            }

            /// Stable identity for this cloud's point ownership, retained by derived caches to prevent ABA matches.
            const std::shared_ptr<const void>& pointsIdentity() const noexcept
            {
                return _points_identity;
            }

            /// True when legacy mutable point storage has escaped and writes cannot be observed individually.
            bool hasUntrackedMutablePointAlias() const noexcept
            {
                return _mutable_points_alias_issued;
            }

            /// True while one or more scoped point editors are alive.
            bool hasActivePointEdit() const noexcept
            {
                return _active_point_edits != 0;
            }

            /// True when point-derived caches may be reused without inspecting point contents.
            bool pointCachesReusable() const noexcept
            {
                return !_mutable_points_alias_issued && !hasActivePointEdit();
            }

            /// Backward-compatible alias for point-position cache identity.
            std::uint64_t pointsVersion() const noexcept
            {
                return pointsRevision();
            }

            /// Return legacy mutable point storage and invalidate cached CPU mirrors.
            /// The returned reference may outlive the call, so point-derived caches must thereafter
            /// validate conservatively. Prefer editPoints() for bounded mutations.
            MatrixType& points()
            {
                advancePointsRevision();
                _mutable_points_alias_issued = true;
                invalidateCpuMirror();
                return _points;
            }

            /// Return scoped mutable point storage. Callers must preserve the Nx3 shape and keep
            /// any references obtained from the editor within the editor's lifetime.
            PointEdit editPoints()
            {
                return PointEdit(*this);
            }

            /// Replace point positions by copy, advancing the cache identity only after a successful replacement.
            void setPoints(const MatrixType& points)
            {
                if (points.cols() != 3)
                {
                    throw std::runtime_error("DeviceCloud requires Nx3 matrix");
                }
                validateReplacementPointCount(points.rows());

                setPoints(copyMatrix(points));
            }

            /// Replace point positions by move, advancing the cache identity only after a successful replacement.
            void setPoints(MatrixType&& points)
            {
                if (points.cols() != 3)
                {
                    throw std::runtime_error("DeviceCloud requires Nx3 matrix");
                }
                validateReplacementPointCount(points.rows());

                validateMatrixContext(points);
                _points = std::move(points);
                advancePointsRevision();
                invalidateCpuMirror();
            }

            /// Return a CPU-readable view of points. GPU clouds cache while no mutable alias has escaped.
            const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& pointsCpu() const
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    return _points;
                }
                else
                {
                    if (!_points_cpu_cache)
                    {
                        _points_cpu_cache =
                            std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                                _points.toHostMatrix());
                    }
                    else if (_mutable_points_alias_issued || hasActivePointEdit())
                    {
                        *_points_cpu_cache = _points.toHostMatrix();
                    }
                    return *_points_cpu_cache;
                }
            }

            template <plamatrix::internal::Device D = Dev>
            std::enable_if_t<D == plamatrix::internal::Device::CPU,
                             DeviceCloud<Scalar, plamatrix::internal::Device::GPU>>
            toGpu(std::shared_ptr<plamatrix::internal::ExecutionContext> context = {}) const
            {
                validateStructure();
                if (!context)
                {
                    context =
                        plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
                }
                if (context->backend() != plamatrix::internal::Backend::Cuda)
                {
                    throw std::invalid_argument("DeviceCloud GPU transfer requires a CUDA execution context");
                }
                auto points = plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(_points, context);
                DeviceCloud<Scalar, plamatrix::internal::Device::GPU> result(std::move(points), context);
                if (_normals)
                    result._normals = std::make_unique<plamatrix::internal::ResidentMatrix<Scalar>>(
                        plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(*_normals, context));
                if (_colors)
                    result._colors = std::make_unique<plamatrix::internal::ResidentMatrix<uint8_t>>(
                        plamatrix::internal::ResidentMatrix<uint8_t>::copyFrom(*_colors, context));
                if (_intensities)
                    result._intensities = std::make_unique<plamatrix::internal::ResidentMatrix<std::uint16_t>>(
                        plamatrix::internal::ResidentMatrix<std::uint16_t>::copyFrom(*_intensities, context));
                if (_scalarFields)
                {
                    result._scalarFields = std::make_unique<plamatrix::internal::ResidentMatrix<Scalar>>(
                        plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(*_scalarFields, context));
                    result._scalarFieldNames = _scalarFieldNames;
                }
                if (_textureCoords)
                    result._textureCoords = std::make_unique<plamatrix::internal::ResidentMatrix<Scalar>>(
                        plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(*_textureCoords, context));
                if (_faces)
                    result._faces = std::make_unique<plamatrix::internal::ResidentMatrix<int>>(
                        plamatrix::internal::ResidentMatrix<int>::copyFrom(*_faces, context));
                if (_faceTextureIndices)
                    result._faceTextureIndices = std::make_unique<plamatrix::internal::ResidentMatrix<int>>(
                        plamatrix::internal::ResidentMatrix<int>::copyFrom(*_faceTextureIndices, context));
                result.setMaterialLibraryFile(_materialLibraryFile);
                result.setTextureImageFile(_textureImageFile);
                result.validateStructure();
                return result;
            }

            template <plamatrix::internal::Device D = Dev>
            std::enable_if_t<D == plamatrix::internal::Device::GPU,
                             DeviceCloud<Scalar, plamatrix::internal::Device::CPU>>
            toCpu() const
            {
                validateStructure();
                DeviceCloud<Scalar, plamatrix::internal::Device::CPU> result(_points.toHostMatrix());
                if (_normals)
                    result._normals =
                        std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                            _normals->toHostMatrix());
                if (_colors)
                    result._colors =
                        std::make_unique<plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>>(
                            _colors->toHostMatrix());
                if (_intensities)
                    result._intensities =
                        std::make_unique<plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>>(
                            _intensities->toHostMatrix());
                if (_scalarFields)
                {
                    result._scalarFields =
                        std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                            _scalarFields->toHostMatrix());
                    result._scalarFieldNames = _scalarFieldNames;
                }
                if (_textureCoords)
                    result._textureCoords =
                        std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                            _textureCoords->toHostMatrix());
                if (_faces)
                    result._faces = std::make_unique<plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>>(
                        _faces->toHostMatrix());
                if (_faceTextureIndices)
                    result._faceTextureIndices =
                        std::make_unique<plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>>(
                            _faceTextureIndices->toHostMatrix());
                result.setMaterialLibraryFile(_materialLibraryFile);
                result.setTextureImageFile(_textureImageFile);
                result.validateStructure();
                return result;
            }

            /// Set optional normals by copy (Nx3 matrix, must match point count)
            void setNormals(const MatrixType& n)
            {
                if (n.rows() != _points.rows() || n.cols() != 3)
                    throw std::runtime_error("Normals must match point count and be Nx3");
                _normals = std::make_unique<MatrixType>(copyMatrix(n));
            }

            /// Set optional normals by move
            void setNormals(MatrixType&& n)
            {
                if (n.rows() != _points.rows() || n.cols() != 3)
                    throw std::runtime_error("Normals must match point count and be Nx3");
                validateMatrixContext(n);
                _normals = std::make_unique<MatrixType>(std::move(n));
            }

            bool hasNormals() const
            {
                return _normals != nullptr;
            }

            const MatrixType* normals() const
            {
                return _normals.get();
            }

            MatrixType* normals()
            {
                return _normals.get();
            }

            /// Set optional RGB colors by copy (Nx3 uint8 matrix)
            void setColors(const AttributeMatrix<uint8_t>& c)
            {
                if (c.rows() != _points.rows() || c.cols() != 3)
                    throw std::runtime_error("Colors must match point count and be Nx3");
                _colors = std::make_unique<AttributeMatrix<uint8_t>>(copyMatrix(c));
            }

            /// Set optional RGB colors by move
            void setColors(AttributeMatrix<uint8_t>&& c)
            {
                if (c.rows() != _points.rows() || c.cols() != 3)
                    throw std::runtime_error("Colors must match point count and be Nx3");
                validateMatrixContext(c);
                _colors = std::make_unique<AttributeMatrix<uint8_t>>(std::move(c));
            }

            bool hasColors() const
            {
                return _colors != nullptr;
            }

            const AttributeMatrix<uint8_t>* colors() const
            {
                return _colors.get();
            }

            AttributeMatrix<uint8_t>* colors()
            {
                return _colors.get();
            }

            /// Set optional intensity values by copy (Nx1 uint16 matrix)
            void setIntensities(const AttributeMatrix<std::uint16_t>& values)
            {
                if (values.rows() != _points.rows() || values.cols() != 1)
                    throw std::runtime_error("Intensities must match point count and be Nx1");
                _intensities = std::make_unique<AttributeMatrix<std::uint16_t>>(copyMatrix(values));
            }

            /// Set optional intensity values by move
            void setIntensities(AttributeMatrix<std::uint16_t>&& values)
            {
                if (values.rows() != _points.rows() || values.cols() != 1)
                    throw std::runtime_error("Intensities must match point count and be Nx1");
                validateMatrixContext(values);
                _intensities = std::make_unique<AttributeMatrix<std::uint16_t>>(std::move(values));
            }

            bool hasIntensities() const
            {
                return _intensities != nullptr;
            }

            const AttributeMatrix<std::uint16_t>* intensities() const
            {
                return _intensities.get();
            }

            AttributeMatrix<std::uint16_t>* intensities()
            {
                return _intensities.get();
            }

            /// Set optional named scalar fields by copy (NxK matrix, one name per column).
            void setScalarFields(const std::vector<std::string>& names, const MatrixType& values)
            {
                validateScalarFields(names, values);
                auto replacement = std::make_unique<MatrixType>(copyMatrix(values));
                auto replacement_names = names;
                _scalarFields = std::move(replacement);
                _scalarFieldNames = std::move(replacement_names);
            }

            /// Set optional named scalar fields by move (NxK matrix, one name per column).
            void setScalarFields(std::vector<std::string> names, MatrixType&& values)
            {
                validateScalarFields(names, values);
                validateMatrixContext(values);
                _scalarFieldNames = std::move(names);
                _scalarFields = std::make_unique<MatrixType>(std::move(values));
            }

            bool hasScalarFields() const
            {
                return _scalarFields != nullptr && !_scalarFieldNames.empty();
            }

            bool hasScalarField(const std::string& name) const
            {
                return scalarFieldIndex(name) >= 0;
            }

            int scalarFieldIndex(const std::string& name) const
            {
                const auto it = std::find(_scalarFieldNames.begin(), _scalarFieldNames.end(), name);
                if (it == _scalarFieldNames.end())
                {
                    return -1;
                }
                return static_cast<int>(std::distance(_scalarFieldNames.begin(), it));
            }

            const std::vector<std::string>& scalarFieldNames() const
            {
                return _scalarFieldNames;
            }

            const MatrixType* scalarFields() const
            {
                return _scalarFields.get();
            }

            MatrixType* scalarFields()
            {
                return _scalarFields.get();
            }

            /// Set optional texture coordinates by copy (Tx2 UV table).
            void setTextureCoords(const MatrixType& t)
            {
                if (t.cols() != 2)
                    throw std::runtime_error("Texture coords must be Tx2");
                if (_faceTextureIndices)
                    validateIndexMatrix(*_faceTextureIndices, t.rows(), "Face texture");
                _textureCoords = std::make_unique<MatrixType>(copyMatrix(t));
            }

            /// Set optional texture coordinates by move
            void setTextureCoords(MatrixType&& t)
            {
                if (t.cols() != 2)
                    throw std::runtime_error("Texture coords must be Tx2");
                validateMatrixContext(t);
                if (_faceTextureIndices)
                    validateIndexMatrix(*_faceTextureIndices, t.rows(), "Face texture");
                _textureCoords = std::make_unique<MatrixType>(std::move(t));
            }

            bool hasTextureCoords() const
            {
                return _textureCoords != nullptr;
            }

            const MatrixType* textureCoords() const
            {
                return _textureCoords.get();
            }

            MatrixType* textureCoords()
            {
                return _textureCoords.get();
            }

            /// Return true when the UV table can be gathered by point index.
            bool hasPointAlignedTextureCoords() const
            {
                return computePointAlignedTextureCoords();
            }

            /// Set optional faces by copy (Fx3 int matrix)
            void setFaces(const AttributeMatrix<int>& f)
            {
                if (f.cols() != 3)
                    throw std::runtime_error("Faces must be Fx3");
                validateIndexMatrix(f, _points.rows(), "Faces");
                validateFaceCountForExistingTextureIndices(f.rows());
                _faces = std::make_unique<AttributeMatrix<int>>(copyMatrix(f));
            }

            /// Set optional faces by move
            void setFaces(AttributeMatrix<int>&& f)
            {
                if (f.cols() != 3)
                    throw std::runtime_error("Faces must be Fx3");
                validateMatrixContext(f);
                validateIndexMatrix(f, _points.rows(), "Faces");
                validateFaceCountForExistingTextureIndices(f.rows());
                _faces = std::make_unique<AttributeMatrix<int>>(std::move(f));
            }

            bool hasFaces() const
            {
                return _faces != nullptr;
            }

            const AttributeMatrix<int>* faces() const
            {
                return _faces.get();
            }

            AttributeMatrix<int>* faces()
            {
                return _faces.get();
            }

            /// Set optional face texture indices by copy (Fx3 int matrix)
            void setFaceTextureIndices(const AttributeMatrix<int>& ft)
            {
                if (ft.cols() != 3)
                    throw std::runtime_error("Face texture indices must be Fx3");
                validateFaceTextureIndices(ft);
                _faceTextureIndices = std::make_unique<AttributeMatrix<int>>(copyMatrix(ft));
            }

            /// Set optional face texture indices by move
            void setFaceTextureIndices(AttributeMatrix<int>&& ft)
            {
                if (ft.cols() != 3)
                    throw std::runtime_error("Face texture indices must be Fx3");
                validateMatrixContext(ft);
                validateFaceTextureIndices(ft);
                _faceTextureIndices = std::make_unique<AttributeMatrix<int>>(std::move(ft));
            }

            bool hasFaceTextureIndices() const
            {
                return _faceTextureIndices != nullptr;
            }

            const AttributeMatrix<int>* faceTextureIndices() const
            {
                return _faceTextureIndices.get();
            }

            AttributeMatrix<int>* faceTextureIndices()
            {
                return _faceTextureIndices.get();
            }

            const std::string& materialLibraryFile() const
            {
                return _materialLibraryFile;
            }
            void setMaterialLibraryFile(const std::string& f)
            {
                _materialLibraryFile = f;
            }

            const std::string& textureImageFile() const
            {
                return _textureImageFile;
            }
            void setTextureImageFile(const std::string& f)
            {
                _textureImageFile = f;
            }

            /// Validate all point, attribute, topology, and texture-index shapes.
            /// Call this at API boundaries before accessing attribute storage directly.
            void validate() const
            {
                validateStructure();
            }

        private:
            static std::shared_ptr<plamatrix::internal::ExecutionContext> makeDefaultContext()
            {
                if constexpr (Dev == plamatrix::internal::Device::GPU)
                {
                    return plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
                }
                return {};
            }

            template <typename Value>
            static AttributeMatrix<Value>
            makeMatrix(plamatrix::Index rows,
                       plamatrix::Index cols,
                       const std::shared_ptr<plamatrix::internal::ExecutionContext>& context)
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    return AttributeMatrix<Value>(rows, cols);
                }
                else
                {
                    return AttributeMatrix<Value>::copyFrom(
                        plamatrix::Matrix<Value, plamatrix::Dynamic, plamatrix::Dynamic>::Zero(rows, cols), context);
                }
            }

            template <typename Matrix>
            static auto readValue(const Matrix& matrix, plamatrix::Index row, plamatrix::Index col)
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    return matrix(row, col);
                }
                else
                {
                    return matrix.toHostMatrix()(row, col);
                }
            }

            template <typename Matrix> void validateMatrixContext(const Matrix& matrix) const
            {
                if constexpr (Dev == plamatrix::internal::Device::GPU)
                {
                    matrix.validateContext(*_context);
                }
            }

            /// Return point storage for a bounded library-internal write without publishing a
            /// mutable alias. The caller must preserve the Nx3 shape and order asynchronous writes
            /// before recording or reusing point-derived caches.
            MatrixType& pointsForInternalWrite()
            {
                advancePointsRevision();
                invalidateCpuMirror();
                return _points;
            }

            /// Create an independent same-device matrix using one contiguous transfer.
            template <typename Matrix> Matrix copyMatrix(const Matrix& source) const
            {
                if constexpr (Dev == plamatrix::internal::Device::CPU)
                {
                    return source;
                }
                else
                {
                    validateMatrixContext(source);
                    return source.clone();
                }
            }

            void invalidateCpuMirror()
            {
                if constexpr (Dev == plamatrix::internal::Device::GPU)
                {
                    _points_cpu_cache.reset();
                }
            }

            void beginPointEdit() noexcept
            {
                ++_active_point_edits;
                advancePointsRevision();
                invalidateCpuMirror();
            }

            void finishPointEdit() noexcept
            {
                if (_active_point_edits > 0)
                {
                    --_active_point_edits;
                }
                advancePointsRevision();
                invalidateCpuMirror();
            }

            void advancePointsRevision() const noexcept
            {
                ++_points_revision;
                if (_points_revision == 0)
                {
                    _points_revision = 1;
                }
            }

            void validateReplacementPointCount(plamatrix::Index point_count) const
            {
                if (point_count == _points.rows())
                {
                    return;
                }
                if (_normals || _colors || _intensities || _scalarFields || _textureCoords || _faces ||
                    _faceTextureIndices)
                {
                    throw std::runtime_error(
                        "DeviceCloud cannot change point count while point or face attributes are present");
                }
            }

            static void
            validateIndexMatrix(const AttributeMatrix<int>& m, plamatrix::Index exclusive_limit, const char* label)
            {
                if constexpr (Dev == plamatrix::internal::Device::GPU)
                {
                    validateCpuIndexMatrix(m.toHostMatrix(), exclusive_limit, label);
                }
                else
                {
                    validateCpuIndexMatrix(m, exclusive_limit, label);
                }
            }

            static void validateCpuIndexMatrix(const plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>& m,
                                               plamatrix::Index exclusive_limit,
                                               const char* label)
            {
                for (plamatrix::Index r = 0; r < m.rows(); ++r)
                {
                    for (int c = 0; c < m.cols(); ++c)
                    {
                        const int idx = m(r, c);
                        if (idx < 0 || idx >= exclusive_limit)
                        {
                            throw std::out_of_range(std::string(label) + " index out of range");
                        }
                    }
                }
            }

            static bool isReservedScalarFieldName(const std::string& name)
            {
                return name == "x" || name == "y" || name == "z" || name == "nx" || name == "ny" || name == "nz" ||
                       name == "red" || name == "green" || name == "blue" || name == "intensity";
            }

            void validateScalarFields(const std::vector<std::string>& names, const MatrixType& values) const
            {
                if (values.rows() != _points.rows())
                {
                    throw std::runtime_error("Scalar fields must match point count");
                }
                if (values.cols() != static_cast<plamatrix::Index>(names.size()))
                {
                    throw std::runtime_error("Scalar field names must match scalar field columns");
                }
                for (std::size_t i = 0; i < names.size(); ++i)
                {
                    if (names[i].empty())
                    {
                        throw std::runtime_error("Scalar field names must be non-empty");
                    }
                    if (names[i].find_first_of(" \t\r\n") != std::string::npos)
                    {
                        throw std::runtime_error("Scalar field names must not contain whitespace: " + names[i]);
                    }
                    if (isReservedScalarFieldName(names[i]))
                    {
                        throw std::runtime_error("Scalar field name conflicts with a built-in point property: " +
                                                 names[i]);
                    }
                    if (std::find(names.begin(), names.begin() + static_cast<std::ptrdiff_t>(i), names[i]) !=
                        names.begin() + static_cast<std::ptrdiff_t>(i))
                    {
                        throw std::runtime_error("Scalar field names must be unique: " + names[i]);
                    }
                }
            }

            void validateFaceTextureIndices(const AttributeMatrix<int>& ft) const
            {
                if (ft.cols() != 3)
                {
                    throw std::runtime_error("Face texture indices must be Fx3");
                }
                if (!_faces)
                {
                    throw std::runtime_error("Face texture indices require faces");
                }
                if (!_textureCoords)
                {
                    throw std::runtime_error("Face texture indices require texture coordinates");
                }
                if (ft.rows() != _faces->rows())
                {
                    throw std::runtime_error("Face texture indices must match face count");
                }

                validateIndexMatrix(ft, _textureCoords->rows(), "Face texture");
            }

            void validateFaceCountForExistingTextureIndices(plamatrix::Index face_count) const
            {
                if (_faceTextureIndices && _faceTextureIndices->rows() != face_count)
                {
                    throw std::runtime_error("Faces must match face texture index count");
                }
            }

            void validateStructure() const
            {
                if constexpr (Dev == plamatrix::internal::Device::GPU)
                {
                    if (!_context || _context->backend() != plamatrix::internal::Backend::Cuda)
                        throw std::runtime_error("DeviceCloud GPU storage requires a CUDA execution context");
                    _points.validateContext(*_context);
                    if (_normals)
                        _normals->validateContext(*_context);
                    if (_colors)
                        _colors->validateContext(*_context);
                    if (_intensities)
                        _intensities->validateContext(*_context);
                    if (_scalarFields)
                        _scalarFields->validateContext(*_context);
                    if (_textureCoords)
                        _textureCoords->validateContext(*_context);
                    if (_faces)
                        _faces->validateContext(*_context);
                    if (_faceTextureIndices)
                        _faceTextureIndices->validateContext(*_context);
                }
                if (_points.cols() != 3)
                    throw std::runtime_error("DeviceCloud points must be Nx3");
                if (_normals && (_normals->rows() != _points.rows() || _normals->cols() != 3))
                    throw std::runtime_error("Normals must match point count and be Nx3");
                if (_colors && (_colors->rows() != _points.rows() || _colors->cols() != 3))
                    throw std::runtime_error("Colors must match point count and be Nx3");
                if (_intensities && (_intensities->rows() != _points.rows() || _intensities->cols() != 1))
                    throw std::runtime_error("Intensities must match point count and be Nx1");
                if (_scalarFields)
                    validateScalarFields(_scalarFieldNames, *_scalarFields);
                if (_textureCoords && _textureCoords->cols() != 2)
                    throw std::runtime_error("Texture coords must be Tx2");
                if (_faces)
                {
                    if (_faces->cols() != 3)
                        throw std::runtime_error("Faces must be Fx3");
                    validateIndexMatrix(*_faces, _points.rows(), "Faces");
                }
                if (_faceTextureIndices)
                    validateFaceTextureIndices(*_faceTextureIndices);
            }

            bool computePointAlignedTextureCoords() const
            {
                if (!_textureCoords || _textureCoords->rows() != _points.rows())
                {
                    return false;
                }
                if (!_faceTextureIndices)
                {
                    return true;
                }
                if (!_faces || _faceTextureIndices->rows() != _faces->rows())
                {
                    return false;
                }
                for (plamatrix::Index r = 0; r < _faces->rows(); ++r)
                {
                    for (int col = 0; col < 3; ++col)
                    {
                        if (readValue(*_faceTextureIndices, r, col) != readValue(*_faces, r, col))
                        {
                            return false;
                        }
                    }
                }
                return true;
            }

            std::shared_ptr<plamatrix::internal::ExecutionContext> _context;
            MatrixType _points;
            std::unique_ptr<MatrixType> _normals;
            std::unique_ptr<AttributeMatrix<uint8_t>> _colors;
            std::unique_ptr<AttributeMatrix<std::uint16_t>> _intensities;
            std::unique_ptr<MatrixType> _scalarFields;
            std::vector<std::string> _scalarFieldNames;
            std::unique_ptr<MatrixType> _textureCoords;
            std::unique_ptr<AttributeMatrix<int>> _faces;
            std::unique_ptr<AttributeMatrix<int>> _faceTextureIndices;
            std::string _materialLibraryFile;
            std::string _textureImageFile;
            mutable std::uint64_t _points_revision = 1;
            bool _mutable_points_alias_issued = false;
            std::size_t _active_point_edits = 0;
            std::shared_ptr<const void> _points_identity = std::make_shared<char>(0);
            mutable std::unique_ptr<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>
                _points_cpu_cache;
        };

    } // namespace internal

    namespace detail
    {

        template <typename PointT, typename CloudT> class PointCloudStorage
        {
        public:
            using PointType = PointT;
            using VectorType = std::vector<PointT, Eigen::aligned_allocator<PointT>>;
            using CloudVectorType = std::vector<CloudT, Eigen::aligned_allocator<CloudT>>;
            using Ptr = std::shared_ptr<CloudT>;
            using ConstPtr = std::shared_ptr<const CloudT>;
            using value_type = PointT;
            using reference = PointT&;
            using const_reference = const PointT&;
            using difference_type = typename VectorType::difference_type;
            using size_type = typename VectorType::size_type;
            using iterator = typename VectorType::iterator;
            using const_iterator = typename VectorType::const_iterator;
            using reverse_iterator = typename VectorType::reverse_iterator;
            using const_reverse_iterator = typename VectorType::const_reverse_iterator;

            PCLHeader header;
            VectorType points;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            bool is_dense = true;
            Eigen::Vector4f sensor_origin_ = Eigen::Vector4f::Zero();
            Eigen::Quaternionf sensor_orientation_ = Eigen::Quaternionf::Identity();

            PointCloudStorage() = default;

            PointCloudStorage(const CloudT& source, const Indices& indices)
                : header(source.header), points(indices.size()), width(checkedWidth(indices.size())), height(1),
                  is_dense(source.is_dense), sensor_origin_(source.sensor_origin_),
                  sensor_orientation_(source.sensor_orientation_)
            {
                for (std::size_t row = 0; row < indices.size(); ++row)
                {
                    const int index = indices[row];
                    if (index < 0)
                    {
                        throw std::out_of_range("PointCloud: subset index is negative");
                    }
                    points[row] = source.points.at(static_cast<std::size_t>(index));
                }
            }

            explicit PointCloudStorage(std::size_t point_count, const PointT& value = PointT())
                : points(point_count, value), width(checkedWidth(point_count)), height(point_count == 0 ? 0u : 1u)
            {
            }

            PointCloudStorage(std::uint32_t cloud_width, std::uint32_t cloud_height, const PointT& value = PointT())
                : points(checkedProduct(cloud_width, cloud_height), value), width(cloud_width), height(cloud_height)
            {
            }

            std::size_t size() const noexcept
            {
                return points.size();
            }
            index_t max_size() const noexcept
            {
                return static_cast<index_t>(points.max_size());
            }
            bool empty() const noexcept
            {
                return points.empty();
            }
            bool isOrganized() const noexcept
            {
                return height > 1;
            }
            PointT* data() noexcept
            {
                return points.data();
            }
            const PointT* data() const noexcept
            {
                return points.data();
            }

            /// View float-backed point storage as an Eigen matrix without copying it.
            using FloatMap = Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>;
            using ConstFloatMap = Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>;

            FloatMap getMatrixXfMap(int dim, int stride, int offset)
            {
                validateFloatMap(dim, stride, offset);
                float* data_ptr = points.empty() ? nullptr : reinterpret_cast<float*>(points.data()) + offset;
                return FloatMap(data_ptr, dim, static_cast<Eigen::Index>(size()), Eigen::OuterStride<>(stride));
            }

            const ConstFloatMap getMatrixXfMap(int dim, int stride, int offset) const
            {
                validateFloatMap(dim, stride, offset);
                const float* data_ptr =
                    points.empty() ? nullptr : reinterpret_cast<const float*>(points.data()) + offset;
                return ConstFloatMap(data_ptr, dim, static_cast<Eigen::Index>(size()), Eigen::OuterStride<>(stride));
            }

            FloatMap getMatrixXfMap()
            {
                const auto stride = static_cast<int>(sizeof(PointT) / sizeof(float));
                return getMatrixXfMap(stride, stride, 0);
            }

            const ConstFloatMap getMatrixXfMap() const
            {
                const auto stride = static_cast<int>(sizeof(PointT) / sizeof(float));
                return getMatrixXfMap(stride, stride, 0);
            }

            void reserve(std::size_t capacity)
            {
                points.reserve(capacity);
            }

            CloudT& operator+=(const CloudT& other)
            {
                concatenate(static_cast<CloudT&>(*this), other);
                return static_cast<CloudT&>(*this);
            }

            CloudT operator+(const CloudT& other) const
            {
                CloudT combined(static_cast<const CloudT&>(*this));
                combined += other;
                return combined;
            }

            static bool concatenate(CloudT& lhs, const CloudT& rhs)
            {
                lhs.header.stamp = std::max(lhs.header.stamp, rhs.header.stamp);
                lhs.points.insert(lhs.points.end(), rhs.points.begin(), rhs.points.end());
                lhs.width = checkedWidth(lhs.size());
                lhs.height = 1;
                lhs.is_dense = lhs.is_dense && rhs.is_dense;
                return true;
            }

            static bool concatenate(const CloudT& lhs, const CloudT& rhs, CloudT& output)
            {
                output = lhs;
                return concatenate(output, rhs);
            }

            void swap(CloudT& other) noexcept
            {
                using std::swap;
                swap(header, other.header);
                points.swap(other.points);
                swap(width, other.width);
                swap(height, other.height);
                swap(is_dense, other.is_dense);
                swap(sensor_origin_, other.sensor_origin_);
                swap(sensor_orientation_, other.sensor_orientation_);
            }

            void resize(std::size_t point_count)
            {
                const auto new_width = checkedWidth(point_count);
                points.resize(point_count);
                if (static_cast<std::size_t>(width) * height != point_count)
                {
                    width = new_width;
                    height = 1;
                }
            }

            void resize(std::uint32_t cloud_width, std::uint32_t cloud_height)
            {
                points.resize(checkedProduct(cloud_width, cloud_height));
                width = cloud_width;
                height = cloud_height;
            }

            void resize(index_t point_count, const PointT& value)
            {
                if (point_count < 0)
                {
                    throw std::invalid_argument("PointCloud: point count must be non-negative");
                }
                const auto count = static_cast<std::size_t>(point_count);
                const auto new_width = checkedWidth(count);
                points.resize(count, value);
                if (static_cast<std::size_t>(width) * height != count)
                {
                    width = new_width;
                    height = 1;
                }
            }

            void resize(index_t cloud_width, index_t cloud_height, const PointT& value)
            {
                const auto dimensions = checkedDimensions(cloud_width, cloud_height);
                points.resize(dimensions.count, value);
                width = dimensions.width;
                height = dimensions.height;
            }

            void clear() noexcept
            {
                points.clear();
                width = 0;
                height = 0;
            }

            void push_back(const PointT& point)
            {
                const auto new_width = checkedWidth(points.size() + 1);
                points.push_back(point);
                width = new_width;
                height = 1;
            }

            void push_back(PointT&& point)
            {
                const auto new_width = checkedWidth(points.size() + 1);
                points.push_back(std::move(point));
                width = new_width;
                height = 1;
            }

            void transient_push_back(const PointT& point)
            {
                points.push_back(point);
            }

            template <typename... Args> PointT& emplace_back(Args&&... args)
            {
                const auto new_width = checkedWidth(points.size() + 1);
                points.emplace_back(std::forward<Args>(args)...);
                width = new_width;
                height = 1;
                return points.back();
            }

            template <typename... Args> PointT& transient_emplace_back(Args&&... args)
            {
                points.emplace_back(std::forward<Args>(args)...);
                return points.back();
            }

            void assign(index_t count, const PointT& value)
            {
                if (count < 0)
                {
                    throw std::invalid_argument("PointCloud: point count must be non-negative");
                }
                points.assign(static_cast<std::size_t>(count), value);
                setUnorganizedSize();
            }

            void assign(index_t cloud_width, index_t cloud_height, const PointT& value)
            {
                const auto dimensions = checkedDimensions(cloud_width, cloud_height);
                points.assign(dimensions.count, value);
                width = dimensions.width;
                height = dimensions.height;
            }

            template <typename InputIterator> void assign(InputIterator first, InputIterator last)
            {
                points.assign(first, last);
                setUnorganizedSize();
            }

            template <typename InputIterator> void assign(InputIterator first, InputIterator last, index_t new_width)
            {
                points.assign(first, last);
                setAssignedWidth(new_width);
            }

            void assign(std::initializer_list<PointT> values)
            {
                points.assign(values);
                setUnorganizedSize();
            }

            void assign(std::initializer_list<PointT> values, index_t new_width)
            {
                points.assign(values);
                setAssignedWidth(new_width);
            }

            iterator insert(iterator position, const PointT& point)
            {
                auto inserted = points.insert(position, point);
                setUnorganizedSize();
                return inserted;
            }

            iterator transient_insert(iterator position, const PointT& point)
            {
                return points.insert(position, point);
            }

            void insert(iterator position, std::size_t count, const PointT& point)
            {
                points.insert(position, count, point);
                setUnorganizedSize();
            }

            void transient_insert(iterator position, std::size_t count, const PointT& point)
            {
                points.insert(position, count, point);
            }

            template <typename InputIterator> void insert(iterator position, InputIterator first, InputIterator last)
            {
                points.insert(position, first, last);
                setUnorganizedSize();
            }

            template <typename InputIterator>
            void transient_insert(iterator position, InputIterator first, InputIterator last)
            {
                points.insert(position, first, last);
            }

            template <typename... Args> iterator emplace(iterator position, Args&&... args)
            {
                auto inserted = points.emplace(position, std::forward<Args>(args)...);
                setUnorganizedSize();
                return inserted;
            }

            template <typename... Args> iterator transient_emplace(iterator position, Args&&... args)
            {
                return points.emplace(position, std::forward<Args>(args)...);
            }

            iterator erase(iterator position)
            {
                auto following = points.erase(position);
                setUnorganizedSize();
                return following;
            }

            iterator erase(iterator first, iterator last)
            {
                auto following = points.erase(first, last);
                setUnorganizedSize();
                return following;
            }

            iterator transient_erase(iterator position)
            {
                return points.erase(position);
            }

            iterator transient_erase(iterator first, iterator last)
            {
                return points.erase(first, last);
            }

            PointT& operator[](std::size_t index)
            {
                return points[index];
            }
            const PointT& operator[](std::size_t index) const
            {
                return points[index];
            }
            PointT& at(std::size_t index)
            {
                return points.at(index);
            }
            const PointT& at(std::size_t index) const
            {
                return points.at(index);
            }

            PointT& at(int column, int row)
            {
                return points.at(organizedIndex(column, row));
            }

            const PointT& at(int column, int row) const
            {
                return points.at(organizedIndex(column, row));
            }

            PointT& operator()(std::size_t column, std::size_t row)
            {
                return points[row * width + column];
            }
            const PointT& operator()(std::size_t column, std::size_t row) const
            {
                return points[row * width + column];
            }

            PointT& front()
            {
                return points.front();
            }
            const PointT& front() const
            {
                return points.front();
            }
            PointT& back()
            {
                return points.back();
            }
            const PointT& back() const
            {
                return points.back();
            }

            iterator begin() noexcept
            {
                return points.begin();
            }
            iterator end() noexcept
            {
                return points.end();
            }
            const_iterator begin() const noexcept
            {
                return points.begin();
            }
            const_iterator end() const noexcept
            {
                return points.end();
            }
            const_iterator cbegin() const noexcept
            {
                return points.cbegin();
            }
            const_iterator cend() const noexcept
            {
                return points.cend();
            }
            reverse_iterator rbegin() noexcept
            {
                return points.rbegin();
            }
            reverse_iterator rend() noexcept
            {
                return points.rend();
            }
            const_reverse_iterator rbegin() const noexcept
            {
                return points.rbegin();
            }
            const_reverse_iterator rend() const noexcept
            {
                return points.rend();
            }
            const_reverse_iterator crbegin() const noexcept
            {
                return points.crbegin();
            }
            const_reverse_iterator crend() const noexcept
            {
                return points.crend();
            }

            Ptr makeShared() const
            {
                return std::make_shared<CloudT>(static_cast<const CloudT&>(*this));
            }

        private:
            static void validateFloatMap(int dim, int stride, int offset)
            {
                if (sizeof(PointT) % sizeof(float) != 0 || dim <= 0 || stride <= 0 || offset < 0 ||
                    stride > static_cast<int>(sizeof(PointT) / sizeof(float)) || offset > stride - dim)
                {
                    throw std::invalid_argument("PointCloud: invalid float matrix view layout");
                }
            }

            struct Dimensions
            {
                std::uint32_t width;
                std::uint32_t height;
                std::size_t count;
            };

            static Dimensions checkedDimensions(index_t cloud_width, index_t cloud_height)
            {
                if (cloud_width < 0 || cloud_height < 0)
                {
                    throw std::invalid_argument("PointCloud: organized dimensions must be non-negative");
                }
                const auto new_width = static_cast<std::uint32_t>(cloud_width);
                const auto new_height = static_cast<std::uint32_t>(cloud_height);
                return {new_width, new_height, checkedProduct(new_width, new_height)};
            }

            void setUnorganizedSize()
            {
                width = checkedWidth(points.size());
                height = 1;
            }

            void setAssignedWidth(index_t requested_width)
            {
                if (requested_width <= 0 || points.size() % static_cast<std::size_t>(requested_width) != 0)
                {
                    setUnorganizedSize();
                    return;
                }
                width = static_cast<std::uint32_t>(requested_width);
                height = checkedWidth(points.size() / width);
            }

            static std::uint32_t checkedWidth(std::size_t count)
            {
                if (count > std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::overflow_error("PointCloud: point count exceeds width range");
                }
                return static_cast<std::uint32_t>(count);
            }

            static std::size_t checkedProduct(std::uint32_t lhs, std::uint32_t rhs)
            {
                if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
                {
                    throw std::overflow_error("PointCloud: organized dimensions exceed size range");
                }
                return static_cast<std::size_t>(lhs) * rhs;
            }

            std::size_t organizedIndex(int column, int row) const
            {
                if (!isOrganized())
                {
                    throw UnorganizedPointCloudException("Can't use 2D indexing with an unorganized point cloud");
                }
                if (column < 0 || row < 0 || static_cast<std::size_t>(column) >= width ||
                    static_cast<std::size_t>(row) >= height)
                {
                    throw std::out_of_range("PointCloud: organized index is outside the cloud");
                }
                return static_cast<std::size_t>(row) * width + static_cast<std::size_t>(column);
            }
        };

    } // namespace detail

    /// Point-type cloud storage, including user-defined point types as in PCL.
    template <typename PointT> class PointCloud : public detail::PointCloudStorage<PointT, PointCloud<PointT>>
    {
        static_assert(!std::is_arithmetic_v<PointT>, "PointCloud requires a point structure type");

    public:
        using detail::PointCloudStorage<PointT, PointCloud>::PointCloudStorage;
    };

} // namespace plapoint
