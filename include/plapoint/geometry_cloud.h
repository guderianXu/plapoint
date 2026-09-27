#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <plamatrix/dense/matrix.h>

namespace plapoint
{

/// CPU-owned Nx3 geometry with optional attributes, topology, and materials.
template <typename Scalar>
class GeometryCloud
{
    static_assert(std::is_floating_point_v<Scalar>, "GeometryCloud scalar must be floating point");
public:
    using MatrixType = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;
    template <typename Value>
    using AttributeMatrix = plamatrix::Matrix<Value, plamatrix::Dynamic, plamatrix::Dynamic>;

    /// Scoped mutable access to point positions. The cloud must outlive the editor and
    /// remain at the same address while the editor is alive.
    /// Point-derived caches are invalidated when editing begins and again when it ends.
    class PointEdit
    {
        friend class GeometryCloud;

    public:
        PointEdit(const PointEdit&) = delete;
        PointEdit& operator=(const PointEdit&) = delete;

        PointEdit(PointEdit&& other) noexcept
            : _cloud(std::exchange(other._cloud, nullptr))
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
                throw std::logic_error("GeometryCloud point editor has been moved from");
            }
            return _cloud->_points;
        }

        MatrixType& operator*() { return get(); }
        MatrixType* operator->() { return &get(); }

    private:
        explicit PointEdit(GeometryCloud& cloud) : _cloud(&cloud)
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

        GeometryCloud* _cloud = nullptr;
    };

    /// Lightweight read-only accessor for one point and its optional attributes.
    class PointView
    {
    public:
        Scalar x() const { return _cloud.readValue(_cloud.points(), static_cast<plamatrix::Index>(_idx), 0); }
        Scalar y() const { return _cloud.readValue(_cloud.points(), static_cast<plamatrix::Index>(_idx), 1); }
        Scalar z() const { return _cloud.readValue(_cloud.points(), static_cast<plamatrix::Index>(_idx), 2); }

        uint8_t r() const { requireColors(); return _cloud.readValue(*_cloud.colors(), static_cast<plamatrix::Index>(_idx), 0); }
        uint8_t g() const { requireColors(); return _cloud.readValue(*_cloud.colors(), static_cast<plamatrix::Index>(_idx), 1); }
        uint8_t b() const { requireColors(); return _cloud.readValue(*_cloud.colors(), static_cast<plamatrix::Index>(_idx), 2); }

        std::uint16_t intensity() const { requireIntensities(); return _cloud.readValue(*_cloud.intensities(), static_cast<plamatrix::Index>(_idx), 0); }

        Scalar nx() const { requireNormals(); return _cloud.readValue(*_cloud.normals(), static_cast<plamatrix::Index>(_idx), 0); }
        Scalar ny() const { requireNormals(); return _cloud.readValue(*_cloud.normals(), static_cast<plamatrix::Index>(_idx), 1); }
        Scalar nz() const { requireNormals(); return _cloud.readValue(*_cloud.normals(), static_cast<plamatrix::Index>(_idx), 2); }

        Scalar u() const { requireTextureCoords(); return _cloud.readValue(*_cloud.textureCoords(), static_cast<plamatrix::Index>(_idx), 0); }
        Scalar v() const { requireTextureCoords(); return _cloud.readValue(*_cloud.textureCoords(), static_cast<plamatrix::Index>(_idx), 1); }

        Scalar scalar(const std::string& name) const
        {
            const int field_index = _cloud.scalarFieldIndex(name);
            if (field_index < 0 || !_cloud.scalarFields())
            {
                throw std::runtime_error("PointView: cloud has no scalar field named " + name);
            }
            return _cloud.readValue(*_cloud.scalarFields(),
                static_cast<plamatrix::Index>(_idx), field_index);
        }

    private:
        friend class GeometryCloud;
        PointView(const GeometryCloud& cloud, size_t idx) : _cloud(cloud), _idx(idx) {}

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

        const GeometryCloud& _cloud;
        size_t _idx;
    };

    PointView operator[](size_t idx) const
    {
        if (idx >= size())
        {
            throw std::out_of_range("GeometryCloud: point index out of range");
        }
        return PointView(*this, idx);
    }

    /// Construct an empty Nx3 point cloud.
    GeometryCloud() : _points(0, 3) {}

    /// Construct an Nx3 point cloud with all coordinates value-initialized.
    explicit GeometryCloud(size_t num_points)
        : _points(static_cast<plamatrix::Index>(num_points), 3) {}

    /// Construct from an Nx3 matrix, throwing if the matrix does not have three columns.
    explicit GeometryCloud(MatrixType&& pts)
        : _points(std::move(pts))
    {
        if (_points.cols() != 3)
        {
            throw std::runtime_error("GeometryCloud requires Nx3 matrix");
        }
    }

    GeometryCloud(GeometryCloud&&) noexcept = default;

    GeometryCloud& operator=(GeometryCloud&& other) noexcept
    {
        if (this != &other)
        {
            GeometryCloud next(std::move(other));
            swap(next);
        }
        return *this;
    }

    void swap(GeometryCloud& other) noexcept
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
    }

    size_t size() const { return _points.rows(); }

    const MatrixType& points() const { return _points; }

    /// Nonzero generation for caches derived from point positions; wraps from UINT64_MAX to one.
    std::uint64_t pointsRevision() const noexcept { return _points_revision; }

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
    bool hasActivePointEdit() const noexcept { return _active_point_edits != 0; }

    /// True when point-derived caches may be reused without inspecting point contents.
    bool pointCachesReusable() const noexcept
    {
        return !_mutable_points_alias_issued && !hasActivePointEdit();
    }

    /// Return mutable point storage and invalidate derived point caches.
    /// The returned reference may outlive the call, so point-derived caches must thereafter
    /// validate conservatively. Prefer editPoints() for bounded mutations.
    MatrixType& points()
    {
        advancePointsRevision();
        _mutable_points_alias_issued = true;
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
            throw std::runtime_error("GeometryCloud requires Nx3 matrix");
        }
        validateReplacementPointCount(points.rows());

        setPoints(MatrixType(points));
    }

    /// Replace point positions by move, advancing the cache identity only after a successful replacement.
    void setPoints(MatrixType&& points)
    {
        if (points.cols() != 3)
        {
            throw std::runtime_error("GeometryCloud requires Nx3 matrix");
        }
        validateReplacementPointCount(points.rows());

        _points = std::move(points);
        advancePointsRevision();
    }

    /// Set optional normals by copy (Nx3 matrix, must match point count)
    void setNormals(const MatrixType& n)
    {
        if (n.rows() != _points.rows() || n.cols() != 3)
            throw std::runtime_error("Normals must match point count and be Nx3");
        _normals = std::make_unique<MatrixType>(n);
    }

    /// Set optional normals by move
    void setNormals(MatrixType&& n)
    {
        if (n.rows() != _points.rows() || n.cols() != 3)
            throw std::runtime_error("Normals must match point count and be Nx3");
        _normals = std::make_unique<MatrixType>(std::move(n));
    }

    bool hasNormals() const { return _normals != nullptr; }

    const MatrixType* normals() const { return _normals.get(); }

    MatrixType* normals() { return _normals.get(); }

    /// Set optional RGB colors by copy (Nx3 uint8 matrix)
    void setColors(const AttributeMatrix<uint8_t>& c)
    {
        if (c.rows() != _points.rows() || c.cols() != 3)
            throw std::runtime_error("Colors must match point count and be Nx3");
        _colors = std::make_unique<AttributeMatrix<uint8_t>>(c);
    }

    /// Set optional RGB colors by move
    void setColors(AttributeMatrix<uint8_t>&& c)
    {
        if (c.rows() != _points.rows() || c.cols() != 3)
            throw std::runtime_error("Colors must match point count and be Nx3");
        _colors = std::make_unique<AttributeMatrix<uint8_t>>(std::move(c));
    }

    bool hasColors() const { return _colors != nullptr; }

    const AttributeMatrix<uint8_t>* colors() const { return _colors.get(); }

    AttributeMatrix<uint8_t>* colors() { return _colors.get(); }

    /// Set optional intensity values by copy (Nx1 uint16 matrix)
    void setIntensities(const AttributeMatrix<std::uint16_t>& values)
    {
        if (values.rows() != _points.rows() || values.cols() != 1)
            throw std::runtime_error("Intensities must match point count and be Nx1");
        _intensities = std::make_unique<AttributeMatrix<std::uint16_t>>(values);
    }

    /// Set optional intensity values by move
    void setIntensities(AttributeMatrix<std::uint16_t>&& values)
    {
        if (values.rows() != _points.rows() || values.cols() != 1)
            throw std::runtime_error("Intensities must match point count and be Nx1");
        _intensities = std::make_unique<AttributeMatrix<std::uint16_t>>(std::move(values));
    }

    bool hasIntensities() const { return _intensities != nullptr; }

    const AttributeMatrix<std::uint16_t>* intensities() const { return _intensities.get(); }

    AttributeMatrix<std::uint16_t>* intensities() { return _intensities.get(); }

    /// Set optional named scalar fields by copy (NxK matrix, one name per column).
    void setScalarFields(const std::vector<std::string>& names, const MatrixType& values)
    {
        validateScalarFields(names, values);
        auto replacement = std::make_unique<MatrixType>(values);
        auto replacement_names = names;
        _scalarFields = std::move(replacement);
        _scalarFieldNames = std::move(replacement_names);
    }

    /// Set optional named scalar fields by move (NxK matrix, one name per column).
    void setScalarFields(std::vector<std::string> names, MatrixType&& values)
    {
        validateScalarFields(names, values);
        _scalarFieldNames = std::move(names);
        _scalarFields = std::make_unique<MatrixType>(std::move(values));
    }

    bool hasScalarFields() const { return _scalarFields != nullptr && !_scalarFieldNames.empty(); }

    bool hasScalarField(const std::string& name) const { return scalarFieldIndex(name) >= 0; }

    int scalarFieldIndex(const std::string& name) const
    {
        const auto it = std::find(_scalarFieldNames.begin(), _scalarFieldNames.end(), name);
        if (it == _scalarFieldNames.end())
        {
            return -1;
        }
        return static_cast<int>(std::distance(_scalarFieldNames.begin(), it));
    }

    const std::vector<std::string>& scalarFieldNames() const { return _scalarFieldNames; }

    const MatrixType* scalarFields() const { return _scalarFields.get(); }

    MatrixType* scalarFields() { return _scalarFields.get(); }

    /// Set optional texture coordinates by copy (Tx2 UV table).
    void setTextureCoords(const MatrixType& t)
    {
        if (t.cols() != 2)
            throw std::runtime_error("Texture coords must be Tx2");
        if (_faceTextureIndices)
            validateIndexMatrix(*_faceTextureIndices, t.rows(), "Face texture");
        _textureCoords = std::make_unique<MatrixType>(t);
    }

    /// Set optional texture coordinates by move
    void setTextureCoords(MatrixType&& t)
    {
        if (t.cols() != 2)
            throw std::runtime_error("Texture coords must be Tx2");
        if (_faceTextureIndices)
            validateIndexMatrix(*_faceTextureIndices, t.rows(), "Face texture");
        _textureCoords = std::make_unique<MatrixType>(std::move(t));
    }

    bool hasTextureCoords() const { return _textureCoords != nullptr; }

    const MatrixType* textureCoords() const { return _textureCoords.get(); }

    MatrixType* textureCoords() { return _textureCoords.get(); }

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
        _faces = std::make_unique<AttributeMatrix<int>>(f);
    }

    /// Set optional faces by move
    void setFaces(AttributeMatrix<int>&& f)
    {
        if (f.cols() != 3)
            throw std::runtime_error("Faces must be Fx3");
        validateIndexMatrix(f, _points.rows(), "Faces");
        validateFaceCountForExistingTextureIndices(f.rows());
        _faces = std::make_unique<AttributeMatrix<int>>(std::move(f));
    }

    bool hasFaces() const { return _faces != nullptr; }

    const AttributeMatrix<int>* faces() const { return _faces.get(); }

    AttributeMatrix<int>* faces() { return _faces.get(); }

    /// Set optional face texture indices by copy (Fx3 int matrix)
    void setFaceTextureIndices(const AttributeMatrix<int>& ft)
    {
        if (ft.cols() != 3)
            throw std::runtime_error("Face texture indices must be Fx3");
        validateFaceTextureIndices(ft);
        _faceTextureIndices = std::make_unique<AttributeMatrix<int>>(ft);
    }

    /// Set optional face texture indices by move
    void setFaceTextureIndices(AttributeMatrix<int>&& ft)
    {
        if (ft.cols() != 3)
            throw std::runtime_error("Face texture indices must be Fx3");
        validateFaceTextureIndices(ft);
        _faceTextureIndices = std::make_unique<AttributeMatrix<int>>(std::move(ft));
    }

    bool hasFaceTextureIndices() const { return _faceTextureIndices != nullptr; }

    const AttributeMatrix<int>* faceTextureIndices() const { return _faceTextureIndices.get(); }

    AttributeMatrix<int>* faceTextureIndices() { return _faceTextureIndices.get(); }

    const std::string& materialLibraryFile() const { return _materialLibraryFile; }
    void setMaterialLibraryFile(const std::string& f) { _materialLibraryFile = f; }

    const std::string& textureImageFile() const { return _textureImageFile; }
    void setTextureImageFile(const std::string& f) { _textureImageFile = f; }

    /// Validate all point, attribute, topology, and texture-index shapes.
    /// Call this at API boundaries before accessing attribute storage directly.
    void validate() const
    {
        validateStructure();
    }

private:
    template <typename Matrix>
    static auto readValue(const Matrix& matrix, plamatrix::Index row, plamatrix::Index col)
    {
        return matrix(row, col);
    }

    void beginPointEdit() noexcept
    {
        ++_active_point_edits;
        advancePointsRevision();
    }

    void finishPointEdit() noexcept
    {
        if (_active_point_edits > 0)
        {
            --_active_point_edits;
        }
        advancePointsRevision();
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
        if (_normals || _colors || _intensities || _scalarFields || _textureCoords
            || _faces || _faceTextureIndices)
        {
            throw std::runtime_error(
                "GeometryCloud cannot change point count while point or face attributes are present");
        }
    }

    static void validateIndexMatrix(const AttributeMatrix<int>& m,
                                    plamatrix::Index exclusive_limit,
                                    const char* label)
    {
        validateCpuIndexMatrix(m, exclusive_limit, label);
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
        return name == "x" || name == "y" || name == "z" ||
               name == "nx" || name == "ny" || name == "nz" ||
               name == "red" || name == "green" || name == "blue" ||
               name == "intensity";
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
                throw std::runtime_error("Scalar field name conflicts with a built-in point property: " + names[i]);
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
        if (_points.cols() != 3)
            throw std::runtime_error("GeometryCloud points must be Nx3");
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
};

} // namespace plapoint
