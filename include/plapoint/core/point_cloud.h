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

#include <plamatrix/dense/dense_matrix.h>

namespace plapoint
{

/// Nx3 point cloud with optional normals, colors, texture coordinates, faces, and CPU/GPU transfer helpers.
template <typename Scalar, plamatrix::Device Dev>
class PointCloud
{
    template <typename, plamatrix::Device>
    friend class PointCloud;

public:
    using MatrixType = plamatrix::DenseMatrix<Scalar, Dev>;

    /// Lightweight read-only accessor for one point and its optional attributes.
    class PointView
    {
    public:
        Scalar x() const { return _cloud.points().getValue(static_cast<plamatrix::Index>(_idx), 0); }
        Scalar y() const { return _cloud.points().getValue(static_cast<plamatrix::Index>(_idx), 1); }
        Scalar z() const { return _cloud.points().getValue(static_cast<plamatrix::Index>(_idx), 2); }

        uint8_t r() const { requireColors(); return _cloud.colors()->getValue(static_cast<plamatrix::Index>(_idx), 0); }
        uint8_t g() const { requireColors(); return _cloud.colors()->getValue(static_cast<plamatrix::Index>(_idx), 1); }
        uint8_t b() const { requireColors(); return _cloud.colors()->getValue(static_cast<plamatrix::Index>(_idx), 2); }

        std::uint16_t intensity() const { requireIntensities(); return _cloud.intensities()->getValue(static_cast<plamatrix::Index>(_idx), 0); }

        Scalar nx() const { requireNormals(); return _cloud.normals()->getValue(static_cast<plamatrix::Index>(_idx), 0); }
        Scalar ny() const { requireNormals(); return _cloud.normals()->getValue(static_cast<plamatrix::Index>(_idx), 1); }
        Scalar nz() const { requireNormals(); return _cloud.normals()->getValue(static_cast<plamatrix::Index>(_idx), 2); }

        Scalar u() const { requireTextureCoords(); return _cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(_idx), 0); }
        Scalar v() const { requireTextureCoords(); return _cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(_idx), 1); }

        Scalar scalar(const std::string& name) const
        {
            const int field_index = _cloud.scalarFieldIndex(name);
            if (field_index < 0 || !_cloud.scalarFields())
            {
                throw std::runtime_error("PointView: cloud has no scalar field named " + name);
            }
            return _cloud.scalarFields()->getValue(
                static_cast<plamatrix::Index>(_idx), field_index);
        }

    private:
        friend class PointCloud;
        PointView(const PointCloud& cloud, size_t idx) : _cloud(cloud), _idx(idx) {}

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

        const PointCloud& _cloud;
        size_t _idx;
    };

    PointView operator[](size_t idx) const
    {
        if (idx >= size())
        {
            throw std::out_of_range("PointCloud: point index out of range");
        }
        return PointView(*this, idx);
    }

    /// Construct an empty Nx3 point cloud.
    PointCloud() : _points(0, 3) {}

    /// Construct an Nx3 point cloud with all coordinates value-initialized.
    explicit PointCloud(size_t num_points) : _points(num_points, 3) {}

    /// Construct from an Nx3 matrix, throwing if the matrix does not have three columns.
    explicit PointCloud(MatrixType&& pts)
    {
        if (pts.cols() != 3)
        {
            throw std::runtime_error("PointCloud requires Nx3 matrix");
        }
        _points = std::move(pts);
    }

    size_t size() const { return _points.rows(); }

    const MatrixType& points() const { return _points; }

    std::uint64_t pointsVersion() const { return _points_version; }

    /// Return mutable point storage and invalidate cached CPU mirrors; callers must preserve Nx3 shape.
    MatrixType& points()
    {
        ++_points_version;
        invalidateCpuMirror();
        return _points;
    }

    /// Return a CPU-readable view of points. GPU clouds cache the transfer until mutable points() is requested.
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& pointsCpu() const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return _points;
        }
        else
        {
            if (!_points_cpu_cache)
            {
                _points_cpu_cache = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(
                    _points.toCpu());
            }
            return *_points_cpu_cache;
        }
    }

    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::CPU, PointCloud<Scalar, plamatrix::Device::GPU>>
    toGpu() const
    {
        validateStructure();
        PointCloud<Scalar, plamatrix::Device::GPU> result(_points.toGpu());
        if (_normals) result._normals = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>>(_normals->toGpu());
        if (_colors) result._colors = std::make_unique<plamatrix::DenseMatrix<uint8_t, plamatrix::Device::GPU>>(_colors->toGpu());
        if (_intensities) result._intensities = std::make_unique<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::GPU>>(_intensities->toGpu());
        if (_scalarFields)
        {
            result._scalarFields = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>>(_scalarFields->toGpu());
            result._scalarFieldNames = _scalarFieldNames;
        }
        if (_textureCoords) result._textureCoords = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>>(_textureCoords->toGpu());
        if (_faces) result._faces = std::make_unique<plamatrix::DenseMatrix<int, plamatrix::Device::GPU>>(_faces->toGpu());
        if (_faceTextureIndices) result._faceTextureIndices = std::make_unique<plamatrix::DenseMatrix<int, plamatrix::Device::GPU>>(_faceTextureIndices->toGpu());
        result.setMaterialLibraryFile(_materialLibraryFile);
        result.setTextureImageFile(_textureImageFile);
        result.validateStructure();
        return result;
    }

    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::GPU, PointCloud<Scalar, plamatrix::Device::CPU>>
    toCpu() const
    {
        validateStructure();
        PointCloud<Scalar, plamatrix::Device::CPU> result(_points.toCpu());
        if (_normals) result._normals = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(_normals->toCpu());
        if (_colors) result._colors = std::make_unique<plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU>>(_colors->toCpu());
        if (_intensities) result._intensities = std::make_unique<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>>(_intensities->toCpu());
        if (_scalarFields)
        {
            result._scalarFields = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(_scalarFields->toCpu());
            result._scalarFieldNames = _scalarFieldNames;
        }
        if (_textureCoords) result._textureCoords = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(_textureCoords->toCpu());
        if (_faces) result._faces = std::make_unique<plamatrix::DenseMatrix<int, plamatrix::Device::CPU>>(_faces->toCpu());
        if (_faceTextureIndices) result._faceTextureIndices = std::make_unique<plamatrix::DenseMatrix<int, plamatrix::Device::CPU>>(_faceTextureIndices->toCpu());
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
        _normals = std::make_unique<MatrixType>(n.rows(), n.cols());
        for (plamatrix::Index r = 0; r < n.rows(); ++r)
            for (int c = 0; c < 3; ++c)
                _normals->setValue(r, c, pointGet(n, r, c));
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
    void setColors(const plamatrix::DenseMatrix<uint8_t, Dev>& c)
    {
        if (c.rows() != _points.rows() || c.cols() != 3)
            throw std::runtime_error("Colors must match point count and be Nx3");
        _colors = std::make_unique<plamatrix::DenseMatrix<uint8_t, Dev>>(c.rows(), c.cols());
        for (plamatrix::Index r = 0; r < c.rows(); ++r)
            for (int col = 0; col < 3; ++col)
                _colors->setValue(r, col, pointGet(c, r, col));
    }

    /// Set optional RGB colors by move
    void setColors(plamatrix::DenseMatrix<uint8_t, Dev>&& c)
    {
        if (c.rows() != _points.rows() || c.cols() != 3)
            throw std::runtime_error("Colors must match point count and be Nx3");
        _colors = std::make_unique<plamatrix::DenseMatrix<uint8_t, Dev>>(std::move(c));
    }

    bool hasColors() const { return _colors != nullptr; }

    const plamatrix::DenseMatrix<uint8_t, Dev>* colors() const { return _colors.get(); }

    plamatrix::DenseMatrix<uint8_t, Dev>* colors() { return _colors.get(); }

    /// Set optional intensity values by copy (Nx1 uint16 matrix)
    void setIntensities(const plamatrix::DenseMatrix<std::uint16_t, Dev>& values)
    {
        if (values.rows() != _points.rows() || values.cols() != 1)
            throw std::runtime_error("Intensities must match point count and be Nx1");
        _intensities = std::make_unique<plamatrix::DenseMatrix<std::uint16_t, Dev>>(values.rows(), values.cols());
        for (plamatrix::Index r = 0; r < values.rows(); ++r)
            _intensities->setValue(r, 0, pointGet(values, r, 0));
    }

    /// Set optional intensity values by move
    void setIntensities(plamatrix::DenseMatrix<std::uint16_t, Dev>&& values)
    {
        if (values.rows() != _points.rows() || values.cols() != 1)
            throw std::runtime_error("Intensities must match point count and be Nx1");
        _intensities = std::make_unique<plamatrix::DenseMatrix<std::uint16_t, Dev>>(std::move(values));
    }

    bool hasIntensities() const { return _intensities != nullptr; }

    const plamatrix::DenseMatrix<std::uint16_t, Dev>* intensities() const { return _intensities.get(); }

    plamatrix::DenseMatrix<std::uint16_t, Dev>* intensities() { return _intensities.get(); }

    /// Set optional named scalar fields by copy (NxK matrix, one name per column).
    void setScalarFields(const std::vector<std::string>& names, const MatrixType& values)
    {
        validateScalarFields(names, values);
        _scalarFields = std::make_unique<MatrixType>(values.rows(), values.cols());
        for (plamatrix::Index r = 0; r < values.rows(); ++r)
        {
            for (int c = 0; c < values.cols(); ++c)
            {
                _scalarFields->setValue(r, c, pointGet(values, r, c));
            }
        }
        _scalarFieldNames = names;
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
        _textureCoords = std::make_unique<MatrixType>(t.rows(), t.cols());
        for (plamatrix::Index r = 0; r < t.rows(); ++r)
            for (int col = 0; col < 2; ++col)
                _textureCoords->setValue(r, col, pointGet(t, r, col));
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

    /// Set optional faces by copy (Fx3 int matrix)
    void setFaces(const plamatrix::DenseMatrix<int, Dev>& f)
    {
        if (f.cols() != 3)
            throw std::runtime_error("Faces must be Fx3");
        validateIndexMatrix(f, _points.rows(), "Faces");
        validateFaceCountForExistingTextureIndices(f.rows());
        _faces = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(f.rows(), f.cols());
        for (plamatrix::Index r = 0; r < f.rows(); ++r)
            for (int col = 0; col < 3; ++col)
                _faces->setValue(r, col, pointGet(f, r, col));
    }

    /// Set optional faces by move
    void setFaces(plamatrix::DenseMatrix<int, Dev>&& f)
    {
        if (f.cols() != 3)
            throw std::runtime_error("Faces must be Fx3");
        validateIndexMatrix(f, _points.rows(), "Faces");
        validateFaceCountForExistingTextureIndices(f.rows());
        _faces = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(std::move(f));
    }

    bool hasFaces() const { return _faces != nullptr; }

    const plamatrix::DenseMatrix<int, Dev>* faces() const { return _faces.get(); }

    plamatrix::DenseMatrix<int, Dev>* faces() { return _faces.get(); }

    /// Set optional face texture indices by copy (Fx3 int matrix)
    void setFaceTextureIndices(const plamatrix::DenseMatrix<int, Dev>& ft)
    {
        if (ft.cols() != 3)
            throw std::runtime_error("Face texture indices must be Fx3");
        validateFaceTextureIndices(ft);
        _faceTextureIndices = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(ft.rows(), ft.cols());
        for (plamatrix::Index r = 0; r < ft.rows(); ++r)
            for (int col = 0; col < 3; ++col)
                _faceTextureIndices->setValue(r, col, pointGet(ft, r, col));
    }

    /// Set optional face texture indices by move
    void setFaceTextureIndices(plamatrix::DenseMatrix<int, Dev>&& ft)
    {
        if (ft.cols() != 3)
            throw std::runtime_error("Face texture indices must be Fx3");
        validateFaceTextureIndices(ft);
        _faceTextureIndices = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(std::move(ft));
    }

    bool hasFaceTextureIndices() const { return _faceTextureIndices != nullptr; }

    const plamatrix::DenseMatrix<int, Dev>* faceTextureIndices() const { return _faceTextureIndices.get(); }

    plamatrix::DenseMatrix<int, Dev>* faceTextureIndices() { return _faceTextureIndices.get(); }

    const std::string& materialLibraryFile() const { return _materialLibraryFile; }
    void setMaterialLibraryFile(const std::string& f) { _materialLibraryFile = f; }

    const std::string& textureImageFile() const { return _textureImageFile; }
    void setTextureImageFile(const std::string& f) { _textureImageFile = f; }

private:
    template <typename T>
    static T pointGet(const plamatrix::DenseMatrix<T, Dev>& m, plamatrix::Index r, int c)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return m(r, c);
        else
            return m.getValue(r, c);
    }

    void invalidateCpuMirror()
    {
        if constexpr (Dev == plamatrix::Device::GPU)
        {
            _points_cpu_cache.reset();
        }
    }

    static void validateIndexMatrix(const plamatrix::DenseMatrix<int, Dev>& m,
                                    plamatrix::Index exclusive_limit,
                                    const char* label)
    {
        if constexpr (Dev == plamatrix::Device::GPU)
        {
            validateCpuIndexMatrix(m.toCpu(), exclusive_limit, label);
        }
        else
        {
            validateCpuIndexMatrix(m, exclusive_limit, label);
        }
    }

    static void validateCpuIndexMatrix(const plamatrix::DenseMatrix<int, plamatrix::Device::CPU>& m,
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

    void validateFaceTextureIndices(const plamatrix::DenseMatrix<int, Dev>& ft) const
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
            throw std::runtime_error("PointCloud points must be Nx3");
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

    bool hasPointAlignedTextureCoords() const
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
                if (_faceTextureIndices->getValue(r, col) != _faces->getValue(r, col))
                {
                    return false;
                }
            }
        }
        return true;
    }

    MatrixType _points;
    std::unique_ptr<MatrixType> _normals;
    std::unique_ptr<plamatrix::DenseMatrix<uint8_t, Dev>> _colors;
    std::unique_ptr<plamatrix::DenseMatrix<std::uint16_t, Dev>> _intensities;
    std::unique_ptr<MatrixType> _scalarFields;
    std::vector<std::string> _scalarFieldNames;
    std::unique_ptr<MatrixType> _textureCoords;
    std::unique_ptr<plamatrix::DenseMatrix<int, Dev>> _faces;
    std::unique_ptr<plamatrix::DenseMatrix<int, Dev>> _faceTextureIndices;
    std::string _materialLibraryFile;
    std::string _textureImageFile;
    std::uint64_t _points_version = 0;
    mutable std::unique_ptr<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>> _points_cpu_cache;
};

} // namespace plapoint
