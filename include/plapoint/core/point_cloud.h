#pragma once

#include <plamatrix/dense/dense_matrix.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace plapoint
{

template <typename Scalar, plamatrix::Device Dev>
class PointCloud
{
public:
    using MatrixType = plamatrix::DenseMatrix<Scalar, Dev>;

    PointCloud() : _points(0, 3) {}

    explicit PointCloud(size_t num_points) : _points(num_points, 3) {}

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

    MatrixType& points() { return _points; }

    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::CPU, PointCloud<Scalar, plamatrix::Device::GPU>>
    toGpu() const
    {
        return PointCloud<Scalar, plamatrix::Device::GPU>(_points.toGpu());
    }

    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::GPU, PointCloud<Scalar, plamatrix::Device::CPU>>
    toCpu() const
    {
        return PointCloud<Scalar, plamatrix::Device::CPU>(_points.toCpu());
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

    /// Set optional texture coordinates by copy (Nx2 matrix, must match point count)
    void setTextureCoords(const MatrixType& t)
    {
        if (t.rows() != _points.rows() || t.cols() != 2)
            throw std::runtime_error("Texture coords must match point count and be Nx2");
        _textureCoords = std::make_unique<MatrixType>(t.rows(), t.cols());
        for (plamatrix::Index r = 0; r < t.rows(); ++r)
            for (int col = 0; col < 2; ++col)
                _textureCoords->setValue(r, col, pointGet(t, r, col));
    }

    /// Set optional texture coordinates by move
    void setTextureCoords(MatrixType&& t)
    {
        if (t.rows() != _points.rows() || t.cols() != 2)
            throw std::runtime_error("Texture coords must match point count and be Nx2");
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
        _faceTextureIndices = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(std::move(ft));
    }

    bool hasFaceTextureIndices() const { return _faceTextureIndices != nullptr; }

    const plamatrix::DenseMatrix<int, Dev>* faceTextureIndices() const { return _faceTextureIndices.get(); }

    plamatrix::DenseMatrix<int, Dev>* faceTextureIndices() { return _faceTextureIndices.get(); }

private:
    template <typename T>
    static T pointGet(const plamatrix::DenseMatrix<T, Dev>& m, plamatrix::Index r, int c)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return m(r, c);
        else
            return m.getValue(r, c);
    }

    MatrixType _points;
    std::unique_ptr<MatrixType> _normals;
    std::unique_ptr<plamatrix::DenseMatrix<uint8_t, Dev>> _colors;
    std::unique_ptr<MatrixType> _textureCoords;
    std::unique_ptr<plamatrix::DenseMatrix<int, Dev>> _faces;
    std::unique_ptr<plamatrix::DenseMatrix<int, Dev>> _faceTextureIndices;
};

} // namespace plapoint
