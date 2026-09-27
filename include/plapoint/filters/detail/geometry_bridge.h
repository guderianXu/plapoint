#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/geometry_cloud.h>

namespace plapoint::detail
{

    template <typename Scalar>
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>
    toDeviceCloud(const GeometryCloud<Scalar>& source)
    {
        source.validate();
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> coordinates = source.points();
        plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> cloud(std::move(coordinates));
        if (source.hasNormals())
            cloud.setNormals(*source.normals());
        if (source.hasColors())
            cloud.setColors(*source.colors());
        if (source.hasIntensities())
            cloud.setIntensities(*source.intensities());
        if (source.hasScalarFields())
            cloud.setScalarFields(source.scalarFieldNames(), *source.scalarFields());
        if (source.hasTextureCoords())
            cloud.setTextureCoords(*source.textureCoords());
        if (source.hasFaces())
            cloud.setFaces(*source.faces());
        if (source.hasFaceTextureIndices())
            cloud.setFaceTextureIndices(*source.faceTextureIndices());
        cloud.setMaterialLibraryFile(source.materialLibraryFile());
        cloud.setTextureImageFile(source.textureImageFile());
        return cloud;
    }

    template <typename Scalar>
    GeometryCloud<Scalar>
    fromDeviceCloud(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& source)
    {
        source.validate();
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> coordinates = source.pointsCpu();
        GeometryCloud<Scalar> cloud(std::move(coordinates));
        if (source.hasNormals())
            cloud.setNormals(*source.normals());
        if (source.hasColors())
            cloud.setColors(*source.colors());
        if (source.hasIntensities())
            cloud.setIntensities(*source.intensities());
        if (source.hasScalarFields())
            cloud.setScalarFields(source.scalarFieldNames(), *source.scalarFields());
        if (source.hasTextureCoords())
            cloud.setTextureCoords(*source.textureCoords());
        if (source.hasFaces())
            cloud.setFaces(*source.faces());
        if (source.hasFaceTextureIndices())
            cloud.setFaceTextureIndices(*source.faceTextureIndices());
        cloud.setMaterialLibraryFile(source.materialLibraryFile());
        cloud.setTextureImageFile(source.textureImageFile());
        return cloud;
    }

} // namespace plapoint::detail
