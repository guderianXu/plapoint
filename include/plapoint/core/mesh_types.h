#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <plapoint/core/point_cloud_blob.h>

namespace plapoint
{

    struct Vertices
    {
        using Ptr = std::shared_ptr<Vertices>;
        using ConstPtr = std::shared_ptr<const Vertices>;

        Indices vertices;
    };

    using VerticesPtr = Vertices::Ptr;
    using VerticesConstPtr = Vertices::ConstPtr;

    struct PolygonMesh
    {
        using Ptr = std::shared_ptr<PolygonMesh>;
        using ConstPtr = std::shared_ptr<const PolygonMesh>;

        PCLHeader header;
        PCLPointCloud2 cloud;
        std::vector<Vertices> polygons;

        static bool concatenate(PolygonMesh& lhs, const PolygonMesh& rhs)
        {
            const std::uint64_t point_offset = static_cast<std::uint64_t>(lhs.cloud.width) * lhs.cloud.height;
            if (!PCLPointCloud2::concatenate(lhs.cloud, rhs.cloud))
            {
                return false;
            }
            lhs.header.stamp = std::max(lhs.header.stamp, rhs.header.stamp);
            for (auto polygon : rhs.polygons)
            {
                for (auto& vertex : polygon.vertices)
                {
                    vertex += static_cast<index_t>(point_offset);
                }
                lhs.polygons.push_back(std::move(polygon));
            }
            return true;
        }

        static bool concatenate(const PolygonMesh& lhs, const PolygonMesh& rhs, PolygonMesh& output)
        {
            output = lhs;
            return concatenate(output, rhs);
        }

        PolygonMesh& operator+=(const PolygonMesh& rhs)
        {
            concatenate(*this, rhs);
            return *this;
        }

        const PolygonMesh operator+(const PolygonMesh& rhs)
        {
            PolygonMesh output(*this);
            output += rhs;
            return output;
        }
    };

    using PolygonMeshPtr = PolygonMesh::Ptr;
    using PolygonMeshConstPtr = PolygonMesh::ConstPtr;

} // namespace plapoint
