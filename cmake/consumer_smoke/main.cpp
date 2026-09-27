#include <type_traits>

#include <plapoint/filters/voxel_grid.h>
#include <plapoint/io/pcd_io.h>
#include <plapoint/kdtree/kdtree_flann.h>
#include <plapoint/memory.h>
#include <plapoint/opencl/opencl_runtime.h>
#include <plapoint/point_cloud.h>
#include <plapoint/register_point_struct.h>
#include <plapoint/registration/icp.h>

struct EIGEN_ALIGN16 ConsumerPoint
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    PCL_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(ConsumerPoint, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity))

int main()
{
    using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
    plapoint::VoxelGrid<plapoint::PointXYZ> voxel;
    voxel.setLeafSize(Eigen::Vector4f::Ones());
    static_assert(std::is_same_v<decltype(voxel.getLeafSize()), Eigen::Vector3f>);

    plapoint::IterativeClosestPoint<plapoint::PointXYZ, plapoint::PointXYZ> icp;
    static_assert(std::is_same_v<decltype(icp.getFinalTransformation()), Eigen::Matrix4f>);
    Cloud cloud;
    cloud.emplace_back(0.0f, 0.0f, 0.0f);
    plapoint::KdTreeFLANN<plapoint::PointXYZ> tree;
    static_assert(std::is_void_v<decltype(tree.setInputCloud(cloud.makeShared()))>);
    tree.setInputCloud(cloud.makeShared());

    plapoint::PointCloud<ConsumerPoint> custom_cloud;
    custom_cloud.resize(1);
    custom_cloud.front().intensity = 1.0f;
    plapoint::PCLPointCloud2 blob;
    plapoint::toPCLPointCloud2(custom_cloud, blob);
    if (blob.fields.size() != 4)
    {
        return 1;
    }

    static_cast<void>(plapoint::opencl::hasUsableOpenClDevice());
    return 0;
}
