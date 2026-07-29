#include <gtest/gtest.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/search/kdtree.h>

#include <plamatrix/plamatrix.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/filter_compaction.h>
#include <plapoint/gpu/filter_indices.h>

namespace
{

bool hasCudaDeviceForFilterIndices()
{
    return plapoint::gpu::hasUsableCudaDevice();
}

template <typename T>
void expectVectorEq(const std::vector<T>& actual, const std::vector<T>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_EQ(actual[i], expected[i]) << "at index " << i;
    }
}

template <typename Scalar>
std::shared_ptr<plapoint::PointCloud<Scalar, plamatrix::Device::CPU>> makeRadiusCloudWithAttributes()
{
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(4, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(4, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(4, 1);
    for (int i = 0; i < 4; ++i)
    {
        points.setValue(i, 0, i < 3 ? Scalar(i) * Scalar(0.1) : Scalar(10));
        points.setValue(i, 1, Scalar(0));
        points.setValue(i, 2, Scalar(0));
        colors.setValue(i, 0, static_cast<std::uint8_t>(10 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(20 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(30 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(100 + i));
    }

    auto cloud = std::make_shared<CpuCloud>(std::move(points));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));
    return cloud;
}

template <typename Scalar>
std::shared_ptr<plapoint::PointCloud<Scalar, plamatrix::Device::CPU>> makeSorCloudWithAttributes()
{
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(5, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(5, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(5, 1);
    for (int i = 0; i < 4; ++i)
    {
        points.setValue(i, 0, Scalar(i) * Scalar(0.01));
        points.setValue(i, 1, Scalar(0));
        points.setValue(i, 2, Scalar(0));
    }
    points.setValue(4, 0, Scalar(100));
    points.setValue(4, 1, Scalar(0));
    points.setValue(4, 2, Scalar(0));
    for (int i = 0; i < 5; ++i)
    {
        colors.setValue(i, 0, static_cast<std::uint8_t>(40 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(50 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(60 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(200 + i));
    }

    auto cloud = std::make_shared<CpuCloud>(std::move(points));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));
    return cloud;
}

template <typename Scalar>
plapoint::PointCloud<Scalar, plamatrix::Device::CPU> makeCompleteAttributeCloud()
{
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    constexpr int count = 6;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(count, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(count, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(count, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(count, 1);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> scalar_fields(count, 2);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> texture_coords(count, 2);
    const Scalar x[count] = {
        Scalar(0), Scalar(0), Scalar(1), Scalar(2), Scalar(10),
        std::numeric_limits<Scalar>::quiet_NaN()};
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, x[i]);
        points.setValue(i, 1, Scalar(0));
        points.setValue(i, 2, Scalar(0));
        normals.setValue(i, 0, Scalar(100 + i));
        normals.setValue(i, 1, Scalar(200 + i));
        normals.setValue(i, 2, Scalar(300 + i));
        colors.setValue(i, 0, static_cast<std::uint8_t>(10 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(20 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(30 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(1000 + i));
        scalar_fields.setValue(i, 0, Scalar(i) + Scalar(0.25));
        scalar_fields.setValue(i, 1, Scalar(500 + i));
        texture_coords.setValue(i, 0, Scalar(i) + Scalar(0.1));
        texture_coords.setValue(i, 1, Scalar(i) + Scalar(0.2));
    }
    CpuCloud cloud(std::move(points));
    cloud.setNormals(std::move(normals));
    cloud.setColors(std::move(colors));
    cloud.setIntensities(std::move(intensities));
    cloud.setScalarFields({"quality", "source_id"}, std::move(scalar_fields));
    cloud.setTextureCoords(std::move(texture_coords));
    return cloud;
}

} // namespace

TEST(FilterIndicesGpuTest, RadiusMaskAndRemovedIndicesMatchCpuAndCopyAttributes)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto cpu_cloud = makeRadiusCloudWithAttributes<Scalar>();
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_ror;
    cpu_ror.setInputCloud(cpu_cloud);
    cpu_ror.setRadius(Scalar(0.25));
    cpu_ror.setMinNeighbors(2);
    CpuCloud cpu_output;
    std::vector<int> cpu_removed;
    cpu_ror.filter(cpu_output, cpu_removed);

    const auto keep_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDeviceColumnMajor(
        gpu_cloud->points().data(), static_cast<int>(gpu_cloud->size()), Scalar(0.25), 2);
    expectVectorEq<std::uint8_t>(keep_mask, {1, 1, 1, 0});
    expectVectorEq(plapoint::gpu::removedIndicesFromKeepMask(keep_mask), cpu_removed);

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::GPU> gpu_ror;
    gpu_ror.setInputCloud(gpu_cloud);
    gpu_ror.setRadius(Scalar(0.25));
    gpu_ror.setMinNeighbors(2);
    GpuCloud gpu_output;
    std::vector<int> gpu_removed;
    gpu_ror.filter(gpu_output, gpu_removed);
    const auto gpu_output_cpu = gpu_output.toCpu();

    expectVectorEq(gpu_removed, cpu_removed);
    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    ASSERT_TRUE(gpu_output_cpu.hasColors());
    ASSERT_TRUE(gpu_output_cpu.hasIntensities());
    EXPECT_EQ(gpu_output_cpu.colors()->getValue(2, 0), 12);
    EXPECT_EQ(gpu_output_cpu.colors()->getValue(2, 2), 32);
    EXPECT_EQ(gpu_output_cpu.intensities()->getValue(2, 0), 102);
}

TEST(FilterIndicesGpuTest, RadiusKeepMaskCanStayInPlaMatrixGpuStorage)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto cpu_cloud = makeRadiusCloudWithAttributes<Scalar>();
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    const auto keep_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud->points(), Scalar(0.25), 2);
    const auto keep_mask_cpu = keep_mask.toCpu();

    ASSERT_EQ(keep_mask_cpu.rows(), 4);
    ASSERT_EQ(keep_mask_cpu.cols(), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(0, 0), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(1, 0), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(2, 0), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(3, 0), 0);
}

TEST(FilterIndicesGpuTest, SorMaskAndRemovedIndicesMatchCpuAndCopyAttributes)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto cpu_cloud = makeSorCloudWithAttributes<Scalar>();
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    auto cpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();
    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_sor;
    cpu_sor.setInputCloud(cpu_cloud);
    cpu_sor.setSearchMethod(cpu_tree);
    cpu_sor.setMeanK(2);
    cpu_sor.setStddevMulThresh(Scalar(0.5));
    CpuCloud cpu_output;
    std::vector<int> cpu_removed;
    cpu_sor.filter(cpu_output, cpu_removed);

    const auto keep_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
        gpu_cloud->points().data(), static_cast<int>(gpu_cloud->size()), 2, Scalar(0.5));
    expectVectorEq<std::uint8_t>(keep_mask, {1, 1, 1, 1, 0});
    expectVectorEq(plapoint::gpu::removedIndicesFromKeepMask(keep_mask), cpu_removed);

    auto gpu_tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::GPU>>();
    gpu_tree->setInputCloud(gpu_cloud);
    gpu_tree->build();
    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::GPU> gpu_sor;
    gpu_sor.setInputCloud(gpu_cloud);
    gpu_sor.setSearchMethod(gpu_tree);
    gpu_sor.setMeanK(2);
    gpu_sor.setStddevMulThresh(Scalar(0.5));
    GpuCloud gpu_output;
    std::vector<int> gpu_removed;
    gpu_sor.filter(gpu_output, gpu_removed);
    const auto gpu_output_cpu = gpu_output.toCpu();

    expectVectorEq(gpu_removed, cpu_removed);
    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    ASSERT_TRUE(gpu_output_cpu.hasColors());
    ASSERT_TRUE(gpu_output_cpu.hasIntensities());
    EXPECT_EQ(gpu_output_cpu.colors()->getValue(3, 0), 43);
    EXPECT_EQ(gpu_output_cpu.colors()->getValue(3, 2), 63);
    EXPECT_EQ(gpu_output_cpu.intensities()->getValue(3, 0), 203);
}

TEST(FilterIndicesGpuTest, SorKeepMaskCanStayInPlaMatrixGpuStorage)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto cpu_cloud = makeSorCloudWithAttributes<Scalar>();
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    const auto keep_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud->points(), 2, Scalar(0.5));
    const auto keep_mask_cpu = keep_mask.toCpu();

    ASSERT_EQ(keep_mask_cpu.rows(), 5);
    ASSERT_EQ(keep_mask_cpu.cols(), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(0, 0), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(1, 0), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(2, 0), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(3, 0), 1);
    EXPECT_EQ(keep_mask_cpu.getValue(4, 0), 0);
}

TEST(FilterIndicesGpuTest, RadiusIndexedWorkspaceReusesMatchingRevisionWithoutHostPointStaging)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    auto gpu_cloud = makeCompleteAttributeCloud<Scalar>().toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;

    auto first = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud, Scalar(1), 2, workspace);
    expectVectorEq<std::uint8_t>(
        plapoint::gpu::keepMaskToHost(first), {1, 1, 1, 1, 0, 0});
    EXPECT_EQ(
        plapoint::gpu::removedIndicesFromKeepMaskDevice(first),
        (std::vector<int>{4, 5}));
    EXPECT_EQ(workspace.lastBackend(), plapoint::gpu::GpuOutlierRemovalBackend::UniformGrid);
    EXPECT_EQ(workspace.indexBuildCount(), 1u);

    auto second = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud, Scalar(1), 2, workspace);
    expectVectorEq<std::uint8_t>(
        plapoint::gpu::keepMaskToHost(second), {1, 1, 1, 1, 0, 0});
    EXPECT_EQ(workspace.indexBuildCount(), 1u);

    gpu_cloud.points().setValue(4, 0, Scalar(2));
    auto rebuilt = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud, Scalar(1), 2, workspace);
    expectVectorEq<std::uint8_t>(
        plapoint::gpu::keepMaskToHost(rebuilt), {1, 1, 1, 1, 1, 0});
    EXPECT_EQ(workspace.indexBuildCount(), 2u);
}

TEST(FilterIndicesGpuTest, IndexedMasksOutliveWorkspaceAndOwningStream)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    using Mask = plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    const auto gpu_cloud = makeCompleteAttributeCloud<Scalar>().toGpu();
    Mask radius_mask;
    Mask sor_mask;
    GpuCloud compacted_cloud;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
        radius_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
            gpu_cloud, Scalar(1), 2, workspace, stream);
        compacted_cloud = plapoint::gpu::compactPointCloudByKeepMask(
            gpu_cloud, radius_mask, stream);
        EXPECT_EQ(workspace.indexBuildCount(), 1u);
    }
    {
        plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
        sor_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
            gpu_cloud, 2, Scalar(0.5), workspace, stream);
        EXPECT_EQ(workspace.indexBuildCount(), 1u);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    expectVectorEq<std::uint8_t>(
        plapoint::gpu::keepMaskToHost(radius_mask), {1, 1, 1, 1, 0, 0});
    EXPECT_EQ(plapoint::gpu::keepMaskToHost(sor_mask).size(), gpu_cloud.size());
    const auto compacted_cpu = compacted_cloud.toCpu();
    EXPECT_EQ(compacted_cpu.size(), 4u);
    EXPECT_TRUE(compacted_cpu.hasScalarFields());
    EXPECT_TRUE(compacted_cpu.hasTextureCoords());
}

TEST(FilterIndicesGpuTest, SorIndexedWorkspaceReusesMatchingRevision)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    auto gpu_cloud = makeCompleteAttributeCloud<Scalar>().toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;

    const auto first = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud, 2, Scalar(0.5), workspace);
    const auto first_mask = plapoint::gpu::keepMaskToHost(first);
    EXPECT_EQ(workspace.indexBuildCount(), 1u);

    const auto reused = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud, 2, Scalar(0.5), workspace);
    EXPECT_EQ(plapoint::gpu::keepMaskToHost(reused), first_mask);
    EXPECT_EQ(workspace.indexBuildCount(), 1u);

    gpu_cloud.points().setValue(4, 0, Scalar(2));
    const auto rebuilt = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud, 2, Scalar(0.5), workspace);
    EXPECT_EQ(plapoint::gpu::keepMaskToHost(rebuilt).size(), gpu_cloud.size());
    EXPECT_EQ(workspace.indexBuildCount(), 2u);
}

TEST(FilterIndicesGpuTest, IndexedBackendsSupportDouble)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = double;
    const auto gpu_cloud = makeCompleteAttributeCloud<Scalar>().toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;

    const auto radius_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud, Scalar(1), 2, workspace);
    EXPECT_EQ(
        plapoint::gpu::removedIndicesFromKeepMaskDevice(radius_mask),
        (std::vector<int>{4, 5}));

    const auto sor_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud, 2, Scalar(0.5), workspace);
    EXPECT_EQ(plapoint::gpu::keepMaskToHost(sor_mask).size(), gpu_cloud.size());
    EXPECT_EQ(workspace.lastBackend(), plapoint::gpu::GpuOutlierRemovalBackend::UniformGrid);
}

TEST(FilterIndicesGpuTest, SorDoubleDistanceOverflowMatchesCpuRejection)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = double;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(3, 3);
    points.fill(Scalar(0));
    points.setValue(0, 0, -std::numeric_limits<Scalar>::max());
    points.setValue(1, 0, std::numeric_limits<Scalar>::max());
    points.setValue(2, 0, std::numeric_limits<Scalar>::max() / Scalar(2));
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(points));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cpu_cloud);
    tree->build();
    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_filter;
    cpu_filter.setInputCloud(cpu_cloud);
    cpu_filter.setSearchMethod(tree);
    cpu_filter.setMeanK(1);
    cpu_filter.setStddevMulThresh(Scalar(1));
    std::vector<int> cpu_removed;
    cpu_filter.filter(cpu_removed);

    const auto gpu_cloud = cpu_cloud->toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
    const auto gpu_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud, 1, Scalar(1), workspace);
    const auto gpu_removed = plapoint::gpu::removedIndicesFromKeepMaskDevice(gpu_mask);

    EXPECT_EQ(cpu_removed, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(gpu_removed, cpu_removed);
}

TEST(FilterIndicesGpuTest, SorRepeatedDoubleDistanceOverflowMatchesCpuRejection)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = double;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(5, 3);
    points.fill(Scalar(0));
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    points.setValue(0, 0, -maximum);
    points.setValue(1, 0, Scalar(0.5) * maximum);
    points.setValue(2, 0, Scalar(0.6) * maximum);
    points.setValue(3, 0, Scalar(0.7) * maximum);
    points.setValue(4, 0, Scalar(0.8) * maximum);
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(points));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cpu_cloud);
    tree->build();
    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_filter;
    cpu_filter.setInputCloud(cpu_cloud);
    cpu_filter.setSearchMethod(tree);
    cpu_filter.setMeanK(3);
    cpu_filter.setStddevMulThresh(Scalar(1));
    std::vector<int> cpu_removed;
    cpu_filter.filter(cpu_removed);

    const auto gpu_cloud = cpu_cloud->toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
    const auto gpu_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud, 3, Scalar(1), workspace);
    const auto gpu_removed = plapoint::gpu::removedIndicesFromKeepMaskDevice(gpu_mask);

    EXPECT_EQ(cpu_removed, (std::vector<int>{0, 1, 2, 3, 4}));
    EXPECT_EQ(gpu_removed, cpu_removed);
}

TEST(FilterIndicesGpuTest, SorDoubleGlobalStatisticsRemainFiniteAfterScaling)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = double;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(4, 3);
    points.fill(Scalar(0));
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    points.setValue(0, 0, Scalar(-0.75) * maximum);
    points.setValue(1, 0, Scalar(-0.25) * maximum);
    points.setValue(2, 0, Scalar(0.25) * maximum);
    points.setValue(3, 0, Scalar(0.75) * maximum);
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(points));

    auto tree = std::make_shared<plapoint::search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cpu_cloud);
    tree->build();
    plapoint::StatisticalOutlierRemoval<Scalar, plamatrix::Device::CPU> cpu_filter;
    cpu_filter.setInputCloud(cpu_cloud);
    cpu_filter.setSearchMethod(tree);
    cpu_filter.setMeanK(1);
    cpu_filter.setStddevMulThresh(Scalar(0));
    std::vector<int> cpu_removed;
    cpu_filter.filter(cpu_removed);

    const auto gpu_cloud = cpu_cloud->toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
    const auto gpu_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDevice(
        gpu_cloud, 1, Scalar(0), workspace);

    EXPECT_TRUE(cpu_removed.empty());
    EXPECT_TRUE(plapoint::gpu::removedIndicesFromKeepMaskDevice(gpu_mask).empty());
}

TEST(FilterIndicesGpuTest, DeviceMaskCompactionPreservesEveryPointAttributeInStableOrder)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    const auto gpu_cloud = makeCompleteAttributeCloud<Scalar>().toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
    const auto keep_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud, Scalar(1), 2, workspace);

    const auto compacted = plapoint::gpu::compactPointCloudByKeepMask(gpu_cloud, keep_mask).toCpu();

    ASSERT_EQ(compacted.size(), 4u);
    ASSERT_TRUE(compacted.hasNormals());
    ASSERT_TRUE(compacted.hasColors());
    ASSERT_TRUE(compacted.hasIntensities());
    ASSERT_TRUE(compacted.hasScalarFields());
    ASSERT_TRUE(compacted.hasTextureCoords());
    EXPECT_EQ(compacted.scalarFieldNames(), (std::vector<std::string>{"quality", "source_id"}));
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(compacted.points().getValue(i, 0), i < 2 ? 0.0f : float(i - 1));
        EXPECT_FLOAT_EQ(compacted.normals()->getValue(i, 2), float(300 + i));
        EXPECT_EQ(compacted.colors()->getValue(i, 1), 20 + i);
        EXPECT_EQ(compacted.intensities()->getValue(i, 0), 1000 + i);
        EXPECT_FLOAT_EQ(compacted.scalarFields()->getValue(i, 0), float(i) + 0.25f);
        EXPECT_FLOAT_EQ(compacted.scalarFields()->getValue(i, 1), float(500 + i));
        EXPECT_FLOAT_EQ(compacted.textureCoords()->getValue(i, 0), float(i) + 0.1f);
        EXPECT_FLOAT_EQ(compacted.textureCoords()->getValue(i, 1), float(i) + 0.2f);
    }
}

TEST(FilterIndicesGpuTest, DeviceMaskCompactionPreservesPointAlignedFaceTextureData)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    auto cpu_cloud = makeCompleteAttributeCloud<Scalar>();
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    for (int col = 0; col < 3; ++col)
    {
        faces.setValue(0, col, col);
        face_texture_indices.setValue(0, col, col);
    }
    cpu_cloud.setFaces(std::move(faces));
    cpu_cloud.setFaceTextureIndices(std::move(face_texture_indices));
    cpu_cloud.setMaterialLibraryFile("surface.mtl");
    cpu_cloud.setTextureImageFile("surface.png");
    const auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
    const auto keep_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud, Scalar(1), 2, workspace);

    const auto compacted = plapoint::gpu::compactPointCloudByKeepMask(gpu_cloud, keep_mask);

    ASSERT_TRUE(compacted.hasTextureCoords());
    EXPECT_EQ(compacted.textureCoords()->rows(), 4);
    EXPECT_FLOAT_EQ(compacted.textureCoords()->getValue(2, 0), 2.1f);
    EXPECT_FALSE(compacted.hasFaces());
    EXPECT_FALSE(compacted.hasFaceTextureIndices());
    EXPECT_EQ(compacted.materialLibraryFile(), "surface.mtl");
    EXPECT_EQ(compacted.textureImageFile(), "surface.png");
}

TEST(FilterIndicesGpuTest, DeviceMaskCompactionDropsNonPointAlignedFaceTextureData)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    auto cpu_cloud = makeCompleteAttributeCloud<Scalar>();
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    face_texture_indices.setValue(0, 0, 2);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 0);
    cpu_cloud.setFaces(std::move(faces));
    cpu_cloud.setFaceTextureIndices(std::move(face_texture_indices));
    cpu_cloud.setMaterialLibraryFile("surface.mtl");
    cpu_cloud.setTextureImageFile("surface.png");
    const auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::OutlierRemovalGpuWorkspace<Scalar> workspace;
    const auto keep_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDevice(
        gpu_cloud, Scalar(1), 2, workspace);

    const auto compacted = plapoint::gpu::compactPointCloudByKeepMask(gpu_cloud, keep_mask);

    EXPECT_FALSE(compacted.hasTextureCoords());
    EXPECT_FALSE(compacted.hasFaces());
    EXPECT_FALSE(compacted.hasFaceTextureIndices());
    EXPECT_TRUE(compacted.materialLibraryFile().empty());
    EXPECT_TRUE(compacted.textureImageFile().empty());
}

TEST(FilterIndicesGpuTest, RadiusBoundaryCases)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto empty_cpu = std::make_shared<CpuCloud>(0);
    auto empty_gpu = std::make_shared<GpuCloud>(empty_cpu->toGpu());
    const auto empty_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDeviceColumnMajor(
        empty_gpu->points().data(), 0, Scalar(1), 1);
    EXPECT_TRUE(empty_mask.empty());

    plapoint::RadiusOutlierRemoval<Scalar, plamatrix::Device::GPU> empty_ror;
    empty_ror.setInputCloud(empty_gpu);
    empty_ror.setRadius(Scalar(1));
    empty_ror.setMinNeighbors(1);
    GpuCloud empty_output;
    std::vector<int> empty_removed;
    empty_ror.filter(empty_output, empty_removed);
    EXPECT_EQ(empty_output.size(), 0u);
    EXPECT_TRUE(empty_removed.empty());

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> kept_points(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        kept_points.setValue(i, 0, Scalar(i) * Scalar(0.1));
        kept_points.setValue(i, 1, Scalar(0));
        kept_points.setValue(i, 2, Scalar(0));
    }
    auto kept_cpu = std::make_shared<CpuCloud>(std::move(kept_points));
    auto kept_gpu = std::make_shared<GpuCloud>(kept_cpu->toGpu());
    const auto no_outlier_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDeviceColumnMajor(
        kept_gpu->points().data(), static_cast<int>(kept_gpu->size()), Scalar(1), 1);
    expectVectorEq<std::uint8_t>(no_outlier_mask, {1, 1, 1});
    EXPECT_TRUE(plapoint::gpu::removedIndicesFromKeepMask(no_outlier_mask).empty());

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> isolated_points(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        isolated_points.setValue(i, 0, Scalar(i) * Scalar(10));
        isolated_points.setValue(i, 1, Scalar(0));
        isolated_points.setValue(i, 2, Scalar(0));
    }
    auto isolated_cpu = std::make_shared<CpuCloud>(std::move(isolated_points));
    auto isolated_gpu = std::make_shared<GpuCloud>(isolated_cpu->toGpu());
    const auto all_outlier_mask = plapoint::gpu::radiusOutlierRemovalKeepMaskDeviceColumnMajor(
        isolated_gpu->points().data(), static_cast<int>(isolated_gpu->size()), Scalar(0.1), 2);
    expectVectorEq<std::uint8_t>(all_outlier_mask, {0, 0, 0});
    expectVectorEq(plapoint::gpu::removedIndicesFromKeepMask(all_outlier_mask), {0, 1, 2});
}

TEST(FilterIndicesGpuTest, SorBoundaryCases)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter-index test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto empty_cpu = std::make_shared<CpuCloud>(0);
    auto empty_gpu = std::make_shared<GpuCloud>(empty_cpu->toGpu());
    const auto empty_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
        empty_gpu->points().data(), 0, 2, Scalar(1));
    EXPECT_TRUE(empty_mask.empty());

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> repeated_points(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        repeated_points.setValue(i, 0, Scalar(2));
        repeated_points.setValue(i, 1, Scalar(-1));
        repeated_points.setValue(i, 2, Scalar(0.5));
    }
    auto repeated_cpu = std::make_shared<CpuCloud>(std::move(repeated_points));
    auto repeated_gpu = std::make_shared<GpuCloud>(repeated_cpu->toGpu());
    const auto no_outlier_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
        repeated_gpu->points().data(), static_cast<int>(repeated_gpu->size()), 2, Scalar(0));
    expectVectorEq<std::uint8_t>(no_outlier_mask, {1, 1, 1});
    EXPECT_TRUE(plapoint::gpu::removedIndicesFromKeepMask(no_outlier_mask).empty());

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> non_finite_points(2, 3);
    non_finite_points.setValue(0, 0, std::numeric_limits<Scalar>::quiet_NaN());
    non_finite_points.setValue(0, 1, Scalar(0));
    non_finite_points.setValue(0, 2, Scalar(0));
    non_finite_points.setValue(1, 0, std::numeric_limits<Scalar>::infinity());
    non_finite_points.setValue(1, 1, Scalar(0));
    non_finite_points.setValue(1, 2, Scalar(0));
    auto non_finite_cpu = std::make_shared<CpuCloud>(std::move(non_finite_points));
    auto non_finite_gpu = std::make_shared<GpuCloud>(non_finite_cpu->toGpu());
    const auto all_outlier_mask = plapoint::gpu::statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
        non_finite_gpu->points().data(), static_cast<int>(non_finite_gpu->size()), 1, Scalar(1));
    expectVectorEq<std::uint8_t>(all_outlier_mask, {0, 0});
    expectVectorEq(plapoint::gpu::removedIndicesFromKeepMask(all_outlier_mask), {0, 1});
}

TEST(FilterIndicesGpuTest, GatherPointCloudByIndicesCopiesDeviceAttributes)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter compaction test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(4, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(4, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(4, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(4, 1);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> scalar_fields(4, 2);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> texture_coords(4, 2);
    for (int i = 0; i < 4; ++i)
    {
        points.setValue(i, 0, Scalar(10 + i));
        points.setValue(i, 1, Scalar(20 + i));
        points.setValue(i, 2, Scalar(30 + i));
        normals.setValue(i, 0, Scalar(1 + i));
        normals.setValue(i, 1, Scalar(2 + i));
        normals.setValue(i, 2, Scalar(3 + i));
        colors.setValue(i, 0, static_cast<std::uint8_t>(40 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(50 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(60 + i));
        intensities.setValue(i, 0, static_cast<std::uint16_t>(700 + i));
        scalar_fields.setValue(i, 0, Scalar(0.25f * i));
        scalar_fields.setValue(i, 1, Scalar(100 + i));
        texture_coords.setValue(i, 0, Scalar(i) + Scalar(0.1));
        texture_coords.setValue(i, 1, Scalar(i) + Scalar(0.2));
    }
    CpuCloud cpu_cloud(std::move(points));
    cpu_cloud.setNormals(std::move(normals));
    cpu_cloud.setColors(std::move(colors));
    cpu_cloud.setIntensities(std::move(intensities));
    cpu_cloud.setScalarFields({"error", "confidence"}, std::move(scalar_fields));
    cpu_cloud.setTextureCoords(std::move(texture_coords));
    const auto gpu_cloud = cpu_cloud.toGpu();

    const GpuCloud gathered_gpu = plapoint::gpu::gatherPointCloudByIndices(gpu_cloud, {3, 1});
    const auto gathered = gathered_gpu.toCpu();

    ASSERT_EQ(gathered.size(), 2u);
    ASSERT_TRUE(gathered.hasNormals());
    ASSERT_TRUE(gathered.hasColors());
    ASSERT_TRUE(gathered.hasIntensities());
    ASSERT_TRUE(gathered.hasScalarFields());
    ASSERT_TRUE(gathered.hasTextureCoords());
    EXPECT_EQ(gathered.scalarFieldNames(), (std::vector<std::string>{"error", "confidence"}));
    EXPECT_FLOAT_EQ(gathered.points().getValue(0, 0), 13.0f);
    EXPECT_FLOAT_EQ(gathered.points().getValue(0, 2), 33.0f);
    EXPECT_FLOAT_EQ(gathered.points().getValue(1, 0), 11.0f);
    EXPECT_FLOAT_EQ(gathered.normals()->getValue(0, 1), 5.0f);
    EXPECT_EQ(gathered.colors()->getValue(0, 2), 63);
    EXPECT_EQ(gathered.colors()->getValue(1, 0), 41);
    EXPECT_EQ(gathered.intensities()->getValue(0, 0), 703);
    EXPECT_EQ(gathered.intensities()->getValue(1, 0), 701);
    EXPECT_FLOAT_EQ(gathered.scalarFields()->getValue(0, 0), 0.75f);
    EXPECT_FLOAT_EQ(gathered.scalarFields()->getValue(1, 0), 0.25f);
    EXPECT_FLOAT_EQ(gathered.scalarFields()->getValue(0, 1), 103.0f);
    EXPECT_FLOAT_EQ(gathered.textureCoords()->getValue(0, 0), 3.1f);
    EXPECT_FLOAT_EQ(gathered.textureCoords()->getValue(1, 1), 1.2f);
}

TEST(FilterIndicesGpuTest, GatherPointCloudByIndicesRejectsInvalidIndex)
{
    if (!hasCudaDeviceForFilterIndices())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU filter compaction test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = CpuCloud(2).toGpu();

    EXPECT_THROW(
        (void)plapoint::gpu::gatherPointCloudByIndices(cloud, {0, -1}),
        std::out_of_range);
    EXPECT_THROW(
        (void)plapoint::gpu::gatherPointCloudByIndices(cloud, {0, 2}),
        std::out_of_range);
}
#endif // PLAPOINT_WITH_CUDA
