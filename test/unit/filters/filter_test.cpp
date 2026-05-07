#include <gtest/gtest.h>
#include <plapoint/filters/filter.h>
#include <plamatrix/plamatrix.h>

namespace
{

class MockFilter : public plapoint::Filter<float, plamatrix::Device::CPU>
{
protected:
    void applyFilter(PointCloudType& output) override
    {
        output = PointCloudType(5);
    }
};

} // namespace

TEST(FilterTest, SetInputAndFilter)
{
    auto cloud = std::make_shared<plapoint::PointCloud<float, plamatrix::Device::CPU>>(10);

    MockFilter mf;
    mf.setInputCloud(cloud);

    plapoint::PointCloud<float, plamatrix::Device::CPU> output;
    mf.filter(output);
    EXPECT_EQ(output.size(), 5);
}

TEST(FilterTest, ThrowsIfNoInput)
{
    MockFilter mf;
    plapoint::PointCloud<float, plamatrix::Device::CPU> output;
    EXPECT_THROW(mf.filter(output), std::runtime_error);
}
