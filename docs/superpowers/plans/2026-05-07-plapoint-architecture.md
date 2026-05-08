# Plapoint Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the foundational data structures and base algorithm classes for the `plapoint` point cloud library, leveraging `plamatrix` as the math backend.

**Architecture:** Create a `plapoint::PointCloud<Scalar, Dev>` class wrapping `plamatrix::DenseMatrix` to manage point data and device location. Implement basic filters and point type concepts mirroring PCL, with template parameters for precision and device binding.

**Key Design Decision:** `plapoint` is a **separate repository** from `plamatrix`. The build system uses `find_package(plamatrix)` to locate the pre-built `libplamatrix.a`. During development, point CMake at the plamatrix build dir via `CMAKE_PREFIX_PATH`.

**Tech Stack:** C++17, CUDA, CMake, Google Test, `plamatrix`.

---

### Task 1: Setup Build System and GitHub Remote

**Files:**
- Create: `CMakeLists.txt`
- Create: `test/CMakeLists.txt`
- Create: `.gitignore`
- Create: `cmake/plapointConfig.cmake.in`

- [ ] **Step 1: Create root `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(plapoint
    VERSION 0.1.0
    DESCRIPTION "Point cloud processing library built on plamatrix"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build unit tests" OFF)

# Find plamatrix
find_package(plamatrix REQUIRED)

# plapoint is header-only for now, so no compiled library target.
# Create an interface target for dependency propagation.
add_library(plapoint INTERFACE)
add_library(plapoint::plapoint ALIAS plapoint)

target_include_directories(plapoint
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(plapoint
    INTERFACE plamatrix::plamatrix
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

# Install
include(GNUInstallDirs)
install(DIRECTORY include/plapoint
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/plapointConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/plapointConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plapoint
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/plapointConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plapoint
)
```

- [ ] **Step 2: Create `cmake/plapointConfig.cmake.in`**

```cmake
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/plapointTargets.cmake")
check_required_components(plapoint)
```

- [ ] **Step 3: Create `test/CMakeLists.txt`**

```cmake
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(plapoint_tests ${TEST_SOURCES})
target_link_libraries(plapoint_tests PRIVATE plapoint::plapoint GTest::gtest GTest::gtest_main)
target_include_directories(plapoint_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
```

- [ ] **Step 4: Create `.gitignore`**

```gitignore
build/
.cache/
```

- [ ] **Step 5: Verify build**

```bash
cd build && cmake .. -DBUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/build \
  && cmake --build . -j$(nproc)
```
Expected: configure + build OK.

- [ ] **Step 6: Set up GitHub remote and push**

```bash
git remote add origin https://github.com/guderianXu/plapoint.git
git add -A && git commit -m "build: add CMake build system for plapoint"
git push -u origin master
```

---

### Task 2: Core `PointCloud` Data Structure

**Files:**
- Create: `include/plapoint/core/point_cloud.h`
- Create: `test/unit/core/point_cloud_test.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// test/unit/core/point_cloud_test.cpp
#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointCloudTest, CpuCreation)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(100);
    EXPECT_EQ(cloud.size(), 100);
    EXPECT_EQ(cloud.points().rows(), 100);
    EXPECT_EQ(cloud.points().cols(), 3);
}

TEST(PointCloudTest, GpuTransfer)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cpu_cloud(10);
    cpu_cloud.points().fill(1.0f);

    auto gpu_cloud = cpu_cloud.toGpu();
    EXPECT_EQ(gpu_cloud.size(), 10);

    auto cpu_cloud_back = gpu_cloud.toCpu();
    EXPECT_FLOAT_EQ(cpu_cloud_back.points().getValue(0, 0), 1.0f);
}

TEST(PointCloudTest, MoveFromMatrix)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> mat(50, 3);
    mat.fill(3.0f);
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(mat));
    EXPECT_EQ(cloud.size(), 50);
    EXPECT_FLOAT_EQ(cloud.points().getValue(0, 0), 3.0f);
}

TEST(PointCloudTest, RejectsNonNx3Matrix)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> bad(5, 4);
    EXPECT_THROW(
        plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(bad),
        std::runtime_error
    );
}
```

- [ ] **Step 2: Verify test fails**
Run: `cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/build && cmake --build . -j$(nproc)`
Expected: Compilation failure (file not found).

- [ ] **Step 3: Write implementation**

```cpp
// include/plapoint/core/point_cloud.h
#pragma once

#include <plamatrix/dense/dense_matrix.h>
#include <cstddef>
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

    explicit PointCloud(const MatrixType& pts)
    {
        if (pts.cols() != 3)
        {
            throw std::runtime_error("PointCloud requires Nx3 matrix");
        }
        _points = pts;
    }

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

private:
    MatrixType _points;
};

} // namespace plapoint
```

- [ ] **Step 4: Run tests**
Run: `cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter=PointCloudTest.*`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit and push**

```bash
git add include/plapoint/core/point_cloud.h test/unit/core/point_cloud_test.cpp
git commit -m "feat: add base PointCloud class with GPU transfer"
git push origin master
```

---

### Task 3: Base Filter Interface

**Files:**
- Create: `include/plapoint/filters/filter.h`

- [ ] **Step 1: Write implementation**

```cpp
// include/plapoint/filters/filter.h
#pragma once

#include <plapoint/core/point_cloud.h>
#include <memory>
#include <vector>

namespace plapoint
{

template <typename Scalar, plamatrix::Device Dev>
class Filter
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;
    using PointCloudConstPtr = std::shared_ptr<const PointCloudType>;

    Filter() = default;
    virtual ~Filter() = default;

    void setInputCloud(const PointCloudConstPtr& cloud)
    {
        _input = cloud;
    }

    void filter(PointCloudType& output)
    {
        if (!_input)
        {
            throw std::runtime_error("Input cloud not set");
        }
        applyFilter(output);
    }

    virtual void filter(std::vector<int>& removed_indices)
    {
        (void)removed_indices;
        throw std::runtime_error("Not implemented");
    }

protected:
    virtual void applyFilter(PointCloudType& output) = 0;

    PointCloudConstPtr _input;
};

} // namespace plapoint
```

- [ ] **Step 2: Write test for Filter (compile-time only, no implementation)**

```cpp
// test/unit/filters/filter_test.cpp
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
```

- [ ] **Step 3: Run tests**
Run: `cd build && cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/home/guderian/code/plamatrix/build && cmake --build . -j$(nproc) && ./test/plapoint_tests`
Expected: All 6 tests PASS (4 from Task 2 + 2 from Task 3).

- [ ] **Step 4: Commit and push**

```bash
git add include/plapoint/filters/filter.h test/unit/filters/filter_test.cpp
git commit -m "feat: add base Filter interface"
git push origin master
```
