cmake_minimum_required(VERSION 3.18)
project(plapoint
    VERSION 0.1.0
    DESCRIPTION "Point cloud processing library built on plamatrix"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build unit tests" OFF)
option(PLAPOINT_WITH_CUDA "Build with CUDA GPU acceleration" ON)

# Detect CUDA
find_package(CUDAToolkit QUIET)
if(PLAPOINT_WITH_CUDA AND CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(PLAPOINT_CUDA_ARCH "native" CACHE STRING "CUDA architecture target")
    set(CMAKE_CUDA_ARCHITECTURES ${PLAPOINT_CUDA_ARCH})
else()
    set(PLAPOINT_WITH_CUDA OFF)
endif()

# OpenMP (required by plamatrix)
find_package(OpenMP REQUIRED)

# Find plamatrix (or reuse target when included as submodule)
if(NOT TARGET plamatrix::plamatrix)
    find_package(plamatrix REQUIRED)
endif()

# plapoint library (compiled, not header-only)
add_library(plapoint)
add_library(plapoint::plapoint ALIAS plapoint)

target_include_directories(plapoint
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(plapoint
    PUBLIC plamatrix::plamatrix
)

if(PLAPOINT_WITH_CUDA)
    target_compile_definitions(plapoint PUBLIC PLAPOINT_WITH_CUDA=1)
    target_link_libraries(plapoint PRIVATE CUDA::cudart)
endif()

# Source files
add_subdirectory(src)

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

# Install
include(GNUInstallDirs)
install(TARGETS plapoint
    EXPORT plapointTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/plapoint
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT plapointTargets
    NAMESPACE plapoint::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plapoint
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

message(STATUS "PlaPoint Configuration:")
message(STATUS "  CUDA support: ${PLAPOINT_WITH_CUDA}")
message(STATUS "  Build tests:  ${BUILD_TESTS}")
