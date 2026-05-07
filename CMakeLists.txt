cmake_minimum_required(VERSION 3.18)
project(plapoint
    VERSION 0.1.0
    DESCRIPTION "Point cloud processing library built on plamatrix"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build unit tests" OFF)

# Detect CUDA (plamatrix may require it)
find_package(CUDAToolkit QUIET)

# OpenMP (required by plamatrix)
find_package(OpenMP REQUIRED)

# Find plamatrix
find_package(plamatrix REQUIRED)

# plapoint is header-only for now
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
