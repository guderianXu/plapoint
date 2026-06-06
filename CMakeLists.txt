cmake_minimum_required(VERSION 3.18)
project(plapoint
    VERSION 0.1.0
    DESCRIPTION "Point cloud processing library built on plamatrix"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_PLAPOINT_BUILD_TESTS_USER_DEFINED OFF)
if(DEFINED PLAPOINT_BUILD_TESTS)
    set(_PLAPOINT_BUILD_TESTS_USER_DEFINED ON)
endif()
option(PLAPOINT_BUILD_TESTS "Build PlaPoint unit tests" OFF)
option(PLAPOINT_BUILD_BENCHMARKS "Build PlaPoint benchmark executable" OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR
        AND DEFINED BUILD_TESTS
        AND NOT _PLAPOINT_BUILD_TESTS_USER_DEFINED)
    set(PLAPOINT_BUILD_TESTS ${BUILD_TESTS} CACHE BOOL "Build PlaPoint unit tests" FORCE)
endif()

set(_PLAPOINT_WITH_CUDA_USER_DEFINED OFF)
if(DEFINED PLAPOINT_WITH_CUDA)
    set(_PLAPOINT_WITH_CUDA_USER_DEFINED ON)
endif()
option(PLAPOINT_WITH_CUDA "Build with CUDA GPU acceleration" ON)

# OpenMP (required by plamatrix)
find_package(OpenMP REQUIRED)

# Find plamatrix (or reuse target when included as submodule)
if(NOT TARGET plamatrix::plamatrix)
    find_package(plamatrix REQUIRED)
endif()

get_target_property(_PLAPOINT_PLAMATRIX_DEFS plamatrix::plamatrix INTERFACE_COMPILE_DEFINITIONS)
if(NOT _PLAPOINT_PLAMATRIX_DEFS)
    set(_PLAPOINT_PLAMATRIX_DEFS "")
endif()
set(_PLAPOINT_PLAMATRIX_WITH_CUDA OFF)
foreach(_PLAPOINT_DEF IN LISTS _PLAPOINT_PLAMATRIX_DEFS)
    if(_PLAPOINT_DEF MATCHES "^PLAMATRIX_WITH_CUDA(=1)?$")
        set(_PLAPOINT_PLAMATRIX_WITH_CUDA ON)
    endif()
endforeach()

# Detect CUDA. PlaPoint CUDA kernels require a CUDA-enabled PlaMatrix target.
find_package(CUDAToolkit QUIET)
if(PLAPOINT_WITH_CUDA AND NOT _PLAPOINT_PLAMATRIX_WITH_CUDA)
    if(_PLAPOINT_WITH_CUDA_USER_DEFINED)
        message(FATAL_ERROR "PLAPOINT_WITH_CUDA=ON requires plamatrix::plamatrix built with CUDA. "
                            "Use a CUDA PlaMatrix package or configure PlaPoint with -DPLAPOINT_WITH_CUDA=OFF.")
    endif()
    message(STATUS "PlaPoint CUDA disabled because plamatrix::plamatrix is CPU-only")
    set(PLAPOINT_WITH_CUDA OFF CACHE BOOL "Build with CUDA GPU acceleration" FORCE)
endif()

if(PLAPOINT_WITH_CUDA AND CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED PLAPOINT_CUDA_ARCHITECTURES)
        if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
            set(PLAPOINT_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                CACHE STRING "CUDA architectures for PlaPoint")
        else()
            set(PLAPOINT_CUDA_ARCHITECTURES "75;86;89"
                CACHE STRING "CUDA architectures for PlaPoint")
        endif()
    endif()
else()
    set(PLAPOINT_WITH_CUDA OFF)
endif()

# plapoint library (compiled, not header-only)
add_library(plapoint)
add_library(plapoint::plapoint ALIAS plapoint)

if(PLAPOINT_WITH_CUDA)
    set_target_properties(plapoint PROPERTIES
        CUDA_ARCHITECTURES "${PLAPOINT_CUDA_ARCHITECTURES}"
    )
endif()

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

if(PLAPOINT_BUILD_TESTS)
    target_compile_definitions(plapoint PRIVATE PLAPOINT_ENABLE_TESTING=1)
endif()

# Source files
add_subdirectory(src)

if(PLAPOINT_BUILD_TESTS OR PLAPOINT_BUILD_BENCHMARKS)
    enable_testing()
endif()

# Tests
if(PLAPOINT_BUILD_TESTS)
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

if(PLAPOINT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
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
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/plapointConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/plapointConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/plapointConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plapoint
)

message(STATUS "PlaPoint Configuration:")
message(STATUS "  CUDA support: ${PLAPOINT_WITH_CUDA}")
if(PLAPOINT_WITH_CUDA)
    message(STATUS "  CUDA arch:    ${PLAPOINT_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Build tests:  ${PLAPOINT_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${PLAPOINT_BUILD_BENCHMARKS}")
