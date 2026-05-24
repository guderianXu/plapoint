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
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR
        AND DEFINED BUILD_TESTS
        AND NOT _PLAPOINT_BUILD_TESTS_USER_DEFINED)
    set(PLAPOINT_BUILD_TESTS ${BUILD_TESTS} CACHE BOOL "Build PlaPoint unit tests" FORCE)
endif()
option(PLAPOINT_WITH_CUDA "Build with CUDA GPU acceleration" ON)

# Detect CUDA
find_package(CUDAToolkit QUIET)
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

# OpenMP (required by plamatrix)
find_package(OpenMP REQUIRED)

# Find plamatrix (or reuse target when included as submodule)
if(NOT TARGET plamatrix::plamatrix)
    find_package(plamatrix REQUIRED)
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

# Source files
add_subdirectory(src)

# Tests
if(PLAPOINT_BUILD_TESTS)
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
