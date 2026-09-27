if(NOT IS_DIRECTORY "${PLAPOINT_SOURCE_ROOT}")
    message(FATAL_ERROR "PLAPOINT_SOURCE_ROOT must name the PlaPoint source tree")
endif()

set(_legacy_type_pattern
    "plamatrix::(DenseMatrix|CSRMatrix|Vec[234]|Mat[234])([^A-Za-z0-9_]|$)")
set(_legacy_header_pattern
    "^[ \t]*#[ \t]*include[ \t]*[<\"]plamatrix/dense/dense_matrix\\.h[>\"]")
set(_old_internal_header_pattern
    "^[ \t]*#[ \t]*include[ \t]*[<\"]plamatrix/(ops/|device/|opencl/|cuda/|vulkan/|core/(error_check|execution_context|types)[.]h|dense/matrix_view[.]h|sparse/(iterative_solver|sparse_ops)[.]h)")
set(_old_internal_symbols
    Backend Device ExecutionContext ResidentMatrix ResidentVector ResidentCsrMatrix
    MatrixView ConstMatrixView makeColumnMajorView
    IndexingWorkspace ReductionWorkspace GroupingWorkspace SymmetricEigh3x3Workspace
    ReductionAxis GroupReduction Int32Key3 SolveOptions IterativeSolverReport pcg
    sumAsync maxAsync compactRowsAsync gatherRowsAsync
    finiteColumnBoundsAsync finiteColumnBoundsWithMaskAsync
    sortByKeyAsync runLengthEncodeAsync segmentedReduceAsync
    exclusiveScanAsync exclusiveScan symmetricEigh3x3BatchedAsync svd3x3
    opencl cuda)
list(JOIN _old_internal_symbols "|" _old_internal_symbols_pattern)
set(_old_internal_symbol_pattern
    "plamatrix::(${_old_internal_symbols_pattern})([^A-Za-z0-9_]|$)")
set(_using_namespace_pattern
    "using[ \t]+namespace[ \t]+plamatrix[ \t]*;")
set(_single_argument_matrix_pattern
    "plamatrix::Matrix[ \t]*<[^<>,]+>")

# Fail closed if a pattern changes so that it misses an old API or matches a new one.
foreach(_legacy_sample
        "plamatrix::DenseMatrix<float>"
        "plamatrix::CSRMatrix<double>"
        "plamatrix::Vec3<float>"
        "plamatrix::Mat4<double>")
    if(NOT "${_legacy_sample}" MATCHES "${_legacy_type_pattern}")
        message(FATAL_ERROR "Legacy PlaMatrix type pattern missed ${_legacy_sample}")
    endif()
endforeach()
foreach(_allowed_sample
        "plamatrix::MatrixXf"
        "plamatrix::Matrix<float, plamatrix::Dynamic, plamatrix::Dynamic>"
        "plamatrix::Vector3d"
        "plamatrix::SparseMatrix<double>"
        "plamatrix::Mathematics")
    if("${_allowed_sample}" MATCHES "${_legacy_type_pattern}")
        message(FATAL_ERROR "Legacy PlaMatrix type pattern rejected ${_allowed_sample}")
    endif()
endforeach()
if(NOT "plamatrix::Matrix<float>" MATCHES "${_single_argument_matrix_pattern}"
        OR "plamatrix::MatrixXf" MATCHES "${_single_argument_matrix_pattern}"
        OR "plamatrix::Matrix<float, plamatrix::Dynamic, plamatrix::Dynamic>"
            MATCHES "${_single_argument_matrix_pattern}")
    message(FATAL_ERROR "PlaMatrix matrix-arity boundary pattern is invalid")
endif()
if(NOT "#include <plamatrix/dense/dense_matrix.h>" MATCHES "${_legacy_header_pattern}"
        OR "#include <plamatrix/dense/matrix.h>" MATCHES "${_legacy_header_pattern}"
        OR NOT "using namespace plamatrix;" MATCHES "${_using_namespace_pattern}"
        OR "using namespace plamatrix_detail;" MATCHES "${_using_namespace_pattern}")
    message(FATAL_ERROR "PlaMatrix include/namespace boundary pattern is invalid")
endif()
if(NOT "plamatrix::ResidentMatrix<float>" MATCHES "${_old_internal_symbol_pattern}"
        OR NOT "plamatrix::Device::GPU" MATCHES "${_old_internal_symbol_pattern}"
        OR "plamatrix::internal::ResidentMatrix<float>" MATCHES "${_old_internal_symbol_pattern}"
        OR "plamatrix::Index" MATCHES "${_old_internal_symbol_pattern}"
        OR NOT "#include <plamatrix/ops/indexing.h>" MATCHES "${_old_internal_header_pattern}"
        OR "#include <plamatrix/internal/ops/indexing.h>" MATCHES "${_old_internal_header_pattern}")
    message(FATAL_ERROR "PlaMatrix internal boundary pattern is invalid")
endif()

set(_sources)
foreach(_directory include src test tools benchmarks)
    if(IS_DIRECTORY "${PLAPOINT_SOURCE_ROOT}/${_directory}")
        file(GLOB_RECURSE _directory_sources LIST_DIRECTORIES FALSE
            "${PLAPOINT_SOURCE_ROOT}/${_directory}/*.h"
            "${PLAPOINT_SOURCE_ROOT}/${_directory}/*.hpp"
            "${PLAPOINT_SOURCE_ROOT}/${_directory}/*.cpp"
            "${PLAPOINT_SOURCE_ROOT}/${_directory}/*.cu"
            "${PLAPOINT_SOURCE_ROOT}/${_directory}/*.cuh")
        list(APPEND _sources ${_directory_sources})
    endif()
endforeach()
if(NOT _sources)
    message(FATAL_ERROR "No PlaPoint C++ sources found in ${PLAPOINT_SOURCE_ROOT}")
endif()

foreach(_source IN LISTS _sources)
    file(STRINGS "${_source}" _old_types REGEX "${_legacy_type_pattern}")
    file(STRINGS "${_source}" _old_headers REGEX "${_legacy_header_pattern}")
    file(STRINGS "${_source}" _old_internal_headers REGEX "${_old_internal_header_pattern}")
    file(STRINGS "${_source}" _old_internal_symbols REGEX "${_old_internal_symbol_pattern}")
    file(STRINGS "${_source}" _namespace_imports REGEX "${_using_namespace_pattern}")
    file(STRINGS "${_source}" _single_argument_matrices REGEX "${_single_argument_matrix_pattern}")
    if(_old_types OR _old_headers OR _old_internal_headers OR _old_internal_symbols OR _namespace_imports
            OR _single_argument_matrices)
        file(RELATIVE_PATH _relative "${PLAPOINT_SOURCE_ROOT}" "${_source}")
        message(FATAL_ERROR "Disallowed PlaMatrix API in ${_relative}: "
            "${_old_types} ${_old_headers} ${_old_internal_headers} "
            "${_old_internal_symbols} ${_namespace_imports} ${_single_argument_matrices}")
    endif()
endforeach()
