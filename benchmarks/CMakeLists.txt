add_executable(plapoint_benchmarks
    plapoint_benchmarks.cpp
)

target_link_libraries(plapoint_benchmarks PRIVATE plapoint::plapoint)

set_target_properties(plapoint_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
)
