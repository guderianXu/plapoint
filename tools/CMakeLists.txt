add_executable(plapoint_mesh_quality_report
    mesh_quality_report.cpp
)

target_link_libraries(plapoint_mesh_quality_report PRIVATE plapoint::plapoint)
target_include_directories(plapoint_mesh_quality_report PRIVATE
    ${PROJECT_SOURCE_DIR}/test
)
