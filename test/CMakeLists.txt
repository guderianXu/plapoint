file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(plapoint_tests ${TEST_SOURCES})
target_link_libraries(plapoint_tests PRIVATE plapoint::plapoint GTest::gtest GTest::gtest_main)
target_include_directories(plapoint_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Set working directory to repo root for validation data access
set_target_properties(plapoint_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

