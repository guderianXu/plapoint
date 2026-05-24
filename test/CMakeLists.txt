file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(plapoint_tests ${TEST_SOURCES})
target_link_libraries(plapoint_tests PRIVATE plapoint::plapoint GTest::gtest GTest::gtest_main)
target_include_directories(plapoint_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(plapoint_tests PRIVATE
    PLAPOINT_TEST_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/validation/reference_data"
)
set_target_properties(plapoint_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

include(GoogleTest)
gtest_discover_tests(plapoint_tests
    TEST_PREFIX plapoint.
    DISCOVERY_MODE PRE_TEST
)
