cmake_minimum_required(VERSION 3.18)

project(plapoint_consumer_smoke LANGUAGES CXX)

find_package(plapoint CONFIG REQUIRED)

add_executable(plapoint_consumer_smoke main.cpp)
target_link_libraries(plapoint_consumer_smoke PRIVATE plapoint::plapoint)
target_compile_features(plapoint_consumer_smoke PRIVATE cxx_std_17)
