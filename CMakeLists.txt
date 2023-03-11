
# CMakeLists.txt

cmake_minimum_required(VERSION 3.8)

project(my_lenet5 C CXX)

set(CMAKE_DEBUG_POSTFIX "d")

enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

function(add_gtest_executable target_name target_source)
  add_executable(${target_name} ${target_source})
  target_link_libraries(${target_name}
    my_nn_lib GTest::gtest GTest::gtest_main)
  gtest_discover_tests(${target_name})
endfunction()

add_library(my_nn_lib)

add_subdirectory(src)
add_subdirectory(examples)

target_include_directories(my_nn_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(my_nn_lib PUBLIC m)

target_compile_options(my_nn_lib PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wno-psabi>
  $<$<CONFIG:Release>:-O3>
  $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-O0 -ggdb>)
set_target_properties(my_nn_lib PROPERTIES
  DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

add_gtest_executable(test_tensor
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.cpp)
add_gtest_executable(test_conv2d
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv2d.cpp)
add_gtest_executable(test_maxpool2d
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_maxpool2d.cpp)
