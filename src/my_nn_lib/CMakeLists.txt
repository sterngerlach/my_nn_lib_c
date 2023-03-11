
# CMakeLists.txt

target_sources(my_nn_lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/activation.c
  ${CMAKE_CURRENT_SOURCE_DIR}/conv2d.c
  ${CMAKE_CURRENT_SOURCE_DIR}/flatten.c
  ${CMAKE_CURRENT_SOURCE_DIR}/linear.c
  ${CMAKE_CURRENT_SOURCE_DIR}/linked_list.c
  ${CMAKE_CURRENT_SOURCE_DIR}/logger.c
  ${CMAKE_CURRENT_SOURCE_DIR}/loss.c
  ${CMAKE_CURRENT_SOURCE_DIR}/maxpool2d.c
  ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_list.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_ops.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_util.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor.c
  ${CMAKE_CURRENT_SOURCE_DIR}/util.c)

add_subdirectory(dataset)
