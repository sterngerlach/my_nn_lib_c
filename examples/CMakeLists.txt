
# CMakeLists.txt

add_executable(read_mnist
  ${CMAKE_CURRENT_SOURCE_DIR}/read_mnist.c)
target_link_libraries(read_mnist my_nn_lib)

add_executable(train_dnn_mnist
  ${CMAKE_CURRENT_SOURCE_DIR}/train_dnn_mnist.c)
target_link_libraries(train_dnn_mnist my_nn_lib)

add_executable(train_lenet5_mnist
  ${CMAKE_CURRENT_SOURCE_DIR}/train_lenet5_mnist.c)
target_link_libraries(train_lenet5_mnist my_nn_lib)
