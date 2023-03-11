
// test_maxpool2d.cpp

#include <gtest/gtest.h>

#include "my_nn_lib/maxpool2d.h"
#include "my_nn_lib/tensor.h"

static void MaxPool2dForwardBackward(const int batch_size,
                                     const int channels,
                                     const int kernel_size,
                                     const int in_width,
                                     const int in_height,
                                     const int out_width,
                                     const int out_height,
                                     const int stride,
                                     const int padding)
{
  // Forward operation
  FloatTensor* x = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, channels, in_height, in_width);
  FloatTensor* y = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  Index2dTensor* mask = (Index2dTensor*)TensorAllocate(TENSOR_TYPE_INDEX2D);

  MaxPool2dParams params;
  params.kernel_height_ = kernel_size;
  params.kernel_width_ = kernel_size;
  params.stride_ = stride;
  params.padding_ = padding;

  MaxPool2dForward(x, y, mask, &params);

  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)y, 4,
    batch_size, channels, out_height, out_width));
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)mask, 4,
    batch_size, channels, out_height, out_width));

  // Backward operation
  FloatTensor* dy = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, channels, out_height, out_width);
  FloatTensor* dx = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  MaxPool2dBackward(dy, mask, x, dx, &params);

  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)dx, 4,
    batch_size, channels, in_height, in_width));
}

TEST(TestMaxPool2d, ForwardAndBackward)
{
  MaxPool2dForwardBackward(16, 32, 2, 8, 8, 4, 4, 2, 0);
  MaxPool2dForwardBackward(16, 32, 2, 16, 16, 8, 8, 2, 0);
  MaxPool2dForwardBackward(16, 32, 4, 16, 16, 4, 4, 4, 0);
  MaxPool2dForwardBackward(16, 32, 5, 8, 8, 4, 4, 2, 2);
  MaxPool2dForwardBackward(16, 32, 5, 16, 16, 8, 8, 2, 2);
}
