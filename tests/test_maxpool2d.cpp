
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
  MaxPool2dParams params;
  MaxPool2dParamsInitialize(&params, kernel_size, kernel_size,
                            stride, padding);
  MaxPool2dOutputs outputs;
  MaxPool2dOutputsInitialize(&outputs, false);

  // Forward operation
  FloatTensor* x = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, channels, in_height, in_width);
  MaxPool2dForward(x, &outputs, &params);

  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)outputs.y_, 4,
    batch_size, channels, out_height, out_width));
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)outputs.mask_, 4,
    batch_size, channels, out_height, out_width));

  // Backward operation
  FloatTensor* dy = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, channels, out_height, out_width);
  MaxPool2dBackward(dy, x, &outputs, &params);

  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)outputs.dx_, 4,
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
