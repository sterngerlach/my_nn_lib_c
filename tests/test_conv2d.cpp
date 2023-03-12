
// test_conv2d.cpp

#include <gtest/gtest.h>

#include "my_nn_lib/conv2d.h"
#include "my_nn_lib/tensor.h"

static void Conv2dForwardAndBackward(const int batch_size,
                                     const int in_channels,
                                     const int out_channels,
                                     const int kernel_size,
                                     const int in_width,
                                     const int in_height,
                                     const int out_width,
                                     const int out_height,
                                     const int stride,
                                     const int padding)
{
  Conv2dParams params;
  Conv2dParamsInitialize(&params, in_channels, out_channels,
                         kernel_size, kernel_size, stride, padding, false);
  Conv2dOutputs outputs;
  Conv2dOutputsInitialize(&outputs, false);

  // Forward operation
  FloatTensor* x = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, in_channels, in_height, in_width);
  Conv2dForward(x, &outputs, &params);

  EXPECT_EQ(outputs.y_->base_.ndim_, 4);
  EXPECT_EQ(outputs.y_->base_.shape_[0], batch_size);
  EXPECT_EQ(outputs.y_->base_.shape_[1], out_channels);
  EXPECT_EQ(outputs.y_->base_.shape_[2], out_height);
  EXPECT_EQ(outputs.y_->base_.shape_[3], out_width);

  // Backward operation
  FloatTensor* dy = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, out_channels, out_height, out_width);
  Conv2dBackward(dy, x, &outputs, &params);

  EXPECT_TRUE(TensorIsShapeEqual(
    (Tensor*)x, (Tensor*)outputs.dx_));
  EXPECT_TRUE(TensorIsShapeEqual(
    (Tensor*)params.weight_, (Tensor*)params.dweight_));
  EXPECT_TRUE(TensorIsShapeEqual(
    (Tensor*)params.bias_, (Tensor*)params.dbias_));

  Conv2dParamsFree(&params);
  Conv2dOutputsFree(&outputs);
  TensorFree((Tensor**)&x);
  TensorFree((Tensor**)&dy);
}

TEST(TestConv2d, ForwardAndBackward)
{
  Conv2dForwardAndBackward(16, 32, 64, 5, 8, 8, 8, 8, 1, 2);
  Conv2dForwardAndBackward(16, 32, 64, 5, 8, 8, 4, 4, 2, 2);
  Conv2dForwardAndBackward(16, 32, 64, 5, 16, 16, 8, 8, 2, 2);
}
