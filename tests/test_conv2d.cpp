
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
  // Forward operation
  FloatTensor* x = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, in_channels, in_height, in_width);
  FloatTensor* weight = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, out_channels, in_channels, kernel_size, kernel_size);
  FloatTensor* bias = (FloatTensor*)TensorEmpty1d(
    TENSOR_TYPE_FLOAT, out_channels);

  Conv2dParams params;
  params.weight_ = weight;
  params.bias_ = bias;
  params.stride_ = stride;
  params.padding_ = padding;

  FloatTensor* y = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  Conv2dForward(x, y, &params);

  EXPECT_EQ(y->base_.ndim_, 4);
  EXPECT_EQ(y->base_.shape_[0], batch_size);
  EXPECT_EQ(y->base_.shape_[1], out_channels);
  EXPECT_EQ(y->base_.shape_[2], out_height);
  EXPECT_EQ(y->base_.shape_[3], out_width);

  // Backward operation
  FloatTensor* dy = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, batch_size, out_channels, out_height, out_width);

  FloatTensor* dx = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  FloatTensor* dweight = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  FloatTensor* dbias = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  Conv2dParams dparams;
  dparams.weight_ = dweight;
  dparams.bias_ = dbias;

  Conv2dBackward(dy, x, dx, &dparams, &params);

  EXPECT_TRUE(TensorIsShapeEqual((Tensor*)x, (Tensor*)dx));
  EXPECT_TRUE(TensorIsShapeEqual((Tensor*)weight, (Tensor*)dweight));
  EXPECT_TRUE(TensorIsShapeEqual((Tensor*)bias, (Tensor*)dbias));

  TensorFree((Tensor**)&x);
  TensorFree((Tensor**)&y);
  TensorFree((Tensor**)&weight);
  TensorFree((Tensor**)&bias);
  TensorFree((Tensor**)&dx);
  TensorFree((Tensor**)&dy);
  TensorFree((Tensor**)&dweight);
  TensorFree((Tensor**)&dbias);
}

TEST(TestConv2d, ForwardAndBackward)
{
  Conv2dForwardAndBackward(16, 32, 64, 5, 8, 8, 8, 8, 1, 2);
  Conv2dForwardAndBackward(16, 32, 64, 5, 8, 8, 4, 4, 2, 2);
  Conv2dForwardAndBackward(16, 32, 64, 5, 16, 16, 8, 8, 2, 2);
}
