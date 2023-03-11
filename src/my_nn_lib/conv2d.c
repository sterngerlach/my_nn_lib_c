
// conv2d.c

#include "my_nn_lib/conv2d.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

#include <math.h>
#include <stdlib.h>

// Forward operation for the 2d convolution
// `x` should be of size (B, Cin, Hin, Win)
// `weight` should be of size (Cout, Cin, KH, KW)
// `bias` should be of size (Cout)
// `bias` may be `NULL`
// The returned tensor `y` is of size (B, Cout, Hout, Wout)
void Conv2dForward(const FloatTensor* x,
                   FloatTensor* y,
                   const FloatTensor* weight,
                   const FloatTensor* bias,
                   const int stride,
                   const int padding)
{
  // The input and output tensors should not be NULL except `bias`
  CheckTensor(x);
  CheckTensor(y);
  CheckTensor(weight);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 4);
  CheckTensorDims(weight, 4);
  CheckTensorDims(bias, 1);

  // Check the consistency of the tensor shapes
  // Check the number of input channels
  Assert(x->base_.shape_[1] == weight->base_.shape_[1],
         "The number of input channels is not consistent: "
         "(`x`: %d, `weight`: %d)",
         x->base_.shape_[1], weight->base_.shape_[1]);

  // Check the number of output channels
  Assert(bias == NULL || weight->base_.shape_[0] == bias->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`weight`: %d, `bias`: %d)",
         weight->base_.shape_[0], bias->base_.shape_[0]);

  const int batch_size = x->base_.shape_[0];
  const int in_channels = x->base_.shape_[1];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  const int out_channels = weight->base_.shape_[0];
  const int kernel_height = weight->base_.shape_[2];
  const int kernel_width = weight->base_.shape_[3];

  // Compute the size of an output tensor
  const int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
  const int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

  // Set the shape of the output tensor if necessary
  TensorSetShape((Tensor*)y, 4, batch_size, out_channels,
    out_height, out_width);

  // Perform the 2D convolution for each batch
  for (int b = 0; b < batch_size; ++b) {
    for (int och = 0; och < out_channels; ++och) {
      for (int oh = 0; oh < out_height; ++oh) {
        for (int ow = 0; ow < out_width; ++ow) {
          float val = 0;

          for (int ich = 0; ich < in_channels; ++ich) {
            for (int kh = 0; kh < kernel_height; ++kh) {
              for (int kw = 0; kw < kernel_width; ++kw) {
                const int ih = oh * stride + kh - padding;
                const int iw = ow * stride + kw - padding;

                if (ih < 0 || ih >= in_height ||
                    iw < 0 || iw >= in_width)
                  continue;

                val += TensorAt4d(x, b, ich, ih, iw)
                  * TensorAt4d(weight, och, ich, kh, kw);
              }
            }
          }

          if (bias != NULL)
            TensorAt4d(y, b, och, oh, ow) = val + TensorAt1d(bias, och);
          else
            TensorAt4d(y, b, och, oh, ow) = val;
        }
      }
    }
  }
}

// Backward operation for the 2d convolution
// `dy` should be of size (B, Cout, Hout, Wout)
// `x` should be of size (B, Cin, Hin, Win)
// The returned tensor `dx` is of size (B, Cin, Hin, Win)
// The returned tensor `dweight` is of size (Cout, Cin, KH, KW)
// The returned tensor `dbias` is of size (Cout)
// `weight` should be of size (Cout, Cin, KH, KW)
// `bias` should be of size (Cout)
// `bias` may be `NULL`
void Conv2dBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    FloatTensor* dx,
                    FloatTensor* dweight,
                    FloatTensor* dbias,
                    const FloatTensor* weight,
                    const FloatTensor* bias,
                    const int stride,
                    const int padding)
{
  // The input and output tensors should not be NULL except `dbias` and `bias`
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(dx);
  CheckTensor(dweight);
  CheckTensor(weight);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 4);
  CheckTensorDims(x, 4);
  CheckTensorDims(weight, 4);
  CheckTensorDims(bias, 1);

  // Check the consistency of the tensor shapes
  // Check the batch size
  Assert(dy->base_.shape_[0] == x->base_.shape_[0],
         "The batch size is not consistent: (`dy`: %d, `x`: %d)",
         dy->base_.shape_[0], x->base_.shape_[0]);

  // Check the number of input channels
  Assert(x->base_.shape_[1] == weight->base_.shape_[1],
         "The number of input channels is not consistent: "
         "(`x`: %d, `weight`: %d)",
         x->base_.shape_[1], weight->base_.shape_[1]);

  // Check the number of output channels
  Assert(dy->base_.shape_[1] == weight->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`dy`: %d, `weight`: %d)",
         dy->base_.shape_[1], weight->base_.shape_[0]);

  Assert(bias == NULL || weight->base_.shape_[0] == bias->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`weight`: %d, `bias`: %d)",
         weight->base_.shape_[0], bias->base_.shape_[0]);

  // Check the width and height
  const int out_height_expected = (x->base_.shape_[2]
    + 2 * padding - weight->base_.shape_[2]) / stride + 1;
  const int out_width_expected = (x->base_.shape_[3]
    + 2 * padding - weight->base_.shape_[3]) / stride + 1;

  Assert(dy->base_.shape_[2] == out_height_expected,
         "The height and kernel size are not consistent: "
         "(`x`: %d, `weight`: %d, `dy`: %d, "
         "`stride`: %d, `padding`: %d, expected: %d)",
         x->base_.shape_[2], weight->base_.shape_[2], dy->base_.shape_[2],
         stride, padding, out_height_expected);

  Assert(dy->base_.shape_[3] == out_width_expected,
         "The width and kernel size are not consistent: "
         "(`x`: %d, `weight`: %d, `dy`: %d, "
         "`stride`: %d, `padding`: %d, expected: %d)",
         x->base_.shape_[3], weight->base_.shape_[3], dy->base_.shape_[3],
         stride, padding, out_width_expected);

  const int batch_size = dy->base_.shape_[0];
  const int out_channels = dy->base_.shape_[1];
  const int out_height = dy->base_.shape_[2];
  const int out_width = dy->base_.shape_[3];

  const int in_channels = x->base_.shape_[1];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  const int kernel_height = weight->base_.shape_[2];
  const int kernel_width = weight->base_.shape_[3];

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)dweight, (const Tensor*)weight);

  if (bias != NULL)
    TensorSetShapeLike((Tensor*)dbias, (const Tensor*)bias);

  // Perform the backpropagation for the 2D convolution
  // Compute the gradient for the weight
  for (int och = 0; och < out_channels; ++och) {
    for (int ich = 0; ich < in_channels; ++ich) {
      for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
          float val = 0;

          for (int b = 0; b < batch_size; ++b) {
            for (int oh = 0; oh < out_height; ++oh) {
              for (int ow = 0; ow < out_width; ++ow) {
                const int ih = oh * stride + kh - padding;
                const int iw = ow * stride + kw - padding;

                if (ih < 0 || ih >= in_height ||
                    iw < 0 || iw >= in_width)
                  continue;

                val += TensorAt4d(x, b, ich, ih, iw)
                  * TensorAt4d(dy, b, och, oh, ow);
              }
            }
          }

          TensorAt4d(dweight, och, ich, kh, kw) = val;
        }
      }
    }
  }

  // Compute the gradient for the bias
  if (bias != NULL) {
    for (int och = 0; och < out_channels; ++och) {
      float val = 0;

      for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
          for (int ow = 0; ow < out_width; ++ow) {
            val += TensorAt4d(dy, b, och, oh, ow);
          }
        }
      }

      TensorAt1d(dbias, och) = val;
    }
  }

  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    for (int ich = 0; ich < in_channels; ++ich) {
      for (int ih = 0; ih < in_height; ++ih) {
        for (int iw = 0; iw < in_width; ++iw) {
          float val = 0;

          const int oh_min0 = (ih + stride + padding - kernel_height) / stride;
          const int ow_min0 = (iw + stride + padding - kernel_width) / stride;
          const int oh_max0 = (ih + stride + padding) / stride;
          const int ow_max0 = (iw + stride + padding) / stride;

          const int oh_min = oh_min0 >= 0 ? oh_min0 : 0;
          const int ow_min = ow_min0 >= 0 ? ow_min0 : 0;
          const int oh_max = oh_max0 <= out_height ? oh_max0 : out_height;
          const int ow_max = ow_max0 <= out_width ? ow_max0 : out_width;

          for (int och = 0; och < out_channels; ++och) {
            for (int oh = oh_min; oh < oh_max; ++oh) {
              for (int ow = ow_min; ow < ow_max; ++ow) {
                const int kh = ih - oh * stride + padding;
                const int kw = iw - ow * stride + padding;

                val += TensorAt4d(weight, och, ich, kh, kw)
                  * TensorAt4d(dy, b, och, oh, ow);
              }
            }
          }

          TensorAt4d(dx, b, ich, ih, iw) = val;
        }
      }
    }
  }
}
