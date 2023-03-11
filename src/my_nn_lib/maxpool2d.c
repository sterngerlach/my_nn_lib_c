
// maxpool2d.c

#include "my_nn_lib/maxpool2d.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

#include <float.h>

// Forward operation for the 2D max-pooling
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `y` is of size (B, C, Hout, Wout)
// The returned tensor `mask` is of size (B, C, Hout, Wout)
void MaxPool2dForward(const FloatTensor* x,
                      FloatTensor* y,
                      Index2dTensor* mask,
                      const int kernel_height,
                      const int kernel_width,
                      const int stride,
                      const int padding)
{
  // The input and output tensors should not be NULL
  CheckTensor(x);
  CheckTensor(y);
  CheckTensor(mask);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 4);

  const int batch_size = x->base_.shape_[0];
  const int channels = x->base_.shape_[1];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  // Compute the size of an output tensor
  const int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
  const int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

  // Set the shape of the output tensor if necessary
  TensorSetShape((Tensor*)y, 4, batch_size, channels, out_height, out_width);

  // `mask` is used in the backpropagation
  TensorSetShape((Tensor*)mask, 4, batch_size, channels,
    out_height, out_width);

  // Perform the 2D max-pooling for each batch
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int oh = 0; oh < out_height; ++oh) {
        for (int ow = 0; ow < out_width; ++ow) {
          float val = FLT_MIN;
          int ih_max = -1;
          int iw_max = -1;

          for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
              const int ih = oh * stride + kh - padding;
              const int iw = ow * stride + kw - padding;

              if (ih < 0 || ih >= in_height ||
                  iw < 0 || iw >= in_width)
                continue;

              const float x_val = TensorAt4d(x, b, ch, ih, iw);
              ih_max = x_val >= val ? ih : ih_max;
              iw_max = x_val >= val ? iw : iw_max;
              val = x_val >= val ? x_val : val;
            }
          }

          TensorAt4d(y, b, ch, oh, ow) = val;
          TensorAt4d(mask, b, ch, oh, ow).idx_[0] = ih_max;
          TensorAt4d(mask, b, ch, oh, ow).idx_[1] = iw_max;
        }
      }
    }
  }
}

// Backward operation for the 2D max-pooling
// `dy` should be of size (B, C, Hout, Wout)
// `mask` should be of size (B, C, Hout, Wout)
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `dx` is of size (B, C, Hin, Win)
void MaxPool2dBackward(const FloatTensor* dy,
                       const Index2dTensor* mask,
                       const FloatTensor* x,
                       FloatTensor* dx,
                       const int kernel_height,
                       const int kernel_width,
                       const int stride,
                       const int padding)
{
  // The input and output tensors should not be NULL
  CheckTensor(dy);
  CheckTensor(mask);
  CheckTensor(x);
  CheckTensor(dx);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 4);
  CheckTensorDims(mask, 4);

  // `dy` and `mask` should have the same shape
  Assert(TensorIsShapeEqual((const Tensor*)dy, (const Tensor*)mask),
         "Input tensors `dy` and `mask` should have the same shape");

  // Check the consistency of the tensor shapes
  const int out_height_expected = (x->base_.shape_[2]
    + 2 * padding - kernel_height) / stride + 1;
  const int out_width_expected = (x->base_.shape_[3]
    + 2 * padding - kernel_width) / stride + 1;

  Assert(dy->base_.shape_[2] == out_height_expected,
         "The height and kernel size are not consistent: "
         "(`x`: %d, `kernel_height`: %d, `dy`: %d, "
         "`stride`: %d, `padding`: %d, expected: %d)",
         x->base_.shape_[2], kernel_height, dy->base_.shape_[2],
         stride, padding, out_height_expected);

  Assert(dy->base_.shape_[3] == out_width_expected,
         "The width and kernel size are not consistent: "
         "(`x`: %d, `kernel_width`: %d, `dy`: %d, "
         "`stride`: %d, `padding`: %d, expected: %d)",
         x->base_.shape_[3], kernel_width, dy->base_.shape_[3],
         stride, padding, out_width_expected);

  const int batch_size = dy->base_.shape_[0];
  const int channels = dy->base_.shape_[1];
  const int out_height = dy->base_.shape_[2];
  const int out_width = dy->base_.shape_[3];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)x);

  // Perform the backpropagation for the 2D max-pooling
  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
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

          for (int oh = oh_min; oh < oh_max; ++oh) {
            for (int ow = ow_min; ow < ow_max; ++ow) {
              // The output element (`b`, `ch`, `oh`, `ow`) in `y` is obtained
              // from the input element (`b`, `ch`, `idx_h`, `idx_w`) in `x`
              const int idx_h = TensorAt4d(mask, b, ch, oh, ow).idx_[0];
              const int idx_w = TensorAt4d(mask, b, ch, oh, ow).idx_[1];

              if (idx_h != ih || idx_w != iw)
                continue;

              // If `idx_h` and `idx_w` are equal to `ih` and `iw`, then
              // the output gradient (`b`, `ch`, `oh`, `ow`) in `dy` should
              // propagate to the input (`b`, `ch`, `ih`, `iw`) in `dx`
              val += TensorAt4d(dy, b, ch, oh, ow);
            }
          }

          TensorAt4d(dx, b, ch, ih, iw) = val;
        }
      }
    }
  }
}
