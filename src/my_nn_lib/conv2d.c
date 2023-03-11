
// conv2d.c

#include "my_nn_lib/conv2d.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

#include <math.h>
#include <stdlib.h>

// Initialize the parameters for the 2d convolution layer
void Conv2dParamsInitialize(Conv2dParams* params,
                            const int in_channels,
                            const int out_channels,
                            const int kernel_width,
                            const int kernel_height,
                            const int stride,
                            const int padding)
{
  Assert(params != NULL, "`params` should not be NULL");
  Assert(in_channels > 0, "`in_channels` should be greater than 0");
  Assert(out_channels > 0, "`out_channels` should be greater than 0");
  Assert(kernel_width > 0, "`kernel_width` should be greater than 0");
  Assert(kernel_height > 0, "`kernel_height` should be greater than 0");
  Assert(stride >= 0, "`stride` should be greater than or equal to 0");
  Assert(padding >= 0, "`padding` should be greater than or equal to 0");

  params->weight_ = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, out_channels, in_channels, kernel_height, kernel_width);
  params->bias_ = (FloatTensor*)TensorEmpty1d(
    TENSOR_TYPE_FLOAT, out_channels);
  params->stride_ = stride;
  params->padding_ = padding;
}

// Free the parameters for the 2d convolution layer
void Conv2dParamsFree(Conv2dParams* params)
{
  Assert(params != NULL, "`params` should not be NULL");

  TensorFree((Tensor**)&params->weight_);
  TensorFree((Tensor**)&params->bias_);
  params->stride_ = 0;
  params->padding_ = 0;
}

// Forward operation for the 2d convolution
// `x` should be of size (B, Cin, Hin, Win)
// `params->weight_` should be of size (Cout, Cin, KH, KW)
// `params->bias_` should be of size (Cout)
// `params->bias_` may be `NULL`
// The returned tensor `y` is of size (B, Cout, Hout, Wout)
void Conv2dForward(const FloatTensor* x,
                   FloatTensor* y,
                   const Conv2dParams* params)
{
  // The input and output tensors should not be NULL except `bias`
  CheckTensor(x);
  CheckTensor(y);
  CheckTensor(params->weight_);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 4);
  CheckTensorDims(params->weight_, 4);
  CheckTensorDims(params->bias_, 1);

  // Check the consistency of the tensor shapes
  // Check the number of input channels
  Assert(x->base_.shape_[1] == params->weight_->base_.shape_[1],
         "The number of input channels is not consistent: "
         "(`x`: %d, `params->weight_`: %d)",
         x->base_.shape_[1], params->weight_->base_.shape_[1]);

  // Check the number of output channels
  Assert(params->bias_ == NULL ||
         params->weight_->base_.shape_[0] == params->bias_->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`params->weight_`: %d, `params->bias_`: %d)",
         params->weight_->base_.shape_[0], params->bias_->base_.shape_[0]);

  const int batch_size = x->base_.shape_[0];
  const int in_channels = x->base_.shape_[1];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  const int out_channels = params->weight_->base_.shape_[0];
  const int kernel_height = params->weight_->base_.shape_[2];
  const int kernel_width = params->weight_->base_.shape_[3];

  // Compute the size of an output tensor
  const int out_height = (in_height + 2 * params->padding_ - kernel_height)
    / params->stride_ + 1;
  const int out_width = (in_width + 2 * params->padding_ - kernel_width)
    / params->stride_ + 1;

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
                const int ih = oh * params->stride_ + kh - params->padding_;
                const int iw = ow * params->stride_ + kw - params->padding_;

                if (ih < 0 || ih >= in_height ||
                    iw < 0 || iw >= in_width)
                  continue;

                val += TensorAt4d(x, b, ich, ih, iw)
                  * TensorAt4d(params->weight_, och, ich, kh, kw);
              }
            }
          }

          if (params->bias_ != NULL)
            TensorAt4d(y, b, och, oh, ow) =
              val + TensorAt1d(params->bias_, och);
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
// The returned tensor `dparams->weight_` is of size (Cout, Cin, KH, KW)
// The returned tensor `dparams->bias_` is of size (Cout)
// `params->weight_` should be of size (Cout, Cin, KH, KW)
// `params->bias_` should be of size (Cout)
// `params->bias_` may be `NULL`
void Conv2dBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    FloatTensor* dx,
                    Conv2dParams* dparams,
                    const Conv2dParams* params)
{
  // The input and output tensors should not be NULL except `dbias` and `bias`
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(dx);
  CheckTensor(dparams->weight_);
  CheckTensor(params->weight_);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 4);
  CheckTensorDims(x, 4);
  CheckTensorDims(params->weight_, 4);
  CheckTensorDims(params->bias_, 1);

  // Check the consistency of the tensor shapes
  // Check the batch size
  Assert(dy->base_.shape_[0] == x->base_.shape_[0],
         "The batch size is not consistent: (`dy`: %d, `x`: %d)",
         dy->base_.shape_[0], x->base_.shape_[0]);

  // Check the number of input channels
  Assert(x->base_.shape_[1] == params->weight_->base_.shape_[1],
         "The number of input channels is not consistent: "
         "(`x`: %d, `params->weight_`: %d)",
         x->base_.shape_[1], params->weight_->base_.shape_[1]);

  // Check the number of output channels
  Assert(dy->base_.shape_[1] == params->weight_->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`dy`: %d, `params->weight_`: %d)",
         dy->base_.shape_[1], params->weight_->base_.shape_[0]);

  Assert(params->bias_ == NULL ||
         params->weight_->base_.shape_[0] == params->bias_->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`params->weight_`: %d, `params->bias_`: %d)",
         params->weight_->base_.shape_[0], params->bias_->base_.shape_[0]);

  // Check the width and height
  const int out_height_expected = (x->base_.shape_[2] + 2 * params->padding_
    - params->weight_->base_.shape_[2]) / params->stride_ + 1;
  const int out_width_expected = (x->base_.shape_[3] + 2 * params->padding_
    - params->weight_->base_.shape_[3]) / params->stride_ + 1;

  Assert(dy->base_.shape_[2] == out_height_expected,
         "The height and kernel size are not consistent: "
         "(`x`: %d, `params->weight_`: %d, `dy`: %d, "
         "`params->stride_`: %d, `params->padding_`: %d, expected: %d)",
         x->base_.shape_[2], params->weight_->base_.shape_[2],
         dy->base_.shape_[2],
         params->stride_, params->padding_, out_height_expected);

  Assert(dy->base_.shape_[3] == out_width_expected,
         "The width and kernel size are not consistent: "
         "(`x`: %d, `params->weight_`: %d, `dy`: %d, "
         "`params->stride_`: %d, `params->padding_`: %d, expected: %d)",
         x->base_.shape_[3], params->weight_->base_.shape_[3],
         dy->base_.shape_[3],
         params->stride_, params->padding_, out_width_expected);

  const int batch_size = dy->base_.shape_[0];
  const int out_channels = dy->base_.shape_[1];
  const int out_height = dy->base_.shape_[2];
  const int out_width = dy->base_.shape_[3];

  const int in_channels = x->base_.shape_[1];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  const int kernel_height = params->weight_->base_.shape_[2];
  const int kernel_width = params->weight_->base_.shape_[3];

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)dparams->weight_, (const Tensor*)params->weight_);

  if (params->bias_ != NULL)
    TensorSetShapeLike((Tensor*)dparams->bias_, (const Tensor*)params->bias_);

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
                const int ih = oh * params->stride_ + kh - params->padding_;
                const int iw = ow * params->stride_ + kw - params->padding_;

                if (ih < 0 || ih >= in_height ||
                    iw < 0 || iw >= in_width)
                  continue;

                val += TensorAt4d(x, b, ich, ih, iw)
                  * TensorAt4d(dy, b, och, oh, ow);
              }
            }
          }

          TensorAt4d(dparams->weight_, och, ich, kh, kw) = val;
        }
      }
    }
  }

  // Compute the gradient for the bias
  if (params->bias_ != NULL) {
    for (int och = 0; och < out_channels; ++och) {
      float val = 0;

      for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
          for (int ow = 0; ow < out_width; ++ow) {
            val += TensorAt4d(dy, b, och, oh, ow);
          }
        }
      }

      TensorAt1d(dparams->bias_, och) = val;
    }
  }

  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    for (int ich = 0; ich < in_channels; ++ich) {
      for (int ih = 0; ih < in_height; ++ih) {
        for (int iw = 0; iw < in_width; ++iw) {
          float val = 0;

          const int oh_min0 = (ih + params->stride_ + params->padding_
            - kernel_height) / params->stride_;
          const int ow_min0 = (iw + params->stride_ + params->padding_
            - kernel_width) / params->stride_;
          const int oh_max0 = (ih + params->stride_ + params->padding_)
            / params->stride_;
          const int ow_max0 = (iw + params->stride_ + params->padding_)
            / params->stride_;

          const int oh_min = oh_min0 >= 0 ? oh_min0 : 0;
          const int ow_min = ow_min0 >= 0 ? ow_min0 : 0;
          const int oh_max = oh_max0 <= out_height ? oh_max0 : out_height;
          const int ow_max = ow_max0 <= out_width ? ow_max0 : out_width;

          for (int och = 0; och < out_channels; ++och) {
            for (int oh = oh_min; oh < oh_max; ++oh) {
              for (int ow = ow_min; ow < ow_max; ++ow) {
                const int kh = ih - oh * params->stride_ + params->padding_;
                const int kw = iw - ow * params->stride_ + params->padding_;

                val += TensorAt4d(params->weight_, och, ich, kh, kw)
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
