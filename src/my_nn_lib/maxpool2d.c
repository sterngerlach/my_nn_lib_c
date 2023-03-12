
// maxpool2d.c

#include "my_nn_lib/maxpool2d.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

#include <float.h>

// Initialize the parameters for the 2D max-pooling layer
void MaxPool2dParamsInitialize(MaxPool2dParams* params,
                               const int kernel_height,
                               const int kernel_width,
                               const int stride,
                               const int padding)
{
  Assert(params != NULL, "`params` should not be NULL");
  Assert(kernel_height > 0, "`kernel_height` should be greater than 0");
  Assert(kernel_width > 0, "`kernel_width` should be greater than 0");
  Assert(stride >= 0, "`stride` should be greater than or equal to 0");
  Assert(padding >= 0, "`padding` should be greater than or equal to 0");

  params->kernel_height_ = kernel_height;
  params->kernel_width_ = kernel_width;
  params->stride_ = stride;
  params->padding_ = padding;
}

// Free the parameters for the 2D max-pooling layer
void MaxPool2dParamsFree(MaxPool2dParams* params)
{
  Assert(params != NULL, "`params` should not be NULL");

  params->kernel_height_ = 0;
  params->kernel_width_ = 0;
  params->stride_ = 0;
  params->padding_ = 0;
}

// Initialize the outputs for the 2D max-pooling layer
void MaxPool2dOutputsInitialize(MaxPool2dOutputs* outputs,
                                const bool inference_only)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  outputs->y_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->mask_ = (Index2dTensor*)TensorAllocate(TENSOR_TYPE_INDEX2D);

  if (!inference_only)
    outputs->dx_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  else
    outputs->dx_ = NULL;
}

// Free the outputs for the 2D max-pooling layer
void MaxPool2dOutputsFree(MaxPool2dOutputs* outputs)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  TensorFree((Tensor**)&outputs->y_);
  TensorFree((Tensor**)&outputs->mask_);
  TensorFree((Tensor**)&outputs->dx_);
}

// Forward operation for the 2D max-pooling
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `outputs->y_` is of size (B, C, Hout, Wout)
// The returned tensor `outputs->mask_` is of size (B, C, Hout, Wout)
void MaxPool2dForward(const FloatTensor* x,
                      MaxPool2dOutputs* outputs,
                      const MaxPool2dParams* params)
{
  // The input and output tensors should not be NULL
  CheckTensor(x);
  CheckTensor(outputs->y_);
  CheckTensor(outputs->mask_);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 4);

  const int batch_size = x->base_.shape_[0];
  const int channels = x->base_.shape_[1];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  // Compute the size of an output tensor
  const int out_height = (in_height + 2 * params->padding_
    - params->kernel_height_) / params->stride_ + 1;
  const int out_width = (in_width + 2 * params->padding_
    - params->kernel_width_) / params->stride_ + 1;

  // Set the shape of the output tensor if necessary
  TensorSetShape((Tensor*)outputs->y_, 4, batch_size, channels,
    out_height, out_width);

  // `mask` is used in the backpropagation
  TensorSetShape((Tensor*)outputs->mask_, 4, batch_size, channels,
    out_height, out_width);

  // Perform the 2D max-pooling for each batch
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int oh = 0; oh < out_height; ++oh) {
        for (int ow = 0; ow < out_width; ++ow) {
          float val = FLT_MIN;
          int ih_max = -1;
          int iw_max = -1;

          for (int kh = 0; kh < params->kernel_height_; ++kh) {
            for (int kw = 0; kw < params->kernel_width_; ++kw) {
              const int ih = oh * params->stride_ + kh - params->padding_;
              const int iw = ow * params->stride_ + kw - params->padding_;

              if (ih < 0 || ih >= in_height ||
                  iw < 0 || iw >= in_width)
                continue;

              const float x_val = TensorAt4d(x, b, ch, ih, iw);
              ih_max = x_val >= val ? ih : ih_max;
              iw_max = x_val >= val ? iw : iw_max;
              val = x_val >= val ? x_val : val;
            }
          }

          TensorAt4d(outputs->y_, b, ch, oh, ow) = val;
          TensorAt4d(outputs->mask_, b, ch, oh, ow).idx_[0] = ih_max;
          TensorAt4d(outputs->mask_, b, ch, oh, ow).idx_[1] = iw_max;
        }
      }
    }
  }
}

// Backward operation for the 2D max-pooling
// `dy` should be of size (B, C, Hout, Wout)
// `outputs->mask_` should be of size (B, C, Hout, Wout)
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `outputs->dx_` is of size (B, C, Hin, Win)
void MaxPool2dBackward(const FloatTensor* dy,
                       const FloatTensor* x,
                       MaxPool2dOutputs* outputs,
                       const MaxPool2dParams* params)
{
  // The input and output tensors should not be NULL
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(outputs->mask_);
  CheckTensor(outputs->dx_);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 4);
  CheckTensorDims(outputs->mask_, 4);

  // `dy` and `mask` should have the same shape
  Assert(TensorIsShapeEqual((const Tensor*)dy, (const Tensor*)outputs->mask_),
         "Input tensors `dy` and `mask` should have the same shape");

  // Check the consistency of the tensor shapes
  const int out_height_expected = (x->base_.shape_[2]
    + 2 * params->padding_ - params->kernel_height_) / params->stride_ + 1;
  const int out_width_expected = (x->base_.shape_[3]
    + 2 * params->padding_ - params->kernel_width_) / params->stride_ + 1;

  Assert(dy->base_.shape_[2] == out_height_expected,
         "The height and kernel size are not consistent: "
         "(`x`: %d, `params->kernel_height_`: %d, `dy`: %d, "
         "`params->stride_`: %d, `params->padding_`: %d, expected: %d)",
         x->base_.shape_[2], params->kernel_height_, dy->base_.shape_[2],
         params->stride_, params->padding_, out_height_expected);

  Assert(dy->base_.shape_[3] == out_width_expected,
         "The width and kernel size are not consistent: "
         "(`x`: %d, `params->kernel_width_`: %d, `dy`: %d, "
         "`params->stride_`: %d, `params->padding_`: %d, expected: %d)",
         x->base_.shape_[3], params->kernel_width_, dy->base_.shape_[3],
         params->stride_, params->padding_, out_width_expected);

  const int batch_size = dy->base_.shape_[0];
  const int channels = dy->base_.shape_[1];
  const int out_height = dy->base_.shape_[2];
  const int out_width = dy->base_.shape_[3];
  const int in_height = x->base_.shape_[2];
  const int in_width = x->base_.shape_[3];

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)outputs->dx_, (const Tensor*)x);

  // Perform the backpropagation for the 2D max-pooling
  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int ih = 0; ih < in_height; ++ih) {
        for (int iw = 0; iw < in_width; ++iw) {
          float val = 0;

          const int oh_min0 = (ih + params->stride_ + params->padding_
            - params->kernel_height_) / params->stride_;
          const int ow_min0 = (iw + params->stride_ + params->padding_
            - params->kernel_width_) / params->stride_;
          const int oh_max0 = (ih + params->stride_ + params->padding_)
            / params->stride_;
          const int ow_max0 = (iw + params->stride_ + params->padding_)
            / params->stride_;

          const int oh_min = oh_min0 >= 0 ? oh_min0 : 0;
          const int ow_min = ow_min0 >= 0 ? ow_min0 : 0;
          const int oh_max = oh_max0 <= out_height ? oh_max0 : out_height;
          const int ow_max = ow_max0 <= out_width ? ow_max0 : out_width;

          for (int oh = oh_min; oh < oh_max; ++oh) {
            for (int ow = ow_min; ow < ow_max; ++ow) {
              // The output element (`b`, `ch`, `oh`, `ow`) in `y` is obtained
              // from the input element (`b`, `ch`, `idx_h`, `idx_w`) in `x`
              const int idx_h = TensorAt4d(
                outputs->mask_, b, ch, oh, ow).idx_[0];
              const int idx_w = TensorAt4d(
                outputs->mask_, b, ch, oh, ow).idx_[1];

              if (idx_h != ih || idx_w != iw)
                continue;

              // If `idx_h` and `idx_w` are equal to `ih` and `iw`, then
              // the output gradient (`b`, `ch`, `oh`, `ow`) in `dy` should
              // propagate to the input (`b`, `ch`, `ih`, `iw`) in `dx`
              val += TensorAt4d(dy, b, ch, oh, ow);
            }
          }

          TensorAt4d(outputs->dx_, b, ch, ih, iw) = val;
        }
      }
    }
  }
}
