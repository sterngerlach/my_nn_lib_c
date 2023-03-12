
// batchnorm2d.c

#include "my_nn_lib/batchnorm2d.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

#include <math.h>

// Initialize the parameters for the 2d batch normalization layer
// Initializes `params->weight_` to ones
// Initializes `params->bias_` to zeros
// Initializes `params->running_mean_` to zeros
// Initializes `params->running_var_` to ones
void BatchNorm2dParamsInitialize(BatchNorm2dParams* params,
                                 const int dims,
                                 const float eps,
                                 const float momentum,
                                 const bool inference_only)
{
  Assert(params != NULL, "`params` should not be NULL");
  Assert(dims > 0, "`dims` should be greater than 0");
  Assert(eps > 0.0f, "`eps` should be greater than 0");
  Assert(momentum > 0.0f, "`momentum` should be greater than 0");
  Assert(momentum < 1.0f, "`momentum` should be less than 1");

  params->weight_ = (FloatTensor*)TensorOnes1d(
    TENSOR_TYPE_FLOAT, dims);
  params->bias_ = (FloatTensor*)TensorZeros1d(
    TENSOR_TYPE_FLOAT, dims);
  params->running_mean_ = (FloatTensor*)TensorZeros1d(
    TENSOR_TYPE_FLOAT, dims);
  params->running_var_ = (FloatTensor*)TensorOnes1d(
    TENSOR_TYPE_FLOAT, dims);
  params->eps_ = eps;
  params->momentum_ = momentum;

  if (!inference_only) {
    params->dweight_ = (FloatTensor*)TensorEmpty1d(
      TENSOR_TYPE_FLOAT, dims);
    params->dbias_ = (FloatTensor*)TensorEmpty1d(
      TENSOR_TYPE_FLOAT, dims);
  } else {
    params->dweight_ = NULL;
    params->dbias_ = NULL;
  }
}

// Free the parameters for the 2d batch normalization layer
void BatchNorm2dParamsFree(BatchNorm2dParams* params)
{
  Assert(params != NULL, "`params` should not be NULL");

  TensorFree((Tensor**)&params->weight_);
  TensorFree((Tensor**)&params->bias_);
  TensorFree((Tensor**)&params->running_mean_);
  TensorFree((Tensor**)&params->running_var_);
  params->eps_ = 0.0f;
  params->momentum_ = 0.0f;

  TensorFree((Tensor**)&params->dweight_);
  TensorFree((Tensor**)&params->dbias_);
}

// Initialize the outputs for the 2d batch normalization layer
void BatchNorm2dOutputsInitialize(BatchNorm2dOutputs* outputs,
                                  const bool inference_only)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  outputs->y_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->xc_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->xn_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  if (!inference_only) {
    outputs->dx_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->mean_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->var_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->std_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->dmean_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->dvar_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->dstd_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->dxc_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
    outputs->dxn_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  } else {
    outputs->dx_ = NULL;
    outputs->mean_ = NULL;
    outputs->var_ = NULL;
    outputs->std_ = NULL;
    outputs->dmean_ = NULL;
    outputs->dvar_ = NULL;
    outputs->dstd_ = NULL;
    outputs->dxc_ = NULL;
    outputs->dxn_ = NULL;
  }
}

// Free the outputs for the 2d batch normalization layer
void BatchNorm2dOutputsFree(BatchNorm2dOutputs* outputs)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  TensorFree((Tensor**)&outputs->y_);
  TensorFree((Tensor**)&outputs->dx_);

  TensorFree((Tensor**)&outputs->mean_);
  TensorFree((Tensor**)&outputs->var_);
  TensorFree((Tensor**)&outputs->std_);
  TensorFree((Tensor**)&outputs->xc_);
  TensorFree((Tensor**)&outputs->xn_);

  TensorFree((Tensor**)&outputs->dmean_);
  TensorFree((Tensor**)&outputs->dvar_);
  TensorFree((Tensor**)&outputs->dstd_);
  TensorFree((Tensor**)&outputs->dxc_);
  TensorFree((Tensor**)&outputs->dxn_);
}

// Forward operation for the 2d batch normalization
// `x` should be of size (B, C, H, W)
// `params->weight_` should be of size (C)
// `params->bias_` should be of size (C)
// `params->running_mean_` should be of size (C)
// `params->running_var_` should be of size (C)
// The returned tensor `outputs->y_` is of size (B, C, H, W)
// The returned tensor `outputs->mean_` is of size (C) (used for training)
// The returned tensor `outputs->var_` is of size (C) (used for training)
// The returned tensor `outputs->std_` is of size (C) (used for training)
// The returned tensor `outputs->xc_` is of size (B, C, H, W)
// The returned tensor `outputs->xn_` is of size (B, C, H, W)
void BatchNorm2dForward(const FloatTensor* x,
                        BatchNorm2dOutputs* outputs,
                        BatchNorm2dParams* params,
                        const bool training)
{
  // The input and output tensors should not be NULL
  CheckTensor(x);
  CheckTensor(outputs->y_);
  CheckTensor(outputs->xc_);
  CheckTensor(outputs->xn_);
  CheckTensor(params->weight_);
  CheckTensor(params->bias_);
  CheckTensor(params->running_mean_);
  CheckTensor(params->running_var_);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 4);
  CheckTensorDims(params->weight_, 1);
  CheckTensorDims(params->bias_, 1);
  CheckTensorDims(params->running_mean_, 1);
  CheckTensorDims(params->running_var_, 1);

  // Check the consistency of the tensor shapes
  // Check the number of channels
  Assert(x->base_.shape_[1] == params->weight_->base_.shape_[0],
         "The number of channels is not consistent: "
         "(`x`: %d, `params->weight_`: %d)",
         x->base_.shape_[1], params->weight_->base_.shape_[0]);
  Assert(x->base_.shape_[1] == params->bias_->base_.shape_[0],
         "The number of channels is not consistent: "
         "(`x`: %d, `params->bias_`: %d)",
         x->base_.shape_[1], params->bias_->base_.shape_[0]);
  Assert(x->base_.shape_[1] == params->running_mean_->base_.shape_[0],
         "The number of channels is not consistent: "
         "(`x`: %d, `params->running_mean_`: %d)",
         x->base_.shape_[1], params->running_mean_->base_.shape_[0]);
  Assert(x->base_.shape_[1] == params->running_var_->base_.shape_[0],
         "The number of channels is not consistent: "
         "(`x`: %d, `params->running_var_`: %d)",
         x->base_.shape_[1], params->running_var_->base_.shape_[0]);

  const int batch_size = x->base_.shape_[0];
  const int channels = x->base_.shape_[1];
  const int height = x->base_.shape_[2];
  const int width = x->base_.shape_[3];

  const int num_samples = batch_size * height * width;

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)outputs->y_, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)outputs->xc_, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)outputs->xn_, (const Tensor*)x);

  if (training) {
    // The output tensors should not be NULL
    CheckTensor(outputs->mean_);
    CheckTensor(outputs->var_);
    CheckTensor(outputs->std_);

    // Set the shape of the output tensor if necessary
    TensorSetShape((Tensor*)outputs->mean_, 1, channels);
    TensorSetShape((Tensor*)outputs->var_, 1, channels);
    TensorSetShape((Tensor*)outputs->std_, 1, channels);

    // Compute the mean over the B, H, and W dimensions
    // Update the running mean
    for (int ch = 0; ch < channels; ++ch) {
      float mu = 0.0f;

      for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            mu += TensorAt4d(x, b, ch, h, w);
          }
        }
      }

      mu = mu / (float)num_samples;
      TensorAt1d(outputs->mean_, ch) = mu;
      TensorAt1d(params->running_mean_, ch) =
        (1.0f - params->momentum_) * TensorAt1d(params->running_mean_, ch)
          + params->momentum_ * mu;
    }

    // Compute the centered input `xc`
    for (int b = 0; b < batch_size; ++b) {
      for (int ch = 0; ch < channels; ++ch) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            TensorAt4d(outputs->xc_, b, ch, h, w) = TensorAt4d(x, b, ch, h, w)
              - TensorAt1d(outputs->mean_, ch);
          }
        }
      }
    }

    // Compute the standard deviation over the B, H, and W dimensions
    // Update the running variance
    for (int ch = 0; ch < channels; ++ch) {
      float val = 0.0f;

      for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            val += powf(TensorAt4d(outputs->xc_, b, ch, h, w), 2.0f);
          }
        }
      }

      val = val / (float)num_samples;
      TensorAt1d(outputs->var_, ch) = val + params->eps_;
      TensorAt1d(outputs->std_, ch) = sqrtf(val + params->eps_);
      TensorAt1d(params->running_var_, ch) =
        (1.0f - params->momentum_) * TensorAt1d(params->running_var_, ch)
          + params->momentum_ * val;
    }

    // Compute the normalized input `xn`
    for (int b = 0; b < batch_size; ++b) {
      for (int ch = 0; ch < channels; ++ch) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            TensorAt4d(outputs->xn_, b, ch, h, w) =
              TensorAt4d(outputs->xc_, b, ch, h, w)
                / TensorAt1d(outputs->std_, ch);
          }
        }
      }
    }
  } else {
    // Compute the centered input `xc`
    for (int b = 0; b < batch_size; ++b) {
      for (int ch = 0; ch < channels; ++ch) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            TensorAt4d(outputs->xc_, b, ch, h, w) = TensorAt4d(x, b, ch, h, w)
              - TensorAt1d(params->running_mean_, ch);
          }
        }
      }
    }

    // Compute the normalized input `xn`
    for (int b = 0; b < batch_size; ++b) {
      for (int ch = 0; ch < channels; ++ch) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            TensorAt4d(outputs->xn_, b, ch, h, w) =
              TensorAt4d(outputs->xc_, b, ch, h, w)
                / sqrtf(TensorAt1d(params->running_var_, ch) + params->eps_);
          }
        }
      }
    }
  }

  // Transform the input using the weight and bias
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          TensorAt4d(outputs->y_, b, ch, h, w) =
            TensorAt4d(outputs->xn_, b, ch, h, w)
              * TensorAt1d(params->weight_, ch) + TensorAt1d(params->bias_, ch);
        }
      }
    }
  }
}

// Backward operation for the 2d batch normalization
// `dy` should be of size (B, C, H, W)
// `x` should be of size (B, C, H, W)
// `outputs->mean_` should be of size (C)
// `outputs->var_` should be of size (C)
// `outputs->std_` should be of size (C)
// `outputs->xc_` should be of size (B, C, H, W)
// `outputs->xn_` should be of size (B, C, H, W)
// The returned tensor `outputs->dx_` is of size (B, C, H, W)
// The returned tensor `params->dweight_` is of size (C)
// The returned tensor `params->dbias_` is of size (C)
// `params->weight_` should be of size (C)
// `params->bias_` should be of size (C)
// `params->running_mean_` should be of size (C)
// `params->running_var_` should be of size (C)
void BatchNorm2dBackward(const FloatTensor* dy,
                         const FloatTensor* x,
                         BatchNorm2dOutputs* outputs,
                         BatchNorm2dParams* params)
{
  // The input and output tensors should not be NULL
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(outputs->mean_);
  CheckTensor(outputs->var_);
  CheckTensor(outputs->std_);
  CheckTensor(outputs->xc_);
  CheckTensor(outputs->xn_);
  CheckTensor(outputs->dx_);
  CheckTensor(outputs->dmean_);
  CheckTensor(outputs->dvar_);
  CheckTensor(outputs->dstd_);
  CheckTensor(outputs->dxc_);
  CheckTensor(outputs->dxn_);
  CheckTensor(params->weight_);
  CheckTensor(params->dweight_);
  CheckTensor(params->dbias_);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 4);
  CheckTensorDims(x, 4);
  CheckTensorDims(outputs->mean_, 1);
  CheckTensorDims(outputs->var_, 1);
  CheckTensorDims(outputs->std_, 1);
  CheckTensorDims(outputs->xc_, 4);
  CheckTensorDims(outputs->xn_, 4);

  // Check the consistency of the tensor shapes
  Assert(TensorIsShapeEqual((const Tensor*)dy, (const Tensor*)x),
         "Input tensors `dy` and `x` should have the same shape");
  Assert(TensorIsShapeEqual((const Tensor*)x, (const Tensor*)outputs->xc_),
         "Input tensors `x` and `outputs->xc_` should have the same shape");
  Assert(TensorIsShapeEqual((const Tensor*)x, (const Tensor*)outputs->xn_),
         "Input tensors `x` and `outputs->xn_` should have the same shape");
  Assert(dy->base_.shape_[1] == outputs->mean_->base_.shape_[0],
         "The number of input channels is not consistent: "
         "(`dy`: %d, `outputs->mean_`: %d)",
         dy->base_.shape_[1], outputs->mean_->base_.shape_[0]);
  Assert(dy->base_.shape_[1] == outputs->var_->base_.shape_[0],
         "The number of input channels is not consistent: "
         "(`dy`: %d, `outputs->var_`: %d)",
         dy->base_.shape_[1], outputs->var_->base_.shape_[0]);
  Assert(dy->base_.shape_[1] == outputs->std_->base_.shape_[0],
         "The number of input channels is not consistent: "
         "(`dy`: %d, `outputs->std_`: %d)",
         dy->base_.shape_[1], outputs->std_->base_.shape_[0]);

  const int batch_size = dy->base_.shape_[0];
  const int channels = dy->base_.shape_[1];
  const int height = dy->base_.shape_[2];
  const int width = dy->base_.shape_[3];

  const int num_samples = batch_size * height * width;

  // Set the shape of the output tensors if necessary
  TensorSetShapeLike((Tensor*)outputs->dx_, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)outputs->dxc_, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)outputs->dxn_, (const Tensor*)x);
  TensorSetShape((Tensor*)outputs->dmean_, 1, channels);
  TensorSetShape((Tensor*)outputs->dvar_, 1, channels);
  TensorSetShape((Tensor*)outputs->dstd_, 1, channels);
  TensorSetShape((Tensor*)params->dweight_, 1, channels);
  TensorSetShape((Tensor*)params->dbias_, 1, channels);

  // Perform the backpropagation for the 2d batch normalization
  // Compute the gradient for the weight
  for (int ch = 0; ch < channels; ++ch) {
    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          val += TensorAt4d(dy, b, ch, h, w)
            * TensorAt4d(outputs->xn_, b, ch, h, w);
        }
      }
    }

    TensorAt1d(params->dweight_, ch) = val;
  }

  // Compute the gradient for the bias
  for (int ch = 0; ch < channels; ++ch) {
    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          val += TensorAt4d(dy, b, ch, h, w);
        }
      }
    }

    TensorAt1d(params->dbias_, ch) = val;
  }

  // Compute the gradient for the normalized input `xn`
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          TensorAt4d(outputs->dxn_, b, ch, h, w) = TensorAt4d(dy, b, ch, h, w)
            * TensorAt1d(params->weight_, ch);
        }
      }
    }
  }

  // Compute the gradient for the centered input `xc`
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          TensorAt4d(outputs->dxc_, b, ch, h, w) =
            TensorAt4d(outputs->dxn_, b, ch, h, w)
              / TensorAt1d(outputs->std_, ch);
        }
      }
    }
  }

  // Compute the gradient for the standard deviation `std`
  for (int ch = 0; ch < channels; ++ch) {
    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          val += TensorAt4d(outputs->dxn_, b, ch, h, w)
            * TensorAt4d(outputs->xc_, b, ch, h, w)
              / TensorAt1d(outputs->var_, ch);
        }
      }
    }

    TensorAt1d(outputs->dstd_, ch) = -val;
  }

  // Compute the gradient for the variance
  for (int ch = 0; ch < channels; ++ch) {
    TensorAt1d(outputs->dvar_, ch) = 0.5f * TensorAt1d(outputs->dstd_, ch)
      / TensorAt1d(outputs->std_, ch);
  }

  // Compute the gradient for the mean and input
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          TensorAt4d(outputs->dxc_, b, ch, h, w) += 2.0f
            * TensorAt4d(outputs->xc_, b, ch, h, w)
              * TensorAt1d(outputs->dvar_, ch) / (float)num_samples;
        }
      }
    }
  }

  // Compute the gradient for the mean
  for (int ch = 0; ch < channels; ++ch) {
    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          val += TensorAt4d(outputs->dxc_, b, ch, h, w);
        }
      }
    }

    TensorAt1d(outputs->dmean_, ch) = -val;
  }

  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          TensorAt4d(outputs->dx_, b, ch, h, w) =
            TensorAt4d(outputs->dxc_, b, ch, h, w)
              + TensorAt1d(outputs->dmean_, ch) / (float)num_samples;
        }
      }
    }
  }
}
