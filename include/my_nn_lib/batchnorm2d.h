
// batchnorm2d.h

#ifndef BATCHNORM2D_H
#define BATCHNORM2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include "my_nn_lib/tensor.h"

// Parameters for the 2d batch normalization layer
typedef struct
{
  // Weight parameters of size (C)
  FloatTensor* weight_;
  // Bias parameters of size (C)
  FloatTensor* bias_;
  // Running mean of size (C)
  FloatTensor* running_mean_;
  // Running variance of size (C)
  FloatTensor* running_var_;
  // Epsilon
  float        eps_;
  // Momentum (to keep track the mean and variance)
  float        momentum_;
} BatchNorm2dParams;

// Outputs and intermediate results for the 2d batch normalization layer
typedef struct
{
  // Output of size (B, C, H, W)
  FloatTensor* y_;
  // Gradient for the input of size (B, C, H, W) (used for training)
  FloatTensor* dx_;

  // Input mean of size (C) (used for training)
  FloatTensor* mean_;
  // Input variance of size (C) (used for training)
  FloatTensor* var_;
  // Input standard deviation of size (C) (used for training)
  FloatTensor* std_;
  // Centered input of size (B, C, H, W)
  FloatTensor* xc_;
  // Normalized input of size (B, C, H, W)
  FloatTensor* xn_;

  // Gradient for the mean of size (C) (used for training)
  FloatTensor* dmean_;
  // Gradient for the variance of size (C) (used for training)
  FloatTensor* dvar_;
  // Gradient for the standard deviation of size (C) (used for training)
  FloatTensor* dstd_;
  // Gradient for the centered input of size (B, C, H, W) (used for training)
  FloatTensor* dxc_;
  // Gradient for the normalized input of size (B, C, H, W) (used for training)
  FloatTensor* dxn_;
} BatchNorm2dOutputs;

// Initialize the parameters for the 2d batch normalization layer
// Initializes `params->weight_` to ones
// Initializes `params->bias_` to zeros
// Initializes `params->running_mean_` to zeros
// Initializes `params->running_var_` to ones
void BatchNorm2dParamsInitialize(BatchNorm2dParams* params,
                                 const int dims,
                                 const float eps,
                                 const float momentum);

// Free the parameters for the 2d batch normalization layer
void BatchNorm2dParamsFree(BatchNorm2dParams* params);

// Initialize the outputs for the 2d batch normalization layer
void BatchNorm2dOutputsInitialize(BatchNorm2dOutputs* outputs,
                                  const bool inference_only);

// Free the outputs for the 2d batch normalization layer
void BatchNorm2dOutputsFree(BatchNorm2dOutputs* outputs);

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
                        const bool training);

// Backward operation for the 2d batch normalization
// `dy` should be of size (B, C, H, W)
// `x` should be of size (B, C, H, W)
// `outputs->mean_` should be of size (C)
// `outputs->var_` should be of size (C)
// `outputs->std_` should be of size (C)
// `outputs->xc_` should be of size (B, C, H, W)
// `outputs->xn_` should be of size (B, C, H, W)
// The returned tensor `outputs->dx_` is of size (B, C, H, W)
// The returned tensor `dparams->weight_` is of size (C)
// The returned tensor `dparams->bias_` is of size (C)
// `params->weight_` should be of size (C)
// `params->bias_` should be of size (C)
// `params->running_mean_` should be of size (C)
// `params->running_var_` should be of size (C)
void BatchNorm2dBackward(const FloatTensor* dy,
                         const FloatTensor* x,
                         BatchNorm2dOutputs* outputs,
                         BatchNorm2dParams* dparams,
                         const BatchNorm2dParams* params);

#ifdef __cplusplus
}
#endif

#endif // BATCHNORM2D_H
