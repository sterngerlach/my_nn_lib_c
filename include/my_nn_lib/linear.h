
// linear.h

#ifndef LINEAR_H
#define LINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Parameters for the fully-connected layer
typedef struct
{
  // Weight parameters of size (Dout, Din)
  FloatTensor* weight_;
  // Bias parameters of size (Dout)
  FloatTensor* bias_;

  // Gradient for the weight parameters of size (Dout, Din)
  FloatTensor* dweight_;
  // Gradient for the bias parameters of size (Dout)
  FloatTensor* dbias_;
} LinearParams;

// Outputs for the fully-connected layer
typedef struct
{
  // Output of size (B, Dout)
  FloatTensor* y_;
  // Gradient for the input of size (B, Din) (used for training)
  FloatTensor* dx_;
} LinearOutputs;

// Initialize the parameters for the fully-connected layer
void LinearParamsInitialize(LinearParams* params,
                            const int in_dims,
                            const int out_dims,
                            const bool inference_only);

// Free the parameters for the fully-connected layer
void LinearParamsFree(LinearParams* params);

// Initialize the outputs for the fully-connected layer
void LinearOutputsInitialize(LinearOutputs* outputs,
                             const bool inference_only);

// Free the outputs for the fully-connected layer
void LinearOutputsFree(LinearOutputs* outputs);

// Forward operation for the fully-connected layer
// `x` should be of size (B, Din)
// The returned tensor `outputs->y_` is of size (B, Dout)
// `params->weight_` should be of size (Dout, Din)
// `params->bias_` should be of size (Dout)
// `params->bias_` may be `NULL`
void LinearForward(const FloatTensor* x,
                   LinearOutputs* outputs,
                   const LinearParams* params);

// Backward operation for the fully-connected layer
// `dy` should be of size (B, Dout)
// `x` should be of size (B, Din)
// The returned tensor `outputs->dx_` is of size (B, Din)
// The returned tensor `params->dweight_` is of size (Dout, Din)
// The returned tensor `params->dbias_` is of size (Dout)
// `params->weight_` should be of size (Dout, Din)
// `params->bias_` should be of size (Dout)
// `params->bias_` may be `NULL`
void LinearBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    LinearOutputs* outputs,
                    LinearParams* params);

#ifdef __cplusplus
}
#endif

#endif // LINEAR_H
