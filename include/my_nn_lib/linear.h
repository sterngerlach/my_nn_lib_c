
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
} LinearParams;

// Initialize the parameters for the fully-connected layer
void LinearParamsInitialize(LinearParams* params,
                            const int in_dims,
                            const int out_dims);

// Free the parameters for the fully-connected layer
void LinearParamsFree(LinearParams* params);

// Forward operation for the fully-connected layer
// `x` should be of size (B, Din)
// The returned tensor `y` is of size (B, Dout)
// `params->weight_` should be of size (Dout, Din)
// `params->bias_` should be of size (Dout)
// `params->bias_` may be `NULL`
void LinearForward(const FloatTensor* x,
                   FloatTensor* y,
                   const LinearParams* params);

// Backward operation for the fully-connected layer
// `dy` should be of size (B, Dout)
// `x` should be of size (B, Din)
// The returned tensor `dx` is of size (B, Din)
// The returned tensor `dparams->weight_` is of size (Dout, Din)
// The returned tensor `dparams->bias_` is of size (Dout)
// `params->weight_` should be of size (Dout, Din)
// `params->bias_` should be of size (Dout)
// `params->bias_` may be `NULL`
void LinearBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    FloatTensor* dx,
                    LinearParams* dparams,
                    const LinearParams* params);

#ifdef __cplusplus
}
#endif

#endif // LINEAR_H
