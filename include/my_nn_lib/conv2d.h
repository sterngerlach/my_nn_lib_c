
// conv2d.h

#ifndef CONV2D_H
#define CONV2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Parameters for the 2d convolution layer
typedef struct
{
  // Weight parameters of size (Cout, Cin, KH, KW)
  FloatTensor* weight_;
  // Bias parameters of size (Cout)
  FloatTensor* bias_;
  // Stride
  int          stride_;
  // Padding
  int          padding_;
} Conv2dParams;

// Initialize the parameters for the 2d convolution layer
void Conv2dParamsInitialize(Conv2dParams* params,
                            const int in_channels,
                            const int out_channels,
                            const int kernel_width,
                            const int kernel_height,
                            const int stride,
                            const int padding);

// Free the parameters for the 2d convolution layer
void Conv2dParamsFree(Conv2dParams* params);

// Forward operation for the 2d convolution
// `x` should be of size (B, Cin, Hin, Win)
// `params->weight_` should be of size (Cout, Cin, KH, KW)
// `params->bias_` should be of size (Cout)
// `params->bias_` may be `NULL`
// The returned tensor `y` is of size (B, Cout, Hout, Wout)
void Conv2dForward(const FloatTensor* x,
                   FloatTensor* y,
                   const Conv2dParams* params);

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
                    const Conv2dParams* params);

#ifdef __cplusplus
}
#endif

#endif // CONV2D_H
