
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

  // Gradient for the weight parameters of size (Cout, Cin, KH, KW)
  // (used for training)
  FloatTensor* dweight_;
  // Gradient for the bias parameters of size (Cout) (used for training)
  FloatTensor* dbias_;
} Conv2dParams;

// Outputs for the 2d convolution layer
typedef struct
{
  // Output of size (B, Cout, Hout, Wout)
  FloatTensor* y_;
  // Gradient for the input of size (B, Cin, Hin, Win) (used for training)
  FloatTensor* dx_;
} Conv2dOutputs;

// Initialize the parameters for the 2d convolution layer
void Conv2dParamsInitialize(Conv2dParams* params,
                            const int in_channels,
                            const int out_channels,
                            const int kernel_width,
                            const int kernel_height,
                            const int stride,
                            const int padding,
                            const bool inference_only);

// Free the parameters for the 2d convolution layer
void Conv2dParamsFree(Conv2dParams* params);

// Initialize the outputs for the 2d convolution layer
void Conv2dOutputsInitialize(Conv2dOutputs* outputs,
                             const bool inference_only);

// Free the outputs for the 2d convolution layer
void Conv2dOutputsFree(Conv2dOutputs* outputs);

// Forward operation for the 2d convolution
// `x` should be of size (B, Cin, Hin, Win)
// `params->weight_` should be of size (Cout, Cin, KH, KW)
// `params->bias_` should be of size (Cout)
// `params->bias_` may be `NULL`
// The returned tensor `outputs->y_` is of size (B, Cout, Hout, Wout)
void Conv2dForward(const FloatTensor* x,
                   Conv2dOutputs* outputs,
                   const Conv2dParams* params);

// Backward operation for the 2d convolution
// `dy` should be of size (B, Cout, Hout, Wout)
// `x` should be of size (B, Cin, Hin, Win)
// The returned tensor `outputs->dx_` is of size (B, Cin, Hin, Win)
// The returned tensor `params->dweight_` is of size (Cout, Cin, KH, KW)
// The returned tensor `params->dbias_` is of size (Cout)
// `params->weight_` should be of size (Cout, Cin, KH, KW)
// `params->bias_` should be of size (Cout)
// `params->bias_` may be `NULL`
void Conv2dBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    Conv2dOutputs* outputs,
                    Conv2dParams* params);

#ifdef __cplusplus
}
#endif

#endif // CONV2D_H
