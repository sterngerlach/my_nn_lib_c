
// maxpool2d.h

#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Parameters for the 2D max-pooling layer
typedef struct
{
  // Kernel height
  int kernel_height_;
  // Kernel width
  int kernel_width_;
  // Stride
  int stride_;
  // Padding
  int padding_;
} MaxPool2dParams;

// Outputs for the 2D max-pooling layer
typedef struct
{
  // Output of size (B, C, Hout, Wout)
  FloatTensor*   y_;
  // Output mask of size (B, C, Hout, Wout)
  Index2dTensor* mask_;
  // Gradient for the input of size (B, C, Hin, Win) (used for training)
  FloatTensor*   dx_;
} MaxPool2dOutputs;

// Initialize the parameters for the 2D max-pooling layer
void MaxPool2dParamsInitialize(MaxPool2dParams* params,
                               const int kernel_height,
                               const int kernel_width,
                               const int stride,
                               const int padding);

// Free the parameters for the 2D max-pooling layer
void MaxPool2dParamsFree(MaxPool2dParams* params);

// Initialize the outputs for the 2D max-pooling layer
void MaxPool2dOutputsInitialize(MaxPool2dOutputs* outputs,
                                const bool inference_only);

// Free the outputs for the 2D max-pooling layer
void MaxPool2dOutputsFree(MaxPool2dOutputs* outputs);

// Forward operation for the 2D max-pooling
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `outputs->y_` is of size (B, C, Hout, Wout)
// The returned tensor `outputs->mask_` is of size (B, C, Hout, Wout)
void MaxPool2dForward(const FloatTensor* x,
                      MaxPool2dOutputs* outputs,
                      const MaxPool2dParams* params);

// Backward operation for the 2D max-pooling
// `dy` should be of size (B, C, Hout, Wout)
// `outputs->mask_` should be of size (B, C, Hout, Wout)
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `outputs->dx_` is of size (B, C, Hin, Win)
void MaxPool2dBackward(const FloatTensor* dy,
                       const FloatTensor* x,
                       MaxPool2dOutputs* outputs,
                       const MaxPool2dParams* params);

#ifdef __cplusplus
}
#endif

#endif // MAXPOOL2D_H
