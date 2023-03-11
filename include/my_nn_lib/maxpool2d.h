
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

// Initialize the parameters for the 2D max-pooling layer
void MaxPool2dParamsInitialize(MaxPool2dParams* params,
                               const int kernel_height,
                               const int kernel_width,
                               const int stride,
                               const int padding);

// Free the parameters for the 2D max-pooling layer
void MaxPool2dParamsFree(MaxPool2dParams* params);

// Forward operation for the 2D max-pooling
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `y` is of size (B, C, Hout, Wout)
// The returned tensor `mask` is of size (B, C, Hout, Wout)
void MaxPool2dForward(const FloatTensor* x,
                      FloatTensor* y,
                      Index2dTensor* mask,
                      const MaxPool2dParams* params);

// Backward operation for the 2D max-pooling
// `dy` should be of size (B, C, Hout, Wout)
// `mask` should be of size (B, C, Hout, Wout)
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `dx` is of size (B, C, Hin, Win)
void MaxPool2dBackward(const FloatTensor* dy,
                       const Index2dTensor* mask,
                       const FloatTensor* x,
                       FloatTensor* dx,
                       const MaxPool2dParams* params);

#ifdef __cplusplus
}
#endif

#endif // MAXPOOL2D_H
