
// maxpool2d.h

#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Forward operation for the 2D max-pooling
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `y` is of size (B, C, Hout, Wout)
// The returned tensor `mask` is of size (B, C, Hout, Wout)
void MaxPool2dForward(const FloatTensor* x,
                      FloatTensor* y,
                      Index2dTensor* mask,
                      const int kernel_height,
                      const int kernel_width,
                      const int stride,
                      const int padding);

// Backward operation for the 2D max-pooling
// `dy` should be of size (B, C, Hout, Wout)
// `mask` should be of size (B, C, Hout, Wout)
// `x` should be of size (B, C, Hin, Win)
// The returned tensor `dx` is of size (B, C, Hin, Win)
void MaxPool2dBackward(const FloatTensor* dy,
                       const Index2dTensor* mask,
                       const FloatTensor* x,
                       FloatTensor* dx,
                       const int kernel_height,
                       const int kernel_width,
                       const int stride,
                       const int padding);

#ifdef __cplusplus
}
#endif

#endif // MAXPOOL2D_H
