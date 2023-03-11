
// conv2d.h

#ifndef CONV2D_H
#define CONV2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Forward operation for the 2d convolution
// `x` should be of size (B, Cin, Hin, Win)
// `weight` should be of size (Cout, Cin, KH, KW)
// `bias` should be of size (Cout)
// `bias` may be `NULL`
// The returned tensor `y` is of size (B, Cout, Hout, Wout)
void Conv2dForward(const FloatTensor* x,
                   FloatTensor* y,
                   const FloatTensor* weight,
                   const FloatTensor* bias,
                   const int stride,
                   const int padding);

// Backward operation for the 2d convolution
// `dy` should be of size (B, Cout, Hout, Wout)
// `x` should be of size (B, Cin, Hin, Win)
// The returned tensor `dx` is of size (B, Cin, Hin, Win)
// The returned tensor `dweight` is of size (Cout, Cin, KH, KW)
// The returned tensor `dbias` is of size (Cout)
// `weight` should be of size (Cout, Cin, KH, KW)
// `bias` should be of size (Cout)
// `bias` may be `NULL`
void Conv2dBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    FloatTensor* dx,
                    FloatTensor* dweight,
                    FloatTensor* dbias,
                    const FloatTensor* weight,
                    const FloatTensor* bias,
                    const int stride,
                    const int padding);

#ifdef __cplusplus
}
#endif

#endif // CONV2D_H
