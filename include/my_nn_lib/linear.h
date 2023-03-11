
// linear.h

#ifndef LINEAR_H
#define LINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Forward operation for the fully-connected layer
// `x` should be of size (B, Din)
// The returned tensor `y` is of size (B, Dout)
// `weight` should be of size (Dout, Din)
// `bias` should be of size (Dout)
// `bias` may be `NULL`
void LinearForward(const FloatTensor* x,
                   FloatTensor* y,
                   const FloatTensor* weight,
                   const FloatTensor* bias);

// Backward operation for the fully-connected layer
// `dy` should be of size (B, Dout)
// `x` should be of size (B, Din)
// The returned tensor `dx` is of size (B, Din)
// The returned tensor `dweight` is of size (Dout, Din)
// The returned tensor `dbias` is of size (Dout)
// `weight` should be of size (Dout, Din)
// `bias` should be of size (Dout)
// `bias` may be `NULL`
void LinearBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    FloatTensor* dx,
                    FloatTensor* dweight,
                    FloatTensor* dbias,
                    const FloatTensor* weight,
                    const FloatTensor* bias);

#ifdef __cplusplus
}
#endif

#endif // LINEAR_H
