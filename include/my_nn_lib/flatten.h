
// flatten.h

#ifndef FLATTEN_H
#define FLATTEN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Forward operation for the flattening operation
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `y` is of size (B, C0 * C1 ... * Cn)
void FlattenForward(const FloatTensor* x,
                    FloatTensor* y);

// Backward operation for the flattening operation
// `dy` should be of size (B, C0 * C1 * ... * Cn)
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `dx` is of size (B, C0, C1, ..., Cn)
void FlattenBackward(const FloatTensor* dy,
                     const FloatTensor* x,
                     FloatTensor* dx);

#ifdef __cplusplus
}
#endif

#endif // FLATTEN_H
