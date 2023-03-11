
// loss.h

#ifndef LOSS_H
#define LOSS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Forward operation for the cross entropy loss
// `x` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `y` is of size (B, D)
float CrossEntropyLossForward(const FloatTensor* x,
                              const IntTensor* target,
                              FloatTensor* y);

// Backward operation for the cross entropy loss
// `y` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `dx` is of size (B, D)
void CrossEntropyLossBackward(const FloatTensor* y,
                              const IntTensor* target,
                              FloatTensor* dx);

#ifdef __cplusplus
}
#endif

#endif // LOSS_H
