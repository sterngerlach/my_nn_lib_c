
// activation.h

#ifndef ACTIVATION_H
#define ACTIVATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Forward operation for the Softmax activation
// `x` should be of size (B, D)
// The returned tensor `y` is of size (B, D)
void SoftmaxForward(const FloatTensor* x,
                    FloatTensor* y);

// Backward operation for the Softmax activation
// `dy` should be of size (B, D)
// `y` should be of size (B, D)
// The returned tensor `dx` is of size (B, D)
void SoftmaxBackward(const FloatTensor* dy,
                     const FloatTensor* y,
                     FloatTensor* dx);

// Forward operation for the ReLU activation
// `x` should be of size (*)
// The returned tensor `y` is of size (*)
void ReLUForward(const FloatTensor* x,
                 FloatTensor* y);

// Backward operation for the ReLU activation
// `dy` should be of size (*)
// `x` should be of size (*)
// The returned tensor `dx` is of size (*)
void ReLUBackward(const FloatTensor* dy,
                  const FloatTensor* x,
                  FloatTensor* dx);

// Forward operation for the Sigmoid activation
// `x` should be of size (*)
// The returned tensor `y` is of size (*)
void SigmoidForward(const FloatTensor* x,
                    FloatTensor* y);

// Backward operation for the Sigmoid activation
// `dy` should be of size (*)
// `y` should be of size (*)
// The returned tensor `dx` is of size (*)
void SigmoidBackward(const FloatTensor* dy,
                     const FloatTensor* x,
                     FloatTensor* dx);

#ifdef __cplusplus
}
#endif

#endif // ACTIVATION_H
