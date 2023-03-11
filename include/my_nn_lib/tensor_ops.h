
// tensor_ops.h

#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Subtract the tensor from the float tensor (in-place)
// `y` = `y` - `a` * `b`
void FloatTensorSubScaleI(FloatTensor* y,
                          const FloatTensor* a,
                          const float b);

// Subtract the scalar from the float tensor (in-place)
// `y` = `y` - `a`
void FloatTensorSubScalarI(FloatTensor* y,
                           const float a);

// Argmax for the last dimension of the 2D tensor
// The returned tensor `y` is a int tensor of size (B)
// `x` is a float tensor of size (B, D)
void FloatTensor2dArgMax(IntTensor* y,
                         const FloatTensor* x);

// Find the maximum element in the tensor
float FloatTensorMaxElement(const FloatTensor* tensor);

// Find the minimum element in the tensor
float FloatTensorMinElement(const FloatTensor* tensor);

// Find the minimum and maximum elements in the tensor
void FloatTensorMinMaxElement(const FloatTensor* tensor,
                              float* val_min,
                              float* val_max);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_OPS_H
