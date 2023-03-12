
// flatten.h

#ifndef FLATTEN_H
#define FLATTEN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Outputs for the flattening operation
typedef struct
{
  // Output of size (B, C0 * C1 ... * Cn)
  FloatTensor* y_;
  // Gradient for the input of size (B, C0, C1, ..., Cn) (used for training)
  FloatTensor* dx_;
} FlattenOutputs;

// Initialize the outputs for the flattening operation
void FlattenOutputsInitialize(FlattenOutputs* outputs,
                              const bool inference_only);

// Free the outputs for the flattening operation
void FlattenOutputsFree(FlattenOutputs* outputs);

// Forward operation for the flattening operation
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `outputs->y_` is of size (B, C0 * C1 ... * Cn)
void FlattenForward(const FloatTensor* x,
                    FlattenOutputs* outputs);

// Backward operation for the flattening operation
// `dy` should be of size (B, C0 * C1 * ... * Cn)
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `outputs->dx_` is of size (B, C0, C1, ..., Cn)
void FlattenBackward(const FloatTensor* dy,
                     const FloatTensor* x,
                     FlattenOutputs* outputs);

// Forward operation for the flattening operation
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `y` is of size (B, C0 * C1 ... * Cn)
void FlattenForwardF(const FloatTensor* x,
                     FloatTensor* y);

// Backward operation for the flattening operation
// `dy` should be of size (B, C0 * C1 * ... * Cn)
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `dx` is of size (B, C0, C1, ..., Cn)
void FlattenBackwardF(const FloatTensor* dy,
                      const FloatTensor* x,
                      FloatTensor* dx);

#ifdef __cplusplus
}
#endif

#endif // FLATTEN_H
