
// loss.h

#ifndef LOSS_H
#define LOSS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Outputs for the cross-entropy loss
typedef struct
{
  // Softmax of size (B, D)
  FloatTensor* y_;
  // Output loss
  float        loss_;
  // Gradient for the input of size (B, D) (used for training)
  FloatTensor* dx_;
} CrossEntropyLossOutputs;

// Initialize the outputs for the cross-entropy loss
void CrossEntropyLossOutputsInitialize(CrossEntropyLossOutputs* outputs,
                                       const bool inference_only);

// Free the outputs for the cross-entropy loss
void CrossEntropyLossOutputsFree(CrossEntropyLossOutputs* outputs);

// Forward operation for the cross-entropy loss
// `x` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `outputs->y_` is of size (B, D)
void CrossEntropyLossForward(const FloatTensor* x,
                             const IntTensor* target,
                             CrossEntropyLossOutputs* outputs);

// Backward operation for the cross-entropy loss
// `target` should be a int tensor of size (B)
// `outputs->y_` should be a float tensor of size (B, D)
// The returned float tensor `outputs->dx_` is of size (B, D)
void CrossEntropyLossBackward(const IntTensor* target,
                              CrossEntropyLossOutputs* outputs);

// Forward operation for the cross-entropy loss
// `x` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `y` is of size (B, D)
float CrossEntropyLossForwardF(const FloatTensor* x,
                               const IntTensor* target,
                               FloatTensor* y);

// Backward operation for the cross-entropy loss
// `y` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `dx` is of size (B, D)
void CrossEntropyLossBackwardF(const FloatTensor* y,
                               const IntTensor* target,
                               FloatTensor* dx);

#ifdef __cplusplus
}
#endif

#endif // LOSS_H
