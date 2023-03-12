
// activation.h

#ifndef ACTIVATION_H
#define ACTIVATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Outputs for the activation functions
typedef struct
{
  // Output of size (*)
  FloatTensor* y_;
  // Gradient for the input of size (*) (used for training)
  FloatTensor* dx_;
} ActivationOutputs;

// Initialize the outputs for the Softmax activation
void ActivationOutputsInitialize(ActivationOutputs* outputs,
                                 const bool inference_only);

// Free the outputs for the Softmax activation
void ActivationOutputsFree(ActivationOutputs* outputs);

// Forward operation for the Softmax activation
// `x` should be of size (B, D)
// The returned tensor `outputs->y_` is of size (B, D)
void SoftmaxForward(const FloatTensor* x,
                    ActivationOutputs* outputs);

// Backward operation for the Softmax activation
// `dy` should be of size (B, D)
// `outputs->y_` should be of size (B, D)
// The returned tensor `outputs->dx_` is of size (B, D)
void SoftmaxBackward(const FloatTensor* dy,
                     ActivationOutputs* outputs);

// Forward operation for the Softmax activation
// `x` should be of size (B, D)
// The returned tensor `y` is of size (B, D)
void SoftmaxForwardF(const FloatTensor* x,
                     FloatTensor* y);

// Backward operation for the Softmax activation
// `dy` should be of size (B, D)
// `y` should be of size (B, D)
// The returned tensor `dx` is of size (B, D)
void SoftmaxBackwardF(const FloatTensor* dy,
                      const FloatTensor* y,
                      FloatTensor* dx);

// Forward operation for the ReLU activation
// `x` should be of size (*)
// The returned tensor `outputs->y_` is of size (*)
void ReLUForward(const FloatTensor* x,
                 ActivationOutputs* outputs);

// Backward operation for the ReLU activation
// `dy` should be of size (*)
// `x` should be of size (*)
// The returned tensor `outputs->dx_` is of size (*)
void ReLUBackward(const FloatTensor* dy,
                  const FloatTensor* x,
                  ActivationOutputs* outputs);

// Forward operation for the ReLU activation
// `x` should be of size (*)
// The returned tensor `y` is of size (*)
void ReLUForwardF(const FloatTensor* x,
                  FloatTensor* y);

// Backward operation for the ReLU activation
// `dy` should be of size (*)
// `x` should be of size (*)
// The returned tensor `dx` is of size (*)
void ReLUBackwardF(const FloatTensor* dy,
                   const FloatTensor* x,
                   FloatTensor* dx);

// Forward operation for the Sigmoid activation
// `x` should be of size (*)
// The returned tensor `outputs->y_` is of size (*)
void SigmoidForward(const FloatTensor* x,
                    ActivationOutputs* outputs);

// Backward operation for the Sigmoid activation
// `dy` should be of size (*)
// `outputs->y_` should be of size (*)
// The returned tensor `outputs->dx_` is of size (*)
void SigmoidBackward(const FloatTensor* dy,
                     ActivationOutputs* outputs);

// Forward operation for the Sigmoid activation
// `x` should be of size (*)
// The returned tensor `y` is of size (*)
void SigmoidForwardF(const FloatTensor* x,
                     FloatTensor* y);

// Backward operation for the Sigmoid activation
// `dy` should be of size (*)
// `y` should be of size (*)
// The returned tensor `dx` is of size (*)
void SigmoidBackwardF(const FloatTensor* dy,
                      const FloatTensor* y,
                      FloatTensor* dx);

#ifdef __cplusplus
}
#endif

#endif // ACTIVATION_H
