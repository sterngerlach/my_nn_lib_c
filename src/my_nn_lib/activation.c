
// activation.c

#include "my_nn_lib/activation.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_ops.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

#include <math.h>

// Initialize the outputs for the Softmax activation
void ActivationOutputsInitialize(ActivationOutputs* outputs,
                                 const bool inference_only)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  outputs->y_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  if (!inference_only)
    outputs->dx_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  else
    outputs->dx_ = NULL;
}

// Free the outputs for the Softmax activation
void ActivationOutputsFree(ActivationOutputs* outputs)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  TensorFree((Tensor**)&outputs->y_);
  TensorFree((Tensor**)&outputs->dx_);
}

// Forward operation for the Softmax activation
// `x` should be of size (B, D)
// The returned tensor `outputs->y_` is of size (B, D)
void SoftmaxForward(const FloatTensor* x,
                    ActivationOutputs* outputs)
{
  SoftmaxForwardF(x, outputs->y_);
}

// Backward operation for the Softmax activation
// `dy` should be of size (B, D)
// `outputs->y_` should be of size (B, D)
// The returned tensor `outputs->dx_` is of size (B, D)
void SoftmaxBackward(const FloatTensor* dy,
                     ActivationOutputs* outputs)
{
  SoftmaxBackwardF(dy, outputs->y_, outputs->dx_);
}

// Forward operation for the Softmax activation
// `x` should be of size (B, D)
// The returned tensor `y` is of size (B, D)
void SoftmaxForwardF(const FloatTensor* x,
                     FloatTensor* y)
{
  // The input and output tensor should not be NULL
  CheckTensor(x);
  CheckTensor(y);

  // Check the dimensions of the input tensor
  CheckTensorDims(x, 2);

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)y, (const Tensor*)x);

  const int batch_size = x->base_.shape_[0];
  const int dims = x->base_.shape_[1];

  // Find the maximum element from the input tensor `x`
  const float x_max = FloatTensorMaxElement(x);

  // Compute exponentials
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < dims; ++i) {
      TensorAt2d(y, b, i) = expf(TensorAt2d(x, b, i) - x_max);
    }
  }

  // Normalize the exponentials
  for (int b = 0; b < batch_size; ++b) {
    float sum = 0;
    float sum_inv = 0;

    for (int i = 0; i < dims; ++i)
      sum += TensorAt2d(y, b, i);

    sum_inv = 1.0f / sum;

    for (int i = 0; i < dims; ++i)
      TensorAt2d(y, b, i) *= sum_inv;
  }
}

// Backward operation for the Softmax activation
// `dy` should be of size (B, D)
// `y` should be of size (B, D)
// The returned tensor `dx` is of size (B, D)
void SoftmaxBackwardF(const FloatTensor* dy,
                      const FloatTensor* y,
                      FloatTensor* dx)
{
  // The input and output tensors should not be NULL
  CheckTensor(dy);
  CheckTensor(y);
  CheckTensor(dx);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 2);
  CheckTensorDims(y, 2);

  // `dy` and `y` should have the same shape
  Assert(TensorIsShapeEqual((const Tensor*)dy, (const Tensor*)y),
         "Input tensors `dy` and `y` should have the same shape");

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)dy);

  const int batch_size = dy->base_.shape_[0];
  const int dims = dy->base_.shape_[1];

  // Perform the backpropagation of the Softmax activation
  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    float sum = 0.0f;
    for (int i = 0; i < dims; ++i) {
      sum += TensorAt2d(dy, b, i) * TensorAt2d(y, b, i);
    }

    for (int i = 0; i < dims; ++i) {
      TensorAt2d(dx, b, i) = TensorAt2d(y, b, i)
        * (TensorAt2d(dy, b, i) - sum);
    }
  }
}

// Forward operation for the ReLU activation
// `x` should be of size (*)
// The returned tensor `outputs->y_` is of size (*)
void ReLUForward(const FloatTensor* x,
                 ActivationOutputs* outputs)
{
  ReLUForwardF(x, outputs->y_);
}

// Backward operation for the ReLU activation
// `dy` should be of size (*)
// `x` should be of size (*)
// The returned tensor `outputs->dx_` is of size (*)
void ReLUBackward(const FloatTensor* dy,
                  const FloatTensor* x,
                  ActivationOutputs* outputs)
{
  ReLUBackwardF(dy, x, outputs->dx_);
}

// Forward operation for the ReLU activation
// `x` should be of size (*)
// The returned tensor `y` is of size (*)
void ReLUForwardF(const FloatTensor* x,
                  FloatTensor* y)
{
  // The input and output tensor should not be NULL
  CheckTensor(x);
  CheckTensor(y);

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)y, (const Tensor*)x);

  // We treat both tensors as 1D tensors
  for (int i = 0; i < x->base_.numel_; ++i)
    TensorAt1d(y, i) = TensorAt1d(x, i) > 0 ? TensorAt1d(x, i) : 0;
}

// Backward operation for the ReLU activation
// `dy` should be of size (*)
// `x` should be of size (*)
// The returned tensor `dx` is of size (*)
void ReLUBackwardF(const FloatTensor* dy,
                   const FloatTensor* x,
                   FloatTensor* dx)
{
  // The input and output tensors should not be NULL
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(dx);

  // `dy` and `x` should have the same shape
  Assert(TensorIsShapeEqual((const Tensor*)dy, (const Tensor*)x),
         "Input tensors `dy` and `x` should have the same shape");

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)x);

  // We treat tensors as 1D tensors
  for (int i = 0; i < x->base_.numel_; ++i)
    TensorAt1d(dx, i) = TensorAt1d(x, i) > 0 ? TensorAt1d(dy, i) : 0;
}

// Forward operation for the Sigmoid activation
// `x` should be of size (*)
// The returned tensor `outputs->y_` is of size (*)
void SigmoidForward(const FloatTensor* x,
                    ActivationOutputs* outputs)
{
  SigmoidForwardF(x, outputs->y_);
}

// Backward operation for the Sigmoid activation
// `dy` should be of size (*)
// `outputs->y_` should be of size (*)
// The returned tensor `outputs->dx_` is of size (*)
void SigmoidBackward(const FloatTensor* dy,
                     ActivationOutputs* outputs)
{
  SigmoidBackwardF(dy, outputs->y_, outputs->dx_);
}

// Forward operation for the Sigmoid activation
// `x` should be of size (*)
// The returned tensor `y` is of size (*)
void SigmoidForwardF(const FloatTensor* x,
                     FloatTensor* y)
{
  // The input and output tensor should not be NULL
  CheckTensor(x);
  CheckTensor(y);

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)y, (const Tensor*)x);

  // We treat both tensors as 1D tensors
  for (int i = 0; i < x->base_.numel_; ++i)
    TensorAt1d(y, i) = 1.0f / (1.0f + expf(TensorAt1d(x, i)));
}

// Backward operation for the Sigmoid activation
// `dy` should be of size (*)
// `y` should be of size (*)
// The returned tensor `dx` is of size (*)
void SigmoidBackwardF(const FloatTensor* dy,
                      const FloatTensor* y,
                      FloatTensor* dx)
{
  // The input and output tensors should not be NULL
  CheckTensor(dy);
  CheckTensor(y);
  CheckTensor(dx);

  // `dy` and `y` should have the same shape
  Assert(TensorIsShapeEqual((const Tensor*)dy, (const Tensor*)y),
         "Input tensors `dy` and `y` should have the same shape");

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)dy);

  // We treat tensors as 1D tensors
  for (int i = 0; i < dy->base_.numel_; ++i)
    TensorAt1d(dx, i) = TensorAt1d(dy, i)
      * TensorAt1d(y, i) * (1.0f - TensorAt1d(y, i));
}
