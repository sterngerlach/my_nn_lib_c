
// loss.c

#include "my_nn_lib/activation.h"
#include "my_nn_lib/loss.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"

#include <math.h>

// Initialize the outputs for the cross-entropy loss
void CrossEntropyLossOutputsInitialize(CrossEntropyLossOutputs* outputs,
                                       const bool inference_only)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  outputs->y_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->loss_ = 0.0f;

  if (!inference_only)
    outputs->dx_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  else
    outputs->dx_ = NULL;
}

// Free the outputs for the cross-entropy loss
void CrossEntropyLossOutputsFree(CrossEntropyLossOutputs* outputs)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  TensorFree((Tensor**)&outputs->y_);
  TensorFree((Tensor**)&outputs->dx_);
  outputs->loss_ = 0.0f;
}

// Forward operation for the cross-entropy loss
// `x` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `outputs->y_` is of size (B, D)
void CrossEntropyLossForward(const FloatTensor* x,
                             const IntTensor* target,
                             CrossEntropyLossOutputs* outputs)
{
  outputs->loss_ = CrossEntropyLossForwardF(x, target, outputs->y_);
}

// Backward operation for the cross-entropy loss
// `target` should be a int tensor of size (B)
// `outputs->y_` should be a float tensor of size (B, D)
// The returned float tensor `outputs->dx_` is of size (B, D)
void CrossEntropyLossBackward(const IntTensor* target,
                              CrossEntropyLossOutputs* outputs)
{
  CrossEntropyLossBackwardF(outputs->y_, target, outputs->dx_);
}

// Forward operation for the cross-entropy loss
// `x` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `y` is of size (B, D)
float CrossEntropyLossForwardF(const FloatTensor* x,
                               const IntTensor* target,
                               FloatTensor* y)
{
  // The input and output tensors should not be NULL
  CheckTensor(x);
  CheckTensor(target);
  CheckTensor(y);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 2);
  CheckTensorDims(target, 1);

  const int batch_size = x->base_.shape_[0];

  // Compute the softmax from the logits
  // `y` is of size (B, D)
  SoftmaxForwardF(x, y);

  // Compute the cross-entropy loss
  float loss = 0.0f;

  for (int b = 0; b < batch_size; ++b) {
    const int t = TensorAt1d(target, b);
    const float prob = TensorAt2d(y, b, t);
    loss += logf(prob);
  }

  return -loss / (float)batch_size;
}

// Backward operation for the cross-entropy loss
// `y` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `dx` is of size (B, D)
void CrossEntropyLossBackwardF(const FloatTensor* y,
                               const IntTensor* target,
                               FloatTensor* dx)
{
  // The input and output tensors should not be NULL
  CheckTensor(y);
  CheckTensor(target);
  CheckTensor(dx);

  // Check the dimensions of the input tensors
  CheckTensorDims(y, 2);
  CheckTensorDims(target, 1);

  const int batch_size = y->base_.shape_[0];
  const int dims = y->base_.shape_[1];

  const float batch_size_inv = 1.0f / (float)batch_size;

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)y);

  // Perform the backward operation
  for (int b = 0; b < batch_size; ++b) {
    const int t = TensorAt1d(target, b);
    for (int i = 0; i < dims; ++i) {
      const float loss = (i == t) ?
        (TensorAt2d(y, b, i) - 1.0f) : TensorAt2d(y, b, i);
      TensorAt2d(dx, b, i) = loss * batch_size_inv;
    }
  }
}
