
// loss.c

#include "my_nn_lib/activation.h"
#include "my_nn_lib/loss.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"

#include <math.h>

// Forward operation for the cross entropy loss
// `x` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `y` is of size (B, D)
float CrossEntropyLossForward(const FloatTensor* x,
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
  SoftmaxForward(x, y);

  // Compute the cross entropy loss
  float loss = 0.0f;

  for (int b = 0; b < batch_size; ++b) {
    const int t = TensorAt1d(target, b);
    const float prob = TensorAt2d(y, b, t);
    loss += logf(prob);
  }

  return -loss / (float)batch_size;
}

// Backward operation for the cross entropy loss
// `y` should be a float tensor of size (B, D)
// `target` should be a int tensor of size (B)
// The returned float tensor `dx` is of size (B, D)
void CrossEntropyLossBackward(const FloatTensor* y,
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
