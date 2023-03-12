
// flatten.c

#include "my_nn_lib/flatten.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

// Initialize the outputs for the flattening operation
void FlattenOutputsInitialize(FlattenOutputs* outputs,
                              const bool inference_only)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  outputs->y_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  if (!inference_only)
    outputs->dx_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  else
    outputs->dx_ = NULL;
}

// Free the outputs for the flattening operation
void FlattenOutputsFree(FlattenOutputs* outputs)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  TensorFree((Tensor**)&outputs->y_);
  TensorFree((Tensor**)&outputs->dx_);
}

// Forward operation for the flattening operation
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `outputs->y_` is of size (B, C0 * C1 ... * Cn)
void FlattenForward(const FloatTensor* x,
                    FlattenOutputs* outputs)
{
  FlattenForwardF(x, outputs->y_);
}

// Backward operation for the flattening operation
// `dy` should be of size (B, C0 * C1 * ... * Cn)
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `outputs->dx_` is of size (B, C0, C1, ..., Cn)
void FlattenBackward(const FloatTensor* dy,
                     const FloatTensor* x,
                     FlattenOutputs* outputs)
{
  FlattenBackwardF(dy, x, outputs->dx_);
}

// Forward operation for the flattening operation
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `y` is of size (B, C0 * C1 ... * Cn)
void FlattenForwardF(const FloatTensor* x,
                     FloatTensor* y)
{
  // The input and output tensors should not be NULL
  CheckTensor(x);
  CheckTensor(y);

  // Check the dimensions of the input tensors
  Assert(x->base_.ndim_ >= 2,
         "The dimension of `x` should be greater than 1");

  const int batch_size = x->base_.shape_[0];

  // Compute the size of an output tensor
  int out_dims = 1;
  for (int i = 1; i < x->base_.ndim_; ++i)
    out_dims *= x->base_.shape_[i];

  // Set the shape of the output tensor if necessary
  TensorSetShape((Tensor*)y, 2, batch_size, out_dims);

  // Perform the flattening operation
  // We treat `x` as a 1D tensor
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < out_dims; ++i) {
      TensorAt2d(y, b, i) = TensorAt1d(x, b * out_dims + i);
    }
  }
}

// Backward operation for the flattening operation
// `dy` should be of size (B, C0 * C1 * ... * Cn)
// `x` should be of size (B, C0, C1, ..., Cn)
// The returned tensor `dx` is of size (B, C0, C1, ..., Cn)
void FlattenBackwardF(const FloatTensor* dy,
                      const FloatTensor* x,
                      FloatTensor* dx)
{
  // The input and output tensors should not be NULL
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(dx);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 2);
  Assert(x->base_.ndim_ >= 2,
         "The dimension of `x` should be greater than 1");

  // Check the consistency of the tensor shapes
  int out_dims_expected = 1;
  for (int i = 1; i < x->base_.ndim_; ++i)
    out_dims_expected *= x->base_.shape_[i];

  // Check the batch size
  Assert(dy->base_.shape_[0] == x->base_.shape_[0],
         "The batch size is not consistent: (`dy`: %d, `x`: %d)",
         dy->base_.shape_[0], x->base_.shape_[0]);
  // Check the number of output dimensions
  Assert(dy->base_.shape_[1] == out_dims_expected,
         "The number of output dimensions is not consistent: "
         "(`y`: %d, expected: %d)", dy->base_.shape_[1], out_dims_expected);

  const int batch_size = x->base_.shape_[0];
  const int out_dims = dy->base_.shape_[1];

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)x);

  // Perform the backpropagation for the flattening operation
  // Compute the gradient for the input
  // We treat `dx` as a 1D tensor
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < out_dims; ++i) {
      TensorAt1d(dx, b * out_dims + i) = TensorAt2d(dy, b, i);
    }
  }
}
