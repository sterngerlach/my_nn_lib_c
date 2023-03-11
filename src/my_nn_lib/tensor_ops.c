
// tensor_ops.c

#include "my_nn_lib/tensor_ops.h"
#include "my_nn_lib/tensor_util.h"

#include <float.h>

// Subtract the tensor from the float tensor (in-place)
// `y` = `y` - `a` * `b`
void FloatTensorSubScaleI(FloatTensor* y,
                          const FloatTensor* a,
                          const float b)
{
  CheckTensor(y);
  CheckTensor(a);

  // `y` and `a` should have the same shape
  Assert(TensorIsShapeEqual((const Tensor*)y, (const Tensor*)a),
         "Input tensors `y` and `a` should have the same shape");

  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < y->base_.numel_; ++i)
    TensorAt1d(y, i) -= TensorAt1d(a, i) * b;
}

// Subtract the scalar from the float tensor (in-place)
// `y` = `y` - `a`
void FloatTensorSubScalarI(FloatTensor* y,
                           const float a)
{
  CheckTensor(y);

  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < y->base_.numel_; ++i)
    TensorAt1d(y, i) -= a;
}

// Argmax for the last dimension of the 2D tensor
// The returned tensor `y` is a int tensor of size (B)
// `x` is a float tensor of size (B, D)
void FloatTensor2dArgMax(IntTensor* y,
                         const FloatTensor* x)
{
  CheckTensor(y);
  CheckTensor(x);

  CheckTensorDims(x, 2);

  // Set the size of the output tensor if necessary
  TensorSetShape((Tensor*)y, 1, x->base_.shape_[0]);

  for (int i = 0; i < x->base_.shape_[0]; ++i) {
    float val = FLT_MIN;
    int max_idx = -1;

    for (int j = 0; j < x->base_.shape_[1]; ++j) {
      if (TensorAt2d(x, i, j) > val) {
        val = TensorAt2d(x, i, j);
        max_idx = j;
      }
    }

    TensorAt1d(y, i) = max_idx;
  }
}

// Find the maximum element in the float tensor
float FloatTensorMaxElement(const FloatTensor* tensor)
{
  CheckTensor(tensor);

  float val = FLT_MIN;

  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i)
    val = TensorAt1d(tensor, i) > val ? TensorAt1d(tensor, i) : val;

  return val;
}

// Find the minimum element in the float tensor
float FloatTensorMinElement(const FloatTensor* tensor)
{
  CheckTensor(tensor);

  float val = FLT_MAX;

  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i)
    val = TensorAt1d(tensor, i) < val ? TensorAt1d(tensor, i) : val;

  return val;
}

// Find the minimum and maximum elements in the float tensor
void FloatTensorMinMaxElement(const FloatTensor* tensor,
                              float* val_min,
                              float* val_max)
{
  CheckTensor(tensor);
  Assert(val_min != NULL, "`val_min` should not be NULL");
  Assert(val_max != NULL, "`val_max` should not be NULL");

  *val_min = FLT_MAX;
  *val_max = FLT_MIN;

  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i) {
    *val_max = TensorAt1d(tensor, i) > *val_max
      ? TensorAt1d(tensor, i) : *val_max;
    *val_min = TensorAt1d(tensor, i) < *val_min
      ? TensorAt1d(tensor, i) : *val_min;
  }
}
