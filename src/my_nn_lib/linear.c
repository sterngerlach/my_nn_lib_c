
// linear.c

#include "my_nn_lib/linear.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

// Forward operation for the fully-connected layer
// `x` should be of size (B, Din)
// The returned tensor `y` is of size (B, Dout)
// `weight` should be of size (Dout, Din)
// `bias` should be of size (Dout)
// `bias` may be `NULL`
void LinearForward(const FloatTensor* x,
                   FloatTensor* y,
                   const FloatTensor* weight,
                   const FloatTensor* bias)
{
  // The input and output tensors should not be NULL except `bias`
  CheckTensor(x);
  CheckTensor(y);
  CheckTensor(weight);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 2);
  CheckTensorDims(weight, 2);
  CheckTensorDims(bias, 1);

  // Check the consistency of the tensor shapes
  // Check the number of input dimensions
  Assert(x->base_.shape_[1] == weight->base_.shape_[1],
         "The number of input dimensions is not consistent: "
         "(`x`: %d, `weight`: %d)",
         x->base_.shape_[1], weight->base_.shape_[1]);

  // Check the number of output dimensions
  Assert(bias == NULL || weight->base_.shape_[0] == bias->base_.shape_[0],
         "The number of output dimensions is not consistent: "
         "(`weight`: %d, `bias`: %d)",
         weight->base_.shape_[0], bias->base_.shape_[0]);

  const int batch_size = x->base_.shape_[0];
  const int in_dims = x->base_.shape_[1];
  const int out_dims = weight->base_.shape_[0];

  // Set the shape of the output tensor if necessary
  TensorSetShape((Tensor*)y, 2, batch_size, out_dims);

  // Perform the fully-connected layer
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < out_dims; ++i) {
      float val = 0;

      for (int j = 0; j < in_dims; ++j) {
        val += TensorAt2d(x, b, j) * TensorAt2d(weight, i, j);
      }

      if (bias != NULL)
        TensorAt2d(y, b, i) = val + TensorAt1d(bias, i);
      else
        TensorAt2d(y, b, i) = val;
    }
  }
}

// Backward operation for the fully-connected layer
// `dy` should be of size (B, Dout)
// `x` should be of size (B, Din)
// The returned tensor `dx` is of size (B, Din)
// The returned tensor `dweight` is of size (Dout, Din)
// The returned tensor `dbias` is of size (Dout)
// `weight` should be of size (Dout, Din)
// `bias` should be of size (Dout)
// `bias` may be `NULL`
void LinearBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    FloatTensor* dx,
                    FloatTensor* dweight,
                    FloatTensor* dbias,
                    const FloatTensor* weight,
                    const FloatTensor* bias)
{
  // The input and output tensors should not be NULL except `dbias` and `bias`
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(dx);
  CheckTensor(dweight);
  CheckTensor(weight);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 2);
  CheckTensorDims(x, 2);
  CheckTensorDims(weight, 2);
  CheckTensorDims(bias, 1);

  // Check the consistency of the tensor shapes
  // Check the batch size
  Assert(dy->base_.shape_[0] == x->base_.shape_[0],
         "The batch size is not consistent: (`dy`: %d, `x`: %d)",
         dy->base_.shape_[0], x->base_.shape_[0]);

  // Check the number of input dimensions
  Assert(x->base_.shape_[1] == weight->base_.shape_[1],
         "The number of input dimensions is not consistent: "
         "(`x`: %d, `weight`: %d)",
         x->base_.shape_[1], weight->base_.shape_[1]);

  // Check the number of output dimensions
  Assert(dy->base_.shape_[1] == weight->base_.shape_[0],
         "The number of output dimensions is not consistent: "
         "(`dy`: %d, `weight`: %d)",
         dy->base_.shape_[1], weight->base_.shape_[0]);

  Assert(bias == NULL || weight->base_.shape_[0] == bias->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`weight`: %d, `bias`: %d)",
         weight->base_.shape_[0], bias->base_.shape_[0]);

  const int batch_size = dy->base_.shape_[0];
  const int out_dims = dy->base_.shape_[1];
  const int in_dims = x->base_.shape_[1];

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)dx, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)dweight, (const Tensor*)weight);

  if (bias != NULL)
    TensorSetShapeLike((Tensor*)dbias, (const Tensor*)bias);

  // Perform the backpropagation for the fully-connected layer
  // Compute the gradient for the weight
  for (int i = 0; i < out_dims; ++i) {
    for (int j = 0; j < in_dims; ++j) {
      float val = 0;

      for (int b = 0; b < batch_size; ++b) {
        val += TensorAt2d(x, b, j) * TensorAt2d(dy, b, i);
      }

      TensorAt2d(dweight, i, j) = val;
    }
  }

  // Compute the gradient for the bias
  for (int i = 0; i < out_dims; ++i) {
    float val = 0;

    for (int b = 0; b < batch_size; ++b) {
      val += TensorAt2d(dy, b, i);
    }

    TensorAt1d(dbias, i) = val;
  }

  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    for (int j = 0; j < in_dims; ++j) {
      float val = 0;

      for (int i = 0; i < out_dims; ++i) {
        val += TensorAt2d(weight, i, j) * TensorAt2d(dy, b, i);
      }

      TensorAt2d(dx, b, j) = val;
    }
  }
}
