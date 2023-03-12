
// linear.c

#include "my_nn_lib/linear.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

// Initialize the parameters for the fully-connected layer
void LinearParamsInitialize(LinearParams* params,
                            const int in_dims,
                            const int out_dims)
{
  Assert(params != NULL, "`params` should not be NULL");
  Assert(in_dims > 0, "`in_dims` should be greater than 0");
  Assert(out_dims > 0, "`out_dims` should be greater than 0");

  params->weight_ = (FloatTensor*)TensorEmpty2d(
    TENSOR_TYPE_FLOAT, out_dims, in_dims);
  params->bias_ = (FloatTensor*)TensorEmpty1d(
    TENSOR_TYPE_FLOAT, out_dims);
}

// Free the parameters for the fully-connected layer
void LinearParamsFree(LinearParams* params)
{
  Assert(params != NULL, "`params` should not be NULL");

  TensorFree((Tensor**)&params->weight_);
  TensorFree((Tensor**)&params->bias_);
}

// Initialize the outputs for the fully-connected layer
void LinearOutputsInitialize(LinearOutputs* outputs,
                             const bool inference_only)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  outputs->y_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  if (!inference_only)
    outputs->dx_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  else
    outputs->dx_ = NULL;
}

// Free the outputs for the fully-connected layer
void LinearOutputsFree(LinearOutputs* outputs)
{
  Assert(outputs != NULL, "`outputs` should not be NULL");

  TensorFree((Tensor**)&outputs->y_);
  TensorFree((Tensor**)&outputs->dx_);
}

// Forward operation for the fully-connected layer
// `x` should be of size (B, Din)
// The returned tensor `outputs->y_` is of size (B, Dout)
// `params->weight_` should be of size (Dout, Din)
// `params->bias_` should be of size (Dout)
// `params->bias_` may be `NULL`
void LinearForward(const FloatTensor* x,
                   LinearOutputs* outputs,
                   const LinearParams* params)
{
  // The input and output tensors should not be NULL except `bias`
  CheckTensor(x);
  CheckTensor(outputs->y_);
  CheckTensor(params->weight_);

  // Check the dimensions of the input tensors
  CheckTensorDims(x, 2);
  CheckTensorDims(params->weight_, 2);
  CheckTensorDims(params->bias_, 1);

  // Check the consistency of the tensor shapes
  // Check the number of input dimensions
  Assert(x->base_.shape_[1] == params->weight_->base_.shape_[1],
         "The number of input dimensions is not consistent: "
         "(`x`: %d, `params->weight_`: %d)",
         x->base_.shape_[1], params->weight_->base_.shape_[1]);

  // Check the number of output dimensions
  Assert(params->bias_ == NULL ||
         params->weight_->base_.shape_[0] == params->bias_->base_.shape_[0],
         "The number of output dimensions is not consistent: "
         "(`params->weight_`: %d, `params->bias_`: %d)",
         params->weight_->base_.shape_[0], params->bias_->base_.shape_[0]);

  const int batch_size = x->base_.shape_[0];
  const int in_dims = x->base_.shape_[1];
  const int out_dims = params->weight_->base_.shape_[0];

  // Set the shape of the output tensor if necessary
  TensorSetShape((Tensor*)outputs->y_, 2, batch_size, out_dims);

  // Perform the fully-connected layer
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < out_dims; ++i) {
      float val = 0;

      for (int j = 0; j < in_dims; ++j) {
        val += TensorAt2d(x, b, j) * TensorAt2d(params->weight_, i, j);
      }

      if (params->bias_ != NULL)
        TensorAt2d(outputs->y_, b, i) = val + TensorAt1d(params->bias_, i);
      else
        TensorAt2d(outputs->y_, b, i) = val;
    }
  }
}

// Backward operation for the fully-connected layer
// `dy` should be of size (B, Dout)
// `x` should be of size (B, Din)
// The returned tensor `outputs->dx_` is of size (B, Din)
// The returned tensor `dparams->weight_` is of size (Dout, Din)
// The returned tensor `dparams->bias_` is of size (Dout)
// `params->weight_` should be of size (Dout, Din)
// `params->bias_` should be of size (Dout)
// `params->bias_` may be `NULL`
void LinearBackward(const FloatTensor* dy,
                    const FloatTensor* x,
                    LinearOutputs* outputs,
                    LinearParams* dparams,
                    const LinearParams* params)
{
  // The input and output tensors should not be NULL except `dbias` and `bias`
  CheckTensor(dy);
  CheckTensor(x);
  CheckTensor(outputs->dx_);
  CheckTensor(dparams->weight_);
  CheckTensor(params->weight_);

  // Check the dimensions of the input tensors
  CheckTensorDims(dy, 2);
  CheckTensorDims(x, 2);
  CheckTensorDims(params->weight_, 2);
  CheckTensorDims(params->bias_, 1);

  // Check the consistency of the tensor shapes
  // Check the batch size
  Assert(dy->base_.shape_[0] == x->base_.shape_[0],
         "The batch size is not consistent: (`dy`: %d, `x`: %d)",
         dy->base_.shape_[0], x->base_.shape_[0]);

  // Check the number of input dimensions
  Assert(x->base_.shape_[1] == params->weight_->base_.shape_[1],
         "The number of input dimensions is not consistent: "
         "(`x`: %d, `params->weight_`: %d)",
         x->base_.shape_[1], params->weight_->base_.shape_[1]);

  // Check the number of output dimensions
  Assert(dy->base_.shape_[1] == params->weight_->base_.shape_[0],
         "The number of output dimensions is not consistent: "
         "(`dy`: %d, `params->weight_`: %d)",
         dy->base_.shape_[1], params->weight_->base_.shape_[0]);

  Assert(params->bias_ == NULL ||
         params->weight_->base_.shape_[0] == params->bias_->base_.shape_[0],
         "The number of output channels is not consistent: "
         "(`params->weight_`: %d, `params->bias_`: %d)",
         params->weight_->base_.shape_[0], params->bias_->base_.shape_[0]);

  const int batch_size = dy->base_.shape_[0];
  const int out_dims = dy->base_.shape_[1];
  const int in_dims = x->base_.shape_[1];

  // Set the shape of the output tensor if necessary
  TensorSetShapeLike((Tensor*)outputs->dx_, (const Tensor*)x);
  TensorSetShapeLike((Tensor*)dparams->weight_, (const Tensor*)params->weight_);

  if (params->bias_ != NULL)
    TensorSetShapeLike((Tensor*)dparams->bias_, (const Tensor*)params->bias_);

  // Perform the backpropagation for the fully-connected layer
  // Compute the gradient for the weight
  for (int i = 0; i < out_dims; ++i) {
    for (int j = 0; j < in_dims; ++j) {
      float val = 0;

      for (int b = 0; b < batch_size; ++b) {
        val += TensorAt2d(x, b, j) * TensorAt2d(dy, b, i);
      }

      TensorAt2d(dparams->weight_, i, j) = val;
    }
  }

  // Compute the gradient for the bias
  for (int i = 0; i < out_dims; ++i) {
    float val = 0;

    for (int b = 0; b < batch_size; ++b) {
      val += TensorAt2d(dy, b, i);
    }

    TensorAt1d(dparams->bias_, i) = val;
  }

  // Compute the gradient for the input
  for (int b = 0; b < batch_size; ++b) {
    for (int j = 0; j < in_dims; ++j) {
      float val = 0;

      for (int i = 0; i < out_dims; ++i) {
        val += TensorAt2d(params->weight_, i, j) * TensorAt2d(dy, b, i);
      }

      TensorAt2d(outputs->dx_, b, j) = val;
    }
  }
}
