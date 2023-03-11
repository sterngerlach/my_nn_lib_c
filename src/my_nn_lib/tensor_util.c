
// tensor_util.c

#include "my_nn_lib/tensor_util.h"

#include <stdbool.h>
#include <stdlib.h>

// Convert a tensor shape to the string (e.g., `(256, 512, 6, 6)`)
// The returned string should be freed by the user
// `NULL` is returned in case of failure
char* TensorShapeToString(const Tensor* tensor)
{
  if (tensor == NULL) {
    LogError("`tensor` is NULL");
    return NULL;
  }

  if (tensor->ndim_ < 1 || tensor->ndim_ > 4) {
    LogError("`tensor` should be a 1-4D tensor, but is %dD", tensor->ndim_);
    return NULL;
  }

  switch (tensor->ndim_) {
    case 1:
      return AllocateFormatString("(%d)",
        tensor->shape_[0]);
    case 2:
      return AllocateFormatString("(%d, %d)",
        tensor->shape_[0], tensor->shape_[1]);
    case 3:
      return AllocateFormatString("(%d, %d, %d)",
        tensor->shape_[0], tensor->shape_[1],
        tensor->shape_[2]);
    case 4:
      return AllocateFormatString("(%d, %d, %d, %d)",
        tensor->shape_[0], tensor->shape_[1],
        tensor->shape_[2], tensor->shape_[3]);
  }

  return NULL;
}

// Check the tensor shape
// Specify -1 if you do not care about that dimension
void CheckTensorShapeCore(const Tensor* tensor,
                          const char* name,
                          const int d0,
                          const int d1,
                          const int d2,
                          const int d3)
{
  if (tensor == NULL) {
    LogError("`%s` is NULL", name);
    exit(EXIT_FAILURE);
  }

  const bool is_d0_equal = tensor->ndim_ <= 0 ||
    d0 == -1 || tensor->shape_[0] == d0;
  const bool is_d1_equal = tensor->ndim_ <= 1 ||
    d1 == -1 || tensor->shape_[1] == d1;
  const bool is_d2_equal = tensor->ndim_ <= 2 ||
    d2 == -1 || tensor->shape_[2] == d2;
  const bool is_d3_equal = tensor->ndim_ <= 3 ||
    d3 == -1 || tensor->shape_[3] == d3;
  const bool all_equal = is_d0_equal && is_d1_equal
    && is_d2_equal && is_d3_equal;

  if (all_equal)
    return;

  char* shape_str = TensorShapeToString(tensor);
  char* expected_str = NULL;

  switch (tensor->ndim_) {
    case 1:
      expected_str = AllocateFormatString("(%d)", d0);
      break;
    case 2:
      expected_str = AllocateFormatString("(%d, %d)", d0, d1);
      break;
    case 3:
      expected_str = AllocateFormatString("(%d, %d, %d)", d0, d1, d2);
      break;
    case 4:
      expected_str = AllocateFormatString("(%d, %d, %d, %d)", d0, d1, d2, d3);
      break;
  }

  LogError("`%s` should be of size `%s`, but is `%s`",
           name, shape_str, expected_str);

  free(shape_str);
  free(expected_str);
  shape_str = NULL;
  expected_str = NULL;

  exit(EXIT_FAILURE);
}

// Initialize the tensor from the uniform distribution
void FloatTensorRandomUniform(FloatTensor* tensor,
                              const float dist_min,
                              const float dist_max)
{
  // The input tensor should not be NULL
  CheckTensor(tensor);

  // We treat the tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i) {
    const float r = GenerateUniformFloat();
    TensorAt1d(tensor, i) = dist_min + (dist_max - dist_min) * r;
  }
}

// Initialize the tensor from the normal distribution
void FloatTensorRandomNormal(FloatTensor* tensor,
                             const float mean,
                             const float std,
                             const float clip_min,
                             const float clip_max)
{
  // The input tensor should not be NULL
  CheckTensor(tensor);

  // We treat the tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i) {
    const float r = GenerateNormalFloat() * std + mean;
    TensorAt1d(tensor, i) = Clamp(r, clip_min, clip_max);
  }
}
