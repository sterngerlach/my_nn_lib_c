
// tensor.c

#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// Get a number of bytes for each element from a given data type
static size_t GetDataSize(const enum TensorDataType dtype)
{
  switch (dtype) {
    case TENSOR_TYPE_INT:
      return sizeof(int);
    case TENSOR_TYPE_FLOAT:
      return sizeof(float);
    case TENSOR_TYPE_U8:
      return sizeof(uint8_t);
    case TENSOR_TYPE_INDEX2D:
      return sizeof(Index2d);
    default:
      Assert(0, "`dtype` should be a valid type");
      break;
  }
}

// Get a tensor shape and a number of elements from variadic arguments
static int GetTensorShapeNd(const int ndim,
                            int* shape,
                            va_list args)
{
  // Get the tensor shape and the number of elements
  int numel = 1;

  for (int i = 0; i < ndim; ++i) {
    shape[i] = va_arg(args, int);
    numel *= shape[i];

    Assert(shape[i] > 0,
           "Tensor size should be greater than 0, "
           "but %d given", shape[i]);
  }

  return numel;
}

// Allocate a buffer for the tensor header
static Tensor* TensorAllocateHeader(const enum TensorDataType dtype)
{
  Tensor* tensor = NULL;

  switch (dtype) {
    case TENSOR_TYPE_INT:
      tensor = calloc(1, sizeof(IntTensor));
      break;
    case TENSOR_TYPE_FLOAT:
      tensor = calloc(1, sizeof(FloatTensor));
      break;
    case TENSOR_TYPE_U8:
      tensor = calloc(1, sizeof(U8Tensor));
      break;
    case TENSOR_TYPE_INDEX2D:
      tensor = calloc(1, sizeof(Index2dTensor));
      break;
    default:
      Assert(0, "`dtype` should be a valid type");
      break;
  }

  return tensor;
}

// Allocate a buffer for the tensor elements
static void* TensorAllocateBuffer(const enum TensorDataType dtype,
                                  const int numel,
                                  const bool fill_with_zeros)
{
  // Get a number of bytes for each element
  const size_t dsize = GetDataSize(dtype);

  void* data = NULL;
  if (fill_with_zeros)
    data = calloc(1, dsize * numel);
  else
    data = malloc(dsize * numel);

  return data;
}

// Get a pointer to the data buffer
// `tensor->dtype_` should be a valid data type
void* TensorGetData(Tensor* tensor)
{
  if (tensor == NULL)
    return NULL;

  switch (tensor->dtype_) {
    case TENSOR_TYPE_INT:
      return ((IntTensor*)tensor)->data_;
    case TENSOR_TYPE_FLOAT:
      return ((FloatTensor*)tensor)->data_;
    case TENSOR_TYPE_U8:
      return ((U8Tensor*)tensor)->data_;
    case TENSOR_TYPE_INDEX2D:
      return ((Index2dTensor*)tensor)->data_;
    default:
      Assert(0, "`tensor->dtype_` should be a valid type");
      break;
  }

  return NULL;
}

// Set a pointer to the data buffer
// `tensor->dtype_` should be a valid data type
static void TensorSetData(Tensor* tensor,
                          void* data)
{
  CheckTensor(tensor);

  switch (tensor->dtype_) {
    case TENSOR_TYPE_INT:
      ((IntTensor*)tensor)->data_ = data;
      break;
    case TENSOR_TYPE_FLOAT:
      ((FloatTensor*)tensor)->data_ = data;
      break;
    case TENSOR_TYPE_U8:
      ((U8Tensor*)tensor)->data_ = data;
      break;
    case TENSOR_TYPE_INDEX2D:
      ((Index2dTensor*)tensor)->data_ = data;
      break;
    default:
      Assert(0, "`tensor->dtype_` should be a valid type");
      break;
  }
}

// Fill an int tensor with the given value
static void IntTensorFill(IntTensor* tensor,
                          const int val)
{
  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i)
    TensorAt1d(tensor, i) = val;
}

// Fill a float tensor with the given value
static void FloatTensorFill(FloatTensor* tensor,
                            const float val)
{
  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i)
    TensorAt1d(tensor, i) = val;
}

// Fill a 8-bit unsigned integer tensor with the given value
static void U8TensorFill(U8Tensor* tensor,
                         const uint8_t val)
{
  // We treat the input tensor as a 1D tensor
  for (int i = 0; i < tensor->base_.numel_; ++i)
    TensorAt1d(tensor, i) = val;
}

// Create a new ND tensor
static Tensor* TensorEmptyNdCore(const enum TensorDataType dtype,
                                 const bool fill_with_zeros,
                                 const int ndim,
                                 va_list args)
{
  Assert(ndim >= 1 && ndim <= 4,
         "Number of dimensions should be within 1 to 4, ",
         "but %d given", ndim);

  // Get a tensor shape and a number of elements
  int numel;
  int shape[4];
  numel = GetTensorShapeNd(ndim, shape, args);
  Assert(numel > 0, "The number of elements should be greater than 0");

  // Allocate a buffer for the tensor header
  Tensor* tensor = TensorAllocateHeader(dtype);
  Assert(tensor != NULL, "Failed to allocate a new tensor");

  // Allocate a buffer for tensor elements
  void* data = TensorAllocateBuffer(dtype, numel, fill_with_zeros);
  Assert(data != NULL, "Failed to allocate a buffer for tensor elements");

  // Set the members
  tensor->dtype_ = dtype;
  tensor->ndim_ = ndim;
  tensor->numel_ = numel;
  memcpy(tensor->shape_, shape, sizeof(int) * ndim);

  // Call `TensorSetData` after `tensor->dtype_` is set
  TensorSetData(tensor, data);

  return tensor;
}

// Create a new ND tensor
// Tensor data is filled with ones
static Tensor* TensorOnesNdCore(const enum TensorDataType dtype,
                                const int ndim,
                                va_list args)
{
  // Create an empty tensor
  Tensor* tensor = TensorEmptyNdCore(dtype, false, ndim, args);
  Assert(tensor != NULL, "Failed to create a new empty tensor");

  // Fill with ones
  TensorFillWithOnes(tensor);

  return tensor;
}

// Create a new ND tensor with the same shape
static Tensor* TensorEmptyLikeCore(const bool fill_with_zeros,
                                   const Tensor* src)
{
  CheckTensor(src);

  Assert(src->ndim_ >= 1 && src->ndim_ <= 4,
         "Number of dimensions should be within 1 to 4, "
         "but %d given", src->ndim_);

  // Allocate a buffer for tensor header
  Tensor* tensor = TensorAllocateHeader(src->dtype_);
  Assert(tensor != NULL, "Failed to allocate a new tensor");

  // Allocate a buffer for tensor elements
  void* data = TensorAllocateBuffer(src->dtype_, src->numel_, fill_with_zeros);
  Assert(data != NULL, "Failed to allocate a buffer for tensor elements");

  // Set the members
  tensor->dtype_ = src->dtype_;
  tensor->ndim_ = src->ndim_;
  tensor->numel_ = src->numel_;
  memcpy(tensor->shape_, src->shape_, sizeof(int) * src->ndim_);

  // Call `TensorSetData` after `tensor->dtype_` is set
  TensorSetData(tensor, data);

  return tensor;
}

// Create a new ND tensor with the same shape
// Tensor data is filled with ones
static Tensor* TensorOnesLikeCore(const Tensor* src)
{
  // Create an empty tensor
  Tensor* tensor = TensorEmptyLikeCore(false, src);
  Assert(tensor != NULL, "Failed to create a new empty tensor");

  // Fill with ones
  TensorFillWithOnes(tensor);

  return tensor;
}

// Set the shape of the tensor (expand or shrink the tensor)
static void TensorSetShapeCore(Tensor* tensor,
                               const int ndim,
                               va_list args)
{
  CheckTensor(tensor);

  Assert(ndim >= 1 && ndim <= 4,
         "Number of dimensions should be within 1 to 4, ",
         "but %d given", ndim);

  // Get a requested shape and a number of elements
  int numel;
  int shape[4];
  numel = GetTensorShapeNd(ndim, shape, args);
  Assert(numel > 0, "The number of elements should be greater than 0");

  // We do not need to reallocate a data buffer if the current shape is
  // the same as the request shape
  if (TensorIsShapeEqualA(tensor, ndim, shape))
    return;

  // Get a pointer to the data buffer (may be NULL)
  void* data = TensorGetData(tensor);

  // Reallocate the data buffer
  // `data` is freed by `realloc()`
  const size_t dsize = GetDataSize(tensor->dtype_);
  void* data_new = realloc(data, dsize * numel);
  Assert(data_new != NULL, "Failed to reallocate a buffer for tensor elements");

  // Set the members
  tensor->ndim_ = ndim;
  tensor->numel_ = numel;
  memcpy(tensor->shape_, shape, sizeof(int) * ndim);

  // Call `TensorSetData` after `tensor->dtype_` is set
  TensorSetData(tensor, data_new);
}

// Set the shape of the tensor (expand or shrink the tensor)
static void TensorSetShapeLikeCore(Tensor* tensor,
                                   const Tensor* src)
{
  CheckTensor(tensor);
  CheckTensor(src);

  Assert(src->ndim_ >= 1 && src->ndim_ <= 4,
         "Number of dimensions should be within 1 to 4, ",
         "but %d given", src->ndim_);

  // We do not need to reallocate a data buffer if the current shape is
  // the same as the requested shape
  if (TensorIsShapeEqualA(tensor, src->ndim_, src->shape_))
    return;

  // Get a pointer to the data buffer (may be NULL)
  void* data = TensorGetData(tensor);

  // Reallocate the data buffer
  // `data` is freed by `realloc()`
  const size_t dsize = GetDataSize(tensor->dtype_);
  void* data_new = realloc(data, dsize * src->numel_);
  Assert(data_new != NULL, "Failed to reallocate a buffer for tensor elements");

  // Set the members
  tensor->ndim_ = src->ndim_;
  tensor->numel_ = src->numel_;
  memcpy(tensor->shape_, src->shape_, sizeof(int) * src->ndim_);

  // Call `TensorSetData` after `tensor->dtype_` is set
  TensorSetData(tensor, data_new);
}

// Allocate a new empty tensor
Tensor* TensorAllocate(const enum TensorDataType dtype)
{
  // Allocate a buffer for the tensor header
  Tensor* tensor = TensorAllocateHeader(dtype);

  // Set the members
  tensor->dtype_ = dtype;
  tensor->ndim_ = 0;
  tensor->numel_ = 0;
  memset(tensor->shape_, 0, sizeof(tensor->shape_));

  // Call `TensorSetData` after `tensor->dtype_` is set
  TensorSetData(tensor, NULL);

  return tensor;
}

// Create a new ND tensor
// Tensor data is not initialized
Tensor* TensorEmptyNd(const enum TensorDataType dtype,
                      const int ndim, ...)
{
  Tensor* tensor = NULL;

  va_list args;
  va_start(args, ndim);
  tensor = TensorEmptyNdCore(dtype, false, ndim, args);
  va_end(args);

  return tensor;
}

// Create a new ND tensor
// Tensor data is filled with zeros
Tensor* TensorZerosNd(const enum TensorDataType dtype,
                      const int ndim, ...)
{
  Tensor* tensor = NULL;

  va_list args;
  va_start(args, ndim);
  tensor = TensorEmptyNdCore(dtype, true, ndim, args);
  va_end(args);

  return tensor;
}

// Create a new ND tensor
// Tensor data is filled with ones
Tensor* TensorOnesNd(const enum TensorDataType dtype,
                     const int ndim, ...)
{
  Tensor* tensor = NULL;

  va_list args;
  va_start(args, ndim);
  tensor = TensorOnesNdCore(dtype, ndim, args);
  va_end(args);

  return tensor;
}

// Create a new ND tensor with the same shape
// Tensor data is not initialized
Tensor* TensorEmptyLike(const Tensor* src)
{
  return TensorEmptyLikeCore(false, src);
}

// Create a new ND tensor with the same shape
// Tensor data is filled with zeros
Tensor* TensorZerosLike(const Tensor* src)
{
  return TensorEmptyLikeCore(true, src);
}

// Create a new ND tensor with the same shape
// Tensor data is filled with ones
Tensor* TensorOnesLike(const Tensor* src)
{
  return TensorOnesLikeCore(src);
}

// Set the shape of the tensor
// The expanded part is not initialized
void TensorSetShape(Tensor* tensor,
                    const int ndim, ...)
{
  va_list args;
  va_start(args, ndim);
  TensorSetShapeCore(tensor, ndim, args);
  va_end(args);
}

// Set the shape of the tensor
// The expanded part is not initialized
void TensorSetShapeLike(Tensor* tensor,
                        const Tensor* src)
{
  TensorSetShapeLikeCore(tensor, src);
}

// Fill the tensor with zeros (works on scalar types only)
void TensorFillWithZeros(Tensor* tensor)
{
  CheckTensor(tensor);

  switch (tensor->dtype_) {
    case TENSOR_TYPE_INT:
      IntTensorFill((IntTensor*)tensor, 0);
      break;
    case TENSOR_TYPE_FLOAT:
      FloatTensorFill((FloatTensor*)tensor, 0.0f);
      break;
    case TENSOR_TYPE_U8:
      U8TensorFill((U8Tensor*)tensor, 0);
      break;
    default:
      Assert(0, "`tensor->dtype_` should be a scalar type");
      break;
  }
}

// Fill the tensor with ones (works on scalar types only)
void TensorFillWithOnes(Tensor* tensor)
{
  CheckTensor(tensor);

  switch (tensor->dtype_) {
    case TENSOR_TYPE_INT:
      IntTensorFill((IntTensor*)tensor, 1);
      break;
    case TENSOR_TYPE_FLOAT:
      FloatTensorFill((FloatTensor*)tensor, 1.0f);
      break;
    case TENSOR_TYPE_U8:
      U8TensorFill((U8Tensor*)tensor, 1);
      break;
    default:
      Assert(0, "`tensor->dtype_` should be a scalar type");
      break;
  }
}

// Free a tensor
void TensorFree(Tensor** tensor)
{
  if (*tensor == NULL)
    return;

  void* data = TensorGetData(*tensor);
  free(data);
  data = NULL;

  free(*tensor);
  *tensor = NULL;
}

// Compare the tensor shape
bool TensorIsShapeEqual(const Tensor* lhs,
                        const Tensor* rhs)
{
  if (lhs == NULL && rhs == NULL)
    return true;
  if (lhs == NULL || rhs == NULL)
    return false;
  if (lhs->ndim_ != rhs->ndim_)
    return false;
  if (lhs->numel_ != rhs->numel_)
    return false;

  for (int i = 0; i < lhs->ndim_; ++i)
    if (lhs->shape_[i] != rhs->shape_[i])
      return false;

  return true;
}

// Compare the tensor shape from shape array
bool TensorIsShapeEqualA(const Tensor* tensor,
                         const int ndim,
                         const int* shape)
{
  if (tensor == NULL)
    return false;
  if (tensor->ndim_ != ndim)
    return false;

  for (int i = 0; i < ndim; ++i)
    if (tensor->shape_[i] != shape[i])
      return false;

  return true;
}

// Compare the tensor shape from variadic arguments
bool TensorIsShapeEqualV(const Tensor* tensor,
                         const int ndim,
                         va_list args)
{
  if (tensor == NULL)
    return false;
  if (tensor->ndim_ != ndim)
    return false;

  for (int i = 0; i < ndim; ++i) {
    const int dims = va_arg(args, int);
    if (tensor->shape_[i] != dims)
      return false;
  }

  return true;
}

// Compare the tensor shape
bool TensorIsShapeEqualNd(const Tensor* tensor,
                          const int ndim, ...)
{
  bool result = false;

  va_list args;
  va_start(args, ndim);
  result = TensorIsShapeEqualV(tensor, ndim, args);
  va_end(args);

  return result;
}
