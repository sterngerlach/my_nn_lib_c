
// tensor.h

#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

// 2D index
typedef struct
{
  int idx_[2];
} Index2d;

// Tensor data type
enum TensorDataType
{
  TENSOR_TYPE_UNKNOWN = 0,
  TENSOR_TYPE_INT,
  TENSOR_TYPE_FLOAT,
  TENSOR_TYPE_U8,
  TENSOR_TYPE_INDEX2D,
};

// Base tensor type
// The maximum number of dimensions is 4
typedef struct
{
  // Data type
  enum TensorDataType dtype_;
  // Shape of the tensor
  int                 shape_[4];
  // Number of dimensions
  int                 ndim_;
  // Number of elements
  int                 numel_;
} Tensor;

// Float tensor
typedef struct
{
  // Tensor information
  Tensor base_;
  // Tensor data
  float* data_;
} FloatTensor;

// Int tensor
typedef struct
{
  // Tensor information
  Tensor base_;
  // Tensor data
  int*   data_;
} IntTensor;

// 8-bit unsigned integer tensor
typedef struct
{
  // Tensor information
  Tensor   base_;
  // Tensor data
  uint8_t* data_;
} U8Tensor;

// 2D index tensor
typedef struct
{
  // Tensor information
  Tensor   base_;
  // Tensor data
  Index2d* data_;
} Index2dTensor;

// Allocate a new empty tensor
Tensor* TensorAllocate(const enum TensorDataType dtype);

// Get a pointer to the data buffer
// `tensor->dtype_` should be a valid data type
void* TensorGetData(Tensor* tensor);

// Create a new ND tensor
// Tensor data is not initialized
Tensor* TensorEmptyNd(const enum TensorDataType dtype,
                      const int ndim, ...);

// Create a new ND tensor
// Tensor data is filled with zeros
Tensor* TensorZerosNd(const enum TensorDataType dtype,
                      const int ndim, ...);

// Create a new ND tensor with the same shape
// Tensor data is not initialized
Tensor* TensorEmptyLike(const Tensor* src);

// Create a new ND tensor with the same shape
// Tensor data is filled with zeros
Tensor* TensorZerosLike(const Tensor* src);

// Set the shape of the tensor
// The expanded part is not initialized
void TensorSetShape(Tensor* tensor,
                    const int ndim, ...);

// Set the shape of the tensor
// The expanded part is not initialized
void TensorSetShapeLike(Tensor* tensor,
                        const Tensor* src);

// Free a tensor
void TensorFree(Tensor** tensor);

// Compare the tensor shape
bool TensorIsShapeEqual(const Tensor* lhs,
                        const Tensor* rhs);

// Compare the tensor shape from shape array
bool TensorIsShapeEqualA(const Tensor* tensor,
                         const int ndim,
                         const int* shape);

// Compare the tensor shape from variadic arguments
bool TensorIsShapeEqualV(const Tensor* tensor,
                         const int ndim,
                         va_list args);

// Compare the tensor shape
bool TensorIsShapeEqualNd(const Tensor* tensor,
                          const int ndim, ...);

// TODO: Create a tensor from buffer
// TODO: Load a tensor from NumPy format file

// Create a new 1D tensor
#define TensorEmpty1d(dtype, d0) \
  TensorEmptyNd((dtype), 1, (d0))
// Create a new 2D tensor
#define TensorEmpty2d(dtype, d0, d1) \
  TensorEmptyNd((dtype), 2, (d0), (d1))
// Create a new 3D tensor
#define TensorEmpty3d(dtype, d0, d1, d2) \
  TensorEmptyNd((dtype), 3, (d0), (d1), (d2))
// Create a new 4D tensor
#define TensorEmpty4d(dtype, d0, d1, d2, d3) \
  TensorEmptyNd((dtype), 4, (d0), (d1), (d2), (d3))

// Create a new 1D tensor with zeros
#define TensorZeros1d(dtype, d0) \
  TensorZerosNd((dtype), 1, (d0))
// Create a new 2D tensor with zeros
#define TensorZeros2d(dtype, d0, d1) \
  TensorZerosNd((dtype), 2, (d0), (d1))
// Create a new 3D tensor with zeros
#define TensorZeros3d(dtype, d0, d1, d2) \
  TensorZerosNd((dtype), 3, (d0), (d1), (d2))
// Create a new 4D tensor with zeros
#define TensorZeros4d(dtype, d0, d1, d2, d3) \
  TensorZerosNd((dtype), 4, (d0), (d1), (d2), (d3))

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
