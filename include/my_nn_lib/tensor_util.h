
// tensor_util.h

#ifndef TENSOR_UTIL_H
#define TENSOR_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/util.h"

// Get a flat index (2D)
#define FlatIndex2d(i, j, stride) \
  ((i) * (stride) + (j))

// Get a flat index (3D)
#define FlatIndex3d(i, j, k, stride0, stride1) \
  (FlatIndex2d(FlatIndex2d((i), (j), (stride0)), (k), (stride1)))

// Get a flat index (4D)
#define FlatIndex4d(i, j, k, l, stride0, stride1, stride2) \
  (FlatIndex2d(FlatIndex3d((i), (j), (k), (stride0), (stride1)), \
    (l), (stride2)))

// Get an element of a 1D tensor
// This does not check the inputs
#define TensorAt1d(tensor, i) \
  ((tensor)->data_[(i)])

// Get an element of a 2D tensor
// This does not check the inputs
#define TensorAt2d(tensor, i, j) \
  ((tensor)->data_[FlatIndex2d((i), (j), (tensor)->base_.shape_[1])])

// Get an element of a 3D tensor
// This does not check the inputs
#define TensorAt3d(tensor, i, j, k) \
  ((tensor)->data_[FlatIndex3d((i), (j), (k), \
    (tensor)->base_.shape_[1], (tensor)->base_.shape_[2])])

// Get an element of a 4D tensor
// This does not check the inputs
#define TensorAt4d(tensor, i, j, k, l) \
  ((tensor)->data_[FlatIndex4d((i), (j), (k), (l), \
    (tensor)->base_.shape_[1], (tensor)->base_.shape_[2], \
    (tensor)->base_.shape_[3])])

// Convert a tensor shape to the string (e.g., `(256, 512, 6, 6)`)
// The returned string should be freed by the user
// `NULL` is returned in case of failure
char* TensorShapeToString(const Tensor* tensor);

// Check that the tensor is allocated (i.e., not `NULL`)
// `tensor` should be a pointer
#define CheckTensor(tensor) \
  Assert((tensor) != NULL, "`%s` should not be NULL", #tensor)

// Check the number of dimensions of a tensor
// `tensor` should be a pointer
#define CheckTensorDims(tensor, dims) \
  Assert(((const Tensor*)(tensor)) == NULL \
    || ((const Tensor*)(tensor))->ndim_ == (dims), \
    "`%s` should be a %dD tensor, but %dD tensor is given", \
    #tensor, (dims), ((const Tensor*)(tensor))->ndim_)

// Check the tensor shape
// Specify -1 if you do not care about that dimension
void CheckTensorShapeCore(const Tensor* tensor,
                          const char* name,
                          const int d0,
                          const int d1,
                          const int d2,
                          const int d3);

// Check the tensor shape of a 1D tensor
#define CheckTensorShape1d(tensor, d0) \
  CheckTensorShapeCore((tensor), #tensor, (d0), -1, -1, -1)

// Check the tensor shape of a 2D tensor
#define CheckTensorShape2d(tensor, d0, d1) \
  CheckTensorShapeCore((tensor), #tensor, (d0), (d1), -1, -1)

// Check the tensor shape of a 3D tensor
#define CheckTensorShape3d(tensor, d0, d1, d2) \
  CheckTensorShapeCore((tensor), #tensor, (d0), (d1), (d2), -1)

// Check the tensor shape of a 4D tensor
#define CheckTensorShape4d(tensor, d0, d1, d2, d3) \
  CheckTensorShapeCore((tensor), #tensor, (d0), (d1), (d2), (d3))

// Initialize the tensor from the uniform distribution
void FloatTensorRandomUniform(FloatTensor* tensor,
                              const float dist_min,
                              const float dist_max);

// Initialize the tensor from the normal distribution
void FloatTensorRandomNormal(FloatTensor* tensor,
                             const float mean,
                             const float std,
                             const float clip_min,
                             const float clip_max);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_UTIL_H
