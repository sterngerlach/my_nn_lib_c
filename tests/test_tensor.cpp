
// test_tensor.cpp

#include <gtest/gtest.h>

#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"

TEST(TestTensor, TensorAllocate)
{
  // Create an empty float tensor
  FloatTensor* tensor = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  EXPECT_EQ(tensor->base_.dtype_, TENSOR_TYPE_FLOAT);
  EXPECT_EQ(tensor->base_.ndim_, 0);
  EXPECT_EQ(tensor->base_.numel_, 0);
  EXPECT_EQ(tensor->base_.shape_[0], 0);

  // Free the empty tensor
  TensorFree((Tensor**)&tensor);
  EXPECT_EQ(tensor, nullptr);
}

TEST(TestTensor, TensorEmpty1d)
{
  // Create a 1D float tensor
  FloatTensor* tensor = (FloatTensor*)TensorEmpty1d(TENSOR_TYPE_FLOAT, 16);

  EXPECT_EQ(tensor->base_.dtype_, TENSOR_TYPE_FLOAT);
  EXPECT_EQ(tensor->base_.ndim_, 1);
  EXPECT_EQ(tensor->base_.numel_, 16);
  EXPECT_EQ(tensor->base_.shape_[0], 16);

  // Store values
  TensorAt1d(tensor, 0) = 13;
  TensorAt1d(tensor, 5) = 24;
  TensorAt1d(tensor, 10) = 35;

  EXPECT_FLOAT_EQ(tensor->data_[0], 13);
  EXPECT_FLOAT_EQ(tensor->data_[5], 24);
  EXPECT_FLOAT_EQ(tensor->data_[10], 35);

  // Free the 1D tensor
  TensorFree((Tensor**)&tensor);
  EXPECT_EQ(tensor, nullptr);
}

TEST(TestTensor, TensorEmpty2d)
{
  // Create a 2D float tensor
  FloatTensor* tensor = (FloatTensor*)TensorEmpty2d(
    TENSOR_TYPE_FLOAT, 13, 24);

  EXPECT_EQ(tensor->base_.dtype_, TENSOR_TYPE_FLOAT);
  EXPECT_EQ(tensor->base_.ndim_, 2);
  EXPECT_EQ(tensor->base_.numel_, 13 * 24);
  EXPECT_EQ(tensor->base_.shape_[0], 13);
  EXPECT_EQ(tensor->base_.shape_[1], 24);

  // Store values
  TensorAt2d(tensor, 3, 4) = 42;
  TensorAt2d(tensor, 6, 8) = 53;
  TensorAt2d(tensor, 9, 12) = 64;

  EXPECT_FLOAT_EQ(tensor->data_[3 * 24 + 4], 42);
  EXPECT_FLOAT_EQ(tensor->data_[6 * 24 + 8], 53);
  EXPECT_FLOAT_EQ(tensor->data_[9 * 24 + 12], 64);

  // Free the 2D tensor
  TensorFree((Tensor**)&tensor);
  EXPECT_EQ(tensor, nullptr);
}

TEST(TestTensor, TensorEmpty3d)
{
  // Create a 3D float tensor
  FloatTensor* tensor = (FloatTensor*)TensorEmpty3d(
    TENSOR_TYPE_FLOAT, 13, 24, 35);

  EXPECT_EQ(tensor->base_.dtype_, TENSOR_TYPE_FLOAT);
  EXPECT_EQ(tensor->base_.ndim_, 3);
  EXPECT_EQ(tensor->base_.numel_, 13 * 24 * 35);
  EXPECT_EQ(tensor->base_.shape_[0], 13);
  EXPECT_EQ(tensor->base_.shape_[1], 24);
  EXPECT_EQ(tensor->base_.shape_[2], 35);

  // Store values
  TensorAt3d(tensor, 3, 4, 5) = 42;
  TensorAt3d(tensor, 6, 7, 8) = 53;
  TensorAt3d(tensor, 9, 10, 11) = 64;

  EXPECT_FLOAT_EQ(tensor->data_[(3 * 24 + 4) * 35 + 5], 42);
  EXPECT_FLOAT_EQ(tensor->data_[(6 * 24 + 7) * 35 + 8], 53);
  EXPECT_FLOAT_EQ(tensor->data_[(9 * 24 + 10) * 35 + 11], 64);

  // Free the 3D tensor
  TensorFree((Tensor**)&tensor);
  EXPECT_EQ(tensor, nullptr);
}

TEST(TestTensor, TensorZeros3d)
{
  // Create a 3D float tensor
  FloatTensor* tensor = (FloatTensor*)TensorZeros3d(
    TENSOR_TYPE_FLOAT, 3, 4, 5);

  EXPECT_EQ(tensor->base_.dtype_, TENSOR_TYPE_FLOAT);
  EXPECT_EQ(tensor->base_.ndim_, 3);
  EXPECT_EQ(tensor->base_.numel_, 3 * 4 * 5);
  EXPECT_EQ(tensor->base_.shape_[0], 3);
  EXPECT_EQ(tensor->base_.shape_[1], 4);
  EXPECT_EQ(tensor->base_.shape_[2], 5);

  // Check that all elements are zero
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 5; ++k)
        EXPECT_FLOAT_EQ(TensorAt3d(tensor, i, j, k), 0.0f);

  // Free the 3D tensor
  TensorFree((Tensor**)&tensor);
  EXPECT_EQ(tensor, nullptr);
}

TEST(TestTensor, TensorEmptyLike)
{
  // Create a 3D float tensor
  FloatTensor* tensor0 = (FloatTensor*)TensorEmpty3d(
    TENSOR_TYPE_FLOAT, 3, 4, 5);

  // Create another 3D float tensor
  FloatTensor* tensor1 = (FloatTensor*)TensorEmptyLike((Tensor*)tensor0);

  EXPECT_EQ(tensor0->base_.dtype_, TENSOR_TYPE_FLOAT);
  EXPECT_EQ(tensor0->base_.ndim_, 3);
  EXPECT_EQ(tensor0->base_.numel_, 3 * 4 * 5);
  EXPECT_EQ(tensor0->base_.shape_[0], 3);
  EXPECT_EQ(tensor0->base_.shape_[1], 4);
  EXPECT_EQ(tensor0->base_.shape_[2], 5);

  // Free the 3D tensor
  TensorFree((Tensor**)&tensor0);
  TensorFree((Tensor**)&tensor1);
}

TEST(TestTensor, TensorSetShape)
{
  // Create an empty float tensor
  FloatTensor* tensor = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);

  // Set the shape of the tensor
  TensorSetShape((Tensor*)tensor, 3, 24, 35, 46);

  EXPECT_EQ(tensor->base_.ndim_, 3);
  EXPECT_EQ(tensor->base_.numel_, 24 * 35 * 46);
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)tensor, 3, 24, 35, 46));

  // Set the shape of the tensor
  TensorSetShape((Tensor*)tensor, 2, 13, 24);

  EXPECT_EQ(tensor->base_.ndim_, 2);
  EXPECT_EQ(tensor->base_.numel_, 13 * 24);
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)tensor, 2, 13, 24));

  TensorAt2d(tensor, 5, 5) = 123;
  TensorAt2d(tensor, 10, 10) = 456;

  EXPECT_FLOAT_EQ(TensorAt2d(tensor, 5, 5), 123);
  EXPECT_FLOAT_EQ(TensorAt2d(tensor, 10, 10), 456);

  // Set the shape of the tensor
  TensorSetShape((Tensor*)tensor, 4, 11, 22, 33, 44);

  EXPECT_EQ(tensor->base_.ndim_, 4);
  EXPECT_EQ(tensor->base_.numel_, 11 * 22 * 33 * 44);
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)tensor, 4, 11, 22, 33, 44));

  TensorAt4d(tensor, 1, 3, 5, 7) = 19;
  TensorAt4d(tensor, 2, 4, 6, 8) = 38;
  TensorAt4d(tensor, 3, 5, 7, 9) = 57;

  EXPECT_FLOAT_EQ(TensorAt4d(tensor, 1, 3, 5, 7), 19);
  EXPECT_FLOAT_EQ(TensorAt4d(tensor, 2, 4, 6, 8), 38);
  EXPECT_FLOAT_EQ(TensorAt4d(tensor, 3, 5, 7, 9), 57);

  // Free the tensor
  TensorFree((Tensor**)&tensor);
}

TEST(TestTensor, TensorSetShapeLike)
{
  // Create an empty float tensor
  FloatTensor* tensor0 = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  // Create a 3D float tensor
  FloatTensor* tensor1 = (FloatTensor*)TensorEmpty3d(
    TENSOR_TYPE_FLOAT, 13, 24, 35);
  // Create a 2D float tensor
  FloatTensor* tensor2 = (FloatTensor*)TensorEmpty2d(
    TENSOR_TYPE_FLOAT, 17, 28);
  // Create a 4D float tensor
  FloatTensor* tensor3 = (FloatTensor*)TensorEmpty4d(
    TENSOR_TYPE_FLOAT, 19, 20, 21, 22);

  // Set the shape of the tensor
  TensorSetShapeLike((Tensor*)tensor0, (Tensor*)tensor1);

  EXPECT_EQ(tensor0->base_.ndim_, 3);
  EXPECT_EQ(tensor0->base_.numel_, 13 * 24 * 35);
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)tensor0, 3, 13, 24, 35));

  // Set the shape of the tensor
  TensorSetShapeLike((Tensor*)tensor0, (Tensor*)tensor2);

  EXPECT_EQ(tensor0->base_.ndim_, 2);
  EXPECT_EQ(tensor0->base_.numel_, 17 * 28);
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)tensor0, 2, 17, 28));

  // Set the shape of the tensor
  TensorSetShapeLike((Tensor*)tensor0, (Tensor*)tensor3);

  EXPECT_EQ(tensor0->base_.ndim_, 4);
  EXPECT_EQ(tensor0->base_.numel_, 19 * 20 * 21 * 22);
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)tensor0, 4, 19, 20, 21, 22));
}

TEST(TestTensor, TensorIsShapeEqual)
{
  // Create a 2D float tensor
  FloatTensor* tensor0 = (FloatTensor*)TensorEmpty2d(
    TENSOR_TYPE_FLOAT, 13, 24);
  // Create a 3D float tensor
  FloatTensor* tensor1 = (FloatTensor*)TensorEmpty3d(
    TENSOR_TYPE_FLOAT, 24, 35, 46);
  // Create another 2D float tensor
  FloatTensor* tensor2 = (FloatTensor*)TensorEmpty2d(
    TENSOR_TYPE_FLOAT, 24, 35);
  // Create another 2D float tensor
  FloatTensor* tensor3 = (FloatTensor*)TensorEmpty2d(
    TENSOR_TYPE_FLOAT, 13, 24);

  EXPECT_TRUE(TensorIsShapeEqual(nullptr, nullptr));
  EXPECT_TRUE(TensorIsShapeEqual((Tensor*)tensor0, (Tensor*)tensor3));

  EXPECT_FALSE(TensorIsShapeEqual((Tensor*)tensor0, nullptr));
  EXPECT_FALSE(TensorIsShapeEqual((Tensor*)tensor0, (Tensor*)tensor1));
  EXPECT_FALSE(TensorIsShapeEqual((Tensor*)tensor0, (Tensor*)tensor2));

  // Free the tensors
  TensorFree((Tensor**)&tensor0);
  TensorFree((Tensor**)&tensor1);
  TensorFree((Tensor**)&tensor2);
  TensorFree((Tensor**)&tensor3);
}

TEST(TestTensor, TensorIsShapeEqualA)
{
  const int shape0[] = { 13, 24, 35 };
  const int shape1[] = { 13, 24 };
  const int shape2[] = { 24, 35 };
  const int shape3[] = { 24, 35, 46, 57 };
  const int shape4[] = { 24, 35, 46 };

  // Create a 3D float tensor
  FloatTensor* tensor = (FloatTensor*)TensorEmpty3d(
    TENSOR_TYPE_FLOAT, 24, 35, 46);

  EXPECT_FALSE(TensorIsShapeEqualA((Tensor*)tensor, 3, shape0));
  EXPECT_FALSE(TensorIsShapeEqualA((Tensor*)tensor, 2, shape1));
  EXPECT_FALSE(TensorIsShapeEqualA((Tensor*)tensor, 2, shape2));
  EXPECT_FALSE(TensorIsShapeEqualA((Tensor*)tensor, 4, shape3));
  EXPECT_TRUE(TensorIsShapeEqualA((Tensor*)tensor, 3, shape4));

  // Free the tensor
  TensorFree((Tensor**)&tensor);
}

TEST(TestTensor, TensorIsShapeEqualNd)
{
  // Create a 3D float tensor
  FloatTensor* tensor = (FloatTensor*)TensorEmpty3d(
    TENSOR_TYPE_FLOAT, 24, 35, 46);

  EXPECT_FALSE(TensorIsShapeEqualNd((Tensor*)tensor, 3, 13, 24, 35));
  EXPECT_FALSE(TensorIsShapeEqualNd((Tensor*)tensor, 2, 13, 24));
  EXPECT_FALSE(TensorIsShapeEqualNd((Tensor*)tensor, 2, 24, 35));
  EXPECT_FALSE(TensorIsShapeEqualNd((Tensor*)tensor, 4, 24, 35, 46, 57));
  EXPECT_TRUE(TensorIsShapeEqualNd((Tensor*)tensor, 3, 24, 35, 46));

  // Free the tensor
  TensorFree((Tensor**)&tensor);
}
