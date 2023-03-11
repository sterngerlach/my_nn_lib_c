
// mnist.c

#include "my_nn_lib/dataset/mnist.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/util.h"

#include <stdint.h>
#include <stdio.h>

// Load the label data in MNIST dataset
U8Tensor* MNISTLoadLabels(const char* file_name)
{
  FILE* fp = fopen(file_name, "rb");
  if (fp == NULL) {
    LogError("Failed to open the MNIST label file: %s", file_name);
    return NULL;
  }

  // Read the identifier
  int id;
  if (fread(&id, sizeof(int), 1, fp) != 1) {
    LogError("Failed to read the identifier in the MNIST label file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  // Read the number of labels
  int num_labels;
  if (fread(&num_labels, sizeof(int), 1, fp) != 1) {
    LogError("Failed to read the number of labels in the MNIST label file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  // Convert the big to little endian
  id = SwapEndianI32(id);
  num_labels = SwapEndianI32(num_labels);

  LogInfo("Reading the MNIST label file: %s", file_name);
  LogInfo("Identifier: %d", id);
  LogInfo("Number of labels: %d", num_labels);

  // Allocate a 8-bit unsigned integer tensor
  U8Tensor* labels = (U8Tensor*)TensorEmpty1d(TENSOR_TYPE_U8, num_labels);
  // Read the labels
  if (fread(labels->data_, sizeof(uint8_t), num_labels, fp) != num_labels) {
    LogError("Failed to read the label data in the MNIST label file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  fclose(fp);
  return labels;
}

// Load the image data in MNIST dataset
U8Tensor* MNISTLoadImages(const char* file_name)
{
  FILE* fp = fopen(file_name, "rb");
  if (fp == NULL) {
    LogError("Failed to open the MNIST image file: %s", file_name);
    return NULL;
  }

  // Read the identifier
  int id;
  if (fread(&id, sizeof(int), 1, fp) != 1) {
    LogError("Failed to read the identifier in the MNIST image file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  // Read the number of images
  int num_images;
  if (fread(&num_images, sizeof(int), 1, fp) != 1) {
    LogError("Failed to read the number of images in the MNIST image file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  // Read the image height
  int height;
  if (fread(&height, sizeof(int), 1, fp) != 1) {
    LogError("Failed to read the image height in the MNIST image file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  // Read the image width
  int width;
  if (fread(&width, sizeof(int), 1, fp) != 1) {
    LogError("Failed to read the image width in the MNIST image file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  // Convert the big to little endian
  id = SwapEndianI32(id);
  num_images = SwapEndianI32(num_images);
  height = SwapEndianI32(height);
  width = SwapEndianI32(width);

  // Allocate a 8-bit unsigned integer tensor
  U8Tensor* images = (U8Tensor*)TensorEmpty3d(
    TENSOR_TYPE_U8, num_images, height, width);
  // Read the images
  if (fread(images->data_, sizeof(uint8_t), images->base_.numel_, fp) !=
      images->base_.numel_) {
    LogError("Failed to read the image data in the MNIST image file: %s",
             file_name);
    fclose(fp);
    return NULL;
  }

  fclose(fp);
  return images;
}
