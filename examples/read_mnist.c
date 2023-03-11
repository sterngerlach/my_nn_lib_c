
// read_mnist.c

#include <stdio.h>
#include <stdlib.h>

#include "my_nn_lib/logger.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/dataset/mnist.h"

int main(int argc, char** argv)
{
  if (argc != 3) {
    LogError("Usage: %s <Path to the image file> <Path to the label file>",
             argv[0]);
    return EXIT_FAILURE;
  }

  // Read the MNIST images and labels
  const char* file_path_images = argv[1];
  const char* file_path_labels = argv[2];

  LogInfo("Reading the MNIST image file: %s", file_path_images);
  U8Tensor* images = MNISTLoadImages(file_path_images);
  if (images == NULL) {
    LogError("Failed to read the MNIST image file: %s", file_path_images);
    return EXIT_FAILURE;
  }

  LogInfo("Reading the MNIST label file: %s", file_path_labels);
  U8Tensor* labels = MNISTLoadLabels(file_path_labels);
  if (labels == NULL) {
    LogError("Failed to read the MNIST label file: %s", file_path_labels);
    TensorFree((Tensor**)&images);
    return EXIT_FAILURE;
  }

  LogInfo("Number of images: %d", images->base_.shape_[0]);
  LogInfo("Image height: %d", images->base_.shape_[1]);
  LogInfo("Image width: %d", images->base_.shape_[2]);
  LogInfo("Number of labels: %d", labels->base_.shape_[0]);

  // Print the second image
  LogInfo("Label: %d", TensorAt1d(labels, 1));

  for (int h = 0; h < images->base_.shape_[1]; ++h) {
    for (int w = 0; w < images->base_.shape_[2]; ++w) {
      fputc(TensorAt3d(images, 1, h, w) <= 127 ? ' ' : '*', stderr);
    }
    fprintf(stderr, "\n");
  }

  // Free the tensors
  TensorFree((Tensor**)&images);
  TensorFree((Tensor**)&labels);

  return EXIT_SUCCESS;
}
