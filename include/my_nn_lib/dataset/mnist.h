
// mnist.h

#ifndef DATASET_MNIST_H
#define DATASET_MNIST_H

#ifdef __cplusplus
extern "C" {
#endif

#include "my_nn_lib/tensor.h"

// Load the label data in MNIST dataset
U8Tensor* MNISTLoadLabels(const char* file_name);

// Load the image data in MNIST dataset
U8Tensor* MNISTLoadImages(const char* file_name);

#ifdef __cplusplus
}
#endif

#endif // DATASET_MNIST_H
