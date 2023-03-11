
// train_dnn_mnist.c
// Train a three-layer deep neural network for MNIST digit classification

#include <stdio.h>
#include <stdlib.h>

#include "my_nn_lib/activation.h"
#include "my_nn_lib/linear.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/loss.h"
#include "my_nn_lib/optimizer.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_list.h"
#include "my_nn_lib/tensor_ops.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"
#include "my_nn_lib/dataset/mnist.h"

typedef struct
{
  LinearParams linear0_;
  LinearParams linear1_;
  LinearParams linear2_;
} ModelParams;

typedef struct
{
  FloatTensor* x0_;
  FloatTensor* x1_;
  FloatTensor* x2_;
  FloatTensor* x3_;
  FloatTensor* x4_;
  FloatTensor* x5_;
  FloatTensor* x6_;
} LayerOutputs;

void ModelParamsInitialize(ModelParams* params)
{
  LinearParamsInitialize(&params->linear0_, 784, 512);
  LinearParamsInitialize(&params->linear1_, 512, 128);
  LinearParamsInitialize(&params->linear2_, 128, 10);

  FloatTensorRandomUniform(params->linear0_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear0_.bias_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear1_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear1_.bias_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear2_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear2_.bias_, -0.1f, 0.1f);
}

void ModelParamsDestroy(ModelParams* params)
{
  LinearParamsFree(&params->linear0_);
  LinearParamsFree(&params->linear1_);
  LinearParamsFree(&params->linear2_);
}

void LayerOutputsInitialize(LayerOutputs* outputs)
{
  outputs->x0_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->x1_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->x2_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->x3_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->x4_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->x5_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  outputs->x6_ = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
}

void LayerOutputsDestroy(LayerOutputs* outputs)
{
  TensorFree((Tensor**)&outputs->x0_);
  TensorFree((Tensor**)&outputs->x1_);
  TensorFree((Tensor**)&outputs->x2_);
  TensorFree((Tensor**)&outputs->x3_);
  TensorFree((Tensor**)&outputs->x4_);
  TensorFree((Tensor**)&outputs->x5_);
  TensorFree((Tensor**)&outputs->x6_);
}

void OptimizerInitialize(TensorListEntry* optim_params,
                         TensorListEntry* optim_gradients,
                         ModelParams* params,
                         ModelParams* gradients)
{
  TensorListInitialize(optim_params);
  TensorListInitialize(optim_gradients);

  TensorListAppend(optim_params,
    (Tensor*)params->linear0_.weight_, "linear0_weight");
  TensorListAppend(optim_params,
    (Tensor*)params->linear0_.bias_, "linear0_bias");
  TensorListAppend(optim_params,
    (Tensor*)params->linear1_.weight_, "linear1_weight");
  TensorListAppend(optim_params,
    (Tensor*)params->linear1_.bias_, "linear1_bias");
  TensorListAppend(optim_params,
    (Tensor*)params->linear2_.weight_, "linear2_weight");
  TensorListAppend(optim_params,
    (Tensor*)params->linear2_.bias_, "linear2_bias");

  TensorListAppend(optim_gradients,
    (Tensor*)gradients->linear0_.weight_, "linear0_weight");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->linear0_.bias_, "linear0_bias");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->linear1_.weight_, "linear1_weight");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->linear1_.bias_, "linear1_bias");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->linear2_.weight_, "linear2_weight");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->linear2_.bias_, "linear2_bias");
}

void OptimizerDestroy(TensorListEntry* optim_params,
                      TensorListEntry* optim_gradients)
{
  TensorListFree(optim_params);
  TensorListFree(optim_gradients);
}

float ModelForward(const FloatTensor* x,
                   LayerOutputs* outputs,
                   const IntTensor* target,
                   const ModelParams* params)
{
  // `x` is of size (B, 784)
  // `x0_` and `x1_` are of size (B, 256)
  LinearForward(x, outputs->x0_, &params->linear0_);
  ReLUForward(outputs->x0_, outputs->x1_);

  // `x2_` and `x3_` are of size (B, 64)
  LinearForward(outputs->x1_, outputs->x2_, &params->linear1_);
  ReLUForward(outputs->x2_, outputs->x3_);

  // `x4_` and `x5_` are of size (B, 10)
  LinearForward(outputs->x3_, outputs->x4_, &params->linear2_);
  ReLUForward(outputs->x4_, outputs->x5_);

  // `x6_` is of size (B, 10)
  float loss = CrossEntropyLossForward(outputs->x5_, target, outputs->x6_);
  return loss;
}

void ModelBackward(const ModelParams* params,
                   const LayerOutputs* outputs,
                   const IntTensor* target,
                   const FloatTensor* x,
                   FloatTensor* dx,
                   ModelParams* grad_params,
                   LayerOutputs* grad_outputs)
{
  // `x6_` and `x5_` are of size (B, 10)
  CrossEntropyLossBackward(outputs->x6_, target, grad_outputs->x5_);

  // `x5_` and `x4_` are of size (B, 10)
  ReLUBackward(grad_outputs->x5_, outputs->x4_, grad_outputs->x4_);

  // `x3_` is of size (B, 64)
  LinearBackward(grad_outputs->x4_, outputs->x3_, grad_outputs->x3_,
                 &grad_params->linear2_, &params->linear2_);

  // `x2_` is of size (B, 64)
  ReLUBackward(grad_outputs->x3_, outputs->x2_, grad_outputs->x2_);

  // `x1_` is of size (B, 256)
  LinearBackward(grad_outputs->x2_, outputs->x1_, grad_outputs->x1_,
                 &grad_params->linear1_, &params->linear1_);

  // `x0_` is of size (B, 256)
  ReLUBackward(grad_outputs->x1_, outputs->x0_, grad_outputs->x0_);

  // `x` is of size (B, 784)
  LinearBackward(grad_outputs->x0_, x, dx,
                 &grad_params->linear0_, &params->linear0_);
}

void ModelUpdateParams(Optimizer* optimizer,
                       TensorListEntry* params,
                       TensorListEntry* gradients)
{
  OptimizerSGDUpdate(optimizer, params, gradients);
}

void LoadMNISTDataset(const char* dataset_dir,
                      U8Tensor** train_images,
                      U8Tensor** train_labels,
                      U8Tensor** test_images,
                      U8Tensor** test_labels)
{
  char* file_path_train_images = AllocateFormatString(
    "%s/%s", dataset_dir, "train-images-idx3-ubyte");
  Assert(file_path_train_images != NULL, "Failed to allocate a new string");

  char* file_path_train_labels = AllocateFormatString(
    "%s/%s", dataset_dir, "train-labels-idx1-ubyte");
  Assert(file_path_train_labels != NULL, "Failed to allocate a new string");

  char* file_path_test_images = AllocateFormatString(
    "%s/%s", dataset_dir, "t10k-images-idx3-ubyte");
  Assert(file_path_test_images != NULL, "Failed to allocate a new string");

  char* file_path_test_labels = AllocateFormatString(
    "%s/%s", dataset_dir, "t10k-labels-idx1-ubyte");
  Assert(file_path_test_labels != NULL, "Failed to allocate a new string");

  LogInfo("Reading the MNIST image file: %s", file_path_train_images);
  *train_images = MNISTLoadImages(file_path_train_images);
  LogInfo("Reading the MNIST label file: %s", file_path_train_labels);
  *train_labels = MNISTLoadLabels(file_path_train_labels);

  LogInfo("Reading the MNIST image file: %s", file_path_test_images);
  *test_images = MNISTLoadImages(file_path_test_images);
  LogInfo("Reading the MNIST label file: %s", file_path_test_labels);
  *test_labels = MNISTLoadLabels(file_path_test_labels);

  free(file_path_train_images);
  free(file_path_train_labels);
  free(file_path_test_images);
  free(file_path_test_labels);

  file_path_train_images = NULL;
  file_path_train_labels = NULL;
  file_path_test_images = NULL;
  file_path_test_labels = NULL;
}

void TrainEpoch(const int epoch,
                const int batch_size,
                FloatTensor* x,
                FloatTensor* dx,
                IntTensor* target,
                IntTensor* estimated,
                ModelParams* model_params,
                ModelParams* grad_params,
                LayerOutputs* layer_outputs,
                LayerOutputs* grad_outputs,
                OptimizerSGD* optimizer,
                TensorListEntry* optim_params,
                TensorListEntry* optim_gradients,
                IntTensor* train_samples_perm,
                const U8Tensor* train_images,
                const U8Tensor* train_labels)
{
  const int in_height = train_images->base_.shape_[1];
  const int in_width = train_images->base_.shape_[2];
  const int in_dims = in_height * in_width;

  const int num_samples = train_images->base_.shape_[0];
  const int num_batches = (num_samples + batch_size - 1) / batch_size;

  float train_loss = 0.0f;
  float train_accuracy = 0.0f;
  int num_correct = 0;

  // Set the shape of the input tensor if necessary
  TensorSetShape((Tensor*)x, 2, batch_size, in_dims);
  // Set the shape of the output tensors if necessary
  TensorSetShape((Tensor*)target, 1, batch_size);
  TensorSetShape((Tensor*)estimated, 1, batch_size);

  // Set the shape of the tensor for permutation if necessary
  TensorSetShape((Tensor*)train_samples_perm, 1, num_samples);
  // Create a random permutation for the training
  IotaI32(train_samples_perm->data_, 0, num_samples);
  RandomPermutationI32(train_samples_perm->data_, num_samples);

  for (int b = 0; b < num_batches; ++b) {
    // Compute the number of samples in the current batch
    const int b0 = b * batch_size;
    const int num_samples_remaining = num_samples - b0;
    const int num_samples_in_batch = Min(num_samples_remaining, batch_size);
    const int num_samples_used = Min((b + 1) * batch_size, num_samples);

    // Set the batch input
    for (int b1 = 0; b1 < num_samples_in_batch; ++b1) {
      const int sample_idx = TensorAt1d(train_samples_perm, b0 + b1);
      for (int h = 0; h < in_height; ++h) {
        for (int w = 0; w < in_width; ++w) {
          const uint8_t val = TensorAt3d(train_images, sample_idx, h, w);
          TensorAt2d(x, b1, h * in_width + w) = (float)val / 255.0f;
        }
      }
    }

    // Set the target labels
    for (int b1 = 0; b1 < num_samples_in_batch; ++b1) {
      const int sample_idx = TensorAt1d(train_samples_perm, b0 + b1);
      TensorAt1d(target, b1) = TensorAt1d(train_labels, sample_idx);
    }

    // Perform the forward operation
    const float loss = ModelForward(x, layer_outputs, target, model_params);

    // Perform the backward operation
    ModelBackward(model_params, layer_outputs, target, x, dx,
                  grad_params, grad_outputs);

    // Update the parameters
    ModelUpdateParams((Optimizer*)optimizer, optim_params, optim_gradients);

    // Compute the loss
    if (b == 0)
      train_loss = loss;
    else
      train_loss = train_loss * 0.95f + loss * 0.05f;

    // Compute the accuracy
    FloatTensor2dArgMax(estimated, layer_outputs->x6_);

    for (int b1 = 0; b1 < num_samples_in_batch; ++b1)
      if (TensorAt1d(estimated, b1) == TensorAt1d(target, b1))
        ++num_correct;

    if (b % 50 == 0) {
      train_accuracy = (float)num_correct / (float)num_samples_used;
      LogInfo("Train epoch %d, batch: %d, accuracy: %.3f%% (%d/%d)",
              epoch, b, train_accuracy * 100.0f,
              num_correct, num_samples_used);
    }
  }

  train_accuracy = (float)num_correct / (float)num_samples;
  LogInfo("Train epoch %d, accuracy: %.3f%% (%d/%d)",
          epoch, train_accuracy * 100.0f,
          num_correct, num_samples);
}

void TestEpoch(const int epoch,
               const int batch_size,
               FloatTensor* x,
               IntTensor* target,
               IntTensor* estimated,
               ModelParams* model_params,
               LayerOutputs* layer_outputs,
               const U8Tensor* test_images,
               const U8Tensor* test_labels)
{
  const int in_height = test_images->base_.shape_[1];
  const int in_width = test_images->base_.shape_[2];
  const int in_dims = in_height * in_width;

  const int num_samples = test_images->base_.shape_[0];
  const int num_batches = (num_samples + batch_size - 1) / batch_size;

  float test_loss = 0.0f;
  float test_accuracy = 0.0f;
  int num_correct = 0;

  // Set the shape of the input tensor if necessary
  TensorSetShape((Tensor*)x, 2, batch_size, in_dims);
  // Set the shape of the output tensors if necessary
  TensorSetShape((Tensor*)target, 1, batch_size);
  TensorSetShape((Tensor*)estimated, 1, batch_size);

  for (int b = 0; b < num_batches; ++b) {
    // Compute the number of samples in the current batch
    const int b0 = b * batch_size;
    const int num_samples_remaining = num_samples - b0;
    const int num_samples_in_batch = Min(num_samples_remaining, batch_size);
    const int num_samples_used = Min((b + 1) * batch_size, num_samples);

    // Set the batch input
    for (int b1 = 0; b1 < num_samples_in_batch; ++b1) {
      for (int h = 0; h < in_height; ++h) {
        for (int w = 0; w < in_width; ++w) {
          const uint8_t val = TensorAt3d(test_images, b0 + b1, h, w);
          TensorAt2d(x, b1, h * in_width + w) = (float)val / 255.0f;
        }
      }
    }

    // Set the target labels
    for (int b1 = 0; b1 < num_samples_in_batch; ++b1) {
      TensorAt1d(target, b1) = TensorAt1d(test_labels, b0 + b1);
    }

    // Perform the forward operation
    const float loss = ModelForward(x, layer_outputs, target, model_params);

    // Compute the loss
    if (b == 0)
      test_loss = loss;
    else
      test_loss = test_loss * 0.95f + loss * 0.05f;

    // Compute the accuracy
    FloatTensor2dArgMax(estimated, layer_outputs->x6_);

    for (int b1 = 0; b1 < num_samples_in_batch; ++b1)
      if (TensorAt1d(estimated, b1) == TensorAt1d(target, b1))
        ++num_correct;

    if (b % 50 == 0) {
      test_accuracy = (float)num_correct / (float)num_samples_used;
      LogInfo("Test epoch %d, batch: %d, accuracy: %.3f%% (%d/%d)",
              epoch, b, test_accuracy * 100.0f,
              num_correct, num_samples_used);
    }
  }

  test_accuracy = (float)num_correct / (float)num_samples;
  LogInfo("Test epoch %d, accuracy: %.3f%% (%d/%d)",
          epoch, test_accuracy * 100.0f,
          num_correct, num_samples);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    LogError("Usage: %s <Directory to the MNIST dataset>", argv[0]);
    return EXIT_FAILURE;
  }

  // Load the MNIST digit dataset
  const char* dataset_dir = argv[1];
  U8Tensor* train_images;
  U8Tensor* train_labels;
  U8Tensor* test_images;
  U8Tensor* test_labels;
  LoadMNISTDataset(dataset_dir, &train_images, &train_labels,
                   &test_images, &test_labels);

  // Initialize a model
  ModelParams model_params;
  ModelParams grad_params;
  LayerOutputs layer_outputs;
  LayerOutputs grad_outputs;
  ModelParamsInitialize(&model_params);
  ModelParamsInitialize(&grad_params);
  LayerOutputsInitialize(&layer_outputs);
  LayerOutputsInitialize(&grad_outputs);

  // Initialize a SGD optimizer
  const float learning_rate = 1e-1f;
  OptimizerSGD* optimizer = OptimizerSGDCreate(learning_rate);

  // Create a list of parameters and gradients for the optimizer
  TensorListEntry optim_params;
  TensorListEntry optim_gradients;
  OptimizerInitialize(&optim_params, &optim_gradients,
                      &model_params, &grad_params);

  // Start the training
  const int num_epochs = 20;
  const int batch_size = 16;

  // Create a tensor for the batch input
  FloatTensor* x = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  FloatTensor* dx = (FloatTensor*)TensorAllocate(TENSOR_TYPE_FLOAT);
  IntTensor* target = (IntTensor*)TensorAllocate(TENSOR_TYPE_INT);
  IntTensor* estimated = (IntTensor*)TensorAllocate(TENSOR_TYPE_INT);

  // Create an array to shuffle the training samples
  IntTensor* train_samples_perm = (IntTensor*)TensorAllocate(TENSOR_TYPE_INT);

  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    LogInfo("Epoch %d ...", epoch);

    // Perform the training
    TrainEpoch(epoch, batch_size, x, dx, target, estimated,
               &model_params, &grad_params, &layer_outputs, &grad_outputs,
               optimizer, &optim_params, &optim_gradients,
               train_samples_perm, train_images, train_labels);

    // Perform the test
    TestEpoch(epoch, batch_size, x, target, estimated,
              &model_params, &layer_outputs, test_images, test_labels);
  }

  // Free the tensors
  TensorFree((Tensor**)&x);
  TensorFree((Tensor**)&dx);
  TensorFree((Tensor**)&target);
  TensorFree((Tensor**)&estimated);
  TensorFree((Tensor**)&train_samples_perm);

  // Free the optimizer
  OptimizerDestroy(&optim_params, &optim_gradients);
  OptimizerSGDFree(&optimizer);

  // Free the model
  ModelParamsDestroy(&model_params);
  ModelParamsDestroy(&grad_params);
  LayerOutputsDestroy(&layer_outputs);
  LayerOutputsDestroy(&grad_outputs);

  // Free the dataset
  TensorFree((Tensor**)&train_images);
  TensorFree((Tensor**)&train_labels);
  TensorFree((Tensor**)&test_images);
  TensorFree((Tensor**)&test_labels);

  return EXIT_SUCCESS;
}
