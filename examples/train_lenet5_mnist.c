
// train_lenet5_mnist.c
// Train a LeNet5 model for MNIST digit classification

#include <stdio.h>
#include <stdlib.h>

#include "my_nn_lib/activation.h"
#include "my_nn_lib/conv2d.h"
#include "my_nn_lib/flatten.h"
#include "my_nn_lib/linear.h"
#include "my_nn_lib/logger.h"
#include "my_nn_lib/loss.h"
#include "my_nn_lib/maxpool2d.h"
#include "my_nn_lib/optimizer.h"
#include "my_nn_lib/tensor.h"
#include "my_nn_lib/tensor_list.h"
#include "my_nn_lib/tensor_ops.h"
#include "my_nn_lib/tensor_util.h"
#include "my_nn_lib/util.h"
#include "my_nn_lib/dataset/mnist.h"

typedef struct
{
  Conv2dParams    conv0_;
  MaxPool2dParams pool0_;
  Conv2dParams    conv1_;
  MaxPool2dParams pool1_;
  LinearParams    linear0_;
  LinearParams    linear1_;
  LinearParams    linear2_;
} LeNet5Params;

typedef struct
{
  Conv2dOutputs           conv0_;
  ActivationOutputs       relu0_;
  MaxPool2dOutputs        pool0_;
  Conv2dOutputs           conv1_;
  ActivationOutputs       relu1_;
  MaxPool2dOutputs        pool1_;
  FlattenOutputs          flatten0_;
  LinearOutputs           linear0_;
  ActivationOutputs       relu2_;
  LinearOutputs           linear1_;
  ActivationOutputs       relu3_;
  LinearOutputs           linear2_;
  ActivationOutputs       relu4_;
  CrossEntropyLossOutputs loss_;
} LayerOutputs;

void LeNet5ParamsInitialize(LeNet5Params* params)
{
  Conv2dParamsInitialize(&params->conv0_, 1, 6, 5, 5, 1, 2);
  MaxPool2dParamsInitialize(&params->pool0_, 2, 2, 2, 0);

  Conv2dParamsInitialize(&params->conv1_, 6, 16, 5, 5, 1, 0);
  MaxPool2dParamsInitialize(&params->pool1_, 2, 2, 2, 0);

  LinearParamsInitialize(&params->linear0_, 400, 120);
  LinearParamsInitialize(&params->linear1_, 120, 84);
  LinearParamsInitialize(&params->linear2_, 84, 10);

  FloatTensorRandomUniform(params->conv0_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->conv0_.bias_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->conv1_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->conv1_.bias_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear0_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear0_.bias_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear1_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear1_.bias_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear2_.weight_, -0.1f, 0.1f);
  FloatTensorRandomUniform(params->linear2_.bias_, -0.1f, 0.1f);
}

void LeNet5ParamsDestroy(LeNet5Params* params)
{
  Conv2dParamsFree(&params->conv0_);
  MaxPool2dParamsFree(&params->pool0_);

  Conv2dParamsFree(&params->conv1_);
  MaxPool2dParamsFree(&params->pool1_);

  LinearParamsFree(&params->linear0_);
  LinearParamsFree(&params->linear1_);
  LinearParamsFree(&params->linear2_);
}

void LayerOutputsInitialize(LayerOutputs* outputs)
{
  Conv2dOutputsInitialize(&outputs->conv0_, false);
  ActivationOutputsInitialize(&outputs->relu0_, false);
  MaxPool2dOutputsInitialize(&outputs->pool0_, false);
  Conv2dOutputsInitialize(&outputs->conv1_, false);
  ActivationOutputsInitialize(&outputs->relu1_, false);
  MaxPool2dOutputsInitialize(&outputs->pool1_, false);
  FlattenOutputsInitialize(&outputs->flatten0_, false);

  LinearOutputsInitialize(&outputs->linear0_, false);
  ActivationOutputsInitialize(&outputs->relu2_, false);
  LinearOutputsInitialize(&outputs->linear1_, false);
  ActivationOutputsInitialize(&outputs->relu3_, false);
  LinearOutputsInitialize(&outputs->linear2_, false);
  ActivationOutputsInitialize(&outputs->relu4_, false);

  CrossEntropyLossOutputsInitialize(&outputs->loss_, false);
}

void LayerOutputsDestroy(LayerOutputs* outputs)
{
  Conv2dOutputsFree(&outputs->conv0_);
  ActivationOutputsFree(&outputs->relu0_);
  MaxPool2dOutputsFree(&outputs->pool0_);
  Conv2dOutputsFree(&outputs->conv1_);
  ActivationOutputsFree(&outputs->relu1_);
  MaxPool2dOutputsFree(&outputs->pool1_);
  FlattenOutputsFree(&outputs->flatten0_);

  LinearOutputsFree(&outputs->linear0_);
  ActivationOutputsFree(&outputs->relu2_);
  LinearOutputsFree(&outputs->linear1_);
  ActivationOutputsFree(&outputs->relu3_);
  LinearOutputsFree(&outputs->linear2_);
  ActivationOutputsFree(&outputs->relu4_);

  CrossEntropyLossOutputsFree(&outputs->loss_);
}

void OptimizerInitialize(TensorListEntry* optim_params,
                         TensorListEntry* optim_gradients,
                         LeNet5Params* params,
                         LeNet5Params* gradients)
{
  TensorListInitialize(optim_params);
  TensorListInitialize(optim_gradients);

  TensorListAppend(optim_params,
    (Tensor*)params->conv0_.weight_, "conv0_weight");
  TensorListAppend(optim_params,
    (Tensor*)params->conv0_.bias_, "conv0_bias");
  TensorListAppend(optim_params,
    (Tensor*)params->conv1_.weight_, "conv1_weight");
  TensorListAppend(optim_params,
    (Tensor*)params->conv1_.bias_, "conv1_bias");
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
    (Tensor*)gradients->conv0_.weight_, "conv0_weight");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->conv0_.bias_, "conv0_bias");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->conv1_.weight_, "conv1_weight");
  TensorListAppend(optim_gradients,
    (Tensor*)gradients->conv1_.bias_, "conv1_bias");
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

void LeNet5Forward(const FloatTensor* x,
                   const IntTensor* target,
                   const LeNet5Params* params,
                   LayerOutputs* outputs)
{
  // `x` is of size (B, 1, 28, 28)
  // `outputs->conv0_.y_` and `outputs->relu0_.y_` are of size (B, 6, 28, 28)
  Conv2dForward(x, &outputs->conv0_, &params->conv0_);
  ReLUForward(outputs->conv0_.y_, &outputs->relu0_);

  // `outputs->pool0_.y_` is of size (B, 6, 14, 14)
  MaxPool2dForward(outputs->relu0_.y_, &outputs->pool0_, &params->pool0_);

  // `outputs->conv1_.y_` and `outputs->relu1_.y_` are of size (B, 16, 10, 10)
  Conv2dForward(outputs->pool0_.y_, &outputs->conv1_, &params->conv1_);
  ReLUForward(outputs->conv1_.y_, &outputs->relu1_);

  // `outputs->pool1_.y_` is of size (B, 16, 5, 5)
  MaxPool2dForward(outputs->relu1_.y_, &outputs->pool1_, &params->pool1_);

  // `outputs->flatten0_.y_` is of size (B, 400)
  FlattenForward(outputs->pool1_.y_, &outputs->flatten0_);

  // `outputs->linear0_.y_` and `outputs->relu2_.y_` are of size (B, 120)
  LinearForward(outputs->flatten0_.y_, &outputs->linear0_, &params->linear0_);
  ReLUForward(outputs->linear0_.y_, &outputs->relu2_);

  // `outputs->linear1_.y_` and `outputs->relu3_.y_` are of size (B, 84)
  LinearForward(outputs->relu2_.y_, &outputs->linear1_, &params->linear1_);
  ReLUForward(outputs->linear1_.y_, &outputs->relu3_);

  // `outputs->linear2_.y_` and `outputs->relu4_.y_` are of size (B, 10)
  LinearForward(outputs->relu3_.y_, &outputs->linear2_, &params->linear2_);
  ReLUForward(outputs->linear2_.y_, &outputs->relu4_);

  // `outputs->loss_.y_` is of size (B, 10)
  CrossEntropyLossForward(outputs->relu4_.y_, target, &outputs->loss_);
}

void LeNet5Backward(const LeNet5Params* params,
                    const FloatTensor* x,
                    const IntTensor* target,
                    LayerOutputs* outputs,
                    LeNet5Params* grad_params)
{
  CrossEntropyLossBackward(target, &outputs->loss_);

  ReLUBackward(outputs->loss_.dx_, outputs->linear2_.y_, &outputs->relu4_);
  LinearBackward(outputs->relu4_.dx_, outputs->relu3_.y_, &outputs->linear2_,
                 &grad_params->linear2_, &params->linear2_);

  ReLUBackward(outputs->linear2_.dx_, outputs->linear1_.y_, &outputs->relu3_);
  LinearBackward(outputs->relu3_.dx_, outputs->relu2_.y_, &outputs->linear1_,
                 &grad_params->linear1_, &params->linear1_);

  ReLUBackward(outputs->linear1_.dx_, outputs->linear0_.y_, &outputs->relu2_);
  LinearBackward(outputs->relu2_.dx_, outputs->flatten0_.y_, &outputs->linear0_,
                 &grad_params->linear0_, &params->linear0_);

  FlattenBackward(outputs->linear0_.dx_, outputs->pool1_.y_,
                  &outputs->flatten0_);

  MaxPool2dBackward(outputs->flatten0_.dx_, outputs->relu1_.y_,
                    &outputs->pool1_, &params->pool1_);

  ReLUBackward(outputs->pool1_.dx_, outputs->conv1_.y_, &outputs->relu1_);
  Conv2dBackward(outputs->relu1_.dx_, outputs->pool0_.y_, &outputs->conv1_,
                 &grad_params->conv1_, &params->conv1_);

  MaxPool2dBackward(outputs->conv1_.dx_, outputs->relu0_.y_,
                    &outputs->pool0_, &params->pool0_);

  ReLUBackward(outputs->pool0_.dx_, outputs->conv0_.y_, &outputs->relu0_);
  Conv2dBackward(outputs->relu0_.dx_, x, &outputs->conv0_,
                 &grad_params->conv0_, &params->conv0_);
}

void LeNet5UpdateParams(Optimizer* optimizer,
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
                IntTensor* target,
                IntTensor* estimated,
                LeNet5Params* model_params,
                LeNet5Params* grad_params,
                LayerOutputs* layer_outputs,
                OptimizerSGD* optimizer,
                TensorListEntry* optim_params,
                TensorListEntry* optim_gradients,
                IntTensor* train_samples_perm,
                const U8Tensor* train_images,
                const U8Tensor* train_labels)
{
  const int in_height = train_images->base_.shape_[1];
  const int in_width = train_images->base_.shape_[2];

  const int num_samples = train_images->base_.shape_[0];
  const int num_batches = (num_samples + batch_size - 1) / batch_size;

  float train_loss = 0.0f;
  float train_accuracy = 0.0f;
  int num_correct = 0;

  // Set the shape of the input tensor if necessary
  TensorSetShape((Tensor*)x, 4, batch_size, 1, in_height, in_width);
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
          TensorAt4d(x, b1, 0, h, w) = (float)val / 255.0f;
        }
      }
    }

    // Set the target labels
    for (int b1 = 0; b1 < num_samples_in_batch; ++b1) {
      const int sample_idx = TensorAt1d(train_samples_perm, b0 + b1);
      TensorAt1d(target, b1) = TensorAt1d(train_labels, sample_idx);
    }

    // Perform the forward operation
    LeNet5Forward(x, target, model_params, layer_outputs);

    // Perform the backward operation
    LeNet5Backward(model_params, x, target, layer_outputs, grad_params);

    // Update the parameters
    LeNet5UpdateParams((Optimizer*)optimizer, optim_params, optim_gradients);

    // Compute the loss
    if (b == 0)
      train_loss = layer_outputs->loss_.loss_;
    else
      train_loss = train_loss * 0.95f + layer_outputs->loss_.loss_ * 0.05f;

    // Compute the accuracy
    FloatTensor2dArgMax(estimated, layer_outputs->loss_.y_);

    for (int b1 = 0; b1 < num_samples_in_batch; ++b1)
      if (TensorAt1d(estimated, b1) == TensorAt1d(target, b1))
        ++num_correct;

    if (b % 10 == 0) {
      train_accuracy = (float)num_correct / (float)num_samples_used;
      LogInfo("Train epoch %d, batch: %d, loss: %.3f, "
              "accuracy: %.3f%% (%d/%d)",
              epoch, b, train_loss, train_accuracy * 100.0f,
              num_correct, num_samples_used);
    }
  }

  train_accuracy = (float)num_correct / (float)num_samples;
  LogInfo("Train epoch %d, loss: %.3f, accuracy: %.3f%% (%d/%d)",
          epoch, train_loss, train_accuracy * 100.0f,
          num_correct, num_samples);
}

void TestEpoch(const int epoch,
               const int batch_size,
               FloatTensor* x,
               IntTensor* target,
               IntTensor* estimated,
               LeNet5Params* model_params,
               LayerOutputs* layer_outputs,
               const U8Tensor* test_images,
               const U8Tensor* test_labels)
{
  const int in_height = test_images->base_.shape_[1];
  const int in_width = test_images->base_.shape_[2];

  const int num_samples = test_images->base_.shape_[0];
  const int num_batches = (num_samples + batch_size - 1) / batch_size;

  float test_loss = 0.0f;
  float test_accuracy = 0.0f;
  int num_correct = 0;

  // Set the shape of the input tensor if necessary
  TensorSetShape((Tensor*)x, 4, batch_size, 1, in_height, in_width);
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
          TensorAt4d(x, b1, 0, h, w) = (float)val / 255.0f;
        }
      }
    }

    // Set the target labels
    for (int b1 = 0; b1 < num_samples_in_batch; ++b1) {
      TensorAt1d(target, b1) = TensorAt1d(test_labels, b0 + b1);
    }

    // Perform the forward operation
    LeNet5Forward(x, target, model_params, layer_outputs);

    // Compute the loss
    if (b == 0)
      test_loss = layer_outputs->loss_.loss_;
    else
      test_loss = test_loss * 0.95f + layer_outputs->loss_.loss_ * 0.05f;

    // Compute the accuracy
    FloatTensor2dArgMax(estimated, layer_outputs->loss_.y_);

    for (int b1 = 0; b1 < num_samples_in_batch; ++b1)
      if (TensorAt1d(estimated, b1) == TensorAt1d(target, b1))
        ++num_correct;

    if (b % 10 == 0) {
      test_accuracy = (float)num_correct / (float)num_samples_used;
      LogInfo("Test epoch %d, batch: %d, loss: %.3f, "
              "accuracy: %.3f%% (%d/%d)",
              epoch, b, test_loss, test_accuracy * 100.0f,
              num_correct, num_samples_used);
    }
  }

  test_accuracy = (float)num_correct / (float)num_samples;
  LogInfo("Test epoch %d, loss: %.3f, accuracy: %.3f%% (%d/%d)",
          epoch, test_loss, test_accuracy * 100.0f,
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
  LeNet5Params model_params;
  LeNet5Params grad_params;
  LayerOutputs layer_outputs;
  LeNet5ParamsInitialize(&model_params);
  LeNet5ParamsInitialize(&grad_params);
  LayerOutputsInitialize(&layer_outputs);

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
  IntTensor* target = (IntTensor*)TensorAllocate(TENSOR_TYPE_INT);
  IntTensor* estimated = (IntTensor*)TensorAllocate(TENSOR_TYPE_INT);

  // Create an array to shuffle the training samples
  IntTensor* train_samples_perm = (IntTensor*)TensorAllocate(TENSOR_TYPE_INT);

  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    LogInfo("Epoch %d ...", epoch);

    // Perform the training
    TrainEpoch(epoch, batch_size, x, target, estimated,
               &model_params, &grad_params, &layer_outputs,
               optimizer, &optim_params, &optim_gradients,
               train_samples_perm, train_images, train_labels);

    // Perform the test
    TestEpoch(epoch, batch_size, x, target, estimated,
              &model_params, &layer_outputs, test_images, test_labels);
  }

  // Free the tensors
  TensorFree((Tensor**)&x);
  TensorFree((Tensor**)&target);
  TensorFree((Tensor**)&estimated);
  TensorFree((Tensor**)&train_samples_perm);

  // Free the optimizer
  OptimizerDestroy(&optim_params, &optim_gradients);
  OptimizerSGDFree(&optimizer);

  // Free the model
  LeNet5ParamsDestroy(&model_params);
  LeNet5ParamsDestroy(&grad_params);
  LayerOutputsDestroy(&layer_outputs);

  // Free the dataset
  TensorFree((Tensor**)&train_images);
  TensorFree((Tensor**)&train_labels);
  TensorFree((Tensor**)&test_images);
  TensorFree((Tensor**)&test_labels);

  return EXIT_SUCCESS;
}
