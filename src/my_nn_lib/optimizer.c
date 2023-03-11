
// optimizer.c

#include "my_nn_lib/optimizer.h"
#include "my_nn_lib/tensor_ops.h"
#include "my_nn_lib/util.h"

#include <stdlib.h>

// Create a SGD optimizer
OptimizerSGD* OptimizerSGDCreate(const float lr)
{
  // Allocate a SGD optimizer
  OptimizerSGD* optimizer = calloc(1, sizeof(OptimizerSGD));
  Assert(optimizer != NULL, "Failed to allocate a new optimizer");

  // Set the parameters
  optimizer->base_.type_ = OPTIMIZER_TYPE_SGD;
  optimizer->lr_ = lr;

  return optimizer;
}

// Free a SGD optimizer
void OptimizerSGDFree(OptimizerSGD** optimizer)
{
  if (*optimizer == NULL)
    return;

  free(*optimizer);
  *optimizer = NULL;
}

// Update the parameters using SGD (Stochastic Gradient Descent)
void OptimizerSGDUpdate(Optimizer* optimizer,
                        TensorListEntry* parameters,
                        TensorListEntry* gradients)
{
  // Check that the given optimizer is a SGD optimizer
  Assert(optimizer->type_ == OPTIMIZER_TYPE_SGD,
         "`optimizer` should be a SGD optimizer");

  OptimizerSGD* optim = (OptimizerSGD*)optimizer;

  // Iterate through the parameters to be updated
  TensorListEntry* param = LinkedListDataHead(
    &parameters->entry_, TensorListEntry, entry_);
  TensorListEntry* grad = LinkedListDataHead(
    &gradients->entry_, TensorListEntry, entry_);

  for (; &param->entry_ != &parameters->entry_ &&
       &grad->entry_ != &gradients->entry_;
       param = LinkedListDataNext(param, TensorListEntry, entry_),
       grad = LinkedListDataNext(grad, TensorListEntry, entry_)) {
    Assert(param->tensor_->dtype_ == TENSOR_TYPE_FLOAT,
           "`param->tensor_` should be a float tensor");
    Assert(grad->tensor_->dtype_ == TENSOR_TYPE_FLOAT,
           "`grad->tensor_` should be a float tensor");
    FloatTensorSubScaleI((FloatTensor*)param->tensor_,
                         (FloatTensor*)grad->tensor_, optim->lr_);
  }
}
