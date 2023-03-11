
// optimizer.h

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "my_nn_lib/tensor_list.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optimizer type
enum OptimizerType
{
  OPTIMIZER_TYPE_UNKNOWN,
  OPTIMIZER_TYPE_SGD,
};

// Optimizer base type
typedef struct
{
  // Optimizer type
  enum OptimizerType type_;
} Optimizer;

// SGD (Stochastic Gradient Descent)
typedef struct
{
  // Optimizer information
  Optimizer base_;
  // Learning rate
  float     lr_;
} OptimizerSGD;

// Create a SGD optimizer
OptimizerSGD* OptimizerSGDCreate(const float lr);

// Free a SGD optimizer
void OptimizerSGDFree(OptimizerSGD** optimizer);

// Update the parameters using SGD
void OptimizerSGDUpdate(Optimizer* optimizer,
                        TensorListEntry* parameters,
                        TensorListEntry* gradients);

#ifdef __cplusplus
}
#endif

#endif // OPTIMIZER_H
