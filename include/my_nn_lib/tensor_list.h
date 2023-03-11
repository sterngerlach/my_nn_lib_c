
// tensor_list.h

#ifndef TENSOR_LIST_H
#define TENSOR_LIST_H

#include "linked_list.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Doubly-linked list for the tensors
// This list holds the network parameters to be updated and their gradients
typedef struct
{
  // Tensor data
  Tensor*         tensor_;
  // Tensor name (may be NULL)
  char*           name_;
  // Pointer to the next and previous entry
  LinkedListEntry entry_;
} TensorListEntry;

// Initialize the tensor list
void TensorListInitialize(TensorListEntry* tensors);

// Insert a new tensor to the list
void TensorListAppend(TensorListEntry* tensors,
                      Tensor* tensor,
                      const char* name);

// Free the tensor list
void TensorListFree(TensorListEntry* tensors);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_LIST_H
