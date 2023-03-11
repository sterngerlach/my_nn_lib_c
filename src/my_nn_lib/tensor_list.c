
// tensor_list.c

#include "my_nn_lib/tensor_list.h"
#include "my_nn_lib/util.h"

#include <stdlib.h>
#include <string.h>

// Initialize the tensor list
void TensorListInitialize(TensorListEntry* tensors)
{
  Assert(tensors != NULL, "`tensors` should not be NULL");

  // Initialize the list head
  tensors->tensor_ = NULL;
  tensors->name_ = NULL;
  LinkedListInitialize(&tensors->entry_);
}

// Insert a new tensor to the list
void TensorListAppend(TensorListEntry* tensors,
                      Tensor* tensor,
                      const char* name)
{
  Assert(tensors != NULL, "`tensors` should not be NULL");
  Assert(tensor != NULL, "`tensor` should not be NULL");

  TensorListEntry* entry = calloc(1, sizeof(TensorListEntry));
  Assert(entry != NULL, "Failed to allocate a new entry");

  char* name0 = NULL;
  if (name != NULL) {
    name0 = strdup(name);
    Assert(name0 != NULL, "Failed to allocate a string for the tensor name");
  }

  entry->tensor_ = tensor;
  entry->name_ = name0;
  LinkedListInsertTail(&entry->entry_, &tensors->entry_);
}

// Free the tensor list
void TensorListFree(TensorListEntry* tensors)
{
  Assert(tensors != NULL, "`tensors` should not be NULL");

  // Remove the entries in the tensor list
  TensorListEntry* iter;
  TensorListEntry* iter_next;
  LinkedListForEachSafe(iter, iter_next, &tensors->entry_,
                        TensorListEntry, entry_) {
    // Remove the entry from the tensor list
    LinkedListRemove(&iter->entry_);
    // Free the tensor name
    free(iter->name_);
    iter->name_ = NULL;
    // Free the entry
    free(iter);
    iter = NULL;
  }
}
