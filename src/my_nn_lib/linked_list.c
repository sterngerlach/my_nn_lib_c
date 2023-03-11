
// linked_list.c

#include "my_nn_lib/linked_list.h"

#include <stdlib.h>

// Insert a new entry `new_entry` between `prev_entry` and `next_entry`
static void LinkedListInsertBetween(LinkedListEntry* new_entry,
                                    LinkedListEntry* prev_entry,
                                    LinkedListEntry* next_entry)
{
  next_entry->prev_ = new_entry;
  new_entry->next_ = next_entry;
  new_entry->prev_ = prev_entry;
  prev_entry->next_ = new_entry;
}

// Initialize the doubly-linked list
void LinkedListInitialize(LinkedListEntry* head)
{
  head->next_ = head;
  head->prev_ = head;
}

// Check if the doubly-linked list is empty
bool LinkedListIsEmpty(const LinkedListEntry* head)
{
  return head->next_ == head;
}

// Insert a new entry `new_entry` after `entry`
void LinkedListInsert(LinkedListEntry* new_entry,
                      LinkedListEntry* entry)
{
  LinkedListInsertBetween(new_entry, entry, entry->next_);
}

// Insert a new entry `new_entry` before `entry`
void LinkedListInsertBefore(LinkedListEntry* new_entry,
                            LinkedListEntry* entry)
{
  LinkedListInsertBetween(new_entry, entry->prev_, entry);
}

// Insert a new entry at head
void LinkedListInsertHead(LinkedListEntry* new_entry,
                          LinkedListEntry* head)
{
  LinkedListInsertBetween(new_entry, head, head->next_);
}

// Insert a new entry at tail
void LinkedListInsertTail(LinkedListEntry* new_entry,
                          LinkedListEntry* head)
{
  LinkedListInsertBetween(new_entry, head->prev_, head);
}

// Remove the entry from the doubly-linked list
void LinkedListRemove(LinkedListEntry* entry)
{
  entry->next_->prev_ = entry->prev_;
  entry->prev_->next_ = entry->next_;
  entry->next_ = NULL;
  entry->prev_ = NULL;
}

// Remove the entry from head
// `NULL` is returned if the list is empty
LinkedListEntry* LinkedListRemoveHead(LinkedListEntry* head)
{
  if (LinkedListIsEmpty(head))
    return NULL;

  LinkedListEntry* entry = head->next_;
  LinkedListRemove(entry);
  return entry;
}

// Remove the entry from tail
// `NULL` is returned if the list is empty
LinkedListEntry* LinkedListRemoveTail(LinkedListEntry* head)
{
  if (LinkedListIsEmpty(head))
    return NULL;

  LinkedListEntry* entry = head->prev_;
  LinkedListRemove(entry);
  return entry;
}
