
// linked_list.h

#ifndef LINKED_LIST_H
#define LINKED_LIST_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Entry for the doubly-linked list
typedef struct LinkedListEntry
{
  // Pointer to the next entry
  struct LinkedListEntry* next_;
  // Pointer to the previous entry
  struct LinkedListEntry* prev_;
} LinkedListEntry;

// Initialize the doubly-linked list
void LinkedListInitialize(LinkedListEntry* head);

// Check if the doubly-linked list is empty
bool LinkedListIsEmpty(const LinkedListEntry* head);

// Insert a new entry `new_entry` after `entry`
void LinkedListInsert(LinkedListEntry* new_entry,
                      LinkedListEntry* entry);

// Insert a new entry `new_entry` before `entry`
void LinkedListInsertBefore(LinkedListEntry* new_entry,
                            LinkedListEntry* entry);

// Insert a new entry at head
void LinkedListInsertHead(LinkedListEntry* new_entry,
                          LinkedListEntry* head);

// Insert a new entry at tail
void LinkedListInsertTail(LinkedListEntry* new_entry,
                          LinkedListEntry* head);

// Remove the entry from the doubly-linked list
void LinkedListRemove(LinkedListEntry* entry);

// Remove the entry from head
// `NULL` is returned if the list is empty
LinkedListEntry* LinkedListRemoveHead(LinkedListEntry* head);

// Remove the entry from tail
// `NULL` is returned if the list is empty
LinkedListEntry* LinkedListRemoveTail(LinkedListEntry* head);

// Get the pointer to the struct `type` from a pointer to `LinkedListEntry`
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define ContainerOf(ptr, type, member) \
  ((type*)((char*)(ptr) - ((ptrdiff_t)&(((type*)0)->member))))

// Get the pointer to the struct `type` from `LinkedListEntry` pointing to
// the first entry
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define LinkedListDataHead(head, type, member) \
  ContainerOf((head)->next_, type, member)

// Get the pointer to the struct `type` from `LinkedListEntry` pointing to
// the last entry
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define LinkedListDataTail(head, type, member) \
  ContainerOf((head)->prev_, type, member)

// Get the pointer to the next entry from a pointer to the struct `type`
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define LinkedListDataNext(iter, type, member) \
  ContainerOf((iter)->member.next_, type, member)

// Get the pointer to the previous entry from a pointer to the struct `type`
// The struct `type` contains `LinkedListEntry` as a member named `member`
#define LinkedListDataPrev(iter, type, member) \
  ContainerOf((iter)->member.prev_, type, member)

// Iterate the data in the doubly-linked list
// `iter` is a pointer to the struct `type`
// `head` is a pointer to `LinkedListEntry`
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define LinkedListForEach(iter, head, type, member) \
  for ((iter) = LinkedListDataHead((head), type, member); \
       &(iter)->member != (head); \
       (iter) = LinkedListDataNext((iter), type, member))

// Iterate the data in the doubly-linked list in reverse
// `iter` is a pointer to the struct `type`
// `head` is a pointer to `LinkedListEntry`
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define LinkedListForEachReverse(iter, head, type, member) \
  for ((iter) = LinkedListDataTail((head), type, member); \
       &(iter)->member != (head); \
       (iter) = LinkedListDataPrev((iter), type, member))

// Iterate the data in the doubly-linked list
// The current data can be freed
// `iter` is a pointer to the struct `type`
// `iter_next` is a pointer to the struct `type`
// `head` is a pointer to `LinkedListEntry`
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define LinkedListForEachSafe(iter, iter_next, head, type, member) \
  for ((iter) = LinkedListDataHead((head), type, member), \
       (iter_next) = LinkedListDataNext((iter), type, member); \
       &(iter)->member != (head); \
       (iter) = (iter_next), \
       (iter_next) = LinkedListDataNext((iter_next), type, member))

// Iterate the data in the doubly-linked list in reverse
// The current data can be freed
// `iter` is a pointer to the struct `type`
// `iter_prev` is a pointer to the struct `type`
// `head` is a pointer to `LinkedListEntry`
// The struct `type` should contain `LinkedListEntry` as a member named `member`
#define LinkedListForEachPrev(iter, iter_prev, head, type, member) \
  for ((iter) = LinkedListDataTail((head), type, member), \
       (iter_prev) = LinkedListDataPrev((iter), type, member); \
       &(iter)->member != (head); \
       (iter) = (iter_prev), \
       (iter_prev) = LinkedListDataPrev((iter_prev), type, member))

#ifdef __cplusplus
}
#endif

#endif // LINKED_LIST_H
