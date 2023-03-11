
// util.h

#ifndef UTIL_H
#define UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>

#include "my_nn_lib/logger.h"

// Checked version of `malloc()`
// Exits the program if memory allocation fails
void* xmalloc(size_t size);

// Checked version of `calloc()`
// Exits the program if memory allocation fails
void* xcalloc(size_t num, size_t size);

// Checked version of `realloc()`
// Exits the program if memory allocation fails
void* xrealloc(void* ptr, size_t size);

// Allocate and return a formatted string
// The returned string should be freed by the user
// `NULL` is returned in case of failure
char* AllocateFormatString(const char* fmt, ...);

// Generate a random number from a uniform distribution [0, 1]
float GenerateUniformFloat();

// Generate a random number from a normal distribution
float GenerateNormalFloat();

// Create a sequence with increasing values
void IotaI32(int* values,
             const int start,
             const int num_values);

// Create a random permutation
void RandomPermutationI32(int* values,
                          const int num_values);

// Bit cast the 32-bit signed integer to the unsigned integer
uint32_t BitCastI32ToU32(const int32_t val);

// Bit cast the 32-bit unsigned integer to the signed integer
int32_t BitCastU32ToI32(const uint32_t val);

// Swap the bytes in a 16-bit unsigned integer
uint16_t SwapBytesU16(const uint16_t val);

// Swap the bytes in a 32-bit unsigned integer
uint32_t SwapBytesU32(const uint32_t val);

// Swap the endianness for a 32-bit unsigned integer
uint32_t SwapEndianU32(const uint32_t val);

// Swap the endianness for a 32-bit signed integer
int32_t SwapEndianI32(const int32_t val);

// Max and min macros
#if defined(__GNUC__) && !defined(__clang__)
// Use the safer version in GCC
#define Max(a, b) ({ \
  typeof(a) a_ = (a); \
  typeof(b) b_ = (b); \
  a_ > b_ ? a_ : b_; })
#define Min(a, b) ({ \
  typeof(a) a_ = (a); \
  typeof(b) b_ = (b); \
  a_ < b_ ? a_ : b_; })
#define Clamp(x, a, b) ({ \
  typeof(x) x_ = (x); \
  typeof(a) a_ = (a); \
  typeof(b) b_ = (b); \
  x_ < a_ ? a_ : (x_ > b_ ? b_ : x_); })
#else
#define Max(a, b)       ((a) > (b) ? (a) : (b))
#define Min(a, b)       ((a) < (b) ? (a) : (b))
#define Clamp(x, a, b)  ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#endif

// Assert macro with the custom message
#define Assert(predicate, ...) \
  do { \
    if (!(predicate)) { \
      LogError(__VA_ARGS__); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// Assert macro only enabled in Debug mode
#ifdef DEBUG
#define DebugAssert(predicate, ...) Assert(predicate, __VA_ARGS__)
#else
#define DebugAssert(predicate, ...)
#endif

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
