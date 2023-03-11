
// util.c

#include "my_nn_lib/logger.h"
#include "my_nn_lib/util.h"

#include <float.h>
#include <math.h>
#include <stdio.h>

// Checked version of `malloc()`
// Exits the program if memory allocation fails
void* xmalloc(size_t size)
{
  void* p = malloc(size);

  if (p == NULL) {
    LogError("Memory allocation failed");
    exit(EXIT_FAILURE);
  }

  return p;
}

// Checked version of `calloc()`
// Exits the program if memory allocation fails
void* xcalloc(size_t num, size_t size)
{
  void* p = calloc(num, size);

  if (p == NULL) {
    LogError("Memory allocation failed");
    exit(EXIT_FAILURE);
  }

  return p;
}

// Checked version of `realloc()`
// Exits the program if memory allocation fails
void* xrealloc(void* ptr, size_t size)
{
  void* p = realloc(ptr, size);

  if (p == NULL && (size > 0 || ptr == NULL)) {
    LogError("Memory allocation failed");
    exit(EXIT_FAILURE);
  }

  return p;
}

// Allocate and return a formatted string
// The returned string should be freed by the user
// `NULL` is returned in case of failure
char* AllocateFormatString(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  const int len = vsnprintf(NULL, 0, fmt, args);
  va_end(args);

  char* str = malloc(sizeof(char) * (len + 1));

  if (str == NULL)
    return NULL;

  va_start(args, fmt);
  vsnprintf(str, len + 1, fmt, args);
  va_end(args);

  return str;
}

// Generate a random number from a uniform distribution [0, 1]
float GenerateUniformFloat()
{
  return (float)rand() / (float)RAND_MAX;
}

// Generate a random number from a normal distribution
float GenerateNormalFloat()
{
  const float x = (float)rand() / (float)RAND_MAX;
  const float y = (float)rand() / (float)RAND_MAX;
  const float x0 = x == 0.0f ? FLT_EPSILON : x;
  const float z = sqrtf(-2.0f * logf(x0)) * cosf(2.0f * (float)M_PI * y);
  return z;
}

// Create a sequence with increasing values
void IotaI32(int* values,
             const int start,
             const int num_values)
{
  for (int i = 0, v = start; i < num_values; ++i, ++v) {
    values[i] = v;
  }
}

// Create a random permutation
void RandomPermutationI32(int* values,
                          const int num_values)
{
  for (int i = num_values - 1; i >= 0; --i) {
    const int j = rand() % (i + 1);

    const int x = values[i];
    values[i] = values[j];
    values[j] = x;
  }
}

// Bit cast the 32-bit signed integer to the unsigned integer
uint32_t BitCastI32ToU32(const int32_t val)
{
  union
  {
    int32_t  from_;
    uint32_t to_;
  } conv;

  conv.from_ = val;
  return conv.to_;
}

// Bit cast the 32-bit unsigned integer to the signed integer
int32_t BitCastU32ToI32(const uint32_t val)
{
  union
  {
    uint32_t from_;
    int32_t  to_;
  } conv;

  conv.from_ = val;
  return conv.to_;
}

// Swap the bytes in a 16-bit unsigned integer
uint16_t SwapBytesU16(const uint16_t val)
{
  const uint16_t val0 = (val & 0x00FF) << 8;
  const uint16_t val1 = (val & 0xFF00) >> 8;

  return val0 | val1;
}

// Swap the bytes in a 32-bit unsigned integer
uint32_t SwapBytesU32(const uint32_t val)
{
  const uint32_t val0 = (val & 0x000000FF) << 24;
  const uint32_t val1 = (val & 0x0000FF00) << 8;
  const uint32_t val2 = (val & 0x00FF0000) >> 8;
  const uint32_t val3 = (val & 0xFF000000) >> 24;

  return val0 | val1 | val2 | val3;
}

// Swap the endianness for a 32-bit unsigned integer
uint32_t SwapEndianU32(const uint32_t val)
{
  return SwapBytesU32(val);
}

// Swap the endianness for a 32-bit signed integer
int32_t SwapEndianI32(const int32_t val)
{
  return BitCastU32ToI32(SwapBytesU32(BitCastI32ToU32(val)));
}
