
// logger.c

#include "my_nn_lib/logger.h"

#include <stdio.h>

// Color codes
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_YELLOW  "\x1b[33m"
#define COLOR_BLUE    "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN    "\x1b[36m"
#define COLOR_RESET   "\x1b[0m"

// `kLevelNames` should sync with `enum LogLevel`
static const char* kLevelNames[] = {
  "Info",
  "Warning",
  "Error",
};

// `kLevelColors` should sync with `enum LogLevel`
static const char* kLevelColors[] = {
  COLOR_GREEN,
  COLOR_YELLOW,
  COLOR_RED,
};

void LogWrite(const int level,
              const char* file,
              const int line,
              const char* fmt, ...)
{
  fprintf(stderr, "%s%s%s: %s:%d: ",
          kLevelColors[level], kLevelNames[level], COLOR_RESET, file, line);

  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);

  fprintf(stderr, "\n");
}
