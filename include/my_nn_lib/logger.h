
// logger.h

#ifndef LOGGER_H
#define LOGGER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

enum LogLevel
{
  LOG_LEVEL_INFO,
  LOG_LEVEL_WARNING,
  LOG_LEVEL_ERROR,
};

void LogWrite(const int level,
              const char* file,
              const int line,
              const char* fmt, ...);

#define LogInfo(...) LogWrite(LOG_LEVEL_INFO, \
  __FILE__, __LINE__, __VA_ARGS__)
#define LogWarning(...) LogWrite(LOG_LEVEL_WARNING, \
  __FILE__, __LINE__, __VA_ARGS__)
#define LogError(...) LogWrite(LOG_LEVEL_ERROR, \
  __FILE__, __LINE__, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // LOGGER_H
