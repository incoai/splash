#include "video/h3/engine/Status.hpp"

#include <cstdlib>

namespace h3 {

Status error(Code code, const char *format, ...) {
  char text[1024];
  va_list args;
  va_start(args, format);
  vsnprintf(text, sizeof text, format, args);
  va_end(args);
  return Status(code, text);
}

Logger::Logger() {
  if (const char *value = std::getenv("H3_LOG")) level_ = std::atoi(value);
}

void Logger::info(const char *format, ...) const {
  if (level_ < 1) return;
  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
  fputc('\n', stdout);
  fflush(stdout);
}

void Logger::debug(const char *format, ...) const {
  if (level_ < 2) return;
  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
  fputc('\n', stdout);
  fflush(stdout);
}

} // namespace h3
