#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

namespace h3 {

// Error categories; the C ABI returns them as h3_status (include/h3/h3.h).
enum class Code : int32_t {
  Ok = 0,
  Argument = 1,  // request or inputs inconsistent
  Io = 2,        // file missing, unreadable or malformed
  Metal = 3,     // device, library, pipeline or allocation failure
  Gpu = 4,       // a command buffer reported an error
  State = 5,     // phase called out of order
};

class [[nodiscard]] Status {
public:
  Status() = default;
  Status(Code code, std::string message) : code_(code), message_(std::move(message)) {}

  [[nodiscard]] bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] Code code() const noexcept { return code_; }
  [[nodiscard]] const std::string &message() const noexcept { return message_; }

private:
  Code code_ = Code::Ok;
  std::string message_;
};

[[nodiscard]] Status error(Code code, const char *format, ...) __attribute__((format(printf, 2, 3)));

#define H3_TRY(expr)                    \
  do {                                  \
    ::h3::Status h3_try_status = (expr); \
    if (!h3_try_status.ok()) return h3_try_status; \
  } while (0)

// Log level from H3_LOG (0 quiet, 1 info, 2 debug; default 1), read once per engine.
class Logger {
public:
  Logger();
  [[nodiscard]] int level() const noexcept { return level_; }
  void info(const char *format, ...) const __attribute__((format(printf, 2, 3)));
  void debug(const char *format, ...) const __attribute__((format(printf, 2, 3)));

private:
  int level_ = 1;
};

} // namespace h3
