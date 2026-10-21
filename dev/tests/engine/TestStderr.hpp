#pragma once

// What the code under test writes to stderr, where the runtime's notices go.

#include "TestChecks.hpp"

#include <unistd.h>

#include <cstdio>
#include <functional>
#include <sstream>
#include <string>

namespace splash::test {

// What `write` puts on stderr. Stderr comes back however `write` ends, so a
// check that fails inside it still reports.
inline std::string capturedStderr(const std::function<void()> &write) {
  std::FILE *log = std::tmpfile();
  require(log != nullptr, "no temporary file");
  const int saved = ::dup(STDERR_FILENO);
  ::dup2(::fileno(log), STDERR_FILENO);
  const auto restore = [saved] {
    ::dup2(saved, STDERR_FILENO);
    ::close(saved);
  };
  try {
    write();
  } catch (...) {
    restore();
    std::fclose(log);
    throw;
  }
  restore();
  std::rewind(log);
  std::ostringstream text;
  for (int character; (character = std::fgetc(log)) != EOF;)
    text.put(static_cast<char>(character));
  std::fclose(log);
  return text.str();
}

} // namespace splash::test
