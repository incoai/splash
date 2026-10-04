#pragma once

#include <unistd.h>

#include <cerrno>
#include <string>
#include <string_view>

namespace splash {

// The server and this runtime write to the same stderr. Each line goes out
// in one write, newline included, so that lines written at once stay whole.
inline void writeStderrLine(std::string_view text) noexcept {
  try {
    std::string line(text);
    line += '\n';
    for (std::string_view rest = line; !rest.empty();) {
      const ssize_t written = ::write(STDERR_FILENO, rest.data(), rest.size());
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        return;
      rest.remove_prefix(static_cast<size_t>(written));
    }
  } catch (...) {
    // Diagnostics must not affect startup or serving.
  }
}

} // namespace splash
