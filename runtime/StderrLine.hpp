#pragma once

#include <unistd.h>

#include <cerrno>
#include <ctime>
#include <sstream>
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

// A notice the runtime gives while it starts, serves or stops, as the server
// prints its own (server/diagnostics.py print_status): the local time, then
// the parts on one line, bounded, with control characters, such as those of
// a caught exception's message, as spaces. An error that ends the process
// is written as an "error: ..." line instead (main.mm).
template <typename... Parts> void logLine(const Parts &...parts) noexcept {
  try {
    std::ostringstream text;
    (text << ... << parts);
    const std::string message = text.str();
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    char timestamp[9] = "--:--:--";
    if (localtime_r(&now, &local))
      std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", &local);
    std::ostringstream line;
    line << timestamp << ' ';
    for (unsigned char character : std::string_view(message).substr(0, 768))
      line << (character < 32 || character == 127 ? ' ' : char(character));
    if (message.size() > 768) line << "...";
    writeStderrLine(line.str());
  } catch (...) {
    // Diagnostics must not affect startup or serving.
  }
}

// A notice of a fault the runtime goes on serving past, in the form of the
// server's warnings (server/server.py): "Warning · " and the parts.
template <typename... Parts> void logWarning(const Parts &...parts) noexcept {
  logLine("Warning · ", parts...);
}

} // namespace splash
