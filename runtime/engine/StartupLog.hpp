#pragma once

#include "StderrLine.hpp"

#include <ctime>
#include <sstream>
#include <string>
#include <string_view>

namespace splash::engine {

template <typename... Parts>
void logStartup(const Parts &...parts) noexcept {
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
    // Native stderr is inherited by serve. Keep each optional startup notice
    // bounded and on one line, including messages from caught exceptions.
    for (unsigned char character : std::string_view(message).substr(0, 768))
      line << (character < 32 || character == 127 ? ' ' : char(character));
    if (message.size() > 768) line << "...";
    writeStderrLine(line.str());
  } catch (...) {
    // Optional diagnostics must not affect startup or serving.
  }
}

} // namespace splash::engine
