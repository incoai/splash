#pragma once

// Checks that throw a message naming what failed.

#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::test {

inline void require(bool value, std::string_view message) {
  if (!value) throw std::runtime_error(std::string(message));
}

// run must fail with an error whose message contains expected, so that an
// unrelated failure does not count as the rejection.
template <class F> void rejects(F run, std::string_view expected, const std::string &message) {
  try {
    run();
  } catch (const std::exception &error) {
    if (std::string_view(error.what()).find(expected) != std::string_view::npos) return;
    throw std::runtime_error(message + ": failed with \"" + error.what() + "\", not \"" + std::string(expected) +
                             "\"");
  }
  throw std::runtime_error(message);
}

} // namespace splash::test
