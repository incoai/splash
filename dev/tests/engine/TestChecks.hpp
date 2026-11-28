#pragma once

// Checks that throw a message naming what failed.

#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::test {

inline void require(bool value, std::string_view message) {
  if (!value) throw std::runtime_error(std::string(message));
}

// How many times slower this build runs than the tests' own (-O2, no
// sanitizer), as measured on the cache's suites on m64 at rc/1.7.4:
// ThreadSanitizer ran persistent-cache in 9.0 s against 0.3 s and
// engine-cache in 46 s against 1.7 s; AddressSanitizer with UBSan in 1.4 s
// and 6.8 s.
inline constexpr double kBuildSlowdown =
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
    30.0;
#elif __has_feature(address_sanitizer)
    5.0;
#else
    1.0;
#endif
#else
    1.0;
#endif

// A wall-clock bound set in the tests' own build, as this build holds it: a
// sanitizer build scales time bounds by its measured slowdown
// (kBuildSlowdown), never by a bound of a test's own.
inline constexpr double timeBound(double milliseconds) noexcept { return milliseconds * kBuildSlowdown; }

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
