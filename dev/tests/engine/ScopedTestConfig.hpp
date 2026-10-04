#pragma once

#include "TestConfig.hpp"

#include <utility>

namespace splash::test {

// Puts config in place of the test configuration until it goes out of scope,
// then restores the one before. Construct it before the component that reads
// it.
class ScopedTestConfig final {
public:
  explicit ScopedTestConfig(TestConfig config)
      : previous_(std::exchange(detail::testConfigStorage(), std::move(config))) {}
  ~ScopedTestConfig() { detail::testConfigStorage() = std::move(previous_); }
  ScopedTestConfig(const ScopedTestConfig &) = delete;
  ScopedTestConfig &operator=(const ScopedTestConfig &) = delete;

private:
  TestConfig previous_;
};

} // namespace splash::test
