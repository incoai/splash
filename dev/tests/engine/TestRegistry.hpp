#pragma once

// Named tests: each test file of a binary registers its tests as a group, and
// the binary's main runs, selects or lists them by name.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::test {

struct TestCase final {
  std::string_view name;
  void (*run)();   // a captureless lambda for a parameterized test
};
struct TestGroup final {
  std::string_view name;           // the file's area: "requests", "memory", ...
  std::span<const TestCase> tests;
};

// Runs every test whose "group/name" contains one of --filter's comma-separated substrings (all without
// --filter), each in its own try after beforeEach (which resets process-wide test state), printing
// "FAIL group/name: what()" for each failure and "<program>: P passed, F failed" at the end. --list prints
// "group/name" one per line. EXIT_SUCCESS only when at least one test ran and none failed.
[[nodiscard]] inline int runTests(std::string_view program, std::span<const TestGroup> groups, int argc, char **argv,
                                  void (*beforeEach)() = nullptr) {
  bool list = false;
  std::vector<std::string_view> filters;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--list") {
      list = true;
    } else if (argument == "--filter" && index + 1 < argc) {
      for (std::string_view substrings = argv[++index];;) {
        const size_t comma = substrings.find(',');
        filters.push_back(substrings.substr(0, comma));
        if (comma == std::string_view::npos)
          break;
        substrings.remove_prefix(comma + 1);
      }
    } else {
      std::cerr << "usage: " << program << " [--list] [--filter substring[,substring...]]\n";
      return EXIT_FAILURE;
    }
  }
  uint32_t passed = 0, failed = 0;
  for (const TestGroup &group : groups) {
    for (const TestCase &test : group.tests) {
      const std::string name = std::string(group.name) + '/' + std::string(test.name);
      if (!filters.empty() && std::ranges::none_of(filters, [&](std::string_view filter) {
            return name.find(filter) != std::string::npos;
          }))
        continue;
      if (list) {
        std::cout << name << '\n';
        continue;
      }
      try {
        if (beforeEach)
          beforeEach();
        test.run();
        ++passed;
      } catch (const std::exception &error) {
        ++failed;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
      }
    }
  }
  if (list)
    return EXIT_SUCCESS;
  std::cout << program << ": " << passed << " passed, " << failed << " failed\n";
  return passed && !failed ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace splash::test
