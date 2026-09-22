#include "dev/benchmarks/indirect_dispatch_backend/IndirectDispatchPolicy.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
uint64_t checks = 0;
void require(bool value, std::string_view reason) {
  ++checks;
  if (!value) throw std::runtime_error(std::string(reason));
}
} // namespace

int main() {
  using namespace splash::metal::private_indirect;
  try {
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    struct ArgumentsCase final { uint64_t length, offset; bool valid; };
    constexpr ArgumentsCase arguments[] = {
        {0, 0, false}, {1, 0, false}, {11, 0, false},
        {12, 0, true}, {13, 0, true},
        {12, 1, false}, {16, 1, false}, {16, 2, false}, {16, 3, false},
        {15, 4, false}, {16, 4, true}, {17, 4, true},
        {23, 12, false}, {24, 12, true}, {25, 12, true},
        {24, 16, false}, {24, 24, false}, {24, 28, false},
        {maximum, 0, true}, {maximum, maximum, false},
        {maximum, maximum - 3, false}, {maximum, maximum - 11, false},
        {maximum, maximum - 15, true},
        {maximum - 3, maximum - 15, true},
        {maximum - 4, maximum - 15, false},
        {maximum - 3, maximum - 14, false},
        {12, maximum - 3, false},
    };
    for (const auto test : arguments)
      require(validArgumentsRange(test.length, test.offset) == test.valid,
              "indirect argument alignment or twelve-byte extent differs");

    // Exhaust all offset alignments around an ordinary view. The expected
    // valid starts are the explicit four-byte slots with three uint32 words.
    for (uint64_t offset = 0; offset < 40; ++offset)
      require(validArgumentsRange(32, offset) ==
                  (offset == 0 || offset == 4 || offset == 8 || offset == 12 ||
                   offset == 16 || offset == 20),
              "ordinary indirect argument slot accepted an invalid offset");

    struct ThreadsCase final { uint64_t x, y, z; bool valid; };
    constexpr ThreadsCase threads[] = {
        {0, 0, 0, false}, {0, 1, 1, false}, {1, 0, 1, false}, {1, 1, 0, false},
        {0, maximum, maximum, false}, {maximum, 0, maximum, false},
        {maximum, maximum, 0, false},
        {1, 1, 1, true}, {256, 1, 1, true}, {8, 4, 2, true},
        {maximum, 1, 1, true}, {1, maximum, 1, true}, {1, 1, maximum, true},
        {maximum / 2, 2, 1, true}, {maximum / 2 + 1, 2, 1, false},
        {2, maximum / 2, 1, true}, {2, maximum / 2 + 1, 1, false},
        {1, 2, maximum / 2, true}, {1, 2, maximum / 2 + 1, false},
        {maximum / 3, 3, 1, true}, {maximum / 3 + 1, 3, 1, false},
        {2, 3, maximum / 6, true}, {2, 3, maximum / 6 + 1, false},
        {maximum, 2, 1, false}, {maximum, 1, 2, false},
        {1, maximum, 2, false}, {maximum, maximum, maximum, false},
    };
    for (const auto test : threads) {
      require(validThreads(test.x, test.y, test.z) == test.valid,
              "thread dimensions accepted zero or overflowing product");
      require(validThreads(test.z, test.y, test.x) == test.valid,
              "thread product validation depends on dimension order");
    }

    struct Range final { uintptr_t identity; uint64_t offset, length; };
    struct OverlapCase final { Range left, right; bool overlaps; };
    constexpr OverlapCase ranges[] = {
        {{1, 0, 12}, {1, 0, 12}, true},
        {{1, 0, 12}, {1, 4, 12}, true},
        {{1, 0, 12}, {1, 11, 1}, true},
        {{1, 0, 12}, {1, 12, 1}, false},
        {{1, 0, 12}, {1, 13, 12}, false},
        {{1, 4, 4}, {1, 0, 16}, true},
        {{1, 4, 4}, {1, 8, 8}, false},
        {{1, 0, 12}, {2, 0, 12}, false},
        {{1, 4, 12}, {2, 8, 12}, false},
        {{1, maximum - 12, 12}, {1, maximum - 1, 1}, true},
        {{1, maximum - 12, 12}, {1, maximum - 13, 1}, false},
        {{1, maximum - 12, 12}, {2, maximum - 12, 12}, false},
        // Invalid metadata conservatively rejects before native identity
        // comparison. Empty/null/overflowing ranges never bypass the guard.
        {{0, 0, 12}, {1, 0, 12}, true},
        {{0, 0, 12}, {0, 0, 12}, true},
        {{1, 0, 0}, {1, 0, 12}, true},
        {{1, 0, 0}, {2, 100, 12}, true},
        {{1, 0, 12}, {2, 100, 0}, true},
        {{1, maximum, 1}, {1, 0, 12}, true},
        {{1, maximum, 1}, {2, 0, 12}, true},
        {{1, maximum - 11, 12}, {2, 0, 12}, true},
        {{1, 0, maximum}, {1, maximum - 1, 1}, true},
        {{1, 1, maximum}, {2, 0, 12}, true},
    };
    for (const auto test : ranges) {
      require(overlap(test.left.identity, test.left.offset, test.left.length,
                      test.right.identity, test.right.offset, test.right.length) == test.overlaps,
              "native-base half-open overlap or malformed-range rejection differs");
      require(overlap(test.right.identity, test.right.offset, test.right.length,
                      test.left.identity, test.left.offset, test.left.length) == test.overlaps,
              "metadata overlap is not symmetric");
    }

    std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":" << checks << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL private indirect dispatch policy: " << error.what() << '\n';
    return 1;
  }
}
