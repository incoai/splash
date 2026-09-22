// Independent pure CPU regime, flag and tail audit. No Metal backend creation
// or submission occurs; only inline production-header policy helpers are used.
#include "flash/FlashSharedExpertFused.hpp"
#include "metal/abi/FlashSharedExpertFused.h"

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
uint64_t checks = 0;
void require(bool value, const char *reason) {
  ++checks;
  if (!value) throw std::runtime_error(reason);
}

void strictFlagTests() {
  using namespace splash::flash;
  const auto absent = flashSharedExpertFusedFlag(nullptr);
  require(!absent, "absent opt-in flag must default false");
  const auto off = flashSharedExpertFusedFlag("0");
  const auto on = flashSharedExpertFusedFlag("1");
  require(!off, "literal 0 must disable fusion");
  require(on, "literal 1 must enable fusion");
  const std::array invalid{
      "", "00", "01", "10", "11", "2", "-0", "+0", "-1", "+1",
      "0x0", "0x1", "0X01", "0.0", "1.0", "1e0", "true", "false",
      "True", "False", "TRUE", "FALSE", "yes", "no", "on", "off",
      " 0", " 1", "0 ", "1 ", "\t1", "1\t", "\n1", "1\n", "\r1", "1\r",
      " ", "_", "1_", "nan", "NaN", "inf", "Infinity", "-2147483648",
      "4294967295", "18446744073709551616", "\xef\xbc\x91", "\xc2\xa0"};
  for (const char *value : invalid) {
    bool rejected = false;
    try { (void)flashSharedExpertFusedFlag(value); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "noncanonical boolean flag was accepted");
  }
}

void eligibilityTests() {
  using namespace splash::flash;
  // Scope and dependency are checked separately from arithmetic decomposition.
  for (uint32_t rows = 0; rows <= 8193; ++rows) {
    require(!flashSharedExpertFusedEligible(false, false, rows), "disabled route was eligible");
    require(!flashSharedExpertFusedEligible(false, true, rows), "disabled cached route was eligible");
    require(!flashSharedExpertFusedEligible(true, false, rows), "route without immutable BF16 operands was eligible");
    const bool inScope = rows >= 256 && rows <= 8192;
    require(flashSharedExpertFusedEligible(true, true, rows) == inScope,
            "prefill eligibility changed at a row boundary");
  }
  for (uint32_t rows : {65536u, 1u << 31, std::numeric_limits<uint32_t>::max()})
    require(!flashSharedExpertFusedEligible(true, true, rows), "oversized physical row extent was accepted");
}

void tailCoverageTests() {
  using namespace splash::flash;
  for (uint32_t rows = 256; rows <= 8192; ++rows) {
    const auto plan = flashSharedExpertFusedRows(rows);
    require(plan.fullRows <= rows && plan.fullRows % 32 == 0,
            "main fusion rows must cover complete M32 tiles");
    require(plan.tailWholeRows == 0 || plan.tailWholeRows == 16,
            "baseline M16 tail must contain exactly zero or one whole tile");
    require(plan.tailVectorRows < 16, "vector tail must remain below the canonical M16 boundary");
    const uint64_t covered = uint64_t{plan.fullRows} + plan.tailWholeRows + plan.tailVectorRows;
    require(covered == rows, "row decomposition loses or duplicates real rows");
    require(uint64_t{plan.tailWholeRows} + plan.tailVectorRows < 32,
            "main fusion must use the largest valid M32 prefix");
    // This independently catches the previous M32 fallback pitfall: rows with
    // a 16..31 remainder must not send that whole remainder to vector dots.
    const uint32_t remainder = rows - plan.fullRows;
    require((remainder >= 16) == (plan.tailWholeRows == 16),
            "16..31 real tail rows lost their canonical whole-K M16 matmul");
    if (remainder) {
      const uint64_t inputByteBegin = uint64_t{plan.fullRows} * 2560 * 2;
      const uint64_t inputByteEnd = inputByteBegin + uint64_t{remainder} * 2560 * 2;
      const uint64_t outputByteBegin = uint64_t{plan.fullRows} * 640 * 2;
      const uint64_t outputByteEnd = outputByteBegin + uint64_t{remainder} * 640 * 2;
      require(inputByteEnd == uint64_t{rows} * 2560 * 2, "tail input view is not exactly adjacent to main prefix");
      require(outputByteEnd == uint64_t{rows} * 640 * 2, "tail output view is not exactly adjacent to main prefix");
    }
  }
  struct Expected { uint32_t rows, full, whole, vector; };
  const std::array boundaries{
      Expected{256, 256, 0, 0}, Expected{257, 256, 0, 1}, Expected{271, 256, 0, 15},
      Expected{272, 256, 16, 0}, Expected{273, 256, 16, 1}, Expected{287, 256, 16, 15},
      Expected{288, 288, 0, 0}, Expected{304, 288, 16, 0}, Expected{511, 480, 16, 15},
      Expected{512, 512, 0, 0}, Expected{2047, 2016, 16, 15}, Expected{2048, 2048, 0, 0},
      Expected{8191, 8160, 16, 15}, Expected{8192, 8192, 0, 0}};
  for (const auto &expected : boundaries) {
    const auto actual = flashSharedExpertFusedRows(expected.rows);
    require(actual.fullRows == expected.full && actual.tailWholeRows == expected.whole &&
        actual.tailVectorRows == expected.vector, "handwritten M32/M16/vector decomposition differs");
  }
}

void invalidDecompositionTests() {
  using namespace splash::flash;
  for (uint32_t rows = 0; rows < 256; ++rows) {
    bool rejected = false;
    try { (void)flashSharedExpertFusedRows(rows); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "undersized decomposition bypassed prefill scope");
  }
  // Probe the unsigned endpoints for strict rejection without arithmetic UB.
  for (uint32_t rows : {0u, 1u, 15u, 16u, 31u, 32u, 8193u, 65536u,
                       (1u << 31) - 1, 1u << 31, std::numeric_limits<uint32_t>::max()}) {
    bool rejected = false;
    try { (void)flashSharedExpertFusedRows(rows); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "out-of-scope decomposition was accepted");
  }
}

void producerIdentityAndAbiTests() {
  using namespace splash::flash;
  const std::string_view identity = kFlashSharedExpertFusedSemantics;
  for (std::string_view contract : {"whole-k", "f32accum", "bf16-dots", "compiled-bf16-swiglu",
                                    "m32n128", "min256", "tail-m16n64"})
    require(identity.find(contract) != std::string_view::npos,
            "producer identity omits a material numerical or routing contract");
  const FlashSharedExpertFusedParams fused{2048, 2560, 640, 0, 640, 32, 128, 0};
  const auto serialized = std::bit_cast<std::array<uint32_t, 8>>(fused);
  const std::array<uint32_t, 8> expected{2048, 2560, 640, 0, 640, 32, 128, 0};
  require(serialized == expected, "fused host parameter ABI field ordering differs");
  const auto dense = std::bit_cast<FlashDenseCacheParams>(fused);
  require(dense.rows == 2048 && dense.input_size == 2560 && dense.output_size == 640 &&
      dense.output_begin == 0 && dense.output_count == 640 && dense.tile_rows == 32 &&
      dense.tile_outputs == 128 && dense.reserved == 0,
      "fused ABI no longer preserves qualified dense descriptor field ordering");
}
} // namespace

int main() {
  try {
    strictFlagTests(); eligibilityTests(); tailCoverageTests(); invalidDecompositionTests();
    producerIdentityAndAbiTests();
    std::cout << "PASS shared-expert strict flag, scope/dependency and canonical M32/M16/vector tails: "
              << checks << " checks; GPU work=false\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
