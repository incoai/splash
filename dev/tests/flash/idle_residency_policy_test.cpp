#include "flash/FlashIdleResidencyPolicy.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace policy = splash::flash::idle_maintenance;
namespace {
uint64_t checks = 0;
void require(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejected(F &&function, const char *message) {
  bool invalid = false;
  try { function(); }
  catch (const std::invalid_argument &) { invalid = true; }
  require(invalid, message);
}
}

int main() {
  try {
    require(!policy::parseSwitch(nullptr), "unset must remain off");
    require(!policy::parseSwitch("0"), "zero must remain off");
    require(policy::parseSwitch("1"), "one must enable");
    for (const char *bad : {"", "00", "01", "10", "-1", "2", "true", "false",
                            " 1", "1 ", "1\n", "+1", "yes", "on"})
      rejected([&] { (void)policy::parseSwitch(bad); }, "non-strict switch accepted");

    require(policy::parseInterval(nullptr) == 500, "unset interval changed");
    for (uint64_t interval = 100; interval <= 1000; ++interval) {
      const std::string text = std::to_string(interval);
      require(policy::parseInterval(text.c_str()) == interval,
              "canonical interval rejected");
    }
    for (const char *bad : {"", "0", "99", "1001", "10000", "0100", "0500",
                            "000500", "-500", "+500", "500.0", "5e2", " 500",
                            "500 ", "500\n", "true", "500ms", "18446744073709551615"})
      rejected([&] { (void)policy::parseInterval(bad); }, "invalid interval accepted");

    for (uint32_t bits = 0; bits != 512; ++bits) {
      std::array<bool, 9> b{};
      for (uint32_t i = 0; i != b.size(); ++i) b[i] = (bits >> i) & 1;
      const bool actual = policy::eligible(b[0], b[1], b[2], b[3], b[4], b[5],
                                           b[6], b[7], b[8]);
      const bool expected = bits == (1U | 2U | 4U | 8U | 128U | 256U);
      require(actual == expected, "eligibility truth table differs");
    }
    require(!policy::eligible(true, false, true, false, false, false, false,
                              true, true), "mask-blocked live request accepted");
    require(!policy::eligible(false, true, true, true, false, false, false,
                              true, true), "startup unarmed accepted");

    constexpr uint64_t reserve = 16ULL << 30;
    constexpr uint64_t threshold = reserve + policy::kMarginBytes;
    for (uint32_t bits = 0; bits != 16; ++bits) {
      const bool valid = bits & 1, growth = bits & 2,
          effective = bits & 4, system = bits & 8;
      for (uint64_t available : {uint64_t{0}, reserve, threshold - 1, threshold,
                                 threshold + 1, std::numeric_limits<uint64_t>::max()})
        require(policy::hostAllowed(valid, growth, effective, system, available, reserve)
                    == (bits == 15 && available > threshold),
                "pressure/margin truth table differs");
    }
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    require(!policy::hostAllowed(true, true, true, true, maximum,
                                maximum - policy::kMarginBytes),
            "equal UINT64_MAX threshold accepted");
    require(policy::hostAllowed(true, true, true, true, maximum,
                               maximum - policy::kMarginBytes - 1),
            "safe near-UINT64_MAX margin rejected");
    for (uint64_t nearOverflow : {maximum - policy::kMarginBytes + 1,
                                  maximum - 1, maximum})
      require(!policy::hostAllowed(true, true, true, true, maximum, nearOverflow),
              "overflow threshold accepted");
    require(policy::hostAllowed(true, true, true, true, policy::kMarginBytes + 1, 0),
            "zero-reserve arithmetic changed");

    const auto geometry = [](std::string_view source, std::string_view layout,
        uint64_t count, uint64_t bytes, uint64_t owners, uint64_t ownerBytes,
        uint32_t layers, uint32_t experts, uint32_t hidden) {
      policy::validateGeometry(source, layout, count, bytes, owners, ownerBytes,
                               layers, experts, hidden);
    };
    geometry(policy::kSource, policy::kLayout, policy::kOriginalBases,
             policy::kOriginalBytes, policy::kOwnerCount, policy::kOwnerBytes,
             48, 512, 2560);
    require(true, "qualified geometry unexpectedly rejected");
    require(policy::qualifiedUnionAvailable(policy::kSource, policy::kLayout,
        policy::kOriginalBases, policy::kOriginalBytes, policy::kOwnerCount,
        policy::kOwnerBytes, 48, 512, 2560), "qualified complete union unavailable");
    for (const uint64_t owners : {uint64_t{0}, policy::kOriginalBases,
                                  policy::kOwnerCount - 1, policy::kOwnerCount + 1})
      require(!policy::qualifiedUnionAvailable(policy::kSource, policy::kLayout,
          policy::kOriginalBases, policy::kOriginalBytes, owners,
          policy::kOwnerBytes, 48, 512, 2560), "missing optional owner union did not skip");
    for (const uint64_t bytes : {uint64_t{0}, policy::kOriginalBytes,
                                 policy::kOwnerBytes - 1, policy::kOwnerBytes + 1})
      require(!policy::qualifiedUnionAvailable(policy::kSource, policy::kLayout,
          policy::kOriginalBases, policy::kOriginalBytes, policy::kOwnerCount,
          bytes, 48, 512, 2560), "missing optional byte union did not skip");
    rejected([&] { (void)policy::qualifiedUnionAvailable("unknown", policy::kLayout,
        policy::kOriginalBases, policy::kOriginalBytes, 0, 0, 48, 512, 2560); },
        "different model was permitted as a missing-store fallback");
    for (uint32_t field = 0; field != 9; ++field) {
      for (int direction : {-1, 1}) {
        std::string source(policy::kSource), layout(policy::kLayout);
        uint64_t count = policy::kOriginalBases, bytes = policy::kOriginalBytes,
                 owners = policy::kOwnerCount, ownerBytes = policy::kOwnerBytes;
        uint32_t layers = 48, experts = 512, hidden = 2560;
        switch (field) {
          case 0: source[0] = direction == -1 ? '0' : 'f'; break;
          case 1: layout[0] = direction == -1 ? '0' : 'f'; break;
          case 2: count += direction; break;
          case 3: bytes += direction; break;
          case 4: owners += direction; break;
          case 5: ownerBytes += direction; break;
          case 6: layers += direction; break;
          case 7: experts += direction; break;
          case 8: hidden += direction; break;
        }
        rejected([&] { geometry(source, layout, count, bytes, owners, ownerBytes,
                                layers, experts, hidden); },
                 "altered geometry accepted");
      }
    }

    std::array<uint32_t, policy::kOwnerCount> mixed{};
    std::array<uint32_t, policy::kOutputWords> output{};
    output.fill(policy::kGuard);
    uint32_t checksum = 0;
    for (uint32_t index = 0; index != mixed.size(); ++index) {
      const uint32_t original = index % 4 == 0 ? 0u :
          index % 4 == 1 ? std::numeric_limits<uint32_t>::max() :
          index % 4 == 2 ? policy::kUnwritten : (index * 0xdeadbeefu);
      const uint32_t expected = original ^
          static_cast<uint32_t>(uint64_t{0x9e3779b9} * (uint64_t{index} + 1));
      mixed[index] = policy::mixedWord(original, index);
      require(mixed[index] == expected, "mixed owner ABI differs");
      output[policy::kOutputValueBegin + index] = mixed[index];
      checksum ^= mixed[index];
    }
    output[policy::kChecksumIndex] = checksum;
    output[policy::kCountIndex] = static_cast<uint32_t>(policy::kOwnerCount);
    require(policy::validateOutput(output, mixed), "valid output rejected");
    for (uint32_t index = 0; index != output.size(); ++index) {
      output[index] ^= 1;
      require(!policy::validateOutput(output, mixed), "corrupted output word accepted");
      output[index] ^= 1;
    }
    for (uint32_t index = 0; index != mixed.size(); ++index) {
      const uint32_t saved = output[policy::kOutputValueBegin + index];
      output[policy::kOutputValueBegin + index] = policy::kUnwritten;
      if (saved != policy::kUnwritten)
        require(!policy::validateOutput(output, mixed), "unwritten owner accepted");
      output[policy::kOutputValueBegin + index] = saved;
    }
    // Checksum alone cannot detect permutations: exact individual words must.
    std::swap(output[policy::kOutputValueBegin], output[policy::kOutputValueBegin + 1]);
    require(!policy::validateOutput(output, mixed), "equal-checksum permutation accepted");
    std::swap(output[policy::kOutputValueBegin], output[policy::kOutputValueBegin + 1]);
    require(!policy::validateOutput(std::span(output).first(output.size() - 1), mixed),
            "short output accepted");
    require(!policy::validateOutput(output, std::span(mixed).first(mixed.size() - 1)),
            "short owner inventory accepted");
    auto unwritten = output;
    for (uint32_t index = 0; index != mixed.size(); ++index)
      unwritten[policy::kOutputValueBegin + index] = policy::kUnwritten;
    require(!policy::validateOutput(unwritten, mixed), "fully unwritten values accepted");
    std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":" << checks
              << ",\"exhaustive_eligibility_cases\":512,"
                 "\"strict_switch_cases\":17,\"interval_cases\":920,"
                 "\"host_boundary_cases\":102,\"geometry_cases\":19,"
                 "\"every_output_word_corrupted\":4096,"
                 "\"owner_words_checked\":1134}" << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "policy test failed: " << error.what() << '\n';
    return 1;
  }
}
