#include "PenaltyOracle.hpp"
#include "TestChecks.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

using namespace splash;
using test::rejects;
using test::require;

int main() {
  try {
    const std::array<float, 4> raw{4.0F, -3.0F, 2.0F, 1.0F};
    const std::array<uint8_t, 4> prompt{1, 1, 0, 0};
    const std::array<uint32_t, 4> counts{0, 0, 2, 0};
    const ops::SamplingPenalties policy{2.0F, 1.5F, 0.5F};
    // Prompt repetition: 4/2 and -3*2. Generated token 2:
    // 2/2 - (2*0.5 + 1.5). An unseen token is unchanged.
    const std::array<float, 4> correct{2.0F, -6.0F, -1.5F, 1.0F};
    require(test::requirePenaltyLogits(raw, correct, prompt, counts, policy, 0) == 0.0,
            "explicit penalty fixture was rejected");
    const std::array<float, 4> favoured{4.0F, -3.0F, 8.0F, 1.0F};
    test::requirePenaltyLogits(raw, favoured, prompt, counts, {1.0F, -2.0F, -2.0F}, 2);
    auto wrong = correct;
    wrong[0] = 4.0F;
    rejects([&] { test::requirePenaltyLogits(raw, wrong, prompt, counts, policy, 0); },
            "same-forward fp64 reference", "missing prompt repetition passed");
    wrong = correct;
    wrong[2] = -1.0F; // One generated occurrence was lost; the winner is unchanged.
    rejects([&] { test::requirePenaltyLogits(raw, wrong, prompt, counts, policy, 0); },
            "same-forward fp64 reference", "a stale generated count passed");
    rejects([&] { test::requirePenaltyLogits(raw, correct, prompt, counts, policy, 3); },
            "not the best", "a nonwinning token passed");
    auto prefixCounts = counts;
    prefixCounts[3] = 1;
    auto prefix = correct;
    prefix[3] = -1.5F; // 1/2 - (0.5 + 1.5), after an accepted proposal.
    test::requirePenaltyLogits(raw, prefix, prompt, prefixCounts, policy, 0);
    rejects([&] { test::requirePenaltyLogits(raw, correct, prompt, prefixCounts, policy, 0); },
            "same-forward fp64 reference", "an omitted draft-prefix count passed");
    auto rounded = correct;
    rounded[0] = std::nextafter(rounded[0], std::numeric_limits<float>::infinity());
    test::requirePenaltyLogits(raw, rounded, prompt, counts, policy, 0);
    rounded[0] += 0.0001F;
    rejects([&] { test::requirePenaltyLogits(raw, rounded, prompt, counts, policy, 0); },
            "same-forward fp64 reference", "an error beyond fp32 arithmetic passed");
    const std::array<uint32_t, 4> empty{};
    test::requirePenaltyLogits(raw, raw, prompt, empty, {1.0F, 1.5F, 0.5F}, 0);
    wrong = raw;
    wrong[0] -= 1.5F;
    rejects([&] { test::requirePenaltyLogits(raw, wrong, prompt, empty, {1.0F, 1.5F, 0.5F}, 0); },
            "same-forward fp64 reference", "presence on a prompt-only token passed");
    const std::array<float, 4> exceptional{
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 2.0F, -1.0F};
    const std::array<uint8_t, 4> noPrompt{};
    test::requirePenaltyLogits(exceptional, exceptional, noPrompt, empty, {}, 2);
    auto changed = exceptional;
    changed[1] = 0.0F;
    rejects([&] { test::requirePenaltyLogits(exceptional, changed, noPrompt, empty, {}, 2); },
            "nonfinite", "a changed nonfinite input passed");
    rejects([&] { test::requirePenaltyLogits(raw, correct, prompt, counts, policy, 4); },
            "geometry", "an out-of-range selected token passed");
    rejects([&] { test::requirePenaltyLogits(raw, correct, prompt, counts, {0.0F, 0.0F, 0.0F}, 0); },
            "valid penalties", "an invalid reference policy passed");
    const std::array<float, 2> infinite{std::numeric_limits<float>::infinity(), 1.0F};
    const std::array<uint8_t, 2> seen{1, 0};
    const std::array<uint32_t, 2> noOutput{};
    test::requirePenaltyLogits(infinite, infinite, seen, noOutput, {}, 0);
    const std::array<float, 2> saturated{std::numeric_limits<float>::max(), 1.0F};
    test::requirePenaltyLogits(infinite, saturated, seen, noOutput, {2.0F, 0.0F, 0.0F}, 0);
    // Prompt bits are stored only when repetition is active; presence alone
    // does not rewrite a prompt-only token, including a positive infinity.
    test::requirePenaltyLogits(infinite, infinite, seen, noOutput, {1.0F, 1.5F, 0.0F}, 0);
    std::cout << "penalty oracle tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "penalty oracle test: " << error.what() << '\n';
    return 1;
  }
}
