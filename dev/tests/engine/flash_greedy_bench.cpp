#include "flash/FlashGreedy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kVocabulary = 248320;
uint64_t sink = 0;

[[gnu::noinline]] uint32_t reference(std::span<const uint16_t> values) {
  if (values.empty()) throw std::invalid_argument("empty greedy vocabulary row");
  uint32_t best = 0;
  float maximum = -INFINITY;
  for (uint32_t token = 0; token < values.size(); ++token) {
    const float value = std::bit_cast<float>(uint32_t{values[token]} << 16);
    if (!std::isfinite(value)) throw std::runtime_error("non-finite Flash vocabulary logit");
    if (value > maximum) { maximum = value; best = token; }
  }
  return best;
}
[[gnu::noinline]] uint32_t candidate(std::span<const uint16_t> values) {
  return splash::flash::flashGreedyToken(values);
}
[[gnu::noinline]] double percentile(const std::deque<double> &window, double fraction) {
  if (window.empty()) return 0;
  std::vector<double> sorted(window.begin(), window.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted[static_cast<size_t>((sorted.size() - 1) * fraction)];
}
template <typename F> double measure(F operation, uint32_t repetitions) {
  const auto began = std::chrono::steady_clock::now();
  uint64_t total = 0;
  for (uint32_t repeat = 0; repeat < repetitions; ++repeat) total += operation(repeat);
  sink ^= total;
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count() /
      repetitions;
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end()); return values[values.size() / 2];
}
} // namespace

int main() {
  std::mt19937 random(0xf1a5);
  std::normal_distribution<float> distribution(0.0f, 8.0f);
  std::array<std::vector<uint16_t>, 8> rows;
  for (uint32_t row = 0; row < rows.size(); ++row) {
    rows[row].resize(kVocabulary);
    for (auto &value : rows[row])
      value = static_cast<uint16_t>(std::bit_cast<uint32_t>(distribution(random)) >> 16);
    // Cover an early, middle and late unique maximum; ties deliberately occur
    // in some rows and always resolve to the earlier token.
    const auto first = row % 3 == 0 ? uint32_t{0}
        : row % 3 == 1 ? kVocabulary / 2 : kVocabulary - 1;
    rows[row][first] = 0x42c8; // +100, above the normally distributed background.
    if (first + 16 < kVocabulary && row % 2) rows[row][first + 16] = 0x42c8;
    if (reference(rows[row]) != candidate(rows[row]))
      throw std::runtime_error("benchmark source argmax differs");
  }
  for (uint32_t repeat = 0; repeat < 100; ++repeat) sink ^= candidate(rows[repeat % rows.size()]);
  std::vector<double> controls, candidates;
  for (uint32_t pair = 0; pair < 9; ++pair) {
    const auto control = [&] { return measure([&](uint32_t n) { return reference(rows[n % rows.size()]); }, 512); };
    const auto proposed = [&] { return measure([&](uint32_t n) { return candidate(rows[n % rows.size()]); }, 512); };
    if (pair % 2) { candidates.push_back(proposed()); controls.push_back(control()); }
    else { controls.push_back(control()); candidates.push_back(proposed()); }
  }
  const auto control = median(controls), proposed = median(candidates);
  std::deque<double> ttft, itl, maximumWindow;
  for (uint32_t i = 0; i < 4096; ++i) {
    const double value = distribution(random);
    maximumWindow.push_back(value);
    if (i < 21) ttft.push_back(value);
    if (i < 553) itl.push_back(value);
  }
  const auto percentiles = [&](const std::deque<double> &a, const std::deque<double> &b) {
    return measure([&](uint32_t repeat) {
      // Fractions alternate to prevent invariant-operation hoisting.
      const double p = repeat % 2 ? .5 : .95;
      const double sum = percentile(a, p) + percentile(a, 1 - p) +
          percentile(b, p) + percentile(b, 1 - p);
      return std::bit_cast<uint64_t>(sum);
    }, 1024);
  };
  std::cout << std::setprecision(12)
      << "{\"schema\":\"flash-greedy-cpu-benchmark-v1\",\"gpu_work\":false,"
      << "\"vocabulary\":" << kVocabulary << ",\"rows\":" << rows.size()
      << ",\"paired_samples\":9,\"repeats_per_sample\":512"
      << ",\"literal_reference_microseconds\":" << control * 1e6
      << ",\"candidate_microseconds\":" << proposed * 1e6
      << ",\"speedup\":" << control / proposed
      << ",\"four_percentiles_observed_windows_microseconds\":" << percentiles(ttft, itl) * 1e6
      << ",\"four_percentiles_full_windows_microseconds\":" << percentiles(maximumWindow, maximumWindow) * 1e6
      << ",\"checksum\":" << sink << "}\n";
}
