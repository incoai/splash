// CPU only: no Metal device, model load, or runtime policy changes.
#include "../../benchmarks/FlashFloatBoundaryAudit.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
using namespace splash::flash::benchmark;
uint64_t checks = 0;
void require(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
void roundingTests() {
  for (uint32_t word = 0; word < 65536; ++word) {
    const auto original = uint16_t(word);
    if (std::isfinite(number(original)))
      require(bf16Double(number(original)) == original, "finite BF16 double round-trip failed");
  }
  for (uint32_t order = 0x81; order < 0xff7f; ++order) {
    const auto lo = fromOrderedBF16(order), hi = fromOrderedBF16(order + 1);
    const double midpoint = (double(number(lo)) + double(number(hi))) * .5;
    const auto even = (lo & 1) ? hi : lo;
    require(bf16ULP(bf16Double(midpoint), even) == 0, "double midpoint did not round to even");
    const auto below = bf16Double(std::nextafter(midpoint, -INFINITY));
    const auto above = bf16Double(std::nextafter(midpoint, INFINITY));
    require(bf16ULP(below, lo) == 0 && bf16ULP(above, hi) == 0,
            "double midpoint neighbors rounded to wrong BF16 cells");
  }
  const double midpoint = 1.0 + std::ldexp(1.0, -8);
  const double above = std::nextafter(midpoint, INFINITY);
  require(bf16(float(above)) == 0x3f80 && bf16Double(above) == 0x3f81,
          "double-rounding trap failed to distinguish reference conversion");
  require(bf16Double(INFINITY) == 0x7f80 && bf16Double(-INFINITY) == 0xff80 &&
          (bf16Double(NAN) & 0x7fc0) == 0x7fc0, "nonfinite reference conversion failed");
}
void boundaryTests() {
  constexpr uint16_t lo = 0x3f80, hi = 0x3f81;
  const double midpoint = (double(number(lo)) + double(number(hi))) * .5;
  const double near = midpoint - std::ldexp(1.0, -26);
  const auto gold = bf16Double(near);
  require(gold == lo, "boundary fixture chose wrong independent gold");
  require(cellRelation(hi, gold, near, std::ldexp(1.0, -25)).pass,
          "adjacent cell within accumulation bound was rejected");
  require(!cellRelation(hi, gold, near, std::ldexp(1.0, -27)).pass,
          "adjacent cell outside accumulation bound was accepted");
  require(!cellRelation(0x3f82, gold, near, 10).pass,
          "two-ULP change was accepted by a large error bound");
  require(!cellRelation(hi, lo, 1.0, std::ldexp(1.0, -20)).pass,
          "one-ULP coefficient/output error far from a midpoint was accepted");
  require(cellRelation(0x8000, 0, 0, 0).pass, "signed-zero equivalent cells rejected");
  for (double invalid : {-1.0, double(INFINITY), double(NAN)})
    require(!cellRelation(lo, lo, 1, invalid).pass, "invalid bound accepted");
  require(!cellRelation(0x7f80, 0x7f80, 1, 1).pass &&
          !cellRelation(0x7fc0, lo, 1, 1).pass &&
          !cellRelation(lo, lo, NAN, 1).pass, "nonfinite boundary evidence accepted");
}
void accumulationTests() {
  for (uint32_t count : {32u, 640u, 2560u, 10240u, 32768u}) {
    for (uint32_t sample = 0; sample < 8; ++sample) {
      double sum = 0, magnitude = 0;
      float f32Serial = 0;
      std::array<float, 32> lanes{};
      for (uint32_t k = 0; k < count; ++k) {
        const auto random = randomWord(uint64_t(sample) * count + k);
        const float x = number(bf16(float(int32_t(random & 2047) - 1023) / 1024.0f));
        const float w = float(int32_t((random >> 12) & 65535) - 32767) / 32768.0f;
        sum = std::fma(double(x), double(w), sum);
        magnitude += std::abs(double(x) * w);
        f32Serial = std::fma(x, w, f32Serial);
        lanes[k % 32] = std::fma(x, w, lanes[k % 32]);
      }
      for (uint32_t stride = 16; stride; stride >>= 1)
        for (uint32_t lane = 0; lane < stride; ++lane) lanes[lane] += lanes[lane + stride];
      const double bound = f32DotBound(count, magnitude);
      require(std::abs(double(f32Serial) - sum) <= bound, "serial F32 error exceeded rigorous bound");
      require(std::abs(double(lanes[0]) - sum) <= bound, "tree F32 error exceeded rigorous bound");
      require(gamma(count, -24) >= double(count) * std::ldexp(1.0, -24),
              "gamma bound fell below first-order rounding error");
    }
  }
  const double flushed = double(number(0x0001)) * 2;
  require(f32DotBound(32, flushed, flushed) >= flushed,
          "flushed BF16 input product was omitted from bound");
  for (double invalid : {-1.0, double(INFINITY), double(NAN)}) {
    bool rejected = false;
    try { (void)f32DotBound(32, invalid); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid product magnitude accepted");
  }
  bool rejected = false;
  try { (void)f32DotBound(0, 1); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "zero-term accumulation bound accepted");
}
} // namespace
int main() {
  try {
    roundingTests(); boundaryTests(); accumulationTests();
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_float_boundary_audit_test: " << error.what() << '\n';
    return 1;
  }
}
