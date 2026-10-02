#include "tests/engine/draft_selector_reference.hpp"

#include <iostream>
#include <stdexcept>

namespace {
using namespace splash::test::draft_selector;
using Score = splash::test::draft_selector::Score<double>;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F;
  }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

// The reference DP finds the best of every path on tables small enough to
// enumerate, including -inf scores and exact ties.
void lookaheadReferenceIsOptimal() {
  constexpr uint32_t positions = 5, candidates = 4;
  for (uint32_t trial = 0; trial < 66; ++trial) {
    Random random(0x1a0c + trial);
    std::vector<double> table(positions * candidates * candidates);
    for (double &value : table) {
      const uint32_t kind = random.next() % 8;
      value = kind == 0 ? -INFINITY
              : kind == 1 ? 0.5
                          : 3.0 * random.unit();
      if (trial >= 64)
        value = trial == 64 ? 0.0 : -INFINITY;
    }
    const Score score = [&](uint32_t p, uint32_t s, uint32_t c) {
      return table[(p * candidates + s) * candidates + c];
    };
    const auto [best, path] = referenceLookahead(score, positions, candidates);
    require(std::fabs(expectedLength(score, path, candidates) - best) < 1e-12,
            "reference look-ahead value differs from its own path");
    std::vector<uint32_t> candidatePath(positions, 0);
    double maximum = 0.0;
    for (uint32_t code = 0; code < 1U << (2 * positions); ++code) {
      for (uint32_t position = 0; position < positions; ++position)
        candidatePath[position] = code >> (2 * position) & 3U;
      maximum = std::max(maximum, expectedLength(score, candidatePath, candidates));
    }
    require(std::fabs(maximum - best) < 1e-12,
            "reference look-ahead missed the best path");
    if (trial >= 64)
      require(std::all_of(path.begin(), path.end(),
                          [](uint32_t c) { return c == 0; }),
              "reference ties did not keep the lowest candidate");
  }
}

} // namespace

int main() {
  try {
    lookaheadReferenceIsOptimal();
    std::cout << "draft_selector_reference_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_reference_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
