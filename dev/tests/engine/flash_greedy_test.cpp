#include "flash/FlashGreedy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
uint64_t checks = 0;
void require(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

// This is the existing Worker algorithm, with comparison only in float space.
float value(uint16_t bits) {
  return std::bit_cast<float>(uint32_t{bits} << 16);
}
uint32_t reference(std::span<const uint16_t> values) {
  if (values.empty()) throw std::invalid_argument("empty greedy vocabulary row");
  uint32_t best = 0;
  float maximum = -INFINITY;
  for (uint32_t token = 0; token < values.size(); ++token) {
    const float current = value(values[token]);
    if (!std::isfinite(current))
      throw std::runtime_error("non-finite Flash vocabulary logit");
    if (current > maximum) { maximum = current; best = token; }
  }
  return best;
}
void equivalent(std::span<const uint16_t> values) {
  require(splash::flash::flashGreedyToken(values) == reference(values),
          "greedy token differs from literal float reference");
}
template <class Exception, class Operation>
void rejects(Operation operation, const char *message) {
  bool caught = false;
  try { operation(); }
  catch (const Exception &error) {
    caught = true;
    require(error.what() == std::string(message), "exception message changed");
  }
  require(caught, "expected exception was not thrown");
}

void exhaustiveFiniteOrdering() {
  std::vector<uint16_t> finite;
  for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits)
    if (std::isfinite(value(static_cast<uint16_t>(bits))))
      finite.push_back(static_cast<uint16_t>(bits));
  require(finite.size() == 65280, "finite BF16 domain is incomplete");
  std::stable_sort(finite.begin(), finite.end(), [](uint16_t a, uint16_t b) {
    return value(a) < value(b);
  });
  // Adjacent comparisons span every representable ordering boundary, in both
  // token orders. Singleton tests include negative extrema and subnormals.
  for (size_t index = 0; index < finite.size(); ++index) {
    equivalent(std::span<const uint16_t>(&finite[index], 1));
    std::array<uint16_t, 4> againstZero{finite[index], 0x0000, 0x8000, 0xff7f};
    equivalent(againstZero);
    if (index) {
      std::array<uint16_t, 2> ascending{finite[index - 1], finite[index]};
      equivalent(ascending);
      std::reverse(ascending.begin(), ascending.end());
      equivalent(ascending);
    }
  }
  equivalent(finite);
  std::reverse(finite.begin(), finite.end());
  equivalent(finite);
}

void nonfiniteRejection() {
  rejects<std::invalid_argument>([] {
    (void)splash::flash::flashGreedyToken({});
  }, "empty greedy vocabulary row");
  // Include every infinity and NaN payload/sign, including signalling NaNs.
  // Put each in every vector lane and tail position; an earlier finite maximum
  // must never hide an invalid logit later in the vocabulary.
  for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
    const auto encoded = static_cast<uint16_t>(bits);
    if (std::isfinite(value(encoded))) continue;
    for (size_t position = 0; position < 96; ++position) {
      std::array<uint16_t, 96> row;
      row.fill(0xff7f);
      row[0] = 0x7f7f;
      row[position] = encoded;
      const size_t count = position < 64 ? row.size() : position + 1;
      rejects<std::runtime_error>([&] {
        (void)splash::flash::flashGreedyToken({row.data(), count});
      }, "non-finite Flash vocabulary logit");
    }
  }
}

void tailsAlignmentAndCanaries() {
  std::mt19937 random(0x51972);
  std::vector<uint16_t> finite;
  for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits)
    if (std::isfinite(value(static_cast<uint16_t>(bits))))
      finite.push_back(static_cast<uint16_t>(bits));
  for (size_t offset = 0; offset < 32; ++offset) {
    for (size_t tail = 0; tail < 32; ++tail) {
      for (size_t blocks : {size_t{0}, size_t{1}, size_t{3}, size_t{8}}) {
        const size_t count = blocks * 32 + tail;
        if (!count) continue;
        std::vector<uint16_t> storage(offset + count + 32, 0x7fc1);
        for (size_t index = 0; index < count; ++index)
          storage[offset + index] = finite[random() % finite.size()];
        const auto before = storage;
        equivalent({storage.data() + offset, count});
        require(storage == before, "greedy scan changed input or canary");
        // Every possible winner location, including the final scalar tail,
        // must retain the first token when a maximum is duplicated.
        std::fill_n(storage.data() + offset, count, uint16_t{0xff7f});
        for (size_t position = 0; position < count; ++position) {
          storage[offset + position] = 0x7f7f;
          storage[offset + count - 1] = 0x7f7f;
          equivalent({storage.data() + offset, count});
          storage[offset + position] = 0xff7f;
          storage[offset + count - 1] = 0xff7f;
        }
      }
    }
  }
}

void tiesAndRealVocabulary() {
  for (uint16_t repeated : {uint16_t{0xff7f}, uint16_t{0x7f7f},
                           uint16_t{0x0001}, uint16_t{0x8001},
                           uint16_t{0x3f80}, uint16_t{0xbf80}}) {
    std::vector<uint16_t> row(248320, repeated);
    equivalent(row);
    require(splash::flash::flashGreedyToken(row) == 0, "finite tie lost lower ID");
  }
  for (size_t leadingZero = 0; leadingZero < 32; ++leadingZero) {
    std::array<uint16_t, 96> row;
    row.fill(0x8001);
    for (size_t index = leadingZero; index < row.size(); ++index)
      row[index] = index % 2 ? 0x8000 : 0x0000;
    equivalent(row);
    require(splash::flash::flashGreedyToken(row) == leadingZero,
            "positive and negative zero tie changed lower ID");
    std::swap(row[leadingZero], row[leadingZero + 1]);
    equivalent(row);
  }
  std::mt19937 random(0x248320);
  std::vector<uint16_t> row(248320);
  for (size_t trial = 0; trial < 64; ++trial) {
    for (auto &bits : row) {
      do { bits = static_cast<uint16_t>(random()); }
      while (!std::isfinite(value(bits)));
    }
    const auto before = row;
    equivalent(row);
    require(row == before, "real vocabulary scan mutated input");
    // Strictly negative real vocabularies catch accidental zero initial maxima.
    for (auto &bits : row) bits |= 0x8000;
    equivalent(row);
  }
}
} // namespace

int main() {
  try {
    exhaustiveFiniteOrdering();
    nonfiniteRejection();
    tailsAlignmentAndCanaries();
    tiesAndRealVocabulary();
    std::cout << "Flash BF16 greedy: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Flash BF16 greedy test failed: " << error.what() << '\n';
    return 1;
  }
}
