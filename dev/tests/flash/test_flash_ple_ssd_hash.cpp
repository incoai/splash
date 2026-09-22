// CPU-only canonical PLE hash qualification for optional SSD row streaming.
// Compile the real FlashPLE.cpp; Mach-O dead stripping removes all model and
// backend functions. No Metal framework, backend or GPU commands are needed.
//
// xcrun -sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime \
//   -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 \
//   dev/tests/flash/test_flash_ple_ssd_hash.cpp runtime/flash/FlashPLE.cpp \
//   -Wl,-dead_strip -o build/flash-ple-ssd-hash-cpu/test
// Replace -O3 with -O1 -g -fsanitize=address,undefined \
//   -fno-omit-frame-pointer to run the sanitizer variant.

#include "flash/FlashPLE.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using splash::flash::FlashPLEGeometry;
using splash::flash::computePLENgramIDs;
constexpr uint32_t heads = splash::flash::kFlashPLEHeads;
uint64_t checks = 0, validCases = 0, rejectedCases = 0;
uint64_t overflowProducts = 0, negativeHashWords = 0;

void require(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}

struct Fixture {
  FlashPLEGeometry g;
  uint64_t tableRows = 1000000;
  std::array<int64_t, 3> multipliers = {
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::min() + 127,
      -7046029254386353131LL};
  std::array<int64_t, heads> sizes{};
  std::array<int64_t, heads> offsets{};
  std::vector<int64_t> tokens, history;

  Fixture(uint32_t lanes = 1, uint32_t rows = 16) {
    g.lanes = lanes; g.rows = rows;
    int64_t offset = 0;
    for (uint32_t head = 0; head < heads; ++head) {
      sizes[head] = 3 + 137 * head;
      offsets[head] = offset;
      offset += sizes[head] + 11;
    }
    history.resize(uint64_t(lanes) * 2, g.eosToken);
    tokens.resize(uint64_t(lanes) * rows);
    for (uint32_t lane = 0; lane < lanes; ++lane)
      for (uint32_t row = 0; row < rows; ++row) {
        const auto flat = uint64_t(lane) * rows + row;
        // Consecutive EOS, boundary tokens and independent lane sequences.
        const uint32_t selector = (row + lane * 3) % 13;
        tokens[flat] = selector == 0 || selector == 1 ? g.eosToken :
            selector == 2 ? 0 : selector == 3 ? g.vocabularySize - 1 :
            (uint64_t(row) * 7919 + lane * 104729 + 1) % g.vocabularySize;
      }
  }
};

// Independent 128-bit model: calculate the signed product, then reduce it to
// its low 64 bits. The production routine instead multiplies uint64_t values.
uint64_t wrappedProduct(int64_t token, int64_t multiplier) {
  const __int128 product = __int128(token) * __int128(multiplier);
  if (product < std::numeric_limits<int64_t>::min() ||
      product > std::numeric_limits<int64_t>::max()) ++overflowProducts;
  const auto unsignedProduct = static_cast<unsigned __int128>(product);
  return static_cast<uint64_t>(unsignedProduct &
      static_cast<unsigned __int128>(std::numeric_limits<uint64_t>::max()));
}

int64_t referenceID(int64_t current, int64_t previous, int64_t older,
                    uint32_t head, const Fixture &f) {
  const int64_t context2 = previous == f.g.eosToken ? f.g.eosToken : older;
  uint64_t bits = wrappedProduct(current, f.multipliers[0]) ^
                  wrappedProduct(previous, f.multipliers[1]);
  if (head >= 8) bits ^= wrappedProduct(context2, f.multipliers[2]);
  // Signed I64 interpretation without an out-of-range integer conversion or
  // the production bit_cast. Positive modulo is evaluated entirely in I128.
  __int128 signedWord = static_cast<__int128>(bits);
  if (bits & (uint64_t{1} << 63)) {
    signedWord -= (__int128{1} << 64);
    ++negativeHashWords;
  }
  const __int128 size = f.sizes[head];
  const auto remainder = (signedWord % size + size) % size;
  return static_cast<int64_t>(remainder + f.offsets[head]);
}

std::vector<int64_t> reference(const Fixture &f, std::vector<int64_t> &history) {
  std::vector<int64_t> ids(f.tokens.size() * heads);
  for (uint32_t lane = 0; lane < f.g.lanes; ++lane) {
    int64_t older = history[uint64_t(lane) * 2];
    int64_t previous = history[uint64_t(lane) * 2 + 1];
    for (uint32_t row = 0; row < f.g.rows; ++row) {
      const uint64_t flat = uint64_t(lane) * f.g.rows + row;
      for (uint32_t head = 0; head < heads; ++head)
        ids[flat * heads + head] =
            referenceID(f.tokens[flat], previous, older, head, f);
      older = previous; previous = f.tokens[flat];
    }
    history[uint64_t(lane) * 2] = older;
    history[uint64_t(lane) * 2 + 1] = previous;
  }
  return ids;
}

std::vector<int64_t> call(const Fixture &f, std::vector<int64_t> &history) {
  ++validCases;
  return computePLENgramIDs(f.tokens, history, f.multipliers, f.sizes,
                           f.offsets, f.g, f.tableRows);
}

std::vector<int64_t> checkExact(const Fixture &f) {
  auto expectedHistory = f.history, actualHistory = f.history;
  const auto expected = reference(f, expectedHistory);
  const auto actual = call(f, actualHistory);
  require(actual.size() == f.tokens.size() * heads, "hash result extent differs");
  for (size_t i = 0; i < actual.size(); ++i) {
    require(actual[i] == expected[i], "canonical hash differs from I128 model");
    const size_t head = i % heads;
    require(actual[i] >= f.offsets[head] &&
                actual[i] < f.offsets[head] + f.sizes[head],
            "hash escaped its stored head range");
  }
  require(actualHistory == expectedHistory, "final history differs");
  require(f.history.size() == actualHistory.size(), "history extent changed");
  return actual;
}

// Chunk extraction/reassembly is lane-major. Joining returned byte arrays
// directly would incorrectly interleave lanes for batched SSD row requests.
void checkChunks(const Fixture &whole, const std::vector<uint32_t> &chunks) {
  const auto expected = checkExact(whole);
  auto state = whole.history;
  std::vector<int64_t> merged(expected.size(), -1);
  uint32_t cursor = 0;
  for (auto length : chunks) {
    require(length > 0 && length <= whole.g.rows - cursor,
            "test chunk geometry is invalid");
    Fixture chunk = whole;
    chunk.g.rows = length;
    chunk.tokens.resize(uint64_t(length) * whole.g.lanes);
    for (uint32_t lane = 0; lane < whole.g.lanes; ++lane)
      std::copy_n(whole.tokens.begin() + uint64_t(lane) * whole.g.rows + cursor,
                  length, chunk.tokens.begin() + uint64_t(lane) * length);
    const auto ids = call(chunk, state);
    for (uint32_t lane = 0; lane < whole.g.lanes; ++lane)
      std::copy_n(ids.begin() + uint64_t(lane) * length * heads,
                  uint64_t(length) * heads,
                  merged.begin() + (uint64_t(lane) * whole.g.rows + cursor) * heads);
    cursor += length;
  }
  auto finalState = whole.history;
  (void)reference(whole, finalState);
  require(cursor == whole.g.rows, "chunks did not cover all rows");
  require(merged == expected, "chunked hashes differ from whole command");
  require(state == finalState, "chunked history differs from whole command");
}

void checkLaneIsolation(const Fixture &joint) {
  const auto expected = checkExact(joint);
  auto jointState = joint.history;
  (void)call(joint, jointState);
  for (uint32_t lane = 0; lane < joint.g.lanes; ++lane) {
    Fixture single = joint;
    single.g.lanes = 1;
    single.tokens.assign(joint.tokens.begin() + uint64_t(lane) * joint.g.rows,
                         joint.tokens.begin() + uint64_t(lane + 1) * joint.g.rows);
    single.history.assign(joint.history.begin() + uint64_t(lane) * 2,
                          joint.history.begin() + uint64_t(lane + 1) * 2);
    auto singleState = single.history;
    const auto actual = call(single, singleState);
    for (size_t index = 0; index < actual.size(); ++index)
      require(actual[index] == expected[uint64_t(lane) * joint.g.rows * heads + index],
              "joint lane depends on another lane");
    require(std::equal(singleState.begin(), singleState.end(),
                       jointState.begin() + uint64_t(lane) * 2),
            "joint history differs from singleton lane");
  }
  Fixture reversed = joint;
  for (uint32_t lane = 0; lane < joint.g.lanes; ++lane) {
    const auto source = joint.g.lanes - lane - 1;
    std::copy_n(joint.tokens.begin() + uint64_t(source) * joint.g.rows, joint.g.rows,
                reversed.tokens.begin() + uint64_t(lane) * joint.g.rows);
    std::copy_n(joint.history.begin() + uint64_t(source) * 2, 2,
                reversed.history.begin() + uint64_t(lane) * 2);
  }
  auto reversedState = reversed.history;
  const auto actual = call(reversed, reversedState);
  for (uint32_t lane = 0; lane < joint.g.lanes; ++lane) {
    const auto source = joint.g.lanes - lane - 1;
    require(std::equal(actual.begin() + uint64_t(lane) * joint.g.rows * heads,
                       actual.begin() + uint64_t(lane + 1) * joint.g.rows * heads,
                       expected.begin() + uint64_t(source) * joint.g.rows * heads),
            "lane permutation changed hashes");
    require(std::equal(reversedState.begin() + uint64_t(lane) * 2,
                       reversedState.begin() + uint64_t(lane + 1) * 2,
                       jointState.begin() + uint64_t(source) * 2),
            "lane permutation changed history");
  }
}

void geometryAndChunks() {
  const std::array<std::array<int64_t, 3>, 7> multiplierSets = {{
      {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min() + 127,
       -7046029254386353131LL},
      {std::numeric_limits<int64_t>::min(), -1, 1},
      {-1, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()},
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}};
  for (uint32_t rows : {1U, 4U, 16U, 2048U})
    for (uint32_t lanes : {1U, 2U, 4U})
      for (uint32_t historyKind = 0; historyKind < 4; ++historyKind)
        for (const auto &multipliers : multiplierSets) {
          Fixture f(lanes, rows); f.multipliers = multipliers;
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            f.history[uint64_t(lane) * 2] = historyKind & 1 ? 23 + lane : f.g.eosToken;
            f.history[uint64_t(lane) * 2 + 1] = historyKind & 2 ? 71 + lane : f.g.eosToken;
          }
          (void)checkExact(f);
        }
  for (uint32_t rows : {1U, 4U, 16U, 2048U})
    for (uint32_t lanes : {1U, 2U, 4U}) {
      Fixture f(lanes, rows);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        f.history[uint64_t(lane) * 2] = 13 + lane;
        f.history[uint64_t(lane) * 2 + 1] = lane & 1 ? f.g.eosToken : 43 + lane;
      }
      checkLaneIsolation(f);
      checkChunks(f, std::vector<uint32_t>(rows, 1));
      if (rows <= 16)
        for (uint32_t split = 1; split < rows; ++split)
          checkChunks(f, {split, rows - split});
      else {
        checkChunks(f, {1, 3, 12, 2032});
        checkChunks(f, {2047, 1});
        std::vector<uint32_t> randomChunks;
        uint32_t remaining = rows, seed = 0x19ab42U;
        while (remaining) {
          seed = seed * 1664525U + 1013904223U;
          const auto next = std::min(remaining, 1 + seed % 97);
          randomChunks.push_back(next); remaining -= next;
        }
        checkChunks(f, randomChunks);
      }
    }
}

void eosSemantics() {
  Fixture f(1, 4); f.tokens = {37, f.g.eosToken, 41, 43}; f.history = {11, 13};
  const auto actual = checkExact(f);
  // EOS itself uses the preceding segment. The next row resets the older
  // trigram token to EOS; the second next row includes EOS as older history.
  for (uint32_t head = 0; head < heads; ++head) {
    require(actual[heads + head] == referenceID(f.g.eosToken, 37, 13, head, f),
            "current EOS prematurely reset its context");
    require(actual[2 * heads + head] == referenceID(41, f.g.eosToken, 37, head, f),
            "token following EOS did not reset trigram context");
    require(actual[3 * heads + head] == referenceID(43, 41, f.g.eosToken, head, f),
            "second token following EOS lost chronological history");
  }
  Fixture historyA(1, 1), historyB = historyA;
  historyA.tokens = historyB.tokens = {97};
  historyA.history = {11, historyA.g.eosToken};
  historyB.history = {177, historyB.g.eosToken};
  require(checkExact(historyA) == checkExact(historyB),
          "EOS did not isolate preceding segment history");
  historyA.history = {11, 53}; historyB.history = {177, 53};
  const auto idsA = checkExact(historyA), idsB = checkExact(historyB);
  require(std::equal(idsA.begin(), idsA.begin() + 8, idsB.begin()),
          "bigram unexpectedly depends on older token");
  require(!std::equal(idsA.begin() + 8, idsA.end(), idsB.begin() + 8),
          "trigram fixture did not exercise older token");

  Fixture exactMinimum(1, 1);
  exactMinimum.tokens = {1}; exactMinimum.history = {0, 0};
  exactMinimum.multipliers = {std::numeric_limits<int64_t>::min(), 0, 0};
  (void)checkExact(exactMinimum); // INT64_MIN % positive divisor is defined.

  Fixture literal(1, 1);
  literal.tokens = {1}; literal.history = {0, 0};
  literal.multipliers = {-1, 0, 0};
  literal.sizes.fill(3); literal.offsets.fill(11); literal.tableRows = 14;
  const auto literalIDs = checkExact(literal);
  for (const auto id : literalIDs)
    require(id == 13, "literal (-1 mod 3) + 11 golden differs");
}

void checkpointAndBoundaryMetadata() {
  // Original aligned local checkpoint manifest fingerprint:
  // 0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402.
  // Arrays copied directly from the three tiny stored I64 tensors, not from
  // regenerating expected prime vocabulary sizes or hash multipliers.
  Fixture checkpoint;
  checkpoint.multipliers = {23703573157769LL, 20109073645365LL, 8052911324071LL};
  checkpoint.sizes = {
      20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
      20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
  checkpoint.offsets = {
      0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
      160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
      300001275};
  checkpoint.tableRows = 320001446;
  for (uint32_t rows : {1U, 4U, 16U, 2048U})
    for (uint32_t lanes : {1U, 2U, 4U}) {
      Fixture f(lanes, rows);
      f.multipliers = checkpoint.multipliers; f.sizes = checkpoint.sizes;
      f.offsets = checkpoint.offsets; f.tableRows = checkpoint.tableRows;
      checkLaneIsolation(f);
      if (rows > 1) checkChunks(f, {1, rows - 1});
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        f.history[uint64_t(lane) * 2] = 248319 - lane;
        f.history[uint64_t(lane) * 2 + 1] = 17 + lane;
      }
      (void)checkExact(f);
    }

  Fixture boundary(4, 16);
  boundary.tableRows = std::numeric_limits<int64_t>::max();
  for (uint32_t head = 0; head < heads; ++head) {
    switch (head % 4) {
    case 0: boundary.sizes[head] = 1;
            boundary.offsets[head] = std::numeric_limits<int64_t>::max() - 1; break;
    case 1: boundary.sizes[head] = std::numeric_limits<int64_t>::max();
            boundary.offsets[head] = 0; break;
    case 2: boundary.sizes[head] = (int64_t{1} << 62) - 57;
            boundary.offsets[head] = (int64_t{1} << 62); break;
    case 3: boundary.sizes[head] = 97;
            boundary.offsets[head] = std::numeric_limits<int64_t>::max() - 97; break;
    }
  }
  (void)checkExact(boundary);
  checkChunks(boundary, {3, 1, 12});
}

template <class Operation> void rejects(const Fixture &fixture, Operation operation) {
  auto history = fixture.history;
  const auto before = history;
  bool rejected = false;
  try { operation(history); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid canonical hash input was accepted");
  require(history == before, "rejected canonical hash input mutated history");
  ++rejectedCases;
}

void invalidInputs() {
  // Corrupt the last lane/row/head so an incremental validation implementation
  // would mutate earlier lane histories before discovering the bad input.
  for (uint32_t lanes : {1U, 4U})
    for (uint32_t rows : {1U, 4U, 16U, 2048U}) {
      Fixture f(lanes, rows);
      for (int64_t invalid : {-1LL, int64_t(f.g.vocabularySize),
                              std::numeric_limits<int64_t>::max()}) {
        Fixture bad = f; bad.tokens.back() = invalid;
        rejects(bad, [&](auto &history) {
          (void)computePLENgramIDs(bad.tokens, history, bad.multipliers,
                                  bad.sizes, bad.offsets, bad.g, bad.tableRows);
        });
        bad = f; bad.history.back() = invalid;
        rejects(bad, [&](auto &history) {
          (void)computePLENgramIDs(bad.tokens, history, bad.multipliers,
                                  bad.sizes, bad.offsets, bad.g, bad.tableRows);
        });
      }
      for (uint32_t which = 0; which < 8; ++which) {
        Fixture bad = f;
        switch (which) {
        case 0: bad.sizes.back() = 0; break;
        case 1: bad.sizes.back() = -1; break;
        case 2: bad.offsets.back() = -1; break;
        case 3: bad.offsets.back() = int64_t(bad.tableRows); break;
        case 4: bad.offsets.back() = int64_t(bad.tableRows) - 1;
                bad.sizes.back() = 2; break;
        case 5: bad.sizes.back() = std::numeric_limits<int64_t>::max(); break;
        case 6: bad.tableRows = 0; break;
        case 7: bad.tableRows = uint64_t(std::numeric_limits<int64_t>::max()) + 1; break;
        }
        rejects(bad, [&](auto &history) {
          (void)computePLENgramIDs(bad.tokens, history, bad.multipliers,
                                  bad.sizes, bad.offsets, bad.g, bad.tableRows);
        });
      }
      for (uint32_t which = 0; which < 11; ++which) {
        Fixture bad = f;
        switch (which) {
        case 0: bad.g.lanes = 0; break;
        case 1: bad.g.rows = 0; break;
        case 2: bad.g.width = 0; break;
        case 3: bad.g.streams = 0; break;
        case 4: bad.g.streams = 9; break;
        case 5: bad.g.vocabularySize = 0; break;
        case 6: bad.g.eosToken = bad.g.vocabularySize; break;
        case 7: bad.g.epsilon = 0; break;
        case 8: bad.g.epsilon = std::numeric_limits<float>::infinity(); break;
        case 9: bad.g.epsilon = std::numeric_limits<float>::quiet_NaN(); break;
        case 10: bad.g.epsilon = -1e-6f; break;
        }
        rejects(bad, [&](auto &history) {
          (void)computePLENgramIDs(bad.tokens, history, bad.multipliers,
                                  bad.sizes, bad.offsets, bad.g, bad.tableRows);
        });
      }
      for (uint32_t which = 0; which < 5; ++which)
      {
          auto history = f.history;
          auto tokens = f.tokens;
          auto multipliers = std::vector<int64_t>(f.multipliers.begin(), f.multipliers.end());
          auto sizes = std::vector<int64_t>(f.sizes.begin(), f.sizes.end());
          auto offsets = std::vector<int64_t>(f.offsets.begin(), f.offsets.end());
          switch (which) {
          case 0: tokens.push_back(0); break;
          case 1: history.push_back(0); break;
          case 2: multipliers.push_back(0); break;
          case 3: sizes.push_back(1); break;
          case 4: offsets.push_back(0); break;
          }
          const auto submitted = history;
          bool failed = false;
          try {
            (void)computePLENgramIDs(tokens, history, multipliers, sizes, offsets,
                                    f.g, f.tableRows);
          } catch (const std::invalid_argument &) { failed = true; }
          require(failed, "long-extent canonical input was accepted");
          require(history == submitted, "long-extent rejection mutated history");
          ++rejectedCases;
      }
      for (uint32_t which = 0; which < 6; ++which)
        rejects(f, [&](auto &history) {
          const auto tokens = std::span<const int64_t>(f.tokens);
          const auto multipliers = std::span<const int64_t>(f.multipliers);
          const auto sizes = std::span<const int64_t>(f.sizes);
          const auto offsets = std::span<const int64_t>(f.offsets);
          (void)computePLENgramIDs(
              which == 0 ? tokens.first(tokens.size() - 1) : tokens,
              which == 1 ? std::span<int64_t>(history).first(history.size() - 1) :
                           std::span<int64_t>(history),
              which == 2 ? multipliers.first(2) : multipliers,
              which == 3 ? sizes.first(heads - 1) : sizes,
              which == 4 ? offsets.first(heads - 1) : offsets,
              which == 5 ? FlashPLEGeometry{UINT32_MAX, UINT32_MAX} : f.g,
              f.tableRows);
        });
    }
  Fixture overWidth;
  overWidth.g.width = UINT32_MAX; overWidth.g.streams = 8;
  rejects(overWidth, [&](auto &history) {
    (void)computePLENgramIDs(overWidth.tokens, history, overWidth.multipliers,
                            overWidth.sizes, overWidth.offsets, overWidth.g,
                            overWidth.tableRows);
  });
}
} // namespace

int main() {
  try {
    geometryAndChunks(); eosSemantics(); checkpointAndBoundaryMetadata(); invalidInputs();
    require(overflowProducts > 0, "fixtures missed signed I64 product wraparound");
    require(negativeHashWords > 0, "fixtures missed positive signed modulo");
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
              << ",\"valid_hash_calls\":" << validCases
              << ",\"rejected_input_cases\":" << rejectedCases
              << ",\"overflow_reference_products\":" << overflowProducts
              << ",\"negative_reference_hash_words\":" << negativeHashWords
              << ",\"canonical_production_function\":true"
                 ",\"gpu_commands\":0,\"metal_backend_constructions\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
