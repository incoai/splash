#pragma once

// DeepSeek-V4.1's Engram hash (engram.py's NgramHashState; in
// dev/tools/deepseek41_reference.py engram_hashes): the table rows a token
// reads in each Engram layer. The token's compressed id and those of the
// ngram - 1 tokens before it (the pad token's before the sequence's start),
// each times its layer's multiplier for its place, are XORed in from the
// token back; each n-gram's running XOR takes each of `heads` heads' prime
// modulo (as Python's, never negative) plus the head's offset. The products
// and the XOR wrap as numpy's int64 does. The host computes them: it reads the
// rows from the table's file.

#include <cstdint>
#include <span>

namespace splash::model {

struct EngramHashTables final {
  // The compressed id of each token of the vocabulary.
  std::span<const int32_t> tokenMap;
  // [layers][ngram]
  std::span<const int64_t> multipliers;
  // [layers][(ngram - 1) * heads]
  std::span<const int64_t> primes;
  std::span<const int64_t> offsets;
  uint32_t layers = 0;
  uint32_t ngram = 0;
  uint32_t heads = 0;
  uint32_t padToken = 0;
  [[nodiscard]] constexpr uint32_t rowsPerLayer() const noexcept { return (ngram - 1) * heads; }
};

// out[layer * rowsPerLayer() + (n - 2) * heads + h]: the row head h of the
// n-gram ending at tokens[position] reads in Engram layer `layer`, for n = 2
// to ngram. tokens holds the sequence from its start to position at least.
inline void engramRows(const EngramHashTables &t, std::span<const uint32_t> tokens, uint64_t position,
                       uint32_t *out) {
  const uint32_t rows = t.rowsPerLayer();
  for (uint32_t layer = 0; layer < t.layers; ++layer) {
    uint64_t rolling = 0;
    for (uint32_t n = 0; n < t.ngram; ++n) {
      const int64_t id = t.tokenMap[position >= n ? tokens[position - n] : t.padToken];
      const uint64_t product = uint64_t(id) * uint64_t(t.multipliers[layer * t.ngram + n]);
      rolling = n == 0 ? product : rolling ^ product;
      if (n == 0) continue;
      for (uint32_t h = 0; h < t.heads; ++h) {
        const uint32_t at = layer * rows + (n - 1) * t.heads + h;
        const int64_t prime = t.primes[at];
        int64_t row = int64_t(rolling) % prime;
        if (row < 0) row += prime;
        out[at] = uint32_t(row + t.offsets[at]);
      }
    }
  }
}

} // namespace splash::model
