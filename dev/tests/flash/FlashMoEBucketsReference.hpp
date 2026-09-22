#pragma once

// Independent scalar oracle for exact MoE preprocessing. This file deliberately
// does not include any production header, ABI, builder, or shader helper.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace splash::flash::bucket_reference {

inline constexpr uint32_t kExperts = 512;
inline constexpr uint32_t kWidth = 2560;
inline constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();
using Bits = uint16_t;

struct Job {
  uint32_t expert = kInvalid;
  uint32_t rowBegin = 0;
  bool operator==(const Job &) const = default;
};

struct Packed {
  uint32_t rows = 0;
  uint32_t selections = 0;
  std::array<uint32_t, kExperts> counts{};
  std::array<uint32_t, kExperts + 1> offsets{};
  std::vector<uint32_t> routeMap;
  std::vector<uint32_t> canonicalToPacked;
  std::vector<Bits> inputs;
  uint32_t diagnostic = 0;
};

struct Jobs {
  std::array<uint32_t, kExperts + 1> offsets{};
  std::vector<Job> entries;
  uint32_t count = 0;
};

inline void checkGeometry(uint32_t rows, uint32_t selections) {
  if (rows == 0 || rows > 2048 || selections == 0 || selections > 10)
    throw std::invalid_argument("bucket reference geometry is outside bounds");
}

inline uint32_t jobCapacity(uint32_t rows, uint32_t selections,
                            uint32_t tileRows) {
  checkGeometry(rows, selections);
  if (tileRows != 8 && tileRows != 16 && tileRows != 32)
    throw std::invalid_argument("bucket reference tile size is outside bounds");
  const uint32_t routes = rows * selections;
  return (routes + tileRows - 1) / tileRows + kExperts - 1;
}

inline Packed pack(std::span<const Bits> input, std::span<const int64_t> ids,
                    uint32_t rows, uint32_t selections,
                    uint32_t stickyDiagnostic = 0) {
  checkGeometry(rows, selections);
  const uint32_t routes = rows * selections;
  if (input.size() != uint64_t{rows} * kWidth || ids.size() != routes)
    throw std::invalid_argument("bucket reference input has the wrong extent");

  Packed output;
  output.rows = rows;
  output.selections = selections;
  output.diagnostic = stickyDiagnostic;
  output.routeMap.assign(routes, kInvalid);
  output.canonicalToPacked.assign(routes, kInvalid);
  output.inputs.assign(uint64_t{routes} * kWidth, Bits{0});

  // Scan all hidden rows, including rows for which no expert was selected.
  // Exponent-only testing accepts signed zero and finite BF16 subnormals and
  // reports both infinities and every NaN encoding without changing payloads.
  for (Bits value : input)
    if ((value & 0x7f80U) == 0x7f80U)
      output.diagnostic |= 4;

  std::vector<uint32_t> validRoutes;
  validRoutes.reserve(routes);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t slot = 0; slot < selections; ++slot) {
      const uint32_t flat = row * selections + slot;
      const int64_t id = ids[flat];
      if (id < 0 || id >= int64_t{kExperts}) {
        output.diagnostic |= 1;
        continue;
      }
      for (uint32_t previous = 0; previous < slot; ++previous)
        if (ids[row * selections + previous] == id)
          output.diagnostic |= 1;
      validRoutes.push_back(flat);
    }
  }

  // A library stable sort is intentionally independent of the histogram,
  // prefix, and per-expert shader implementation under test.
  std::stable_sort(validRoutes.begin(), validRoutes.end(),
                   [&](uint32_t a, uint32_t b) { return ids[a] < ids[b]; });
  for (uint32_t flat : validRoutes)
    ++output.counts[static_cast<uint32_t>(ids[flat])];
  for (uint32_t expert = 0; expert < kExperts; ++expert)
    output.offsets[expert + 1] = output.offsets[expert] + output.counts[expert];
  std::copy(validRoutes.begin(), validRoutes.end(), output.routeMap.begin());
  for (uint32_t packedRow = 0; packedRow < validRoutes.size(); ++packedRow) {
    output.canonicalToPacked[validRoutes[packedRow]] = packedRow;
    const uint32_t row = validRoutes[packedRow] / selections;
    std::copy_n(input.begin() + uint64_t{row} * kWidth, kWidth,
                output.inputs.begin() + uint64_t{packedRow} * kWidth);
  }
  return output;
}

inline Jobs makeJobs(const Packed &packed, uint32_t tileRows,
                      uint32_t outputCapacity = 0) {
  const uint32_t minimum = jobCapacity(packed.rows, packed.selections, tileRows);
  if (outputCapacity == 0)
    outputCapacity = minimum;
  if (outputCapacity < minimum)
    throw std::invalid_argument("bucket reference job capacity is insufficient");
  if (packed.offsets[0] != 0 ||
      packed.offsets[kExperts] > packed.rows * packed.selections)
    throw std::invalid_argument("bucket reference offsets are malformed");
  for (uint32_t expert = 0; expert < kExperts; ++expert)
    if (packed.offsets[expert + 1] != packed.offsets[expert] + packed.counts[expert])
      throw std::invalid_argument("bucket reference count and offset disagree");

  Jobs output;
  output.entries.resize(outputCapacity);
  for (uint32_t expert = 0; expert < kExperts; ++expert) {
    output.offsets[expert] = output.count;
    for (uint32_t begin = packed.offsets[expert];
         begin < packed.offsets[expert + 1]; begin += tileRows) {
      if (output.count == output.entries.size())
        throw std::logic_error("bucket reference job bound was exceeded");
      output.entries[output.count++] = Job{expert, begin};
    }
  }
  output.offsets[kExperts] = output.count;
  return output;
}

} // namespace splash::flash::bucket_reference
