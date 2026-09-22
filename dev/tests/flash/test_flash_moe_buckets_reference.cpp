#include "FlashMoEBucketsReference.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace ref = splash::flash::bucket_reference;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try { function(); }
  catch (const std::invalid_argument &) { return; }
  throw std::runtime_error("malformed bucket reference request was accepted");
}

std::vector<ref::Bits> inputBits(uint32_t rows) {
  std::vector<ref::Bits> values(uint64_t{rows} * ref::kWidth);
  for (uint64_t index = 0; index < values.size(); ++index) {
    auto bits = static_cast<ref::Bits>((index * 4051 + index / ref::kWidth * 79) & 0xffffU);
    if ((bits & 0x7f80U) == 0x7f80U) bits ^= 0x0080U;
    values[index] = bits;
  }
  return values;
}

void handGolden() {
  auto input = inputBits(2);
  input[0] = 0x8000;
  input[1] = 0x0001;
  input[2] = 0x7f7f;
  input[ref::kWidth] = 0xff7f;
  // Stable order is expert, then original flat route; slot order is retained
  // even when duplicates appear. Invalid routes consume no packed rows.
  const std::vector<int64_t> ids{511, 3, -1, 3, 0, 511, 512, 0};
  const auto got = ref::pack(input, ids, 2, 4, 0x80);
  require(got.diagnostic == 0x81, "duplicate/invalid flag or sticky seed changed");
  require(got.routeMap == std::vector<uint32_t>({4, 7, 1, 3, 0, 5, ref::kInvalid, ref::kInvalid}),
          "hand-written stable bucket golden changed");
  require(got.canonicalToPacked ==
              std::vector<uint32_t>({4, 2, ref::kInvalid, 3, 0, 5, ref::kInvalid, 1}),
          "hand-written inverse route golden changed");
  require(got.counts[0] == 2 && got.counts[3] == 2 && got.counts[511] == 2,
          "hand-written bucket counts changed");
  require(got.offsets[0] == 0 && got.offsets[1] == 2 && got.offsets[3] == 2 &&
              got.offsets[4] == 4 && got.offsets[511] == 4 && got.offsets[512] == 6,
          "empty bucket prefix or final valid total changed");
  const std::array<uint32_t, 6> expectedRows{1, 1, 0, 0, 0, 1};
  for (uint32_t row = 0; row < expectedRows.size(); ++row)
    require(std::equal(input.begin() + uint64_t{expectedRows[row]} * ref::kWidth,
                       input.begin() + uint64_t{expectedRows[row] + 1} * ref::kWidth,
                       got.inputs.begin() + uint64_t{row} * ref::kWidth),
            "raw BF16 bits were changed by packing");
  require(std::all_of(got.inputs.begin() + 6 * ref::kWidth, got.inputs.end(),
                      [](ref::Bits bits) { return bits == 0; }),
          "invalid tail packed rows were not zero");
  for (uint32_t tile : {8U, 16U, 32U}) {
    const auto jobs = ref::makeJobs(got, tile);
    require(jobs.count == 3 && jobs.entries[0] == ref::Job{0, 0} &&
                jobs.entries[1] == ref::Job{3, 2} && jobs.entries[2] == ref::Job{511, 4},
            "hand-written tile-job golden changed");
    require(jobs.offsets[0] == 0 && jobs.offsets[1] == 1 && jobs.offsets[3] == 1 &&
                jobs.offsets[4] == 2 && jobs.offsets[511] == 2 && jobs.offsets[512] == 3,
            "job prefix across empty expert buckets changed");
    require(std::all_of(jobs.entries.begin() + jobs.count, jobs.entries.end(),
                        [](const ref::Job &job) { return job == ref::Job{}; }),
            "unused tile jobs did not contain the invalid sentinel");
  }
}

void diagnosticGoldens() {
  auto input = inputBits(2);
  const std::array<int64_t, 10> bad{-1, -2, std::numeric_limits<int64_t>::min(),
                                   512, 513, std::numeric_limits<int64_t>::max(),
                                   int64_t{1} << 32, (int64_t{1} << 32) + 511,
                                   99999, -99999};
  auto allInvalid = ref::pack(input, bad, 2, 5, 0x80);
  require(allInvalid.diagnostic == 0x81 && allInvalid.offsets[512] == 0,
          "invalid I64 IDs were truncated or counted");
  require(std::all_of(allInvalid.routeMap.begin(), allInvalid.routeMap.end(),
                      [](uint32_t map) { return map == ref::kInvalid; }) &&
              std::all_of(allInvalid.canonicalToPacked.begin(), allInvalid.canonicalToPacked.end(),
                          [](uint32_t map) { return map == ref::kInvalid; }) &&
              std::all_of(allInvalid.inputs.begin(), allInvalid.inputs.end(),
                          [](ref::Bits bits) { return bits == 0; }),
          "all-invalid fixture left live route data");
  for (uint32_t tile : {8U, 16U, 32U})
    require(ref::makeJobs(allInvalid, tile).count == 0,
            "all-invalid routes generated matrix jobs");

  // Nonfinite hidden data is checked independently of whether a row is routed.
  for (ref::Bits badBits : {ref::Bits{0x7f80}, ref::Bits{0xff80}, ref::Bits{0x7f81},
                            ref::Bits{0xffc1}, ref::Bits{0x7fff}}) {
    input.back() = badBits;
    const auto got = ref::pack(input, bad, 2, 5, 0x80);
    require(got.diagnostic == 0x85, "unrouted nonfinite BF16 did not set bit 4");
    const std::vector<int64_t> valid{0, 511};
    const auto copied = ref::pack(input, valid, 2, 1, 0x81);
    require(copied.diagnostic == 0x85 && copied.inputs.back() == badBits,
            "nonfinite BF16 payload was altered or sticky bits were cleared");
  }
}

void coverage(const ref::Packed &packed, uint32_t tile) {
  const auto jobs = ref::makeJobs(packed, tile, ref::jobCapacity(packed.rows, packed.selections, 8));
  std::vector<uint32_t> covered(packed.offsets[512], 0);
  require(jobs.count <= ref::jobCapacity(packed.rows, packed.selections, tile),
          "GPU job launch bound was exceeded");
  for (uint32_t expert = 0; expert < ref::kExperts; ++expert) {
    require(jobs.offsets[expert + 1] - jobs.offsets[expert] ==
                (packed.counts[expert] + tile - 1) / tile,
            "expert tile-job count is incorrect");
    for (uint32_t index = jobs.offsets[expert]; index < jobs.offsets[expert + 1]; ++index) {
      const auto job = jobs.entries[index];
      require(job.expert == expert && job.rowBegin >= packed.offsets[expert] &&
                  job.rowBegin < packed.offsets[expert + 1],
              "matrix job crossed an expert boundary");
      const uint32_t end = std::min(job.rowBegin + tile, packed.offsets[expert + 1]);
      for (uint32_t row = job.rowBegin; row < end; ++row) ++covered[row];
    }
  }
  require(std::all_of(covered.begin(), covered.end(), [](uint32_t hits) { return hits == 1; }),
          "matrix jobs overlap or miss a valid packed row");
  require(std::all_of(jobs.entries.begin() + jobs.count, jobs.entries.end(),
                      [](const ref::Job &job) { return job == ref::Job{}; }),
          "worst-case allocator tail contains a live job");
}

void geometryTests() {
  for (uint32_t rows : {1U, 2U, 32U, 128U, 2048U}) {
    const auto input = inputBits(rows);
    for (uint32_t selections : {1U, 10U}) {
      std::vector<int64_t> ids(uint64_t{rows} * selections);
      for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t slot = 0; slot < selections; ++slot)
          ids[row * selections + slot] = (row * 73 + slot * 53) % 512;
      const auto spread = ref::pack(input, ids, rows, selections, 0x80);
      require(spread.diagnostic == 0x80 && spread.offsets[512] == rows * selections,
              "valid spread geometry changed diagnostics or route total");
      for (uint32_t expert = 0; expert < 512; ++expert) {
        uint32_t previous = 0;
        for (uint32_t index = spread.offsets[expert]; index < spread.offsets[expert + 1]; ++index) {
          const uint32_t flat = spread.routeMap[index];
          require(ids[flat] == expert &&
                      (index == spread.offsets[expert] || flat > previous) &&
                      spread.canonicalToPacked[flat] == index,
                  "bucket map is not expert-major and flat-route stable");
          previous = flat;
        }
      }
      for (uint32_t tile : {8U, 16U, 32U}) coverage(spread, tile);

      std::fill(ids.begin(), ids.end(), 511);
      const auto concentrated = ref::pack(input, ids, rows, selections, 0x80);
      require(concentrated.diagnostic == (selections == 1 ? 0x80U : 0x81U) &&
                  concentrated.counts[511] == rows * selections &&
                  concentrated.offsets[511] == 0,
              "maximum concentrated expert fixture changed");
      for (uint32_t flat = 0; flat < ids.size(); ++flat)
        require(concentrated.routeMap[flat] == flat,
                "duplicates were discarded or reordered");
      for (uint32_t tile : {8U, 16U, 32U}) coverage(concentrated, tile);
    }
  }
}

void tileBoundaryTests() {
  for (uint32_t tile : {8U, 16U, 32U}) {
    const uint32_t rows = tile * 3;
    const auto input = inputBits(rows);
    std::vector<int64_t> ids;
    ids.insert(ids.end(), tile - 1, 0);
    ids.insert(ids.end(), tile, 255);
    ids.insert(ids.end(), tile + 1, 511);
    const auto packed = ref::pack(input, ids, rows, 1, 0x80);
    const auto jobs = ref::makeJobs(packed, tile);
    require(jobs.count == 4 && jobs.entries[0] == ref::Job{0, 0} &&
                jobs.entries[1] == ref::Job{255, tile - 1} &&
                jobs.entries[2] == ref::Job{511, 2 * tile - 1} &&
                jobs.entries[3] == ref::Job{511, 3 * tile - 1},
            "M-1/M/M+1 tile boundary golden changed");
    coverage(packed, tile);
  }
  const auto input = inputBits(512);
  std::vector<int64_t> ids(512);
  for (uint32_t expert = 0; expert < 512; ++expert) ids[expert] = expert;
  const auto packed = ref::pack(input, ids, 512, 1);
  for (uint32_t tile : {8U, 16U, 32U}) {
    const auto jobs = ref::makeJobs(packed, tile);
    require(jobs.count == 512, "512 singleton buckets were not preserved");
    for (uint32_t expert = 0; expert < 512; ++expert)
      require(jobs.entries[expert] == ref::Job{expert, expert},
              "singleton expert tile job changed");
  }
  // 511 singleton buckets plus one heavy bucket produces near-maximal
  // fragmentation, including an incomplete final tile at maximum geometry.
  std::vector<int64_t> fragmented(20480, 511);
  for (uint32_t expert = 0; expert < 511; ++expert) fragmented[expert] = expert;
  const auto fragmentedPacked = ref::pack(inputBits(2048), fragmented, 2048, 10);
  for (uint32_t tile : {8U, 16U, 32U}) coverage(fragmentedPacked, tile);
  require(ref::jobCapacity(2048, 10, 8) == 3071 &&
              ref::jobCapacity(2048, 10, 16) == 1791 &&
              ref::jobCapacity(2048, 10, 32) == 1151,
          "maximum supported launch bound changed");
}

void rejectionTests() {
  const auto input = inputBits(1);
  const std::vector<int64_t> ids{0};
  rejects([&] { (void)ref::pack(input, ids, 0, 1); });
  rejects([&] { (void)ref::pack(input, ids, 2049, 1); });
  rejects([&] { (void)ref::pack(input, ids, 1, 0); });
  rejects([&] { (void)ref::pack(input, ids, 1, 11); });
  rejects([&] { (void)ref::pack({}, ids, 1, 1); });
  rejects([&] { (void)ref::pack(input, {}, 1, 1); });
  for (uint32_t tile : {0U, 1U, 4U, 7U, 64U, std::numeric_limits<uint32_t>::max()})
    rejects([&] { (void)ref::jobCapacity(1, 1, tile); });
  auto packed = ref::pack(input, ids, 1, 1);
  rejects([&] { (void)ref::makeJobs(packed, 8, 1); });
  packed.offsets[1] = 2;
  rejects([&] { (void)ref::makeJobs(packed, 8); });
}
} // namespace

int main() {
  try {
    handGolden();
    diagnosticGoldens();
    geometryTests();
    tileBoundaryTests();
    rejectionTests();
    std::cout << "PASS exact Flash MoE bucket CPU oracle; stable maps, raw BF16 copies, "
                 "diagnostics, all geometry bounds, M8/16/32 job coverage\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
