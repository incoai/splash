// CPU-only: link the production hash and layout planner with -dead_strip.
// Expected hashes are generated independently by a segment-based Python oracle.
#include "flash/FlashPLE.hpp"
#include "flash/FlashPLESSDLayout.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::flash;
namespace {
uint64_t checks = 0;
void require(bool value, const char *why) {
  ++checks;
  if (!value) throw std::runtime_error(why);
}
template<class F> void rejects(F f, const char *why) {
  bool caught = false;
  try { f(); } catch (const std::invalid_argument &) { caught = true; }
  catch (const std::overflow_error &) { caught = true; }
  require(caught, why);
}
template<class T> T read(std::ifstream &input) {
  T value{};
  input.read(reinterpret_cast<char *>(&value), sizeof(value));
  require(bool(input), "short independent fixture");
  return value;
}
template<class T> std::vector<T> readVector(std::ifstream &input, size_t count) {
  std::vector<T> value(count);
  input.read(reinterpret_cast<char *>(value.data()), count * sizeof(T));
  require(bool(input), "short independent vector");
  return value;
}
void rejectionChecks(const std::array<int64_t, 3> &m,
                     const std::array<int64_t, 16> &s,
                     const std::array<int64_t, 16> &o, uint64_t table) {
  const auto invoke = [&](std::vector<int64_t> t, std::vector<int64_t> h,
                          auto sizes, auto offsets, FlashPLEGeometry g,
                          uint64_t n) {
    const auto before = h;
    rejects([&] { (void)computePLENgramIDs(t, h, m, sizes, offsets, g, n); },
            "invalid host hash input accepted");
    require(h == before, "rejected host hash changed request history");
  };
  FlashPLEGeometry g{};
  const std::vector<int64_t> history{248044, 248044};
  for (int64_t token : {INT64_MIN, int64_t{-1}, int64_t{248320}, INT64_MAX})
    invoke({token}, history, s, o, g, table);
  for (int64_t token : {INT64_MIN, int64_t{-1}, int64_t{248320}, INT64_MAX}) {
    // Host staging is deliberately stricter than a masked unused GPU history.
    invoke({1}, {token, 248044}, s, o, g, table);
    invoke({1}, {248044, token}, s, o, g, table);
  }
  for (size_t head : {size_t{0}, size_t{7}, size_t{8}, size_t{15}}) {
    for (int64_t size : {INT64_MIN, int64_t{-1}, int64_t{0}, INT64_MAX}) {
      auto bad = s; bad[head] = size;
      invoke({1}, history, bad, o, g, table);
    }
    for (int64_t offset : {INT64_MIN, int64_t{-1}, int64_t(table), INT64_MAX}) {
      auto bad = o; bad[head] = offset;
      invoke({1}, history, s, bad, g, table);
    }
  }
  invoke({}, history, s, o, g, table);
  invoke({1, 2}, history, s, o, g, table);
  invoke({1}, {248044}, s, o, g, table);
  invoke({1}, history, s, o, g, 0);
  invoke({1}, history, s, o, g, uint64_t(INT64_MAX) + 1);
  for (uint32_t value : {0u, 9u}) {
    auto bad = g; bad.streams = value;
    invoke({1}, history, s, o, bad, table);
  }
  for (unsigned member = 0; member < 6; ++member) {
    auto bad = g;
    if (member == 0) bad.lanes = 0;
    if (member == 1) bad.rows = 0;
    if (member == 2) bad.width = 0;
    if (member == 3) bad.vocabularySize = 0;
    if (member == 4) bad.eosToken = bad.vocabularySize;
    if (member == 5) bad.epsilon = std::numeric_limits<float>::quiet_NaN();
    invoke({1}, history, s, o, bad, table);
  }
  g.lanes = UINT32_MAX; g.rows = UINT32_MAX;
  invoke({1}, history, s, o, g, table);
}
void layoutAdversaries() {
  constexpr uint64_t a = kFlashPLESSDAlignment;
  const auto valid = flashPLESSDPlan(4*a, {{0, 1, false}, {a, a+1, true},
                                         {2*a, 2*a+1, false}, {3*a, 3*a+1, false}});
  require(valid.windows.size() == 2 && valid.windows[1].end == 4*a,
          "adjacent non-PLE windows did not coalesce");
  require(valid.mappedBytes == 3*a && valid.diskOnlyBytes == a,
          "layout accounting differs from unique page partition");
  for (const auto &ranges : std::vector<std::vector<FlashPLESSDLayoutRange>>{
       {}, {{1, 2, false}}, {{0, 0, false}}, {{0, 4*a+1, false}},
       {{0, 1, false}, {0, 2, true}}, {{0, a+1, false}, {a, a+2, true}},
       {{0, 1, false}, {2*a, 2*a+1, true}}, {{0, 1, false}}})
    rejects([&] { (void)flashPLESSDPlan(4*a, ranges); }, "bad layout accepted");
  rejects([&] { (void)flashPLESSDPlan(0, {{0, 1, false}}); }, "empty payload accepted");
  rejects([&] { (void)flashPLESSDPlan(4*a-1, {{0, 1, false}}); }, "unaligned payload accepted");
  rejects([&] { (void)flashPLESSDAligned(UINT64_MAX); }, "alignment overflow accepted");
  const std::string prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.";
  for (unsigned i = 0; i < 128; ++i)
    for (const auto *suffix : {".weight", ".scales", ".biases"})
      require(flashPLESSDTableTensor(prefix + std::to_string(i) + suffix),
              "valid disk PLE tensor misclassified");
  for (const auto *ending : {"128.weight", "999999999999999999999.weight", "01.weight",
                             "-1.weight", "1.weight.foo", "1", ".weight", "0.norm"})
    require(!flashPLESSDTableTensor(prefix + ending), "invalid PLE name accepted");
  require(!flashPLESSDTableTensor("vision." + prefix + "0.weight"), "unrelated PLE name accepted");
  require(!flashPLESSDStreamingValue(nullptr) && !flashPLESSDStreamingValue("0") &&
          flashPLESSDStreamingValue("1"), "streaming default policy incorrect");
  for (const auto *value : {"", "01", "true", "2", " 1", "1 "})
    rejects([&] { (void)flashPLESSDStreamingValue(value); }, "malformed streaming option accepted");
  require(flashPLESSDCacheBytes(nullptr, false) == (64ULL << 20) &&
          flashPLESSDCacheBytes(nullptr, true) == (64ULL << 20), "default row cache budget incorrect");
  for (uint64_t mib : {0ULL, 1ULL, 64ULL, 1024ULL}) {
    const auto text = std::to_string(mib);
    require(flashPLESSDCacheBytes(text.c_str(), true) == (mib << 20), "valid cache size rejected");
    rejects([&] { (void)flashPLESSDCacheBytes(text.c_str(), false); }, "cache size without SSD mode accepted");
  }
  for (const auto *value : {"", "-1", "+1", "1025", "1048576", "18446744073709551616",
                             "1 ", " 1", "1.5", "true", "64MB"})
    rejects([&] { (void)flashPLESSDCacheBytes(value, true); }, "malformed cache size accepted");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: oracle FIXTURE.bin");
    std::ifstream input(argv[1], std::ios::binary);
    require(bool(input), "independent fixture missing");
    require(read<uint64_t>(input) == 0x31564453534c4550ULL, "bad fixture magic");
    const auto count = read<uint32_t>(input);
    uint64_t comparedIDs = 0, nativeBytes = 0, diskBytes = 0;
    uint64_t nativeWindows = 0, diskTensors = 0, pageComparisons = 0;
    for (uint32_t i = 0; i < count; ++i) {
      const auto lanes = read<uint32_t>(input), rows = read<uint32_t>(input);
      const auto table = read<uint64_t>(input);
      auto m0 = readVector<int64_t>(input, 3), s0 = readVector<int64_t>(input, 16);
      auto o0 = readVector<int64_t>(input, 16), tokens = readVector<int64_t>(input, uint64_t(lanes)*rows);
      auto history = readVector<int64_t>(input, uint64_t(lanes)*2);
      const auto expected = readVector<int64_t>(input, uint64_t(lanes)*rows*16);
      const auto after = readVector<int64_t>(input, uint64_t(lanes)*2);
      FlashPLEGeometry g{}; g.lanes = lanes; g.rows = rows;
      const auto actual = computePLENgramIDs(tokens, history, m0, s0, o0, g, table);
      require(actual == expected, "production host hash differs from independent segment oracle");
      require(history == after, "production host history differs from independent segment oracle");
      for (const auto id : actual) require(id >= 0 && uint64_t(id) < table, "valid host hash escaped source table");
      comparedIDs += actual.size();
      if (i == 0) {
        std::array<int64_t,3> m; std::array<int64_t,16>s,o;
        std::copy(m0.begin(), m0.end(), m.begin());
        std::copy(s0.begin(), s0.end(), s.begin());
        std::copy(o0.begin(), o0.end(), o.begin());
        rejectionChecks(m,s,o,table);
      }
    }
    const auto shards = read<uint32_t>(input);
    require(shards == 21, "wrong aligned shard inventory");
    for (uint32_t shard = 0; shard < shards; ++shard) {
      const auto bytes = read<uint64_t>(input);
      const auto tensors = read<uint32_t>(input);
      std::vector<FlashPLESSDLayoutRange> ranges;
      for (uint32_t j = 0; j < tensors; ++j)
        ranges.push_back({read<uint64_t>(input), read<uint64_t>(input), bool(read<uint32_t>(input))});
      const auto plan = flashPLESSDPlan(bytes, ranges);
      nativeBytes += plan.mappedBytes; diskBytes += plan.diskOnlyBytes;
      nativeWindows += plan.windows.size(); diskTensors += plan.diskTensorCount;
      for (const auto &window : plan.windows)
        for (const auto &range : ranges) {
          ++pageComparisons;
          if (range.diskOnly)
            require(window.end <= range.begin || window.begin >= flashPLESSDAligned(range.end),
                    "native window overlaps disk-only PLE page");
        }
      for (const auto &range : ranges) if (!range.diskOnly) {
        unsigned matches = 0;
        for (const auto &window : plan.windows)
          matches += range.begin >= window.begin && range.end <= window.end;
        require(matches == 1, "non-PLE tensor is not covered by exactly one native window");
      }
    }
    require(nativeBytes == 74317889536ULL && diskBytes == 32002539520ULL &&
            nativeWindows == 28 && diskTensors == 384, "actual model native/disk partition differs");
    require(input.peek() == EOF, "unexpected fixture trailing bytes");
    layoutAdversaries();
    std::cout << "{\"valid\":true,\"gpu_executed\":false,\"fixture_cases\":" << count
      << ",\"compared_source_ids\":" << comparedIDs << ",\"checks\":" << checks
      << ",\"native_bytes\":" << nativeBytes << ",\"disk_page_bytes\":" << diskBytes
      << ",\"native_windows\":" << nativeWindows << ",\"disk_tensor_count\":" << diskTensors
      << ",\"window_tensor_pairs_checked\":" << pageComparisons << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
