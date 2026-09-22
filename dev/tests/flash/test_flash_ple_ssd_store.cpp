#include "flash/FlashPLESSDStore.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <unistd.h>

using splash::flash::FlashPLESSDStore;
namespace {
uint8_t value(uint64_t id, uint64_t byte) {
  return static_cast<uint8_t>((id * 37 + byte * 13 + (id >> 8) * 23) & 255);
}
}
int main() {
  constexpr uint64_t parts = 4, partRows = 1024, planeBytes = 100;
  const auto directory = std::filesystem::temp_directory_path() /
      ("splash-ple-ssd-cpu-" + std::to_string(::getpid()));
  std::filesystem::create_directory(directory);
  const auto path = directory / "fixture.bin";
  std::vector<uint8_t> bytes(parts * partRows * planeBytes);
  std::vector<FlashPLESSDStore::Part> inventory;
  for (uint64_t p = 0; p < parts; ++p) {
    const uint64_t base = p * partRows * planeBytes;
    inventory.push_back({partRows, {0, base, 80},
        {0, base + partRows * 80, 10}, {0, base + partRows * 90, 10}});
    for (uint64_t r = 0; r < partRows; ++r) {
      for (uint64_t b = 0; b < 80; ++b) bytes[base + r * 80 + b] = value(p * partRows + r, b);
      for (uint64_t b = 0; b < 10; ++b) {
        bytes[base + partRows * 80 + r * 10 + b] = value(p * partRows + r, 80 + b);
        bytes[base + partRows * 90 + r * 10 + b] = value(p * partRows + r, 90 + b);
      }
    }
  }
  { std::ofstream stream(path, std::ios::binary); stream.write(
      reinterpret_cast<const char *>(bytes.data()), bytes.size()); }
  uint64_t checks = 0;
  for (const uint64_t budget : {0ULL, 312ULL, 4096ULL, 65536ULL}) {
    FlashPLESSDStore store({{path, bytes.size()}}, inventory, {budget, 256, true});
    assert(store.tableRows() == parts * partRows);
    assert(store.shardRows() == partRows && store.partCount() == parts);
    std::mt19937_64 random(100 + budget);
    std::vector<int64_t> ids{0, 1, 1, 1023, 1024, 2048, 4095};
    for (uint64_t iteration = 0; iteration < 3000; ++iteration) {
      if (iteration) {
        ids.resize(1 + random() % 17);
        for (auto &id : ids) id = static_cast<int64_t>(random() % (parts * partRows));
        if (iteration % 3 == 0) ids.back() = ids.front();
      }
      std::vector<uint8_t> output(ids.size() * planeBytes, 0);
      store.lookupRows(ids, output);
      for (size_t r = 0; r < ids.size(); ++r)
        for (uint64_t b = 0; b < planeBytes; ++b) {
          assert(output[r * planeBytes + b] == value(ids[r], b));
          ++checks;
        }
      const auto stats = store.statistics();
      assert(stats.cacheAccountedBytes <= budget && !stats.poisoned);
      assert(stats.completedReadBytes == stats.requestedReadBytes);
      assert(stats.requestedReadBytes == stats.logicalMissBytes);
      ++checks;
      if (iteration % 500 == 0) store.clearCache();
    }
    std::array<int64_t, 1> bad{-1};
    std::array<uint8_t, 100> unchanged{};
    unchanged.fill(0xA5);
    bool rejected = false;
    try { store.lookupRows(bad, unchanged); } catch (...) { rejected = true; }
    assert(rejected && !store.statistics().poisoned);
    for (auto byte : unchanged) assert(byte == 0xA5);
    store.lookupRows({}, {});
    checks += 102;
  }
  std::filesystem::remove_all(directory);
  std::cout << "{\"valid\":true,\"checks\":" << checks
            << ",\"gpu_execution\":false,\"cache_churn_batches\":12000}\n";
}
