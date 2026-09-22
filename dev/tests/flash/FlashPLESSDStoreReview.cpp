#include "flash/FlashPLESSDStore.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
using Store = splash::flash::FlashPLESSDStore;
uint64_t checks = 0;

void require(bool value, std::string_view reason) {
  ++checks;
  if (!value) throw std::runtime_error(std::string(reason));
}

template <class Function> void reject(Function &&function, std::string_view reason) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  require(rejected, reason);
}

void writeFile(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("cannot write disposable CPU fixture");
  stream.close();
  if (!stream) throw std::runtime_error("cannot close disposable CPU fixture");
}

std::vector<uint8_t> readFile(const std::filesystem::path &path, uint64_t bytes) {
  std::vector<uint8_t> result(static_cast<size_t>(bytes));
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char *>(result.data()),
              static_cast<std::streamsize>(result.size()));
  if (!stream) throw std::runtime_error("cannot read disposable CPU fixture");
  return result;
}

uint8_t pattern(uint64_t part, uint64_t row, uint32_t plane, uint32_t byte) {
  return static_cast<uint8_t>((part * 71 + row * 31 + plane * 13 + byte * 7 + 19) % 251);
}

struct Fixture final {
  std::filesystem::path directory;
  std::vector<Store::Source> sources;
  std::vector<Store::Part> parts;
  std::array<std::vector<uint8_t>, 3> original;
  uint64_t rowsPerPart;

  explicit Fixture(uint64_t rows = 7, uint64_t partCount = 2) : rowsPerPart(rows) {
    std::array<char, 64> temporary{};
    constexpr std::string_view prefix = "/tmp/splash-ple-ssd-review-XXXXXX";
    std::copy(prefix.begin(), prefix.end(), temporary.begin());
    const char *created = ::mkdtemp(temporary.data());
    if (!created) throw std::runtime_error("cannot make disposable CPU fixture directory");
    directory = created;
    constexpr std::array<uint64_t, 3> strides{83, 13, 15};
    constexpr std::array<uint64_t, 3> offsets{47, 17, 91};
    constexpr std::array<uint32_t, 3> counts{80, 10, 10};
    const uint64_t partSpacing = rows * 83 + 256;
    const uint64_t sourceBytes = partCount * partSpacing + 256;
    for (uint32_t plane = 0; plane < 3; ++plane) {
      original[plane].assign(static_cast<size_t>(sourceBytes), 0xff);
      sources.push_back({directory / ("plane-" + std::to_string(plane)), sourceBytes});
    }
    for (uint64_t part = 0; part < partCount; ++part) {
      Store::Part specification;
      specification.rows = rows;
      std::array<Store::Plane *, 3> planes{
          &specification.weights, &specification.scales, &specification.biases};
      for (uint32_t plane = 0; plane < 3; ++plane) {
        *planes[plane] = {plane, offsets[plane] + part * partSpacing, strides[plane]};
        for (uint64_t row = 0; row < rows; ++row)
          for (uint32_t byte = 0; byte < counts[plane]; ++byte)
            original[plane][static_cast<size_t>(planes[plane]->offset +
                row * planes[plane]->rowStride + byte)] = pattern(part, row, plane, byte);
      }
      parts.push_back(specification);
    }
    for (uint32_t plane = 0; plane < 3; ++plane) {
      writeFile(sources[plane].path, original[plane]);
      // A deliberately old timestamp makes mutation detection deterministic,
      // without waiting for the filesystem clock to tick.
      const std::array<timespec, 2> times{{{123456, 0}, {123456, 0}}};
      if (::utimensat(AT_FDCWD, sources[plane].path.c_str(), times.data(), 0) != 0)
        throw std::runtime_error("cannot timestamp disposable CPU fixture");
    }
  }

  ~Fixture() {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }

  std::array<uint8_t, 100> row(uint64_t id) const {
    std::array<uint8_t, 100> result{};
    const uint64_t part = id / rowsPerPart;
    const uint64_t local = id % rowsPerPart;
    for (uint32_t byte = 0; byte < 80; ++byte) result[byte] = pattern(part, local, 0, byte);
    for (uint32_t byte = 0; byte < 10; ++byte) result[80 + byte] = pattern(part, local, 1, byte);
    for (uint32_t byte = 0; byte < 10; ++byte) result[90 + byte] = pattern(part, local, 2, byte);
    return result;
  }

  std::unique_ptr<Store> store(uint64_t cacheBytes = 4096) const {
    Store::Options options;
    options.cacheBytes = cacheBytes;
    options.maxCoalescedReadBytes = 80;
    options.noCache = false;
    return std::make_unique<Store>(sources, parts, options);
  }
};

void exactRows(Store &store, const Fixture &fixture, std::span<const int64_t> ids) {
  std::vector<uint8_t> output(ids.size() * 100, 0xee);
  store.lookupRows(ids, output);
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto expected = fixture.row(static_cast<uint64_t>(ids[i]));
    require(std::equal(expected.begin(), expected.end(), output.begin() + i * 100),
            "padded original planes reconstructed the wrong 80/10/10 bytes");
  }
}

void paddedRowsAndAliasing() {
  Fixture fixture;
  auto store = fixture.store();
  const std::array<int64_t, 7> ids{13, 0, 7, 6, 1, 12, 8};
  exactRows(*store, fixture, ids);
  exactRows(*store, fixture, ids);
  for (uint32_t plane = 0; plane < 3; ++plane)
    require(readFile(fixture.sources[plane].path, fixture.sources[plane].byteCount) ==
                fixture.original[plane], "lookup changed original fixture payload bytes");

  // The first row is warm. Without overlap rejection its cache copy overwrites
  // the second ID before the implementation uses that ID as a part index.
  std::array<int64_t, 32> shared{};
  shared.fill(0);
  shared[0] = 0;
  shared[1] = 1;
  auto *bytes = reinterpret_cast<uint8_t *>(shared.data());
  const auto before = shared;
  reject([&] {
    store->lookupRows(std::span<const int64_t>(shared.data(), 2),
                      std::span<uint8_t>(bytes, 200));
  }, "overlapping ID and output spans were accepted");
  require(shared == before, "alias rejection wrote output bytes");
  require(!store->statistics().poisoned, "invalid caller alias poisoned immutable source store");
  exactRows(*store, fixture, ids);

  const std::array<int64_t, 1> validID{0};
  auto *nearAddressEnd = reinterpret_cast<uint8_t *>(UINTPTR_MAX - 49);
  reject([&] { store->lookupRows(validID, std::span<uint8_t>(nearAddressEnd, 100)); },
         "output span address extent overflow was accepted");
  require(!store->statistics().poisoned, "caller address extent overflow poisoned store");
}

template <class Mutation> void warmMutation(Mutation &&mutation, std::string_view reason) {
  Fixture fixture;
  auto store = fixture.store();
  const std::array<int64_t, 1> id{0};
  exactRows(*store, fixture, id);
  mutation(fixture);
  std::array<uint8_t, 100> output;
  output.fill(0xee);
  reject([&] { store->lookupRows(id, output); }, reason);
  require(std::all_of(output.begin(), output.end(), [](uint8_t value) { return value == 0xee; }),
          "changed-source validation wrote warmed row output");
  const auto stats = store->statistics();
  require(stats.poisoned && stats.sourceValidationFailures == 1 && stats.failedBatches == 1,
          "changed source did not poison store with one validation failure");
  store->clearCache();
  reject([&] { store->lookupRows(id, output); }, "clearCache revived a poisoned source store");
  require(store->statistics().sourceValidationFailures == 1,
          "poison rejection revalidated changed source");
}

void sourceChanges() {
  warmMutation([](const Fixture &fixture) {
    const int fd = ::open(fixture.sources[0].path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open disposable fixture for mutation");
    const uint8_t changed = 0x81;
    const auto count = ::pwrite(fd, &changed, 1,
                               static_cast<off_t>(fixture.parts[0].weights.offset));
    ::close(fd);
    if (count != 1) throw std::runtime_error("cannot mutate disposable fixture byte");
  }, "warm cache concealed same-size source mutation");
  warmMutation([](const Fixture &fixture) {
    const auto replacement = fixture.directory / "replacement";
    writeFile(replacement, fixture.original[0]);
    std::filesystem::rename(replacement, fixture.sources[0].path);
  }, "warm cache concealed same-size identical source pathname replacement");
  warmMutation([](const Fixture &fixture) {
    std::filesystem::resize_file(fixture.sources[2].path, fixture.sources[2].byteCount - 1);
  }, "warm cache concealed source truncation");
  warmMutation([](const Fixture &fixture) {
    std::filesystem::remove(fixture.sources[1].path);
  }, "warm cache concealed removed source pathname");
}

void constructorBounds() {
  Fixture fixture;
  auto sources = fixture.sources;
  auto parts = fixture.parts;
  Store::Options options;
  options.cacheBytes = 0;
  options.maxCoalescedReadBytes = 80;
  options.noCache = false;
  auto construct = [&] { Store store(sources, parts, options); };
  const auto reset = [&] { sources = fixture.sources; parts = fixture.parts; };

  parts[0].weights.offset = UINT64_MAX - 10;
  reject(construct, "overflowing affine plane offset was accepted");
  reset();
  parts[0].weights.rowStride = UINT64_MAX;
  reject(construct, "overflowing affine row multiplication was accepted");
  reset();
  for (auto &part : parts) part.rows = UINT64_MAX;
  reject(construct, "overflowing table inventory was accepted");
  reset();
  for (auto &part : parts) part.rows = static_cast<uint64_t>(INT64_MAX / 2) + 1;
  reject(construct, "table inventory beyond signed IDs was accepted");
  reset();
  const auto maximumOffset = static_cast<uint64_t>(std::numeric_limits<off_t>::max());
  sources[0].byteCount = maximumOffset + 1;
  for (auto &part : parts) part.rows = 1;
  parts[0].weights.offset = maximumOffset - 79;
  reject(construct, "affine plane beyond native pread offset extent was accepted");
  reset();
  sources[0].byteCount = maximumOffset + 1;
  reject(construct, "source byte count beyond native file extent was accepted");
  reset();
  parts[0].weights.source = static_cast<uint32_t>(sources.size());
  reject(construct, "invalid plane source index was accepted");
  reset();
  parts[0].scales.rowStride = 9;
  reject(construct, "short affine scale row stride was accepted");
  reset();
  sources[1].byteCount -= 1;
  reject(construct, "source metadata byte count mismatch was accepted");
  reset();
  sources[1].path = fixture.directory;
  reject(construct, "directory payload source was accepted");
  reset();
  options.maxCoalescedReadBytes = 79;
  reject(construct, "scratch below one weight row was accepted");
  options.maxCoalescedReadBytes = 80;
  options.cacheBytes = 1024ULL * 1024 * 1024 + 1;
  reject(construct, "CPU cache budget above its hard limit was accepted");
}

uint64_t rowHash(uint64_t id) {
  id ^= id >> 30;
  id *= 0xbf58476d1ce4e5b9ULL;
  id ^= id >> 27;
  id *= 0x94d049bb133111ebULL;
  return id ^ (id >> 31);
}

void circularProbeDeletion() {
  Fixture fixture(512, 1);
  // Current fixed-array admission gives three slots and 16 hash buckets. Rows
  // chosen for the last bucket force linear probing to wrap through bucket 0.
  auto store = fixture.store(768);
  std::vector<int64_t> colliding;
  for (uint64_t id = 0; id < store->tableRows(); ++id)
    if ((rowHash(id) & 15) == 15) colliding.push_back(static_cast<int64_t>(id));
  require(colliding.size() >= 8, "fixture lacks wrapped probe collisions");
  exactRows(*store, fixture, std::span<const int64_t>(colliding.data(), 3));
  require(store->statistics().cachedRows == 3, "collision fixture cache admission changed");
  const std::array<int64_t, 1> fourth{colliding[3]};
  exactRows(*store, fixture, fourth);
  const auto readRequests = store->statistics().readRequests;
  const std::array<int64_t, 2> retained{colliding[1], colliding[2]};
  exactRows(*store, fixture, retained);
  require(store->statistics().readRequests == readRequests,
          "backward shift lost reachable entries across the bucket wrap");
  for (size_t cycle = 0; cycle < 200; ++cycle) {
    const std::array<int64_t, 1> evict{colliding[3 + cycle % (colliding.size() - 3)]};
    exactRows(*store, fixture, evict);
    exactRows(*store, fixture, evict);
  }
  const auto stats = store->statistics();
  require(stats.cacheAccountedBytes <= stats.cacheBudgetBytes && stats.cachedRows == 3 &&
              stats.cacheEvictions > 100 && !stats.poisoned,
          "wrapped probe eviction corrupted fixed-array cache bounds or liveness");
}
} // namespace

int main() {
  try {
    paddedRowsAndAliasing();
    sourceChanges();
    constructorBounds();
    circularProbeDeletion();
    std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":" << checks << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL PLE SSD independent CPU review: " << error.what() << '\n';
    return 1;
  }
}
