// Conversion is a Root-only GPU job. --check and --cpu-self-test never create
// a Metal device or load a model and can run independently on the CPU.
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "flash/FlashOperandStore.hpp"
#include "flash/FlashForward.hpp"
#include "engine/Json.hpp"
#include "engine/MemoryGovernor.hpp"

#import <Foundation/Foundation.h>

#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace splash::flash;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
bool within(const std::filesystem::path &path, const std::filesystem::path &root) {
  auto cursor = path.begin();
  for (const auto &part : root) if (cursor == path.end() || *cursor++ != part) return false;
  return true;
}
void cpuSelfTest() {
  struct EnvironmentRestore final {
    const char *name; bool present; std::string value;
    explicit EnvironmentRestore(const char *key) : name(key), present(std::getenv(key) != nullptr),
        value(present ? std::getenv(key) : "") {}
    ~EnvironmentRestore() { if (present) ::setenv(name, value.c_str(), 1); else ::unsetenv(name); }
  } restoreDense("SPLASH_FLASH_OPERAND_STORE"), restoreExpert("SPLASH_FLASH_INT8_EXPERT_STORE"),
    restoreHot("SPLASH_FLASH_HOT_EXPERT_PLAN");
  ::setenv("SPLASH_FLASH_OPERAND_STORE", "0", 1);
  ::setenv("SPLASH_FLASH_INT8_EXPERT_STORE", "0", 1);
  ::unsetenv("SPLASH_FLASH_HOT_EXPERT_PLAN");
  const FlashWeights unloaded;
  require(!FlashOperandStore::fromEnvironment(unloaded), "dense exact-zero opt-out inspected unloaded weights");
  require(FlashForward::expertCachePlannedBytes(unloaded) == 0,
      "expert exact-zero opt-out inspected unloaded weights");
  std::cout << "exact-zero selector opt-out: PASS\n";
  std::string pattern = (std::filesystem::temp_directory_path() / "splash-operand-store-test-XXXXXX").string();
  std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
  require(::mkdtemp(name.data()), "cannot create CPU fixture root");
  const std::filesystem::path root(name.data());
  struct Cleanup final { std::filesystem::path path; ~Cleanup() { std::error_code ignored;
    std::filesystem::remove_all(path, ignored); } } cleanup{root};
  const std::string source(64, '1'), manifest(64, '2');
  FlashOperandSpec spec{"language_model.model.layers.0.ple.value_proj", FlashOperandFormat::BF16,
      1, 64, 64, 4, 64, 32, 2048, 2, 128};
  std::vector<std::byte> bf16(64 * 64 * 2), f32(64 * 64 * 4);
  for (uint32_t i = 0; i < 64 * 64; ++i) {
    const float value = (int(i % 31) - 15) * .03125f;
    const uint32_t word = std::bit_cast<uint32_t>(value);
    const uint16_t half = static_cast<uint16_t>((word + 0x7fff + ((word >> 16) & 1)) >> 16);
    std::memcpy(bf16.data() + i * 2, &half, 2); std::memcpy(f32.data() + i * 4, &word, 4);
  }
  std::string identity;
  {
    FlashOperandStoreWriter writer(root / "store", source, manifest);
    writer.append(spec, bf16); spec.format = FlashOperandFormat::F32;
    writer.append(spec, f32); identity = writer.publish();
  }
  auto store = FlashOperandStore::load(root / "store", source, manifest);
  store.verifyPayloads(); require(store.identitySha256() == identity, "writer/loader identity mismatch");
  require(store.contains(spec.projection, FlashOperandFormat::BF16) &&
      store.contains(spec.projection, FlashOperandFormat::F32), "format lookup failed");
  require(store.plannedBytes(store.specs()) == 32768, "mapped byte plan not exact");
  bool rejected = false;
  try { FlashOperandStoreWriter writer(root / "store", source, manifest); }
  catch (const std::exception &) { rejected = true; }
  require(rejected, "existing destination overwritten");
  {
    FlashOperandStoreWriter writer(root / "abandoned", source, manifest);
    spec.format = FlashOperandFormat::BF16; writer.append(spec, bf16);
  }
  require(!std::filesystem::exists(root / "abandoned"), "abandoned writer published");
  for (const auto &entry : std::filesystem::directory_iterator(root))
    require(entry.path().filename() == "store", "abandoned staging directory leaked");
  rejected = false;
  try { (void)FlashOperandStore::load(root / "store", std::string(64, '3'), manifest); }
  catch (const std::exception &) { rejected = true; }
  require(rejected, "wrong model identity accepted");
  std::cout << "operand-store CPU writer/loader self-test: PASS\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if ((argc == 5 || argc == 6) && std::string_view(argv[1]) == "--check") {
        require(argc == 5 || std::string_view(argv[5]) == "--verify-payloads", "invalid checker argument");
        auto store = FlashOperandStore::load(argv[2], argv[3], argv[4]);
        if (argc == 6) store.verifyPayloads();
        std::cout << "{\"schema\":\"splash-local-affine-operands-v1\",\"manifest_sha256\":"
          << splash::json::quote(store.identitySha256()) << ",\"entries\":" << store.specs().size()
          << ",\"payloads_verified\":" << (argc == 6 ? "true" : "false") << "}\n";
        return 0;
      }
      if (argc < 4) throw std::invalid_argument(
          "usage: flash-operand-export METALLIB PACKAGE NEW_DEST [--bf16-only|--f32-only] [--include-f32-head] [PREFIX ...]\n"
          "       flash-operand-export --check STORE SOURCE_SHA256 WEIGHTS_FINGERPRINT [--verify-payloads]\n"
          "       flash-operand-export --cpu-self-test");
      bool bf16 = true, f32 = true, includeFloatHead = false;
      std::vector<std::string> selected;
      for (int i = 4; i < argc; ++i) {
        const std::string value(argv[i]);
        if (value == "--bf16-only") { require(bf16 && f32, "duplicate format option"); f32 = false; }
        else if (value == "--f32-only") { require(bf16 && f32, "duplicate format option"); bf16 = false; }
        else if (value == "--include-f32-head") includeFloatHead = true;
        else { require(!value.starts_with("--"), "unknown exporter option"); selected.push_back(value); }
      }
      const auto destination = std::filesystem::weakly_canonical(std::filesystem::absolute(argv[3]));
      const auto package = std::filesystem::canonical(argv[2]);
      require(!within(destination, package), "export cannot modify source package directory");
      if (const char *userHome = std::getenv("HOME")) {
        const auto originalModels = std::filesystem::weakly_canonical(std::filesystem::path(userHome) / ".omlx");
        require(!within(destination, originalModels), "export cannot modify original .omlx directory");
      }
      require(!std::filesystem::exists(std::filesystem::symlink_status(destination)), "export destination already exists");
      splash::metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, package);
      const auto bf16Names = selected.empty() ? FlashDenseCache::defaultPrefixes(weights, true) : selected;
      const auto f32Names = selected.empty() ? FlashFloatDenseCache::defaultPrefixes(weights, includeFloatHead) : selected;
      uint64_t planned = 0;
      if (bf16) planned += FlashDenseCache::plannedBytes(weights, bf16Names);
      if (f32) planned += FlashFloatDenseCache::plannedBytes(weights, f32Names);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      require(physical > reserve, "insufficient physical memory for governed conversion");
      splash::engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      auto reservation = governor.tryReserve(planned);
      require(bool(reservation), "governor denied saved operand conversion before allocation");
      const auto space = std::filesystem::space(destination.parent_path());
      require(space.available > planned + (64ULL << 20), "insufficient disk space for saved operands");
      FlashOperandStoreWriter writer(destination, weights.sourceIdentity(), weights.manifestFingerprint());
      std::unique_ptr<FlashDenseCache> bf16Cache;
      std::unique_ptr<FlashFloatDenseCache> f32Cache;
      if (bf16) {
        std::cerr << "Converting selected BF16 operands once...\n";
        bf16Cache = std::make_unique<FlashDenseCache>(backend, weights, bf16Names);
        for (const auto &prefix : bf16Names)
          writer.append(flashOperandSpec(prefix, FlashOperandFormat::BF16, weights.projection(prefix)), bf16Cache->tensor(prefix));
      }
      if (f32) {
        std::cerr << "Converting selected original-coefficient F32 operands once...\n";
        f32Cache = std::make_unique<FlashFloatDenseCache>(backend, weights, f32Names);
        for (const auto &prefix : f32Names)
          writer.append(flashOperandSpec(prefix, FlashOperandFormat::F32, weights.projection(prefix)), f32Cache->tensor(prefix));
      }
      reservation->commit();
      const std::string identity = writer.publish();
      std::cout << "{\"schema\":\"splash-local-affine-operands-v1\",\"directory\":"
        << splash::json::quote(destination.string()) << ",\"manifest_sha256\":" << splash::json::quote(identity)
        << ",\"source_identity_sha256\":" << splash::json::quote(weights.sourceIdentity())
        << ",\"weights_manifest_fingerprint\":" << splash::json::quote(weights.manifestFingerprint())
        << ",\"bf16_entries\":" << (bf16 ? bf16Names.size() : 0)
        << ",\"f32_entries\":" << (f32 ? f32Names.size() : 0)
        << ",\"planned_bytes\":" << planned << ",\"allocated_bytes\":" << backend.memoryStats().allocatedBytes << "}\n";
      return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
