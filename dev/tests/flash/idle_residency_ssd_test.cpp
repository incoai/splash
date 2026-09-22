#include "flash/FlashIdleResidencyPolicy.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace policy = splash::flash::idle_maintenance;
namespace {
uint64_t checks = 0;
void require(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejected(F &&function, const char *message) {
  bool invalid = false;
  try { function(); }
  catch (const std::invalid_argument &) { invalid = true; }
  require(invalid, message);
}
}

int main() {
  try {
    for (const auto mode : {policy::StorageMode::Raw, policy::StorageMode::PLESSD}) {
      const auto geometry = policy::ownerGeometry(mode);
      require(geometry.ownerCount - geometry.originalCount == 1113,
              "derived owner union differs between storage modes");
      require(geometry.ownerBytes - geometry.originalBytes == 38006423552ULL,
              "derived owner bytes differ between storage modes");
      policy::validateGeometry(policy::kSource, policy::kLayout,
          geometry.originalCount, geometry.originalBytes, geometry.ownerCount,
          geometry.ownerBytes, 48, 512, 2560, mode);
      require(policy::qualifiedUnionAvailable(policy::kSource, policy::kLayout,
          geometry.originalCount, geometry.originalBytes, geometry.ownerCount,
          geometry.ownerBytes, 48, 512, 2560, mode), "complete union unavailable");
      const auto other = policy::ownerGeometry(mode == policy::StorageMode::Raw
          ? policy::StorageMode::PLESSD : policy::StorageMode::Raw);
      rejected([&] { policy::validateGeometry(policy::kSource, policy::kLayout,
          other.originalCount, other.originalBytes, other.ownerCount,
          other.ownerBytes, 48, 512, 2560, mode); }, "opposite storage mode geometry accepted");
      for (uint32_t field = 0; field != 9; ++field) {
        for (const int direction : {-1, 1}) {
          auto changed = geometry;
          std::string source(policy::kSource), layout(policy::kLayout);
          uint32_t layers = 48, experts = 512, hidden = 2560;
          switch (field) {
            case 0: source[0] = direction == -1 ? '0' : 'f'; break;
            case 1: layout[0] = direction == -1 ? '0' : 'f'; break;
            case 2: changed.originalCount += direction; break;
            case 3: changed.originalBytes += direction; break;
            case 4: changed.ownerCount += direction; break;
            case 5: changed.ownerBytes += direction; break;
            case 6: layers += direction; break;
            case 7: experts += direction; break;
            case 8: hidden += direction; break;
          }
          rejected([&] { policy::validateGeometry(source, layout,
              changed.originalCount, changed.originalBytes, changed.ownerCount,
              changed.ownerBytes, layers, experts, hidden, mode); }, "altered geometry accepted");
        }
      }
      for (const uint64_t missing : {uint64_t{0}, geometry.originalCount,
              geometry.ownerCount - 1, geometry.ownerCount + 1})
        require(!policy::qualifiedUnionAvailable(policy::kSource, policy::kLayout,
            geometry.originalCount, geometry.originalBytes, missing,
            geometry.ownerBytes, 48, 512, 2560, mode), "missing store did not disable maintenance");
      for (const uint64_t bytes : {uint64_t{0}, geometry.originalBytes,
              geometry.ownerBytes - 1, geometry.ownerBytes + 1})
        require(!policy::qualifiedUnionAvailable(policy::kSource, policy::kLayout,
            geometry.originalCount, geometry.originalBytes, geometry.ownerCount,
            bytes, 48, 512, 2560, mode), "missing store bytes did not disable maintenance");

      std::vector<uint32_t> mixed(geometry.ownerCount);
      std::array<uint32_t, policy::kOutputWords> output{};
      output.fill(policy::kGuard);
      uint32_t checksum = 0;
      for (uint32_t index = 0; index != mixed.size(); ++index) {
        mixed[index] = policy::mixedWord(index * 0xdeadbeefu, index);
        output[policy::kOutputValueBegin + index] = mixed[index];
        checksum ^= mixed[index];
      }
      output[policy::kChecksumIndex] = checksum;
      output[policy::kCountIndex] = static_cast<uint32_t>(mixed.size());
      require(policy::validateOutput(output, mixed, mode), "valid output rejected");
      for (uint32_t index = 0; index != output.size(); ++index) {
        output[index] ^= 1;
        require(!policy::validateOutput(output, mixed, mode), "corrupted output/guard accepted");
        output[index] ^= 1;
      }
      require(!policy::validateOutput(output, mixed,
          mode == policy::StorageMode::Raw ? policy::StorageMode::PLESSD : policy::StorageMode::Raw),
          "wrong reflected owner count accepted");
    }
    require(policy::kOriginalBytes - policy::kSSDOriginalBytes == 32002539520ULL,
            "SSD mode included PLE padded bytes");
    rejected([] { (void)policy::ownerGeometry(static_cast<policy::StorageMode>(99)); },
             "invalid storage enum accepted");
    std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":" << checks
              << ",\"storage_modes\":2,\"original_shader_preserved\":true}"
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "SSD maintenance policy test failed: " << error.what() << '\n';
    return 1;
  }
}
