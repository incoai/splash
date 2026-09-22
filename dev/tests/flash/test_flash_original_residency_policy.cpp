#include "flash/FlashOriginalResidency.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
uint64_t checks = 0;

void require(bool value, std::string_view reason) {
  ++checks;
  if (!value) throw std::runtime_error(std::string(reason));
}

uint32_t category(splash::flash::OriginalResidencyCategory value) {
  return static_cast<uint32_t>(value);
}

uint32_t baseMask(std::initializer_list<std::string_view> names) {
  uint32_t result = 0;
  for (const auto name : names)
    result |= splash::flash::flashOriginalResidencyCategory(name);
  return result;
}
} // namespace

int main() {
  using namespace splash::flash;
  try {
    struct NamedCategory final {
      std::string_view name;
      OriginalResidencyCategory expected;
    };
    for (const auto &test : std::array<NamedCategory, 25>{{
        {"language_model.lm_head.weight", OriginalResidencyCategory::Text},
        {"language_model.model.embed_tokens.weight", OriginalResidencyCategory::Text},
        {"language_model.model.layers.0.linear_attn.in_proj_qkv.scales", OriginalResidencyCategory::Text},
        {"language_model.model.layers.2.mlp.switch_mlp.down_proj.biases", OriginalResidencyCategory::Text},
        {"language_model.model.hyper_connection_mixer.hc_norm.weight", OriginalResidencyCategory::Text},
        {"mtp.layers.0.self_attn.q_proj.weight", OriginalResidencyCategory::MTP},
        {"mtp.fc_embedding.weight", OriginalResidencyCategory::MTP},
        {"mtp.hyper_connection_mixer.input_mix_weight_up.biases", OriginalResidencyCategory::MTP},
        {"language_model.model.layers.1.ple.key_proj.weight", OriginalResidencyCategory::PLE},
        {"language_model.model.layers.1.ple.conv1d.weight", OriginalResidencyCategory::PLE},
        {"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.127.weight", OriginalResidencyCategory::PLE},
        {"mtp.layers.0.ple.key_proj.weight", OriginalResidencyCategory::PLE},
        {"language_model.ngram_adapter.weight", OriginalResidencyCategory::PLE},
        {"unknown.ngram_table.weight", OriginalResidencyCategory::PLE},
        {"unknown.ple.table.weight", OriginalResidencyCategory::PLE},
        {"vision_tower.blocks.0.attn.qkv.weight", OriginalResidencyCategory::Vision},
        {"vision_tower.patch_embed.proj.weight", OriginalResidencyCategory::Vision},
        {"language_model.vision_bridge.weight", OriginalResidencyCategory::Vision},
        {"mtp.layers.0.vision_adapter.weight", OriginalResidencyCategory::Vision},
        {"language_model.model.layers.1.ple.vision_ngram.weight", OriginalResidencyCategory::Vision},
        {"vision_tower.ple.ngram.weight", OriginalResidencyCategory::Vision},
        {"", OriginalResidencyCategory::Unknown},
        {"language_model", OriginalResidencyCategory::Unknown},
        {"language_modelx.model.layers.0.weight", OriginalResidencyCategory::Unknown},
        {"module.mtp.layers.0.weight", OriginalResidencyCategory::Unknown},
    }}) {
      require(flashOriginalResidencyCategory(test.name) == category(test.expected),
              "tensor category or sensitivity precedence differs: " + std::string(test.name));
    }

    // Every legal five-bit union is tested independently. Only the three
    // nonempty Text/MTP masks may select an entire original payload.
    for (uint32_t mask = 0; mask < 32; ++mask) {
      const bool expected = mask == 1 || mask == 2 || mask == 3;
      require(flashOriginalResidencyBaseEligible(mask) == expected,
              "mixed payload eligibility differs for mask " + std::to_string(mask));
    }
    for (const uint32_t other : std::array<uint32_t, 5>{
             32, 64, 1024, uint32_t{1} << 31, std::numeric_limits<uint32_t>::max()}) {
      for (uint32_t textOrMTP = 0; textOrMTP < 4; ++textOrMTP)
        require(!flashOriginalResidencyBaseEligible(other | textOrMTP),
                "unknown high category bit selected an original payload");
    }
    require(flashOriginalResidencyBaseEligible(baseMask({
                "language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight",
                "mtp.layers.0.self_attn.q_proj.weight"})),
            "mixed Text/MTP payload was excluded");
    require(!flashOriginalResidencyBaseEligible(baseMask({
                "language_model.lm_head.weight",
                "language_model.model.layers.1.ple.key_proj.weight"})),
            "text tensor admitted a payload containing PLE");
    require(!flashOriginalResidencyBaseEligible(baseMask({
                "mtp.fc_embedding.weight", "vision_tower.blocks.0.attn.qkv.weight"})),
            "MTP tensor admitted a payload containing vision");
    require(!flashOriginalResidencyBaseEligible(baseMask({
                "language_model.lm_head.weight", "unclassified.tensor.weight"})),
            "text tensor admitted a payload containing unknown weights");
    require(!flashOriginalResidencyBaseEligible(baseMask({})),
            "empty payload selected for residency");

    constexpr uint64_t margin = uint64_t{2} * 1024 * 1024 * 1024;
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t sourceQualifiedMappedBytes = 66060288000ULL;
    const std::array<uint64_t, 5> extents{
        1, 16384, uint64_t{1} << 30, sourceQualifiedMappedBytes, maximum - margin};
    for (const uint64_t originalBytes : extents) {
      const uint64_t boundary = originalBytes + margin;
      require(!flashOriginalResidencyHostAllowed(true, true, true, boundary - 1, originalBytes),
              "headroom below the inclusive two-GiB margin was admitted");
      require(flashOriginalResidencyHostAllowed(true, true, true, boundary, originalBytes),
              "exact two-GiB headroom margin was rejected");
      if (boundary < maximum)
        require(flashOriginalResidencyHostAllowed(true, true, true, boundary + 1, originalBytes),
                "headroom above the two-GiB margin was rejected");
      require(!flashOriginalResidencyHostAllowed(true, true, true, 0, originalBytes),
              "zero headroom admitted nonzero original backing");
    }

    // Invalid measurements, a growth veto, or non-normal pressure must each
    // veto the selection even when the numeric margin is satisfied.
    for (uint32_t flags = 0; flags < 8; ++flags) {
      const bool hostValid = (flags & 1) != 0;
      const bool growthAllowed = (flags & 2) != 0;
      const bool pressureNormal = (flags & 4) != 0;
      for (const uint64_t originalBytes : extents) {
        const bool expected = flags == 7;
        require(flashOriginalResidencyHostAllowed(hostValid, growthAllowed, pressureNormal,
                    originalBytes + margin, originalBytes) == expected,
                "host validity/growth/pressure veto differs");
      }
      require(!flashOriginalResidencyHostAllowed(hostValid, growthAllowed, pressureNormal, maximum, 0),
              "zero original backing was selected");
      require(!flashOriginalResidencyHostAllowed(hostValid, growthAllowed, pressureNormal,
                  maximum, maximum - margin + 1),
              "overflowing margin was admitted");
      require(!flashOriginalResidencyHostAllowed(hostValid, growthAllowed, pressureNormal,
                  maximum, maximum),
              "maximum extent wrapped around the safety margin");
    }
    require(!flashOriginalResidencyHostAllowed(true, true, true, margin, 0),
            "empty original selection was admitted at the exact margin");

    std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":" << checks
              << ",\"source_qualified_mapped_bytes\":" << sourceQualifiedMappedBytes << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL original residency policy: " << error.what() << '\n';
    return 1;
  }
}
