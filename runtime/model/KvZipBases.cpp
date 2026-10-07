#include "model/KvZipBases.hpp"

#include <array>

namespace splash::model {
namespace {

struct Calibration final {
  std::string_view family;
  uint32_t attentionLayers;
  uint32_t kvHeads;
  std::span<const uint8_t> bases;
};

constexpr uint8_t kQwen3_8_27b[] = {
#include "model/kvzip/qwen3_8_27b.inc"
};

constexpr uint8_t kQwen3_6_35b_a3b[] = {
#include "model/kvzip/qwen3_6_35b_a3b.inc"
};

constexpr std::array kCalibrations{
    Calibration{"Qwen3.8-27B", 16, 4, kQwen3_8_27b},
    Calibration{"Qwen3.6-35B-A3B", 10, 2, kQwen3_6_35b_a3b},
};

} // namespace

std::span<const uint8_t> kvZipBases(std::string_view family, const kv::Layout &layout) {
  for (const Calibration &calibration : kCalibrations) {
    if (calibration.family == family && calibration.attentionLayers == layout.attentionLayers &&
        calibration.kvHeads == layout.kvHeads &&
        calibration.bases.size() ==
            uint64_t{layout.attentionLayers} * 2 * layout.kvHeads * SPLASH_KVZIP_BASE_BYTES_PER_HEAD)
      return calibration.bases;
  }
  return {};
}

} // namespace splash::model
