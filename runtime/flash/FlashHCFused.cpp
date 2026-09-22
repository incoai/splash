#include "flash/FlashHCFused.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "metal/abi/FlashHCFused.h"
#include "metal/abi/FlashFloatDenseCache.h"

#include <cmath>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::flash {
bool flashHCUpF32MPPEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_HC_UP_F32_MPP");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("SPLASH_FLASH_HC_UP_F32_MPP must be 0 or 1");
  }();
  return enabled;
}
bool flashHCUpF32MPPGeometry(std::string_view prefix, uint32_t rows,
                            uint32_t outputSize, uint32_t inputSize) noexcept {
  if (rows < 4 || rows > 16 || outputSize != 10240 || inputSize != 320) return false;
  if (prefix == "language_model.model.hyper_connection_mixer.input_mix_weight_up") return true;
  constexpr std::string_view leading = "language_model.model.layers.";
  if (!prefix.starts_with(leading)) return false;
  const auto rest = prefix.substr(leading.size());
  const auto dot = rest.find('.');
  if (dot == std::string_view::npos || !dot || dot > 2) return false;
  uint32_t layer = 0;
  for (char digit : rest.substr(0, dot)) {
    if (digit < '0' || digit > '9') return false;
    layer = layer * 10 + uint32_t(digit - '0');
  }
  if (layer >= 48) return false;
  const auto role = rest.substr(dot + 1);
  return role == "attn_hyper_connection.input_mix_weight_up" ||
         role == "mlp_hyper_connection.input_mix_weight_up";
}
namespace {
uint64_t mul(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("fused HC byte count overflows");
  return a * b;
}
uint64_t add(uint64_t a, uint64_t b) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    throw std::invalid_argument("fused HC byte extent overflows");
  return a + b;
}
void bytes(const metal::MetalBuffer &buffer, uint64_t count) {
  if (!buffer || buffer.sizeBytes() < count)
    throw std::invalid_argument("fused HC buffer is smaller than logical shape");
}
void distinct(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  // The first Flash route uses Shared allocations/views. Exact range metadata
  // is available without reading data; Private views need a backend overlap
  // API before they can enter this fusion safely.
  const auto *ap = a.contents();
  const auto *bp = b.contents();
  if (!ap || !bp)
    throw std::invalid_argument("fused HC requires shared addressable buffer views");
  const uintptr_t aa = reinterpret_cast<uintptr_t>(ap);
  const uintptr_t bb = reinterpret_cast<uintptr_t>(bp);
  const bool overlap = aa <= bb ? uint64_t(bb - aa) < a.sizeBytes()
                                : uint64_t(aa - bb) < b.sizeBytes();
  if (overlap)
    throw std::invalid_argument("unsupported fused HC buffer alias");
}
bool affineFormat(const FlashAffineProjection &p) noexcept {
  return p.experts == 1 && p.inputSize && p.outputSize &&
      (p.bits == 4 || p.bits == 5 || p.bits == 6 || p.bits == 8) &&
      (p.groupSize == 32 || p.groupSize == 64 || p.groupSize == 128) &&
      p.inputSize % p.groupSize == 0;
}
void projection(const FlashAffineProjection &p) {
  if (!affineFormat(p)) throw std::invalid_argument("invalid fused HC affine format");
  const uint64_t packed = (mul(p.inputSize, p.bits) + 7) / 8;
  const uint64_t coefficients = mul(p.inputSize / p.groupSize, 2);
  if (p.weightRowStrideBytes < packed || p.parameterRowStrideBytes < coefficients ||
      p.parameterRowStrideBytes % 2)
    throw std::invalid_argument("invalid fused HC matrix strides");
  const uint64_t wbytes = add(mul(p.outputSize - 1, p.weightRowStrideBytes), packed);
  const uint64_t sbytes = add(mul(p.outputSize - 1, p.parameterRowStrideBytes), coefficients);
  const auto tensor = [](const FlashTensor *t, FlashDType dtype, uint64_t size) {
    if (!t || t->dtype != dtype || t->logicalBytes < size)
      throw std::invalid_argument("invalid fused HC tensor dtype or logical bytes");
    bytes(t->buffer, size);
  };
  tensor(p.weights, FlashDType::U32, wbytes);
  tensor(p.scales, FlashDType::BF16, sbytes);
  tensor(p.biases, FlashDType::BF16, sbytes);
}
FlashHCFusedMatrix matrix(const FlashAffineProjection &p) {
  return {p.inputSize, p.outputSize, p.bits, p.groupSize,
          p.weightRowStrideBytes, p.parameterRowStrideBytes};
}
FlashHCFusedParams params(FlashHCGeometry g, FlashHCFusedConfig c) {
  if (!g.rows || g.rows > 32 || g.width != 2560 || g.streams != 4 ||
      !std::isfinite(g.epsilon) || g.epsilon <= 0 ||
      (c.simdgroups != 4 && c.simdgroups != 8) ||
      (c.arithmetic != FlashHCFusedArithmetic::ExplicitCoefficients &&
       c.arithmetic != FlashHCFusedArithmetic::GroupedAffineExperimental))
    throw std::invalid_argument("invalid fused HC workload/configuration");
  FlashHCFusedParams p{};
  p.rows = g.rows; p.width = g.width; p.streams = g.streams; p.lowrank = 320;
  p.arithmetic_mode = static_cast<uint32_t>(c.arithmetic);
  p.simdgroups = c.simdgroups; p.norm_epsilon = g.epsilon;
  return p;
}
std::string pipeline(const char *phase, const FlashAffineProjection &p,
                     FlashHCFusedConfig c) {
  return std::string("flash_hc_fused_") + phase + "_q" + std::to_string(p.bits) +
      "_g" + std::to_string(p.groupSize) + "_s" + std::to_string(c.simdgroups);
}
uint64_t hyperBytes(FlashHCGeometry g) { return mul(mul(mul(g.rows, g.width), g.streams), 2); }
uint64_t branchBytes(FlashHCGeometry g) { return mul(mul(g.rows, g.width), 2); }
void outputsDisjoint(const metal::MetalBuffer &output, const FlashAffineProjection &p) {
  distinct(output, p.weights->buffer); distinct(output, p.scales->buffer);
  distinct(output, p.biases->buffer);
}
} // namespace

void addHCFusedUpMixF32Cache(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer normalized, metal::MetalBuffer activated,
    const FlashTensor &weight, metal::MetalBuffer mixed, metal::MetalBuffer diagnostics,
    FlashHCGeometry g, FlashFloatDenseSmallRowsWorkspace &workspace,
    metal::MetalBuffer rawDebug) {
  if (!workspace.belongsTo(backend) || workspace.maximumInputSize() < 320 ||
      g.rows < 4 || g.rows > 16 || g.width != 2560 || g.streams != 4 ||
      !std::isfinite(g.epsilon) || g.epsilon <= 0 || weight.dtype != FlashDType::F32 ||
      weight.shape != std::vector<uint64_t>{10240, 320} ||
      weight.logicalBytes != uint64_t{10240} * 320 * 4)
    throw std::invalid_argument("cached F32 HC-up requires owned workspace, R4..16 and original F32[10240,320]");
  bytes(weight.buffer, weight.logicalBytes); bytes(normalized, hyperBytes(g));
  bytes(activated, uint64_t{g.rows} * 320 * 2); bytes(mixed, branchBytes(g)); bytes(diagnostics, 4);
  auto padded = workspace.paddedInput();
  const uint32_t paddedRows = (g.rows + 7) / 8 * 8;
  bytes(padded, uint64_t{paddedRows} * 320 * 2);
  for (const auto &b : {normalized, activated, mixed, diagnostics, padded}) distinct(b, weight.buffer);
  const std::array buffers{normalized, activated, mixed, diagnostics, padded};
  for (size_t a = 0; a < buffers.size(); ++a)
    for (size_t b = a + 1; b < buffers.size(); ++b) distinct(buffers[a], buffers[b]);
  if (rawDebug) {
    bytes(rawDebug, hyperBytes(g)); distinct(rawDebug, weight.buffer);
    for (const auto &b : buffers) distinct(rawDebug, b);
  }
  const FlashFloatDenseSmallRowsParams padding{g.rows, paddedRows, 320, 10240, 0, 10240, 8, 64};
  graph.add("flash_float_dense_small_rows_pad", {activated, padded, diagnostics}, padding,
      {(uint64_t{paddedRows} * 320 + 255) / 256, 1, 1});
  const FlashFloatDenseSmallRowsParams p{g.rows, paddedRows, 320, 2560, 0, 2560, 8, 32};
  graph.add(rawDebug ? "flash_hc_up_f32_mpp_m8_n32_s4_debug" : "flash_hc_up_f32_mpp_m8_n32_s4",
      {padded, weight.buffer, normalized, mixed, rawDebug ? rawDebug : activated, diagnostics},
      p, {80, paddedRows / 8, 1}, {128, 1, 1});
}

bool supportsHCFused(const FlashAffineProjection &down,
                     const FlashAffineProjection &up,
                     const FlashAffineProjection *injection,
                     FlashHCGeometry g) noexcept {
  return g.rows && g.rows <= 32 && g.width == 2560 && g.streams == 4 &&
      affineFormat(down) && affineFormat(up) && down.inputSize == 10240 &&
      down.outputSize == 320 && up.inputSize == 320 && up.outputSize == 10240 &&
      (!injection || (affineFormat(*injection) && injection->inputSize == 10240 &&
                       injection->outputSize == 4));
}

void addHCFusedDown(metal::CommandGraph &graph, metal::MetalBuffer normalized,
                    const FlashAffineProjection &down,
                    const FlashAffineProjection *injection,
                    metal::MetalBuffer activatedDown,
                    metal::MetalBuffer injectionWeights,
                    metal::MetalBuffer diagnostics, FlashHCGeometry geometry,
                    FlashHCFusedConfig config) {
  auto p = params(geometry, config);
  projection(down);
  if (down.inputSize != 10240 || down.outputSize != 320)
    throw std::invalid_argument("fused HC down geometry differs");
  p.down = matrix(down);
  const auto &inj = injection ? *injection : down;
  if (injection) {
    projection(inj);
    if (inj.inputSize != 10240 || inj.outputSize != 4)
      throw std::invalid_argument("fused HC injection geometry differs");
    p.has_injection = 1;
    bytes(injectionWeights, mul(mul(geometry.rows, geometry.streams), 2));
    distinct(injectionWeights, normalized); distinct(injectionWeights, activatedDown);
    outputsDisjoint(injectionWeights, inj); outputsDisjoint(injectionWeights, down);
  }
  p.injection = matrix(inj);
  bytes(normalized, hyperBytes(geometry)); bytes(activatedDown, mul(mul(geometry.rows, 320), 2));
  bytes(diagnostics, 4); distinct(activatedDown, normalized);
  distinct(diagnostics, normalized); distinct(diagnostics, activatedDown);
  outputsDisjoint(diagnostics, down);
  if (injection) {
    distinct(diagnostics, injectionWeights); outputsDisjoint(diagnostics, inj);
  }
  outputsDisjoint(activatedDown, down);
  if (injection) outputsDisjoint(activatedDown, inj);
  const uint64_t outputs = 320 + (injection ? geometry.streams : 0);
  graph.add(pipeline("down", down, config),
            {normalized, down.weights->buffer, down.scales->buffer, down.biases->buffer,
             inj.weights->buffer, inj.scales->buffer, inj.biases->buffer,
             activatedDown, injection ? injectionWeights : activatedDown, diagnostics},
            p, {(outputs + config.simdgroups - 1) / config.simdgroups, geometry.rows, 1},
            {config.simdgroups * 32, 1, 1});
}

void addHCFusedUpMix(metal::CommandGraph &graph, metal::MetalBuffer normalized,
                     metal::MetalBuffer activatedDown,
                     const FlashAffineProjection &up, metal::MetalBuffer mixed,
                     metal::MetalBuffer diagnostics, FlashHCGeometry geometry,
                     FlashHCFusedConfig config, metal::MetalBuffer rawUpDebug) {
  auto p = params(geometry, config);
  projection(up);
  if (up.inputSize != 320 || up.outputSize != 10240)
    throw std::invalid_argument("fused HC up geometry differs");
  p.up = matrix(up);
  bytes(normalized, hyperBytes(geometry)); bytes(activatedDown, mul(mul(geometry.rows, 320), 2));
  bytes(mixed, branchBytes(geometry)); bytes(diagnostics, 4);
  distinct(mixed, normalized); distinct(mixed, activatedDown); outputsDisjoint(mixed, up);
  distinct(diagnostics, normalized); distinct(diagnostics, activatedDown);
  distinct(diagnostics, mixed); outputsDisjoint(diagnostics, up);
  if (rawUpDebug) {
    p.write_raw_up = 1; bytes(rawUpDebug, hyperBytes(geometry));
    distinct(rawUpDebug, normalized); distinct(rawUpDebug, activatedDown);
    distinct(rawUpDebug, mixed); outputsDisjoint(rawUpDebug, up);
    distinct(diagnostics, rawUpDebug);
  }
  graph.add(pipeline("up_mix", up, config),
            {normalized, activatedDown, up.weights->buffer, up.scales->buffer,
             up.biases->buffer, mixed, rawUpDebug ? rawUpDebug : activatedDown, diagnostics},
            p, {(uint64_t{geometry.width} + config.simdgroups - 1) / config.simdgroups,
                geometry.rows, 1}, {config.simdgroups * 32, 1, 1});
}

void addHCFusedInjectNorm(metal::CommandGraph &graph, metal::MetalBuffer hyperInput,
                          metal::MetalBuffer branch, metal::MetalBuffer injectionWeights,
                          const FlashTensor &normWeight, metal::MetalBuffer hyperOutput,
                          metal::MetalBuffer normalized, metal::MetalBuffer diagnostics,
                          FlashHCGeometry geometry, NormConvention convention) {
  auto p = params(geometry, {});
  p.norm_is_float = normWeight.dtype == FlashDType::F32;
  switch (convention) {
  case NormConvention::OnePlusWeight: p.norm_convention = 0; break;
  case NormConvention::DirectGamma: p.norm_convention = 1; break;
  default: throw std::invalid_argument("invalid fused HC norm convention");
  }
  const uint64_t width = mul(geometry.width, geometry.streams);
  const uint64_t wbytes = mul(width, p.norm_is_float ? 4 : 2);
  if ((normWeight.dtype != FlashDType::BF16 && !p.norm_is_float) ||
      normWeight.shape.size() != 1 || normWeight.shape[0] != width ||
      normWeight.logicalBytes != wbytes)
    throw std::invalid_argument("invalid fused HC norm weight");
  bytes(normWeight.buffer, wbytes); bytes(hyperInput, hyperBytes(geometry));
  bytes(hyperOutput, hyperBytes(geometry)); bytes(normalized, hyperBytes(geometry));
  bytes(branch, branchBytes(geometry)); bytes(injectionWeights, mul(mul(geometry.rows, 4), 2));
  bytes(diagnostics, 4);
  distinct(diagnostics, hyperInput); distinct(diagnostics, branch);
  distinct(diagnostics, injectionWeights); distinct(diagnostics, normWeight.buffer);
  distinct(diagnostics, hyperOutput); distinct(diagnostics, normalized);
  distinct(hyperOutput, branch); distinct(hyperOutput, injectionWeights);
  distinct(hyperOutput, normWeight.buffer);
  if (!hyperOutput.sameView(hyperInput)) distinct(hyperOutput, hyperInput);
  distinct(normalized, hyperInput); distinct(normalized, hyperOutput);
  distinct(normalized, branch); distinct(normalized, injectionWeights);
  distinct(normalized, normWeight.buffer);
  graph.add("flash_hc_fused_inject_norm",
            {hyperInput, branch, injectionWeights, normWeight.buffer,
             hyperOutput, normalized, diagnostics},
            p, {geometry.rows, geometry.streams, 1}, {640, 1, 1});
}

} // namespace splash::flash
