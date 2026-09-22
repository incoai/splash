#include "FlashGDNBatchILP.hpp"
#include "metal/abi/FlashGDNBatchILP.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace splash::flash {
namespace {
using namespace splash::flash;
using namespace splash::metal;
[[noreturn]] void fail(const char *text) { throw std::invalid_argument(text); }
void distinct(const MetalBuffer &a, const MetalBuffer &b) {
  const auto aa = reinterpret_cast<uintptr_t>(a.contents()), bb = reinterpret_cast<uintptr_t>(b.contents());
  if (!aa || !bb) fail("batched GDN requires addressable Shared buffers");
  if (aa <= bb ? uint64_t(bb - aa) < a.sizeBytes() : uint64_t(aa - bb) < b.sizeBytes())
    fail("batched GDN input, output or request states overlap");
}
void shared(const MetalBuffer &buffer) {
  if (!buffer || buffer.storage() != BufferStorage::Shared || !buffer.contents())
    fail("batched GDN requires sufficient Shared buffers");
}
void append(CommandGraph &graph, const ComputeDispatch &d) {
  if (d.bytes.size() != 1 || d.bytes[0].sizeBytes != sizeof(FlashGDNParams))
    fail("batched GDN staged parameter ABI changed");
  FlashGDNParams params{}; std::memcpy(&params, d.bytes[0].data, sizeof(params));
  std::vector<MetalBuffer> buffers;
  for (const auto &binding : d.buffers) {
    if (binding.index != buffers.size()) fail("batched GDN buffer order changed");
    buffers.push_back(binding.buffer);
  }
  graph.add(d.pipelineName, std::move(buffers), params, d.threadgroups, d.threadsPerThreadgroup);
}
} // namespace

bool flashGDNBatchILPEnabled() {
  static const bool selected = [] {
    const char *raw = std::getenv("SPLASH_FLASH_GDN_BATCH_ILP");
    if (!raw || std::string_view(raw) == "0") return false;
    if (std::string_view(raw) != "1") fail("SPLASH_FLASH_GDN_BATCH_ILP must be 0 or 1");
    for (const char *name : {"SPLASH_FLASH_GDN_STAGED", "SPLASH_FLASH_BATCH_PREFILL"}) {
      const char *required = std::getenv(name);
      if (!required || std::string_view(required) != "1")
        fail("SPLASH_FLASH_GDN_BATCH_ILP requires GDN_STAGED=1 and BATCH_PREFILL=1");
    }
    return true;
  }();
  return selected;
}
bool flashGDNBatchILPEligible(uint32_t lanes, uint32_t rows) noexcept {
  return lanes >= 2 && lanes <= 4 && rows >= 512 && rows <= kFlashGDNMaximumRows;
}
void validateGDNBatchILPTile(FlashGDNBatchILPTile tile) {
  if (tile.values != 32 || (tile.time != 16 && tile.time != 32) || tile.simds != 8)
    fail("batched GDN supports V32/T16-or-T32/SG8");
}
std::string gdnBatchILPPipelineName(FlashGDNBatchILPTile tile) {
  validateGDNBatchILPTile(tile);
  return "flash_gdn_batch_ilp_v32_t" + std::to_string(tile.time) + "_s8";
}
uint32_t validateGDNBatchILPGeometry(std::span<const FlashGDNBatchILPLane> lanes,
    uint32_t inputRowsStride, FlashGDNBatchILPTile tile, float epsilon) {
  validateGDNBatchILPTile(tile);
  if (lanes.empty() || lanes.size() > 4 || !inputRowsStride || inputRowsStride > 4096 ||
      !std::isfinite(epsilon) || epsilon <= 0.0f) fail("batched GDN invalid scalar geometry");
  uint32_t active = 0;
  for (const auto &lane : lanes) {
    if (lane.rows > kFlashGDNMaximumRows || lane.rows > inputRowsStride)
      fail("batched GDN actual rows exceed the bounded input stride");
    if (!lane.rows) continue;
    if (!lane.state.convolution || !lane.state.recurrent)
      fail("batched GDN live lane has missing state buffers");
    ++active;
  }
  return active;
}
void addGDNBatchILP(MetalBackend &backend, CommandGraph &graph, const FlashGDNWeights &weights,
    const FlashGDNBuffers &flat, std::span<const FlashGDNBatchILPLane> lanes,
    uint32_t inputRowsStride, FlashGDNBatchILPTile tile, float epsilon) {
  const uint32_t active = validateGDNBatchILPGeometry(lanes, inputRowsStride, tile, epsilon);
  if (!active) return;
  const std::array flatPlanes{flat.qkv, flat.z, flat.a, flat.b, flat.mixed, flat.decay,
      flat.beta, flat.recurrentRows, flat.output, flat.diagnostics};
  for (const auto &buffer : flatPlanes) shared(buffer);
  for (size_t i = 0; i < flatPlanes.size(); ++i)
    for (size_t j = i + 1; j < flatPlanes.size(); ++j) distinct(flatPlanes[i], flatPlanes[j]);
  for (const auto *weight : {weights.convolution, weights.aLog, weights.timeBias, weights.norm}) {
    if (!weight) fail("batched GDN has missing immutable weights");
    shared(weight->buffer);
    for (const auto &buffer : flatPlanes) distinct(weight->buffer, buffer);
  }
  std::array<CommandGraph, 4> validated;
  FlashGDNBatchILPParams merged{{inputRowsStride, uint32_t(lanes.size()), 16, 48, 128, 128, 4,
      epsilon, flashGDNConvolutionLaneBytes(), flashGDNRecurrentLaneBytes()}, {0, 0, 0, 0}};
  std::array<MetalBuffer, 4> stateBuffers;
  MetalBuffer fallback;
  for (uint32_t slot = 0; slot < lanes.size(); ++slot) {
    const auto &lane = lanes[slot];
    if (!lane.rows) continue;
    shared(lane.state.convolution); shared(lane.state.recurrent);
    distinct(lane.state.convolution, lane.state.recurrent);
    for (const auto &buffer : flatPlanes) {
      distinct(lane.state.convolution, buffer); distinct(lane.state.recurrent, buffer);
    }
    for (const auto *weight : {weights.convolution, weights.aLog, weights.timeBias, weights.norm}) {
      distinct(lane.state.convolution, weight->buffer); distinct(lane.state.recurrent, weight->buffer);
    }
    for (uint32_t previous = 0; previous < slot; ++previous) {
      if (!lanes[previous].rows) continue;
      distinct(lane.state.convolution, lanes[previous].state.convolution);
      distinct(lane.state.convolution, lanes[previous].state.recurrent);
      distinct(lane.state.recurrent, lanes[previous].state.convolution);
      distinct(lane.state.recurrent, lanes[previous].state.recurrent);
    }
    const auto view = [&](const MetalBuffer &buffer, uint32_t width, uint32_t bytes = 2) {
      const uint64_t stride = uint64_t{inputRowsStride} * width * bytes;
      // Require full slot storage, including untouched dummy row tails.
      (void)backend.view(buffer, 0, uint64_t{lanes.size()} * stride);
      return backend.view(buffer, uint64_t{slot} * stride, uint64_t{lane.rows} * width * bytes);
    };
    const FlashGDNBuffers local{view(flat.qkv, 10240), view(flat.z, 6144), view(flat.a, 48),
        view(flat.b, 48), view(flat.mixed, 10240), view(flat.decay, 48, 4), view(flat.beta, 48),
        view(flat.recurrentRows, 6144), view(flat.output, 6144), flat.diagnostics};
    (void)backend.view(lane.state.convolution, 0, lane.state.convolution.sizeBytes());
    stateBuffers[slot] = backend.view(lane.state.recurrent, 0, lane.state.recurrent.sizeBytes());
    fallback = stateBuffers[slot];
    merged.actual_rows[slot] = lane.rows;
    addGDNStagedPrefill(validated[slot], weights, local, lane.state, lane.rows, 1,
        FlashGDNStageTile::Values16Time16, epsilon);
    if (validated[slot].dispatches().size() != 4) fail("batched GDN staged graph ABI changed");
    const std::array<const char *, 4> expected{"flash_gdn_fused_prepare", "flash_gdn_staged_v16_t16",
        "flash_gdn_output", "flash_gdn_convolution_carry"};
    for (uint32_t phase = 0; phase < 4; ++phase) {
      const auto &dispatch = validated[slot].dispatches()[phase];
      if (dispatch.pipelineName != expected[phase] || dispatch.bytes.size() != 1 ||
          dispatch.bytes[0].sizeBytes != sizeof(FlashGDNParams))
        fail("batched GDN validated producer contract changed");
      for (uint32_t binding = 0; binding < dispatch.buffers.size(); ++binding)
        if (dispatch.buffers[binding].index != binding)
          fail("batched GDN validated producer binding order changed");
    }
  }
  // Inactive bindings use a retained live buffer; the kernel masks the slot
  // before selecting or reading its state. No synthetic state allocation.
  for (auto &buffer : stateBuffers) if (!buffer) buffer = fallback;
  for (uint32_t slot = 0; slot < lanes.size(); ++slot)
    if (lanes[slot].rows) append(graph, validated[slot].dispatches()[0]);
  graph.add(gdnBatchILPPipelineName(tile), {stateBuffers[0], stateBuffers[1], stateBuffers[2], stateBuffers[3],
      flat.mixed, flat.decay, flat.beta, flat.recurrentRows, flat.diagnostics}, merged,
      {48, 4, lanes.size()}, {256, 1, 1});
  for (uint32_t slot = 0; slot < lanes.size(); ++slot) if (lanes[slot].rows) {
    append(graph, validated[slot].dispatches()[2]);
    append(graph, validated[slot].dispatches()[3]);
  }
}

} // namespace splash::flash
