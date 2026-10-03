// Sweeps the decode Q4 projection pipelines over the production projection
// shapes and persistent-group counts. Every pipeline streams the same weight
// bytes, so the effective GB/s is the tuning signal for a device family.
#include "../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kQuantGroup = 64;

enum class Contract : uint8_t { Affine, Residual, GateUp, UpSilu };

struct Pipeline final {
  const char *name;
  Contract contract;
  uint32_t rows;
  uint32_t tileColumns;
};

struct Shape final {
  const char *label;
  uint32_t outputSize;
  uint32_t inputSize;
  uint32_t referenceGroups; // original policy; * marks this reference, not selection
};

constexpr Pipeline kPipelines[] = {
    {"decode_linear_q4_n128", Contract::Affine, 8, 128},
    {"decode_linear_q4_n256", Contract::Affine, 8, 256},
    {"decode_linear_q4_n128_residual", Contract::Residual, 8, 128},
    {"decode_linear_q4_n128_paired", Contract::Affine, 8, 128},
    {"decode_linear_q4_n128_residual_paired", Contract::Residual, 8, 128},
    {"decode_linear_q4_n256_gate_up", Contract::GateUp, 8, 256},
    {"decode_linear_q4_n128_m16", Contract::Affine, 16, 128},
    {"decode_linear_q4_n256_m16", Contract::Affine, 16, 256},
    {"decode_linear_q4_n128_residual_m16", Contract::Residual, 16, 128},
    {"decode_linear_q4_n256_gate_up_m16", Contract::GateUp, 16, 256},
    {"decode_linear_q4_n128_m24", Contract::Affine, 24, 128},
    {"decode_linear_q4_n256_m24", Contract::Affine, 24, 256},
    {"decode_linear_q4_n128_residual_m24", Contract::Residual, 24, 128},
    {"decode_linear_q4_n256_up_silu_m24", Contract::UpSilu, 24, 256},
    {"decode_linear_q4_n128_m32", Contract::Affine, 32, 128},
    {"decode_linear_q4_n256_m32", Contract::Affine, 32, 256},
    {"decode_linear_q4_n128_residual_m32", Contract::Residual, 32, 128},
    {"decode_linear_q4_n256_up_silu_m32", Contract::UpSilu, 32, 256},
};

struct WeightSet final {
  MetalBuffer weights;
  MetalBuffer scales;
  MetalBuffer biases;
  uint64_t bytes = 0;
};

WeightSet allocateWeights(MetalBackend &backend, const Shape &shape,
                          const std::string &label) {
  const uint64_t elements = uint64_t{shape.outputSize} * shape.inputSize;
  const uint64_t parameters = elements / kQuantGroup;
  WeightSet set{
      backend.allocateBuffer(elements / 2, BufferStorage::Shared,
                             label + " weights"),
      backend.allocateBuffer(parameters * sizeof(__bf16), BufferStorage::Shared,
                             label + " scales"),
      backend.allocateBuffer(parameters * sizeof(__bf16), BufferStorage::Shared,
                             label + " biases"),
  };
  // Touch every page so the sweep streams resident memory, not zero fill.
  std::memset(set.weights.contents(), 0x5a, elements / 2);
  std::memset(set.scales.contents(), 0x3c, parameters * sizeof(__bf16));
  std::memset(set.biases.contents(), 0x3c, parameters * sizeof(__bf16));
  set.bytes = elements / 2 + 2 * parameters * sizeof(__bf16);
  return set;
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::vector<uint32_t> groupCandidates(uint32_t tiles, uint32_t referenceGroups) {
  std::vector<uint32_t> groups{referenceGroups, tiles, tiles / 2, tiles / 4,
                               tiles / 8,  20,        40,        60,
                               80,         120,       160,       240,
                               320,        480,       640};
  std::erase_if(groups, [tiles](uint32_t g) { return g == 0 || g > tiles; });
  std::sort(groups.begin(), groups.end());
  groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
  return groups;
}

void sweep(MetalBackend &backend, const Shape &shape, const Pipeline &pipeline,
           const WeightSet &first, const WeightSet &second) {
  const uint32_t tiles = shape.outputSize / pipeline.tileColumns;
  if (tiles * pipeline.tileColumns != shape.outputSize)
    return;
  const std::string label = std::string(shape.label) + " " + pipeline.name;
  MetalBuffer input = backend.allocateBuffer(
      uint64_t{pipeline.rows} * shape.inputSize * sizeof(__bf16),
      BufferStorage::Shared, label + " input");
  MetalBuffer output = backend.allocateBuffer(
      uint64_t{pipeline.rows} * shape.outputSize * sizeof(__bf16),
      BufferStorage::Shared, label + " output");
  MetalBuffer extra = backend.allocateBuffer(
      uint64_t{pipeline.rows} * shape.outputSize * sizeof(__bf16),
      BufferStorage::Shared, label + " extra");

  ComputeDispatch dispatch;
  dispatch.pipelineName = pipeline.name;
  uint32_t parameterIndex = 5;
  switch (pipeline.contract) {
  case Contract::Affine:
    dispatch.buffers = {{0, input}, {1, first.weights}, {2, first.scales},
                        {3, first.biases}, {4, output}};
    break;
  case Contract::Residual:
  case Contract::UpSilu:
    dispatch.buffers = {{0, input}, {1, first.weights}, {2, first.scales},
                        {3, first.biases}, {4, extra}, {5, output}};
    parameterIndex = 6;
    break;
  case Contract::GateUp:
    dispatch.buffers = {{0, input}, {1, first.weights}, {2, first.scales},
                        {3, first.biases}, {4, output}, {5, second.weights},
                        {6, second.scales}, {7, second.biases}};
    parameterIndex = 8;
    break;
  }
  const uint64_t bytes =
      pipeline.contract == Contract::GateUp ? 2 * first.bytes : first.bytes;
  dispatch.threadsPerThreadgroup = {256, 1, 1};

  for (uint32_t groups : groupCandidates(tiles, shape.referenceGroups)) {
    Q4PersistentParams params{shape.outputSize, shape.inputSize, groups};
    dispatch.bytes = {{parameterIndex, &params, sizeof(params)}};
    dispatch.threadgroups = {groups, 1, 1};
    for (uint32_t warmup = 0; warmup < 2; ++warmup)
      static_cast<void>(backend.submit(dispatch));
    std::vector<double> samples;
    for (uint32_t repeat = 0; repeat < 9; ++repeat)
      samples.push_back(backend.submit(dispatch).gpuSeconds);
    const double seconds = median(samples);
    std::cout << shape.label << ' ' << pipeline.name << " groups=" << groups
              << (groups == shape.referenceGroups ? "*" : "")
              << " ms=" << seconds * 1e3
              << " GB/s=" << double(bytes) / seconds / 1e9 << '\n';
  }
}

void run(const std::string &metallibPath) {
  MetalBackend backend(metallibPath);
  const auto &capabilities = backend.capabilities();
  std::cout << "device=\"" << capabilities.deviceName
            << "\" apple_gpu_family=" << capabilities.appleGpuFamily << '\n';
  const Shape shapes[] = {
      {"gdn_input", 16'640, 5'120, 60},
      {"full_input", 14'336, 5'120, 60},
      {"mixer_output", 5'120, 6'144, 40},
      {"ffn_gate_up", 17'408, 5'120, 36},
      {"ffn_down", 5'120, 17'408, 80},
      {"draft_context", 5'120, 25'600, 80},
      {"lm_head", 248'320, 5'120, 1'940},
  };
  for (const Shape &shape : shapes) {
    const WeightSet first = allocateWeights(backend, shape, shape.label);
    const bool gateUp = std::string(shape.label) == "ffn_gate_up";
    const WeightSet second =
        gateUp ? allocateWeights(backend, shape, std::string(shape.label) + " up")
               : WeightSet{};
    for (const Pipeline &pipeline : kPipelines) {
      const bool fused = pipeline.contract == Contract::GateUp ||
                         pipeline.contract == Contract::UpSilu;
      if (fused != gateUp)
        continue;
      sweep(backend, shape, pipeline, first, second);
    }
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: q4_decode_profile <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
