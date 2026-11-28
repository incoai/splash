// GPU time of a decode step's fused message gates (ops::MessageGate) at Inkling-2-Mini's shape, hidden 2304 and
// rank 64: a call of one gate (a loop layer's FFN input) and of three (its q, k and v inputs), at each row count.
//   message-gate-benchmark <metallib> <bf16|f32> [rows[,rows...]] [rounds]
// A call is its two dispatches, down and apply, as a decode step runs them; the mixes before it wrote its rows
// normalized (here random rows, of the strides Inkling-2-Mini's mixes write). The weights stay DRAM-cold: a command
// streams every copy of a ring of at least 384 MiB once, each call's gates the next copies. The cases run in ABBA
// order after a warm-up; printed are the medians of `rounds` (20) commands in us a call, the first and third
// quartiles, each kernel's alone (d/a: a command of that kernel's dispatches of every call), the share of a plain
// read each reaches at the GB/s given in SPLASH_READ_GBPS (if set), and the threadgroups per core of each grid.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/MessageGate.h"
#include "ops/MessageGate.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

namespace {

constexpr uint64_t kRingBytes = 384ull << 20;
constexpr uint32_t kHidden = 2304, kRank = 64;
constexpr double kWarmupSeconds = 0.1;

// The kernels timed apart: the whole call, then a command of one kernel's dispatches of every call, in order.
constexpr std::array<const char *, 3> kParts{"", "message_gate_down", "message_gate_apply"};

struct Case {
  uint32_t rows, gates;
  std::array<std::vector<double>, kParts.size()> us;
};

MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *name) {
  MetalBuffer buffer = backend.allocateBuffer(std::max<uint64_t>(bytes, 16), metal::BufferStorage::Shared, name);
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

MetalBuffer randoms(MetalBackend &backend, uint64_t count, float scale, bool bfloat16, std::mt19937 &rng) {
  std::normal_distribution<float> normal(0.0f, scale);
  MetalBuffer buffer = backend.allocateBuffer(count * (bfloat16 ? 2 : 4), metal::BufferStorage::Shared, "gate rows");
  for (uint64_t i = 0; i < count; ++i) {
    const float value = normal(rng);
    if (bfloat16) static_cast<__bf16 *>(buffer.contents())[i] = __bf16(value);
    else static_cast<float *>(buffer.contents())[i] = value;
  }
  return buffer;
}

std::vector<uint32_t> numbers(const std::string &text) {
  std::vector<uint32_t> values;
  std::stringstream stream(text);
  for (std::string part; std::getline(stream, part, ',');) values.push_back(uint32_t(std::stoul(part)));
  return values;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc < 3 || argc > 5 || (std::string(argv[2]) != "bf16" && std::string(argv[2]) != "f32")) {
        std::cerr << "usage: message-gate-benchmark <metallib> <bf16|f32> [rows[,rows...]] [rounds]\n";
        return 2;
      }
      MetalBackend backend(argv[1]);
      const bool bfloat16 = std::string(argv[2]) == "bf16";
      const std::vector<uint32_t> rowCounts = numbers(argc > 3 ? argv[3] : "1,2,3,4,8,9,16");
      const uint32_t rounds = argc > 4 ? uint32_t(std::stoul(argv[4])) : 20;
      const uint32_t mostRows = *std::max_element(rowCounts.begin(), rowCounts.end());
      if (!rounds || !mostRows || mostRows > MessageGate::kFusedRows)
        throw std::invalid_argument("rows of 1 to MessageGate::kFusedRows, and rounds");
      const char *read = std::getenv("SPLASH_READ_GBPS");
      const double readGbps = read ? std::stod(read) : 0.0;
      // The ring of gates' weights, a gate's three projections a copy.
      const uint64_t weightBytes = bfloat16 ? 2 : 4;
      const uint64_t gateBytes = (uint64_t{kRank} * 2 * kHidden + 2 * uint64_t{kHidden} * kRank) * weightBytes;
      const uint32_t copies = uint32_t(std::max<uint64_t>((kRingBytes + gateBytes - 1) / gateBytes, 6));
      std::mt19937 rng(5);
      const uint32_t format = bfloat16 ? GGUF_FMT_BF16 : QuantizedSegment::kFloat32;
      const auto segment = [&](uint32_t outputs, uint32_t inputs, float scale) {
        const MetalBuffer values = randoms(backend, uint64_t{outputs} * inputs, scale, bfloat16, rng);
        return bfloat16 ? QuantizedSegment::rows(format, outputs, inputs, values)
                        : QuantizedSegment::floats(outputs, inputs, values);
      };
      std::vector<MessageGateWeights> ring;
      for (uint32_t i = 0; i < copies; ++i)
        ring.push_back({segment(kRank, 2 * kHidden, 0.02f), segment(kHidden, kRank, 0.1f),
                        segment(kHidden, kRank, 0.1f)});
      // The rows: x of three queries and four messages a row, as Inkling-2-Mini's mixes write them, and both
      // normalized in bf16 beside them.
      const MetalBuffer x = randoms(backend, uint64_t{mostRows} * 3 * kHidden, 1.5f, false, rng),
                        message = randoms(backend, uint64_t{mostRows} * 4 * kHidden, 0.7f, false, rng),
                        normalizedX = randoms(backend, uint64_t{mostRows} * 3 * kHidden, 1.0f, true, rng),
                        normalizedMessage = randoms(backend, uint64_t{mostRows} * 4 * kHidden, 1.0f, true, rng);
      std::vector<MetalBuffer> norms, outputs;
      for (uint32_t g = 0; g < MessageGate::kMaximumGates; ++g) {
        norms.push_back(randoms(backend, kHidden, 0.5f, false, rng));
        outputs.push_back(zeros(backend, uint64_t{mostRows} * kHidden * 2, "gate output"));
      }
      const uint32_t slices = (kHidden + MESSAGE_GATE_THREADS - 1) / MESSAGE_GATE_THREADS;
      const MessageGateBuffers buffers{
          x,
          3 * kHidden,
          message,
          4 * kHidden,
          normalizedX,
          normalizedMessage,
          zeros(backend, 16, "gate concat"),
          zeros(backend, uint64_t{MessageGate::kMaximumGates} * mostRows * kRank * 2, "gate low"),
          zeros(backend, uint64_t{MessageGate::kMaximumGates} * mostRows * kHidden * 4, "gate gated rows"),
          zeros(backend, uint64_t{MessageGate::kMaximumGates} * (mostRows + MESSAGE_GATE_GROUP_ROWS) * slices * 4,
                "gate partial sums"),
          zeros(backend, MessageGate::kCountersBytes, "gate counters")};
      std::vector<Case> cases;
      for (const uint32_t rows : rowCounts)
        for (const uint32_t gates : {1u, 3u}) cases.push_back({rows, gates, {}});
      uint32_t next = 0;
      // A whole call's time (part 0) or one kernel's alone, us a call.
      const auto time = [&](Case &c, size_t part) {
        CommandGraph graph;
        const uint32_t calls = copies / c.gates;
        for (uint32_t call = 0; call < calls; ++call) {
          std::vector<MessageGateInput> inputs;
          for (uint32_t g = 0; g < c.gates; ++g)
            inputs.push_back({ring[next++ % copies], {norms[g], true}, (c.gates == 3 ? g : 0) * kHidden, g * kHidden,
                              backend.view(outputs[g], 0, uint64_t{c.rows} * kHidden * 2)});
          static_cast<void>(MessageGate::add(graph, buffers, inputs, kHidden, c.rows));
        }
        std::vector<metal::ComputeDispatch> dispatches;
        for (const metal::ComputeDispatch &d : graph.dispatches())
          if (d.pipelineName.starts_with(kParts[part])) dispatches.push_back(d);
        return backend.submitCommandAsync(dispatches).wait().gpuSeconds * 1e6 / calls;
      };
      for (Case &c : cases)
        for (double spent = 0; spent < kWarmupSeconds * 1e6;) spent += time(c, 0) * (copies / c.gates);
      for (uint32_t r = 0; r < rounds; ++r)
        for (size_t i = 0; i < cases.size(); ++i) {
          Case &c = cases[r % 2 ? cases.size() - 1 - i : i];
          for (size_t part = 0; part < kParts.size(); ++part) c.us[part].push_back(time(c, part));
        }
      const DeviceCapabilities &device = backend.capabilities();
      std::printf("%s (GPU family %u, %u cores): message gates %s, hidden %u, rank %u, %u weight copies of %.2f MB, "
                  "median us a call of %u rounds (quartiles), down / apply alone",
                  device.deviceName.c_str(), device.appleGpuFamily, device.gpuCoreCount, argv[2], kHidden, kRank,
                  copies, gateBytes / 1e6, rounds);
      if (readGbps) std::printf(" and their share of a %.0f GB/s read", readGbps);
      std::printf(", down / apply threadgroups per core\n");
      const double halfGate = double(gateBytes) / 2;
      for (Case &c : cases) {
        for (std::vector<double> &us : c.us) std::sort(us.begin(), us.end());
        const auto median = [&](size_t part) { return c.us[part][c.us[part].size() / 2]; };
        const uint32_t groups = (c.rows + MessageGate::groupRows(c.rows) - 1) / MessageGate::groupRows(c.rows);
        const double down = double(kRank / MESSAGE_GATE_DOWN_OUTPUTS) * groups * c.gates / device.gpuCoreCount;
        const double apply = double(slices) * groups * c.gates / device.gpuCoreCount;
        std::printf("  R%u G%u: %.2f (%.2f-%.2f) d/a %.2f/%.2f", c.rows, c.gates, median(0),
                    c.us[0][c.us[0].size() / 4], c.us[0][c.us[0].size() * 3 / 4], median(1), median(2));
        if (readGbps) {
          const double floor = halfGate * c.gates / (readGbps * 1e3);
          std::printf(" (%.0f%%/%.0f%% of a read)", 100 * floor / median(1), 100 * floor / median(2));
        }
        std::printf(" tg/core %.1f/%.1f\n", down, apply);
      }
    } catch (const std::exception &e) {
      std::cerr << "message-gate-benchmark: " << e.what() << '\n';
      return 1;
    }
    return 0;
  }
}
