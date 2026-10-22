#pragma once

// Request and statistics types of the engine (the C ABI in h3.h mirrors them).
#include <cstdint>
#include <string>
#include <vector>

namespace h3 {

enum class Variant : int32_t { Turbo = 0, Default = 1 };
enum class VaePrecision : int32_t { W8 = 0, F16 = 1 };

// User-facing options of one generation request.
struct Request {
  Variant variant = Variant::Turbo;
  int32_t frames = 73;  // 17 n + 5
  int32_t width = 1344, height = 768;
  int32_t windowChunk = 5, windowRadius = 1;  // anchor frames first and last (always)
  bool qkfix = true;
  std::vector<int32_t> qkfixBlocks{45, 48, 49};
  bool stream = false;
  int32_t streamSlots = 8;
  VaePrecision vaePrecision = VaePrecision::W8;
};

struct ForwardStats {
  double wallMs = 0, embedMs = 0, blocksSpanMs = 0, blocksGpuMs = 0, finalMs = 0;
  double gpuBusyMs = 0, gpuIdleMs = 0;  // GPU time of the block commands, and the rest of their span
  int32_t commandBuffers = 0;           // block commands
  double streamWaitMs = 0, streamWaitMaxMs = 0, streamReadGb = 0, streamReadS = 0, streamReadMaxMs = 0;
  int32_t streamWaits = 0, streamBlocks = 0;
  double deviceMb = 0;
};

struct DecodeStats {
  double wallS = 0, gpuS = 0, hostCopyS = 0, uploadS = 0, gpuSpanS = 0;
  int64_t nonfinite = 0;
  double deviceMb = 0;
};

struct AudioStats {
  double wallS = 0, gpuS = 0;
  int64_t samples = 0, nonfinite = 0;  // per channel; non-finite values in the waveform
  double deviceMb = 0;
};

struct PreviewStats {
  double wallS = 0, gpuS = 0, hostCopyS = 0;
  int64_t nonfinite = 0;
  int32_t chunks = 0, frames = 0;
  double deviceMb = 0;
};

} // namespace h3
