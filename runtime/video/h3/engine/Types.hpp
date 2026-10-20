#pragma once

// Request and statistics types of the engine (the C ABI in include/h3/h3.h mirrors them).
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
  std::string streamImage;  // empty: <weights>/dit-<variant>/stream.bin
  VaePrecision vaePrecision = VaePrecision::W8;
};

struct ForwardStats {
  double wallMs = 0, embedMs = 0, blocksSpanMs = 0, blocksGpuMs = 0, finalMs = 0;
  double linearMs = 0, attentionMs = 0, otherMs = 0, branchMs = 0;  // GPU time by step class
  double gpuBusyMs = 0, gpuIdleMs = 0, gpuGapMaxMs = 0;               // union of command-buffer intervals
  int32_t gpuGaps = 0, commandBuffers = 0;
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
