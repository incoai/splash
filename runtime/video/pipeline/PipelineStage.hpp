#pragma once

// A stage of the video runtime (SGLang Diffusion's PipelineStage): one phase of a request, which reads what the
// stages before it left in the batch and leaves its results there. The runtime (VideoRuntime.hpp) runs its stages in
// order and announces each stage's phase as it starts and as it completes; a stage reports the steps in between and
// its outputs, and stops at a step boundary once the request is cancelled.
#include "video/pipeline/VideoBatch.hpp"
#include "video/protocol/VideoProtocol.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

namespace splash::video {

// The first 12 hex digits of the SHA-256 of an array's bytes, as the stages log their inputs and outputs: two runs
// compare from their logs alone, and a digest equals that of the array kept to a file in the same order.
[[nodiscard]] inline std::string logDigest(const std::vector<float> &values) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(values.data());
  const size_t size = values.size() * sizeof(float);
  CC_SHA256_CTX context;
  CC_SHA256_Init(&context);
  // CommonCrypto takes 32-bit lengths.
  constexpr size_t kPieceBytes = size_t(1) << 30;
  for (size_t at = 0; at < size; at += kPieceBytes)
    CC_SHA256_Update(&context, bytes + at, static_cast<CC_LONG>(std::min(kPieceBytes, size - at)));
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &context);
  std::string hex;
  for (int i = 0; i < 6; ++i) hex += std::format("{:02x}", digest[i]);
  return hex;
}

// A request stopped by its Cancel.
struct Cancelled final {};

// What the request's client hears while a stage runs.
class StageEvents {
public:
  virtual ~StageEvents() = default;
  // `step` of the stage's steps are done.
  virtual void progress(uint32_t step) = 0;
  virtual void output(wire::VideoOutputKind kind, uint32_t frames, std::vector<uint8_t> mp4) = 0;
  [[nodiscard]] virtual bool cancelled() const = 0;
};

class PipelineStage {
public:
  virtual ~PipelineStage() = default;
  [[nodiscard]] virtual wire::VideoPhase phase() const = 0;
  // Whether the stage runs for the request, and the steps its phase reports.
  [[nodiscard]] virtual bool runs(const VideoBatch &) const { return true; }
  [[nodiscard]] virtual uint32_t steps(const VideoBatch &) const { return 1; }
  virtual void forward(VideoBatch &batch, StageEvents &events) = 0;
};

} // namespace splash::video
