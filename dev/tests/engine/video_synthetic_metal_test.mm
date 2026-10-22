// The video engine (runtime/video) on deterministic synthetic weights and
// inputs (VideoSynthetic.hpp): one 3 s forward of the DiT's embedding,
// block 0 and final layer, whose video and audio velocities must hash to the
// h3 engine's references (its synthetic_check, fwd/q1_video and fwd/q1_audio).
// The int8 MPP kernels need an Apple10 GPU (M5); earlier GPUs skip the test.
#include "TestChecks.hpp"
#include "VideoSynthetic.hpp"
#include "metal/MetalBackend.hpp"
#include "video/model/Prep.hpp"
#include "video/ops/WindowAttention.hpp"
#include "video/stages/DenoisingStage.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

using splash::test::require;

std::string sha256(const std::vector<float> &values) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(values.data(), static_cast<CC_LONG>(values.size() * sizeof(float)), digest);
  std::string hex;
  for (const unsigned char byte : digest) {
    char pair[3];
    snprintf(pair, sizeof pair, "%02x", byte);
    hex += pair;
  }
  return hex;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: video-synthetic METALLIB");
      if (splash::metal::probeDeviceCapabilities().appleGpuFamily < 10) {
        std::cout << "video_synthetic_metal_test: SKIP (needs an Apple10 GPU)\n";
        return 0;
      }
      using namespace splash::video;
      using namespace splash::video::test;
      splash::metal::MetalBackend backend(argv[1]);
      ops::checkRegisterLayout(backend);
      ClipOptions options; // 3 s turbo, window 5 / 1, QKFIX 45, 48, 49
      options.stream = false;     // synthetic weights are resident
      const auto g = ops::VideoGeometry::of(options.frames, options.width, options.height);
      const auto seed = [](int id) { return syntheticSeed(kSyntheticInputGroup, id); };
      // The request's layout and RoPE for 32 text rows, and synthetic text; block 0 alone.
      constexpr int32_t kTextRows = 32;
      model::Prepared prepared;
      model::Prep::layout(g, kTextRows, prepared);
      model::Prep::rope(backend, prepared);
      std::vector<uint16_t> text(size_t(kTextRows) * 5376);
      syntheticBf16(seed(0), text.size(), 1.0f, 0.0f, text.data());
      Dit dit(backend, options, g, {kTextRows, prepared.tags.data(), prepared.rope.data(), text.data()},
                     syntheticDitWeights(backend, {}, {0}));
      dit.setBlocks(1);
      std::vector<uint16_t> blocks(size_t(50) * 3 * 6 * 5376), final(size_t(3) * 2 * 5376);
      std::vector<float> video(size_t(g.videoRows()) * 96), audio(size_t(g.audioRows) * 32);
      syntheticBf16(seed(1), blocks.size(), 0.1f, 0.0f, blocks.data());
      syntheticBf16(seed(2), final.size(), 0.1f, 0.0f, final.data());
      syntheticF32(seed(3), video.size(), 1.7f, 0.0f, video.data());
      syntheticF32(seed(4), audio.size(), 1.7f, 0.0f, audio.data());
      std::vector<float> videoOut(video.size()), audioOut(audio.size());
      ForwardStats stats;
      dit.forward(blocks.data(), final.data(), video.data(), audio.data(), videoOut.data(), audioOut.data(), stats);
      require(sha256(videoOut) == "48d1344755caf3081c03e3525a904b1e7b6b2cd345a70df97f2ff7f22da55da4" &&
                  sha256(audioOut) == "14835bdc152f36c39aaed79143a74f5a465b687a3b937176f6347aaa400d8ec7",
              "synthetic velocities differ from the h3 references");
      std::cout << "video_synthetic_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "video_synthetic_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
