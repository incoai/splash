// The video engine (runtime/video) on deterministic synthetic weights and
// inputs, without weight files (video/model/Synthetic.hpp): one 3 s forward
// of the DiT's embedding, block 0 and final layer, whose video and audio
// velocities must hash to the h3 engine's references (its synthetic_check,
// fwd/q1_video and fwd/q1_audio), and its wall time. The int8 MPP kernels
// need an Apple10 GPU (M5); on an earlier GPU the test reports a skip.
#import <Metal/Metal.h>

#include "video/h3/engine/Engine.hpp"
#include "video/h3/model/Synthetic.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kVideoSha256 = "48d1344755caf3081c03e3525a904b1e7b6b2cd345a70df97f2ff7f22da55da4";
constexpr const char *kAudioSha256 = "14835bdc152f36c39aaed79143a74f5a465b687a3b937176f6347aaa400d8ec7";

std::string sha256(const std::vector<float> &values) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(values.data(), static_cast<CC_LONG>(values.size() * sizeof(float)), digest);
  std::string hex;
  char byte[3];
  for (const unsigned char value : digest) {
    snprintf(byte, sizeof byte, "%02x", value);
    hex += byte;
  }
  return hex;
}

void check(const h3::Status &status) {
  if (!status.ok()) throw std::runtime_error(status.message());
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::invalid_argument("usage: video-synthetic METALLIB");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device || ![device supportsFamily:MTLGPUFamilyApple10]) {
        std::cout << "video_synthetic_metal_test: SKIP (needs an Apple10 GPU)\n";
        return 0;
      }
      h3::Request request;  // 3 s turbo, window 5 / 1, QKFIX 45, 48, 49
      std::unique_ptr<h3::Engine> engine;
      check(h3::Engine::create({argv[1], ""}, request, engine));
      check(engine->loadDitSynthetic(32, 1));
      const auto seed = [](int id) { return h3::model::syntheticSeed(h3::model::kSyntheticInputGroup, id); };
      std::vector<uint16_t> blocks(size_t(50) * 3 * 6 * 5376), final(size_t(3) * 2 * 5376);
      h3::model::syntheticBf16(seed(1), blocks.size(), 0.1f, 0.0f, blocks.data());
      h3::model::syntheticBf16(seed(2), final.size(), 0.1f, 0.0f, final.data());
      const h3::model::VideoGeometry &g = engine->geometry();
      std::vector<float> video(size_t(g.videoRows()) * 96), audio(size_t(g.audioRows) * 32);
      h3::model::syntheticF32(seed(3), video.size(), 1.7f, 0.0f, video.data());
      h3::model::syntheticF32(seed(4), audio.size(), 1.7f, 0.0f, audio.data());
      std::vector<float> videoOut(video.size()), audioOut(audio.size());
      h3::ForwardStats stats;
      const auto started = std::chrono::steady_clock::now();
      check(engine->forward(blocks.data(), final.data(), video.data(), audio.data(), videoOut.data(),
                            audioOut.data(), stats));
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      const bool same = sha256(videoOut) == kVideoSha256 && sha256(audioOut) == kAudioSha256;
      printf("video synthetic forward: 1 block, %.3f s wall, %.3f s GPU blocks, %.0f MB device, velocities %s\n",
             seconds, stats.blocksGpuMs / 1e3, stats.deviceMb, same ? "identical" : "DIFFER");
      if (!same) throw std::runtime_error("synthetic velocities differ from the h3 references");
      std::cout << "video_synthetic_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "video_synthetic_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
