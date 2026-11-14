// Model-free exact GPU-to-GPU comparison; deliberately collects no timings.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "HyperCaptureContract.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using splash::test::CaptureRegion;
using splash::test::validateHyperCapture;
static void require(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
static uint16_t bf16(float x) { uint32_t bits = std::bit_cast<uint32_t>(x); return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16); }
int main(int argc, const char **argv) {
 @autoreleasepool {
  try {
   require(argc == 2, "usage: hyper-capture-parity TEST_METALLIB");
   id<MTLDevice> device = MTLCreateSystemDefaultDevice(); require(device != nil, "Metal device unavailable");
   NSError *error = nil;
   id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
   require(library != nil, "cannot load test metallib");
   auto pipeline = [&](const char *name) {
    id<MTLFunction> f = [library newFunctionWithName:@(name)]; require(f != nil, "missing test pipeline");
    auto p = [device newComputePipelineStateWithFunction:f error:&error]; require(p != nil, "pipeline creation failed"); return p;
   };
   auto combine = pipeline("hyper_connection_combine"), mean = pipeline("hyper_connection_stream_mean");
   auto fused = pipeline("test_hyper_combine_capture"), bad = pipeline("test_hyper_capture_wrong_order");
   id<MTLCommandQueue> queue = [device newCommandQueue]; require(queue != nil, "command queue unavailable");
   constexpr uint64_t pad = 128;
   unsigned cases = 0, rejected = 0, negativeControls = 0;
   for (uint32_t streamsCount : {1u, 2u, 3u, 4u}) for (uint32_t rows : {1u, 5u, 8u, 16u, 24u, 32u}) for (uint32_t hidden : {1u, 255u, 256u, 257u, 2560u}) for (uint32_t slot : {0u, 1u, 3u}) {
    @autoreleasepool {
     const uint32_t storageRows = rows == 24 ? 32 : rows;
     HyperStreamMeanParams p{rows, hidden, streamsCount, 2, hidden * 4, slot * hidden};
     const uint64_t streamBytes = uint64_t{storageRows} * hidden * streamsCount * 4;
     const uint64_t captureBytes = uint64_t{p.row + storageRows + 1} * p.stride * 2;
     const std::array<uint64_t, 4> sizes{streamBytes, uint64_t{rows} * hidden * 2, uint64_t{rows} * streamsCount * 4, captureBytes};
     std::array<id<MTLBuffer>, 4> reference, candidate;
     for (size_t i = 0; i < 4; ++i) {
      reference[i] = [device newBufferWithLength:sizes[i] + pad * 2 options:MTLResourceStorageModeShared];
      candidate[i] = [device newBufferWithLength:sizes[i] + pad * 2 options:MTLResourceStorageModeShared];
      require(reference[i] != nil && candidate[i] != nil, "test allocation failed");
      std::memset(reference[i].contents, 0xa5, reference[i].length);
     }
     std::mt19937 rng(20261113 + rows * 71 + hidden + slot); std::uniform_real_distribution<float> rand(-8, 8);
     float *streams = reinterpret_cast<float *>(static_cast<char *>(reference[0].contents) + pad);
     auto *branch = reinterpret_cast<uint16_t *>(static_cast<char *>(reference[1].contents) + pad);
     auto *weights = reinterpret_cast<float *>(static_cast<char *>(reference[2].contents) + pad);
     for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t d = 0; d < hidden; ++d) { branch[r * hidden + d] = bf16(rand(rng));
       for (uint32_t s = 0; s < streamsCount; ++s) streams[(uint64_t{r} * streamsCount + s) * hidden + d] = rand(rng); }
      for (uint32_t s = 0; s < streamsCount; ++s) weights[r * streamsCount + s] = rand(rng);
     }
     // Cancellation and signed-zero/rounding boundaries are deterministic.
     for (uint32_t s = 0; s < streamsCount; ++s) { streams[uint64_t{s} * hidden] = std::array<float,4>{16777216.f,1.f,-16777216.f,-0.f}[s]; weights[s] = 0; }
     branch[0] = bf16(-0.f);
     if (hidden > 1) for (uint32_t s = 0; s < streamsCount; ++s) streams[uint64_t{s} * hidden + 1] = std::array<float,4>{1.00390625f, -1.f, 0.0000152587890625f, -0.f}[s];
     for (size_t i = 0; i < 4; ++i) std::memcpy(candidate[i].contents, reference[i].contents, reference[i].length);
     std::array<std::vector<uint8_t>,2> readonly;
     for (size_t i = 0; i < 2; ++i) { auto *b = static_cast<uint8_t *>(reference[i+1].contents); readonly[i].assign(b,b+reference[i+1].length); }
     auto regions = [&](const auto &buffers) {
      std::array<CaptureRegion,4> result;
      for (size_t i=0;i<4;++i) result[i]={reinterpret_cast<uintptr_t>((__bridge void *)buffers[i]),pad,sizes[i],buffers[i].length};
      return result;
     };
     validateHyperCapture(p, regions(reference)); validateHyperCapture(p, regions(candidate));
     // Every bad binding is rejected by the contract before command creation.
     for (size_t i=0;i<4;++i) {
      auto badRegions=regions(candidate); badRegions[i].bytes=0;
      try { validateHyperCapture(p,badRegions); throw std::runtime_error("short buffer accepted"); }
      catch(const std::invalid_argument &) { ++rejected; }
     }
     auto alias=regions(candidate); alias[3]=alias[0];
     try { validateHyperCapture(p,alias); throw std::runtime_error("overlap accepted"); }
     catch(const std::invalid_argument &) { ++rejected; }
     auto run = [&](id<MTLComputePipelineState> state, auto bindings, const void *params, size_t bytes) {
      auto command=[queue commandBuffer]; auto encoder=[command computeCommandEncoder];
      [encoder setComputePipelineState:state];
      for (size_t i=0;i<bindings.size();++i) [encoder setBuffer:bindings[i] offset:pad atIndex:i];
      [encoder setBytes:params length:bytes atIndex:bindings.size()];
      [encoder dispatchThreadgroups:MTLSizeMake(rows,(hidden+255)/256,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
      require(command.status == MTLCommandBufferStatusCompleted, "GPU test command failed");
     };
     const HyperConnectionParams c{rows,hidden,streamsCount,0,0};
     run(combine,std::array{reference[0],reference[1],reference[2]},&c,sizeof(c));
     run(mean,std::array{reference[0],reference[3]},&p,sizeof(p));
     run(fused,candidate,&p,sizeof(p));
     for (size_t i=0;i<4;++i) require(!std::memcmp(reference[i].contents,candidate[i].contents,reference[i].length), "fused/reference bytes differ (state/capture/padding/guard)");
     for (size_t i=0;i<2;++i) require(!std::memcmp(reference[i+1].contents,readonly[i].data(),readonly[i].size()),"read-only input changed");
     if (streamsCount == 4) {
     run(bad,candidate,&p,sizeof(p)); // Zero weights keep row-zero state fixed.
     auto *a=static_cast<uint8_t *>(reference[3].contents), *b=static_cast<uint8_t *>(candidate[3].contents);
     const uint64_t at=pad+(uint64_t{p.row}*p.stride+p.column)*2;
     require(std::memcmp(a+at,b+at,2)!=0,"parity fixture failed to detect reordered FP32 reduction");
     ++negativeControls;
     }
     ++cases;
    }
   }
   std::cout << cases << " exact GPU parity cases, " << rejected << " invalid bindings rejected, " << negativeControls << " faulty reductions detected. Device: " << device.name.UTF8String << ". No model or timing measurements.\n";
   return 0;
  } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
 }
}
