// Validates the native-row embedding gathers (kq_embed_q4k / q6k / q80) against a CPU
// reference on random tables. Compiles the inlined kernel source at runtime, so no Xcode
// Metal toolchain is needed:
//   python3 dev/benchmarks/kquant/inline_metal.py metal/kernels/shared/kquant.metal > /tmp/kquant.metal
//   clang++ -std=c++20 -fobjc-arc -O2 -framework Foundation -framework Metal -I runtime \
//       dev/benchmarks/kquant/harness_embed.mm -o /tmp/harness_embed && /tmp/harness_embed /tmp/kquant.metal
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "metal/abi/KQuant.h"

static uint16_t halfBits(float f) { _Float16 h = (_Float16)f; uint16_t b; memcpy(&b, &h, 2); return b; }
static float halfValue(uint16_t b) { _Float16 h; memcpy(&h, &b, 2); return (float)h; }
static float bf16Round(float f) { uint32_t u; memcpy(&u, &f, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; float r; memcpy(&r, &u, 4); return r; }
static float bf16Value(uint16_t b) { uint32_t u = uint32_t(b) << 16; float r; memcpy(&r, &u, 4); return r; }

struct Table { std::string kernel; uint32_t type; uint32_t rowBytes; std::vector<uint8_t> bytes; };

static Table makeTable(uint32_t type, uint32_t vocab, uint32_t hidden, std::mt19937 &rng) {
  std::uniform_int_distribution<int> byte(0, 255), sc(-64, 63);
  std::uniform_real_distribution<float> scale(0.001f, 0.05f);
  Table t; t.type = type;
  const uint32_t blocks = type == 8 ? hidden / 32 : hidden / 256;
  const uint32_t blockBytes = type == 12 ? 144 : type == 14 ? 210 : 34;
  t.kernel = type == 12 ? "kq_embed_q4k" : type == 14 ? "kq_embed_q6k" : "kq_embed_q80";
  t.rowBytes = blocks * blockBytes;
  t.bytes.resize(size_t(vocab) * t.rowBytes);
  for (size_t i = 0; i < t.bytes.size(); ++i) t.bytes[i] = uint8_t(byte(rng));
  for (uint32_t row = 0; row < vocab; ++row)
    for (uint32_t b = 0; b < blocks; ++b) {
      uint8_t *blk = t.bytes.data() + size_t(row) * t.rowBytes + size_t(b) * blockBytes;
      uint16_t d = halfBits(scale(rng)), dmin = halfBits(scale(rng));
      if (type == 12) { memcpy(blk, &d, 2); memcpy(blk + 2, &dmin, 2); }
      else if (type == 14) { for (int i = 0; i < 16; ++i) blk[192 + i] = uint8_t(int8_t(sc(rng))); memcpy(blk + 208, &d, 2); }
      else memcpy(blk, &d, 2);
    }
  return t;
}

static float reference(const Table &t, uint32_t token, uint32_t dim, uint32_t hidden) {
  const uint8_t *row = t.bytes.data() + size_t(token) * t.rowBytes;
  if (t.type == 12) {
    const uint8_t *blk = row + (dim / 256) * 144;
    const float d = halfValue(uint16_t(blk[0] | (blk[1] << 8))), dmin = halfValue(uint16_t(blk[2] | (blk[3] << 8)));
    const uint32_t j = (dim % 256) / 32, l = dim % 32; const uint8_t *sc = blk + 4; uint8_t s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    const uint8_t q = (blk[16 + (j / 2) * 32 + l] >> ((j % 2) * 4)) & 15;
    return d * float(s) * float(q) - dmin * float(m);
  }
  if (t.type == 14) {  // llama.cpp dequantize_row_q6_K
    const uint8_t *blk = row + (dim / 256) * 210;
    const uint32_t l = dim % 256, n = l / 128, r = l % 128, quarter = r / 32, pos = r % 32;
    const uint8_t *ql = blk + n * 64, *qh = blk + 128 + n * 32; const int8_t *sc = (const int8_t *)(blk + 192 + n * 8);
    int q;
    if (quarter == 0) q = int((ql[pos] & 0xF) | (((qh[pos] >> 0) & 3) << 4)) - 32;
    else if (quarter == 1) q = int((ql[pos + 32] & 0xF) | (((qh[pos] >> 2) & 3) << 4)) - 32;
    else if (quarter == 2) q = int((ql[pos] >> 4) | (((qh[pos] >> 4) & 3) << 4)) - 32;
    else q = int((ql[pos + 32] >> 4) | (((qh[pos] >> 6) & 3) << 4)) - 32;
    const float d = halfValue(uint16_t(blk[208] | (blk[209] << 8)));
    return d * float(sc[pos / 16 + 2 * quarter]) * float(q);
  }
  const uint8_t *blk = row + (dim / 32) * 34;
  return halfValue(uint16_t(blk[0] | (blk[1] << 8))) * float(int8_t(blk[2 + dim % 32]));
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: harness_embed <inlined kquant.metal>\n"); return 2; }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    if (!source) { fprintf(stderr, "read failed\n"); return 1; }
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = (MTLLanguageVersion)(4 << 16);
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr, "compile failed: %s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::mt19937 rng(7);
    const uint32_t vocab = 64, hidden = 5120, rows = 8;
    const uint32_t tokens[rows] = {0, 1, 63, 7, 70000u, 5, 63, 2};
    int failures = 0;
    for (uint32_t type : {12u, 14u, 8u}) {
      Table t = makeTable(type, vocab, hidden, rng);
      id<MTLFunction> fn = [library newFunctionWithName:@(t.kernel.c_str())];
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
      if (!pso) { fprintf(stderr, "%s: pipeline failed\n", t.kernel.c_str()); return 1; }
      id<MTLBuffer> tokenBuffer = [device newBufferWithBytes:tokens length:sizeof tokens options:MTLResourceStorageModeShared];
      id<MTLBuffer> table = [device newBufferWithBytes:t.bytes.data() length:t.bytes.size() options:MTLResourceStorageModeShared];
      id<MTLBuffer> output = [device newBufferWithLength:size_t(rows) * hidden * 2 options:MTLResourceStorageModeShared];
      KQEmbedParams params{rows, vocab, hidden};
      id<MTLCommandBuffer> commands = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:tokenBuffer offset:0 atIndex:0];
      [encoder setBuffer:table offset:0 atIndex:1];
      [encoder setBuffer:output offset:0 atIndex:2];
      [encoder setBytes:&params length:sizeof params atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake((rows * hidden + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [commands commit];
      [commands waitUntilCompleted];
      const uint16_t *out = (const uint16_t *)output.contents;
      size_t exact = 0, offByOne = 0, bad = 0; double worst = 0;
      for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t token = tokens[r] < vocab ? tokens[r] : 0;
        for (uint32_t dim = 0; dim < hidden; ++dim) {
          const float ref = reference(t, token, dim, hidden), refBf = bf16Round(ref), got = bf16Value(out[size_t(r) * hidden + dim]);
          if (got == refBf) { ++exact; continue; }
          const float tolerance = std::fabs(refBf) * (1.0f / 128) + 1e-6f;  // one bf16 ulp
          if (std::fabs(got - refBf) <= tolerance) ++offByOne; else { ++bad; worst = std::max(worst, double(std::fabs(got - ref))); }
        }
      }
      printf("%-13s rows %u x %u: exact %zu, within 1 bf16 ulp %zu, wrong %zu (worst abs err %.3g)\n", t.kernel.c_str(), rows, hidden, exact, offByOne, bad, worst);
      failures += bad != 0;
    }
    printf(failures ? "FAILED\n" : "all embedding gathers match the CPU reference\n");
    return failures ? 1 : 0;
  }
}
