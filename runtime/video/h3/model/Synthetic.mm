#include "video/h3/model/Synthetic.hpp"

#include <cmath>
#include <cstring>

namespace h3::model {

uint64_t syntheticSeed(int group, int id) {
  return (uint64_t(h3_syn_hash(0x48335359ull, uint64_t(group))) << 32) | h3_syn_hash(0x48335359ull + 1, uint64_t(id));
}

float syntheticFloat(uint64_t seed, uint64_t i, float scale, float offset) {
  const float r = static_cast<float>(static_cast<int32_t>(h3_syn_hash(seed, i)) >> 8) * 0x1p-23f;
  return std::fma(scale, r, offset);
}

void syntheticF32(uint64_t seed, size_t n, float scale, float offset, float *out) {
  for (size_t i = 0; i < n; ++i) out[i] = syntheticFloat(seed, i, scale, offset);
}

void syntheticBf16(uint64_t seed, size_t n, float scale, float offset, uint16_t *out) {
  for (size_t i = 0; i < n; ++i) {
    const float f = syntheticFloat(seed, i, scale, offset);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    out[i] = static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
  }
}

Status SyntheticFiller::allocate(size_t count, int type, uint64_t seed, float scale, float offset,
                                 __strong id<MTLBuffer> &out, const char *what) {
  const size_t elem = type == H3_SYN_I8 ? 1 : type == H3_SYN_F32 ? 4 : 2;
  H3_TRY(b_.allocate(count * elem, out, what));
  fill(out, 0, count, type, seed, scale, offset);
  return {};
}

void SyntheticFiller::fill(id<MTLBuffer> buffer, size_t offsetBytes, size_t count, int type, uint64_t seed,
                           float scale, float offset) {
  const size_t elem = type == H3_SYN_I8 ? 1 : type == H3_SYN_F32 ? 4 : 2;
  jobs_.push_back({buffer, offsetBytes,
                   SyntheticFill{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                                 static_cast<uint32_t>(count), type, scale, offset}});
  bytes_ += count * elem;
}

Status SyntheticFiller::run() {
  H3_TRY(b_.prepare({"synthetic_fill"}));
  H3_TRY(b_.runSync([&](id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:b_.pipeline("synthetic_fill")];
    for (const Job &j : jobs_) {
      [e setBuffer:j.buffer offset:j.offset atIndex:0];
      metal::bindBytes(e, j.fill, 1);
      [e dispatchThreads:MTLSizeMake(j.fill.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  }));
  jobs_.clear();
  return {};
}

} // namespace h3::model
