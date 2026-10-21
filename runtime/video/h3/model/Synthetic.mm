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

metal::MetalBuffer SyntheticFiller::allocate(size_t count, int type, uint64_t seed, float scale, float offset,
                                             std::string_view what) {
  const size_t elem = type == H3_SYN_I8 ? 1 : type == H3_SYN_F32 ? 4 : 2;
  metal::MetalBuffer out = h3::allocate(b_, count * elem, what);
  fill(out, 0, count, type, seed, scale, offset);
  return out;
}

void SyntheticFiller::fill(const metal::MetalBuffer &buffer, size_t offsetBytes, size_t count, int type, uint64_t seed,
                           float scale, float offset) {
  const size_t elem = type == H3_SYN_I8 ? 1 : type == H3_SYN_F32 ? 4 : 2;
  jobs_.push_back({buffer, offsetBytes,
                   SyntheticFill{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                                 static_cast<uint32_t>(count), type, scale, offset}});
  bytes_ += count * elem;
}

void SyntheticFiller::run() {
  Encoder e(b_);
  e.pipeline("synthetic_fill");
  for (const Job &j : jobs_) {
    e.bind({j.buffer, j.offset}, 0);
    e.bytes(j.fill, 1);
    e.dispatchThreads(size3(j.fill.count), size3(256));
  }
  h3::run(b_, e);
  jobs_.clear();
}

} // namespace h3::model
