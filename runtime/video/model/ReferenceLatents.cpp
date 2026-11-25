#include "video/model/ReferenceLatents.hpp"

#include "video/model/Latents.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace splash::video::model {
namespace {

// at::uniform_real_distribution<float>(0, 1): the low 24 bits of one 32-bit draw, times 2^-24
float uniform(std::mt19937 &gen) {
  return static_cast<float>(gen() & ((1u << 24) - 1)) * (1.0f / static_cast<float>(1u << 24));
}

// ATen normal_fill_16 (mean 0, std 1): Box-Muller over data[j], data[j + 8]
void fill16(float *data) {
  for (int j = 0; j < 8; ++j) {
    const float u1 = 1.0f - data[j];
    const float u2 = data[j + 8];
    const float radius = std::sqrt(-2.0f * std::log(u1));
    const float theta = static_cast<float>(2.0f * std::numbers::pi_v<double> * u2);
    data[j] = radius * std::cos(theta);
    data[j + 8] = radius * std::sin(theta);
  }
}

} // namespace

void torchRandn(uint64_t seed, size_t count, float *out) {
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed & 0xffffffffu));
  for (size_t i = 0; i < count; ++i) out[i] = uniform(gen);
  for (size_t i = 0; i + 16 <= count; i += 16) fill16(out + i);
  if (count % 16) {
    float *tail = out + count - 16;
    for (int i = 0; i < 16; ++i) tail[i] = uniform(gen);
    fill16(tail);
  }
}

void referenceLatents(const std::vector<float> &moments, int32_t F, int32_t h, int32_t w, uint64_t seed,
                      std::vector<float> &out) {
  std::vector<float> noise(size_t(24) * F * h * w);
  torchRandn(seed, noise.size(), noise.data());
  constexpr int32_t C = 24;
  const size_t plane = size_t(F) * h * w;
  out.resize(plane * C);
  for (int32_t c = 0; c < C; ++c) {
    const float lm = static_cast<float>(kVaeLatentMean[c]), ls = static_cast<float>(kVaeLatentStd[c]);
    for (size_t p = 0; p < plane; ++p) {
      const size_t idx = size_t(c) * plane + p;
      const float mean = moments[p * 2 * C + c];
      const float logvar = std::clamp(moments[p * 2 * C + C + c], -30.0f, 20.0f);
      const float sd = std::exp(0.5f * logvar);
      const float n = sd * noise[idx];
      float v = mean + n;
      v = static_cast<float>(static_cast<_Float16>(v));
      const float d = v - lm;
      out[idx] = d / ls;
    }
  }
}

} // namespace splash::video::model
