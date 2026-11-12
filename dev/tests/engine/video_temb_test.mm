// The adaLN tables' time embedder (model::Prep::timestepEmbedding), on the CPU: silu(time_embedder(sinusoid(t))) of a
// few timesteps against an fp64 evaluation of the same module (sinusoid, proj_in, silu, proj_out, silu), on synthetic
// weights of the real shapes; each row within 1e-6 (relative RMS), and its bf16 bits the RNE rounding of its fp32 values.
#include "TestChecks.hpp"
#include "video/model/Prep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

using splash::test::require;
using namespace splash::video;

constexpr int kIn = 256, kHidden = 5376, kDim = 2688;

uint16_t rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

std::vector<float> gaussian(std::mt19937_64 &rng, size_t n, double scale) {
  std::normal_distribution<double> d(0.0, scale);
  std::vector<float> v(n);
  for (float &x : v) x = static_cast<float>(d(rng));
  return v;
}

std::vector<double> reference(double t, const std::vector<float> &w1, const std::vector<float> &b1,
                              const std::vector<float> &w2, const std::vector<float> &b2) {
  std::vector<double> x(kIn), h(kHidden), y(kDim);
  for (int i = 0; i < kIn / 2; ++i) {  // timestep_embedding(t, 256), flip_sin_to_cos: [cos | sin]
    const double a = t * std::exp(-std::log(10000.0) * i / 128.0);
    x[size_t(i)] = std::cos(a);
    x[size_t(kIn / 2 + i)] = std::sin(a);
  }
  for (int j = 0; j < kHidden; ++j) {
    double acc = b1[size_t(j)];
    for (int k = 0; k < kIn; ++k) acc += x[size_t(k)] * double(w1[size_t(j) * kIn + size_t(k)]);
    h[size_t(j)] = acc / (1.0 + std::exp(-acc));
  }
  for (int j = 0; j < kDim; ++j) {
    double acc = b2[size_t(j)];
    for (int k = 0; k < kHidden; ++k) acc += h[size_t(k)] * double(w2[size_t(j) * kHidden + size_t(k)]);
    y[size_t(j)] = acc / (1.0 + std::exp(-acc));
  }
  return y;
}

} // namespace

int main() {
  try {
    std::mt19937_64 rng(0x48335445);  // "H3TE"
    // proj_in / proj_out at nn.Linear's scale, biases that move the pre-activations off zero
    const std::vector<float> w1 = gaussian(rng, size_t(kHidden) * kIn, 1.0 / std::sqrt(double(kIn)));
    const std::vector<float> b1 = gaussian(rng, kHidden, 0.5);
    const std::vector<float> w2 = gaussian(rng, size_t(kDim) * kHidden, 2.0 / std::sqrt(double(kHidden)));
    const std::vector<float> b2 = gaussian(rng, kDim, 0.5);
    // the ends of the schedules, a FastH3 rung, mid timesteps and the keyframe timestep
    const std::vector<float> t = {0.0f, 8.344650268554688e-05f, 0.10048043727874756f, 0.5f, 0.874f, 0.999f, 1.0f};
    std::vector<float> st(t.size() * kDim);
    std::vector<uint16_t> bf(t.size() * kDim);
    model::Prep::timestepEmbedding(t, w1, b1, w2, b2, st.data(), bf.data());
    double worst = 0;
    for (size_t r = 0; r < t.size(); ++r) {
      const std::vector<double> want = reference(double(t[r]), w1, b1, w2, b2);
      double err = 0, norm = 0;
      for (int j = 0; j < kDim; ++j) {
        const size_t at = r * kDim + size_t(j);
        const double d = double(st[at]) - want[size_t(j)];
        err += d * d;
        norm += want[size_t(j)] * want[size_t(j)];
        require(bf[at] == rne(st[at]), "the bf16 bits are not the RNE rounding of the fp32 values");
      }
      const double rel = std::sqrt(err / norm);
      worst = std::max(worst, rel);
      require(norm > 0 && rel <= 1e-6, "silu(temb) is more than 1e-6 (relative RMS) off the fp64 reference");
    }
    std::cout << "video_temb_test: PASS (worst relative RMS " << worst << ")\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "video_temb_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
