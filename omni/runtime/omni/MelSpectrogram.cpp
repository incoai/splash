#include "omni/MelSpectrogram.hpp"

// The current CBLAS interface; the one the header offers by default is deprecated.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace splash::omni {
namespace {

constexpr uint32_t kBins = kMelFftSize / 2 + 1; // 201 frequency bins
constexpr uint32_t kDftColumns = 2 * kBins;     // cosine and sine per bin
constexpr uint32_t kPad = kMelFftSize / 2;      // centered frames: 200 each side
constexpr double kLogMelFloor = 1e-10;
constexpr double kDynamicRange = 8.0;

// Slaney's mel scale as transformers' mel_filter_bank computes it.
double hertzToMel(double hertz) noexcept {
  constexpr double minLogHertz = 1000.0, minLogMel = 15.0;
  const double logStep = 27.0 / std::log(6.4);
  if (hertz >= minLogHertz) return minLogMel + std::log(hertz / minLogHertz) * logStep;
  return 3.0 * hertz / 200.0;
}
double melToHertz(double mel) noexcept {
  constexpr double minLogHertz = 1000.0, minLogMel = 15.0;
  const double logStep = std::log(6.4) / 27.0;
  if (mel >= minLogMel) return minLogHertz * std::exp(logStep * (mel - minLogMel));
  return 200.0 * mel / 3.0;
}

// [201][128]: triangular filters between consecutive mel-spaced edges,
// normalized to unit area (Slaney), in double, rounded once to fp32.
std::vector<float> melFilterBank() {
  const double melMax = hertzToMel(kMelSampleRate / 2.0);
  std::array<double, kMelBins + 2> edges{};
  for (uint32_t i = 0; i < edges.size(); ++i) edges[i] = melToHertz(double(i) * melMax / double(kMelBins + 1));
  std::vector<float> filters(size_t(kBins) * kMelBins, 0.0F);
  for (uint32_t bin = 0; bin < kBins; ++bin) {
    const double hertz = double(bin) * (kMelSampleRate / 2.0) / double(kBins - 1);
    for (uint32_t m = 0; m < kMelBins; ++m) {
      const double down = -(edges[m] - hertz) / (edges[m + 1] - edges[m]);
      const double up = (edges[m + 2] - hertz) / (edges[m + 2] - edges[m + 1]);
      const double weight = std::max(0.0, std::min(down, up));
      filters[size_t(bin) * kMelBins + m] = float(weight * 2.0 / (edges[m + 2] - edges[m]));
    }
  }
  return filters;
}

// [400][402]: column 2k is cos(2 pi k n / 400), column 2k + 1 is -sin.
std::vector<float> dftMatrix() {
  std::vector<float> matrix(size_t(kMelFftSize) * kDftColumns);
  for (uint32_t n = 0; n < kMelFftSize; ++n)
    for (uint32_t k = 0; k < kBins; ++k) {
      const double angle = 2.0 * std::numbers::pi * double(k) * double(n) / double(kMelFftSize);
      matrix[size_t(n) * kDftColumns + 2 * k] = float(std::cos(angle));
      matrix[size_t(n) * kDftColumns + 2 * k + 1] = float(-std::sin(angle));
    }
  return matrix;
}

std::array<float, kMelFftSize> periodicHann() {
  std::array<float, kMelFftSize> window{};
  for (uint32_t n = 0; n < kMelFftSize; ++n)
    window[n] = float(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * double(n) / double(kMelFftSize)));
  return window;
}

const std::vector<float> &filters() {
  static const std::vector<float> table = melFilterBank();
  return table;
}
const std::vector<float> &dft() {
  static const std::vector<float> table = dftMatrix();
  return table;
}
const std::array<float, kMelFftSize> &window() {
  static const std::array<float, kMelFftSize> table = periodicHann();
  return table;
}

} // namespace

std::vector<float> logMelSpectrogram(std::span<const float> samples16k) {
  const uint32_t frames = melFrames(samples16k.size());
  if (!frames) return {};
  // Reflect-pad the signal by half a window on each side; a signal shorter
  // than a window plus one sample is zero-extended first.
  std::vector<float> signal(samples16k.begin(), samples16k.end());
  if (signal.size() < kMelFftSize + 1) signal.resize(kMelFftSize + 1, 0.0F);
  const size_t length = signal.size();
  std::vector<float> padded(length + 2 * kPad);
  for (uint32_t i = 0; i < kPad; ++i) padded[i] = signal[kPad - i];
  std::copy(signal.begin(), signal.end(), padded.begin() + kPad);
  for (uint32_t i = 0; i < kPad; ++i) padded[kPad + length + i] = signal[length - 2 - i];

  // Windowed frames [frames][400].
  std::vector<float> framed(size_t(frames) * kMelFftSize);
  const std::array<float, kMelFftSize> &hann = window();
  for (uint32_t t = 0; t < frames; ++t)
    vDSP_vmul(padded.data() + size_t(t) * kMelHop, 1, hann.data(), 1, framed.data() + size_t(t) * kMelFftSize, 1,
              kMelFftSize);

  // Spectrum [frames][402] = frames x DFT, then the power of each bin.
  std::vector<float> spectrum(size_t(frames) * kDftColumns);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(frames), int(kDftColumns), int(kMelFftSize), 1.0F,
              framed.data(), int(kMelFftSize), dft().data(), int(kDftColumns), 0.0F, spectrum.data(),
              int(kDftColumns));
  std::vector<float> power(size_t(frames) * kBins);
  for (uint32_t t = 0; t < frames; ++t) {
    const float *row = spectrum.data() + size_t(t) * kDftColumns;
    float *out = power.data() + size_t(t) * kBins;
    for (uint32_t k = 0; k < kBins; ++k) out[k] = row[2 * k] * row[2 * k] + row[2 * k + 1] * row[2 * k + 1];
  }

  // Mel energies [frames][128] = power x filters, then the log compression.
  std::vector<float> mel(size_t(frames) * kMelBins);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(frames), int(kMelBins), int(kBins), 1.0F, power.data(),
              int(kBins), filters().data(), int(kMelBins), 0.0F, mel.data(), int(kMelBins));
  float maximum = -INFINITY;
  for (float &value : mel) {
    value = float(std::log10(std::max(double(value), kLogMelFloor)));
    maximum = std::max(maximum, value);
  }
  const float floor = maximum - float(kDynamicRange);
  for (float &value : mel) value = (std::max(value, floor) + 4.0F) / 4.0F;
  return mel;
}

} // namespace splash::omni
