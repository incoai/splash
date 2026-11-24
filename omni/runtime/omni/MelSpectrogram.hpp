#pragma once

// The Whisper log-mel features of the ASR front end, on the CPU: 16 kHz mono
// samples to [frames][128] fp32 exactly as transformers' WhisperFeatureExtractor
// computes them for an unpadded signal (n_fft 400, hop 160, periodic Hann,
// centered with reflect padding, power spectrum, the last STFT frame dropped,
// 128 Slaney mel filters to 8 kHz with Slaney area normalization, log10 with
// a 1e-10 floor, clamped to 8 below the global maximum, (x + 4) / 4).
//
// The STFT is two Accelerate GEMMs rather than an FFT: a 400-point DFT is
// not a vDSP size (400 = 16 * 25), so the windowed frames multiply a
// [400][402] cosine/sine matrix, the power spectrum multiplies the
// [201][128] filter bank. For 30 s of audio that is 0.5 G multiply-adds, a
// few milliseconds.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::omni {

inline constexpr uint32_t kMelSampleRate = 16000;
inline constexpr uint32_t kMelBins = 128;
inline constexpr uint32_t kMelFftSize = 400;
inline constexpr uint32_t kMelHop = 160;

// The frames the features of `samples` samples hold: one per hop, the
// trailing centered frame dropped as the reference does.
[[nodiscard]] constexpr uint32_t melFrames(size_t samples) noexcept {
  return static_cast<uint32_t>(samples / kMelHop);
}

// The features of a 16 kHz mono signal, frame-major: frame t's 128 mels at
// [t * 128, t * 128 + 128). Empty for fewer than 160 samples. A signal
// shorter than one window is zero-extended to the window before the
// transform (the reference's reflect padding needs more than half a window).
[[nodiscard]] std::vector<float> logMelSpectrogram(std::span<const float> samples16k);

} // namespace splash::omni
