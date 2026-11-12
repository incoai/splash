#include "video/model/Latents.hpp"

#include <cstddef>

namespace splash::video::model {

void unpatchifyVideo(const float *rows, const ops::VideoGeometry &g, const float *std, const float *mean, float *out) {
  const int32_t H = g.latentHeight, W = g.latentWidth, C = 24;
  for (int32_t t = 0; t < g.latentFrames; ++t)
    for (int32_t y = 0; y < g.gridH; ++y)
      for (int32_t x = 0; x < g.gridW; ++x) {
        const float *row = rows + ((static_cast<size_t>(t) * g.gridH + y) * g.gridW + x) * (C * 4);
        for (int32_t py = 0; py < 2; ++py)
          for (int32_t px = 0; px < 2; ++px) {
            float *dst = out + ((static_cast<size_t>(t) * H + 2 * y + py) * W + 2 * x + px) * C;
            for (int32_t c = 0; c < C; ++c) {
              float v = row[c * 4 + py * 2 + px];
              if (std) {
                v = v * std[c];
                v = v + mean[c];
              }
              dst[c] = v;
            }
          }
      }
}

} // namespace splash::video::model
