// Video VAE decoder input and output.
//   vae_embed   folded post_quant_conv + x_embedder (fp32), register tokens, zero token -> fp32 residual
//   vae_finish  stitch the tiles of one temporal clip, cross-fade with the previous clip, de-normalise -> uint8 frames
// Rows: a batch of tile-clips, S = 1797 rows each (7 x 16 x 16 latent tokens, 4 registers, 1 zero token).
#include "metal/abi/video/VaeABI.h"
#include "metal/kernels/video/common/tensor.h"

using namespace h3;

// One thread per (row, 4 output channels). lat: fp32 channels-last [T][LH][LW][24] (de-normalised, time-padded).
// tiles: int4 (t0, y0, x0, -) latent origin of each tile-clip of the batch.
// rows r < n = S - 5: We' z + be' (We' = We Wpq, be' = We bpq + be, folded in fp64 by the converter);
// rows n..n+3: register tokens; row n+4: zero (the reference's all-zero suffix token).
kernel void vae_embed(device const float *lat [[buffer(0)]], device const float *We [[buffer(1)]],
                      device const float *be [[buffer(2)]], device const float *reg [[buffer(3)]],
                      device float *X [[buffer(4)]], device const int4 *tiles [[buffer(5)]],
                      constant VaeParams &p [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  const int c4 = int(gid.x) * 4, row = int(gid.y);
  if (row >= p.M || c4 >= 2048) return;
  const int b = row / p.S, r = row - b * p.S, TS = p.TS, n = p.S - 5;
  float4 y = float4(0.0f);
  if (r < n) {
    int4 t = tiles[b];
    int d = r / (TS * TS), hh = (r / TS) % TS, ww = r % TS;
    device const float *z = lat + ((ulong(t.x + d) * p.LH + (t.y + hh)) * p.LW + (t.z + ww)) * 24;
    float4 acc = ld4(be + c4);
    for (int c = 0; c < 24; ++c) {
      float zc = z[c];
      acc += zc * float4(We[(c4 + 0) * 24 + c], We[(c4 + 1) * 24 + c], We[(c4 + 2) * 24 + c], We[(c4 + 3) * 24 + c]);
    }
    y = acc;
  } else if (r < n + 4) {
    y = ld4(reg + (r - n) * 2048 + c4);
  }
  *(device float4 *)(X + ulong(row) * 2048 + c4) = y;
}

// VideoVAE._stitch_tiles semantics: blend with the ORIGINAL upper tile over the top overlap, then with the ORIGINAL
// left tile over the left overlap, crop bottom / right overlaps; temporal cross-fade with the previous clip's overlap
// frames (VideoVAE.decode / _blend); pixel de-normalisation clip(x * std + mean, 0, 1) * 255 + 0.5 -> uint8.
// OUT: fp32 proj_out tokens [tile][S][3072] (tile = i NTX + j); channel n = c 1024 + ft 256 + py 16 + px;
// token = d TS^2 + (y / 16) TS + x / 16.
// Thread (x, y, k): k < NF0 -> output frame k = clip frame FPP + k (cross-faded with PO[k] for k < FOV if has_prev);
// k >= NF0 -> clip frame FPP + 20 + (k - NF0), stitched fp32 into NO[k - NF0] (next overlap).
// mode 1: only convert PO[k], k < FOV, to uint8 (tail after the last clip).
static inline float tval(device const float *OUT, constant VaeFinishParams &p, int ti, int f, int r, int q, int c) {
  int tok = (f >> 2) * p.TS * p.TS + (r >> 4) * p.TS + (q >> 4);
  int n = c * 1024 + (f & 3) * 256 + (r & 15) * 16 + (q & 15);
  return OUT[(ulong(ti) * p.S + tok) * 3072 + n];
}
kernel void vae_finish(device const float *OUT [[buffer(0)]], device const float *PO [[buffer(1)]],
                       device float *NO [[buffer(2)]], device uchar *U8 [[buffer(3)]],
                       device atomic_uint *bad [[buffer(4)]], constant VaeFinishParams &p [[buffer(5)]],
                       uint3 gid [[thread_position_in_grid]]) {
  // bad[0]: count of non-finite stitched values (bit test: survives fast-math)
  const int x = gid.x, y = gid.y, k = gid.z;
  if (x >= p.WP || y >= p.HP) return;
  const ulong pix = ulong(y) * p.WP + x, fsz = ulong(p.HP) * p.WP * 3;
  if (p.mode == 1) {
    if (k >= p.FOV) return;
    for (int c = 0; c < 3; ++c) {
      float v = PO[k * fsz + pix * 3 + c];
      float z = clamp(v * p.ps[c] + p.pm[c], 0.0f, 1.0f);
      U8[k * fsz + pix * 3 + c] = uchar(z * 255.0f + 0.5f);
    }
    return;
  }
  if (k >= p.NF0 + p.FOV) return;
  const int f = k < p.NF0 ? p.FPP + k : p.CF + p.FPP + (k - p.NF0);
  int i = 0, j = 0;
  const int TP = 16 * p.TS;  // tile side in pixels (256)
  while (i < p.NTY - 1 && y >= p.ys[i] + TP - p.ovy[i]) ++i;
  while (j < p.NTX - 1 && x >= p.xs[j] + TP - p.ovx[j]) ++j;
  const int r = y - p.ys[i], q = x - p.xs[j];
  for (int c = 0; c < 3; ++c) {
    float v = tval(OUT, p, i * p.NTX + j, f, r, q, c);
    if (i > 0 && r < p.ovy[i - 1]) {
      int e = p.ovy[i - 1];
      float a = tval(OUT, p, (i - 1) * p.NTX + j, f, r + TP - e, q, c);
      float w = float(r) / float(e);
      v = a * (1.0f - w) + v * w;
    }
    if (j > 0 && q < p.ovx[j - 1]) {
      int e = p.ovx[j - 1];
      float a = tval(OUT, p, i * p.NTX + j - 1, f, r, q + TP - e, c);
      float w = float(q) / float(e);
      v = a * (1.0f - w) + v * w;
    }
    if ((as_type<uint>(v) & 0x7f800000u) == 0x7f800000u) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
    if (k < p.NF0) {
      if (p.has_prev && k < p.FOV) {
        float a = PO[k * fsz + pix * 3 + c];
        float w = float(k) / float(p.FOV);
        v = a * (1.0f - w) + v * w;
      }
      float z = clamp(v * p.ps[c] + p.pm[c], 0.0f, 1.0f);
      U8[k * fsz + pix * 3 + c] = uchar(z * 255.0f + 0.5f);
    } else {
      NO[(k - p.NF0) * fsz + pix * 3 + c] = v;
    }
  }
}
