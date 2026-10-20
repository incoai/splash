#include "video/h3/ops/PrepOps.hpp"

#include "metal/abi/video/PrepABI.h"

#include <algorithm>

namespace h3::ops {
namespace {

MTLSize grid1(int n) { return MTLSizeMake((n + 255) / 256, 1, 1); }

} // namespace

std::vector<std::string> PrepOps::kernels() {
  std::vector<std::string> k{"prep_gemm_f32_bias", "prep_gemm_bf16_bias", "prep_gemm_bf16",    "prep_rms_looped",
                             "prep_rms_row",       "prep_attention",      "prep_silu_f32",     "prep_swiglu_bf16",
                             "prep_add_bf16",      "prep_cast_bf16",      "prep_rope_inv_freq", "prep_rope_trig",
                             "prep_temb_sinusoid", "prep_normal"};
  for (const char *nk : {"2_32", "3_32", "4_32", "5_32", "3_16", "4_16", "5_16"}) {
    k.push_back(std::string("prep_gemv_wide_") + nk);
    k.push_back(std::string("prep_gemv_wide_") + nk + "_bias");
  }
  return k;
}

void PrepOps::linear(id<MTLComputeCommandEncoder> e, bool f32, View x, View w, View bias, View y, int M, int N, int K) {
  const bool hasBias = static_cast<bool>(bias);
  if (!f32 && M >= 2 && M <= 15 && K % 4 == 0) {
    // MLX gemv_wide_config: passes = ceil(M / 5) <= 3, vectors per threadgroup ceil(M / passes), 32 K-lanes when one
    // pass (or N <= 64), else 16; one grid column for N >= 65536, else `passes`.
    const int passes = (M + 4) / 5, nv = (M + passes - 1) / passes;
    const int kl = (passes == 1 || N <= 64) ? 32 : 16, gridX = N >= 65536 ? 1 : passes;
    const PrepGemvParams p{K, N, M, K, K, 1.0f, 1.0f};
    [e setComputePipelineState:b_.pipeline("prep_gemv_wide_" + std::to_string(nv) + "_" + std::to_string(kl) +
                                           (hasBias ? "_bias" : ""))];
    metal::bind(e, w, 0);
    metal::bind(e, x, 1);
    metal::bind(e, hasBias ? bias : w, 2);
    metal::bind(e, y, 3);
    metal::bindBytes(e, p, 4);
    metal::dispatch(e, MTLSizeMake(gridX, (N + 3) / 4, 1), MTLSizeMake(32, kl / 8, 1));
    return;
  }
  const PrepGemmParams p{M, N, K, K, K, N};
  const char *name = f32 ? "prep_gemm_f32_bias" : (hasBias ? "prep_gemm_bf16_bias" : "prep_gemm_bf16");
  [e setComputePipelineState:b_.pipeline(name)];
  metal::bind(e, x, 0);
  metal::bind(e, w, 1);
  metal::bind(e, hasBias ? bias : w, 2);
  metal::bind(e, y, 3);
  metal::bindBytes(e, p, 4);
  metal::dispatch(e, MTLSizeMake(N / 32, (M + 15) / 16, 1), MTLSizeMake(32, 1, 1));
}

void PrepOps::rms(id<MTLComputeCommandEncoder> e, View x, View w, View y, int rows, int axis, int inLd, int outLd,
                  float eps) {
  const PrepRmsParams p{eps, static_cast<uint32_t>(axis), inLd, outLd};
  const bool looped = axis > 4096;  // RMS_LOOPED_LIMIT
  [e setComputePipelineState:b_.pipeline(looped ? "prep_rms_looped" : "prep_rms_row")];
  metal::bind(e, x, 0);
  metal::bind(e, w, 1);
  metal::bind(e, y, 2);
  metal::bindBytes(e, p, 3);
  // looped: MLX dispatches the kernel's maximum threadgroup size (1024); single row: 32 x ceil(ceil(axis / 4) / 32)
  const int threads = looped ? 1024 : 32 * (((axis + 3) / 4 + 31) / 32);
  metal::dispatch(e, MTLSizeMake(rows, 1, 1), MTLSizeMake(threads, 1, 1));
}

void PrepOps::attention(id<MTLComputeCommandEncoder> e, View q, View k, View v, View o, int L, int heads, float scale,
                        const AttnStrides &s) {
  constexpr int bq = 64, bk = 32;
  PrepAttnParams p{};
  p.qL = p.kL = L;
  p.NQ = (L + bq - 1) / bq;
  p.NK = (L + bk - 1) / bk;
  p.NQ_aligned = L / bq;
  p.NK_aligned = L / bk;
  p.qL_rem = L - p.NQ_aligned * bq;
  p.kL_rem = L - p.NK_aligned * bk;
  p.scale = scale;
  p.q_h = s.qH;
  p.q_l = s.qL;
  p.k_h = s.kH;
  p.k_l = s.kL;
  p.v_h = s.vH;
  p.v_l = s.vL;
  p.o_h = s.oH;
  p.o_l = s.oL;
  p.align_q = L % bq == 0;
  p.align_k = L % bk == 0;
  [e setComputePipelineState:b_.pipeline("prep_attention")];
  metal::bind(e, q, 0);
  metal::bind(e, k, 1);
  metal::bind(e, v, 2);
  metal::bind(e, o, 3);
  metal::bindBytes(e, p, 4);
  metal::dispatch(e, MTLSizeMake(p.NQ, heads, 1), MTLSizeMake(32, 4, 1));
}

void PrepOps::siluF32(id<MTLComputeCommandEncoder> e, View x, View y, int n) {
  const PrepElemParams p{n, 0, 0, 0};
  [e setComputePipelineState:b_.pipeline("prep_silu_f32")];
  metal::bind(e, x, 0);
  metal::bind(e, y, 1);
  metal::bindBytes(e, p, 2);
  metal::dispatch(e, grid1(n), MTLSizeMake(256, 1, 1));
}

void PrepOps::swiglu(id<MTLComputeCommandEncoder> e, View fused, View y, int rows, int ffn) {
  const PrepElemParams p{rows * ffn, ffn, 0, 0};
  [e setComputePipelineState:b_.pipeline("prep_swiglu_bf16")];
  metal::bind(e, fused, 0);
  metal::bind(e, y, 1);
  metal::bindBytes(e, p, 2);
  metal::dispatch(e, MTLSizeMake((ffn + 255) / 256, rows, 1), MTLSizeMake(256, 1, 1));
}

void PrepOps::add(id<MTLComputeCommandEncoder> e, View a, View b, View y, int n) {
  const PrepElemParams p{n, 0, 0, 0};
  [e setComputePipelineState:b_.pipeline("prep_add_bf16")];
  metal::bind(e, a, 0);
  metal::bind(e, b, 1);
  metal::bind(e, y, 2);
  metal::bindBytes(e, p, 3);
  metal::dispatch(e, grid1(n), MTLSizeMake(256, 1, 1));
}

void PrepOps::castBf16(id<MTLComputeCommandEncoder> e, View x, View y, int n) {
  const PrepElemParams p{n, 0, 0, 0};
  [e setComputePipelineState:b_.pipeline("prep_cast_bf16")];
  metal::bind(e, x, 0);
  metal::bind(e, y, 1);
  metal::bindBytes(e, p, 2);
  metal::dispatch(e, grid1(n), MTLSizeMake(256, 1, 1));
}

void PrepOps::ropeInvFreq(id<MTLComputeCommandEncoder> e, View inv) {
  [e setComputePipelineState:b_.pipeline("prep_rope_inv_freq")];
  metal::bind(e, inv, 0);
  metal::dispatch(e, MTLSizeMake(1, 1, 1), MTLSizeMake(16, 1, 1));
}

void PrepOps::ropeTrig(id<MTLComputeCommandEncoder> e, View pos, View inv, View cos, View sin, int rows) {
  const PrepRopeParams p{rows, 0, 0, 0};
  [e setComputePipelineState:b_.pipeline("prep_rope_trig")];
  metal::bind(e, pos, 0);
  metal::bind(e, inv, 1);
  metal::bind(e, cos, 2);
  metal::bind(e, sin, 3);
  metal::bindBytes(e, p, 4);
  metal::dispatch(e, MTLSizeMake(1, (rows + 3) / 4, 1), MTLSizeMake(64, 4, 1));
}

void PrepOps::tembSinusoid(id<MTLComputeCommandEncoder> e, View t, View y, int T) {
  const PrepTembParams p{T, 0, 0, 0};
  [e setComputePipelineState:b_.pipeline("prep_temb_sinusoid")];
  metal::bind(e, t, 0);
  metal::bind(e, y, 1);
  metal::bindBytes(e, p, 2);
  metal::dispatch(e, MTLSizeMake(1, T, 1), MTLSizeMake(128, 1, 1));
}

void PrepOps::normal(id<MTLComputeCommandEncoder> e, View bits, View y, int n) {
  const float lo = -0x1.fffffep-1f;  // nextafter(-1, 0)
  const float hi = 1.0f;
  const PrepNormalParams p{n, 0, lo, hi - lo};
  [e setComputePipelineState:b_.pipeline("prep_normal")];
  metal::bind(e, bits, 0);
  metal::bind(e, y, 1);
  metal::bindBytes(e, p, 2);
  metal::dispatch(e, grid1(n), MTLSizeMake(256, 1, 1));
}

} // namespace h3::ops
