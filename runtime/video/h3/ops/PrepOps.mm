#include "video/h3/ops/PrepOps.hpp"

#include "metal/abi/video/PrepABI.h"

#include <algorithm>

namespace h3::ops {
namespace {

metal::DispatchSize grid1(int n) { return size3((n + 255) / 256, 1, 1); }

} // namespace

void PrepOps::linear(Encoder &e, bool f32, View x, View w, View bias, View y, int M, int N, int K) {
  const bool hasBias = static_cast<bool>(bias);
  if (!f32 && M >= 2 && M <= 15 && K % 4 == 0) {
    // MLX gemv_wide_config: passes = ceil(M / 5) <= 3, vectors per threadgroup ceil(M / passes), 32 K-lanes when one
    // pass (or N <= 64), else 16; one grid column for N >= 65536, else `passes`.
    const int passes = (M + 4) / 5, nv = (M + passes - 1) / passes;
    const int kl = (passes == 1 || N <= 64) ? 32 : 16, gridX = N >= 65536 ? 1 : passes;
    const PrepGemvParams p{K, N, M, K, K, 1.0f, 1.0f};
    e.pipeline("prep_gemv_wide_" + std::to_string(nv) + "_" + std::to_string(kl) +
                                           (hasBias ? "_bias" : ""));
    e.bind(w, 0);
    e.bind(x, 1);
    e.bind(hasBias ? bias : w, 2);
    e.bind(y, 3);
    e.bytes(p, 4);
    e.dispatch(size3(gridX, (N + 3) / 4, 1), size3(32, kl / 8, 1));
    return;
  }
  const PrepGemmParams p{M, N, K, K, K, N};
  const char *name = f32 ? "prep_gemm_f32_bias" : (hasBias ? "prep_gemm_bf16_bias" : "prep_gemm_bf16");
  e.pipeline(name);
  e.bind(x, 0);
  e.bind(w, 1);
  e.bind(hasBias ? bias : w, 2);
  e.bind(y, 3);
  e.bytes(p, 4);
  e.dispatch(size3(N / 32, (M + 15) / 16, 1), size3(32, 1, 1));
}

void PrepOps::rms(Encoder &e, View x, View w, View y, int rows, int axis, int inLd, int outLd, float eps) {
  const PrepRmsParams p{eps, static_cast<uint32_t>(axis), inLd, outLd};
  const bool looped = axis > 4096;  // RMS_LOOPED_LIMIT
  e.pipeline(looped ? "prep_rms_looped" : "prep_rms_row");
  e.bind(x, 0);
  e.bind(w, 1);
  e.bind(y, 2);
  e.bytes(p, 3);
  // looped: MLX dispatches the kernel's maximum threadgroup size (1024); single row: 32 x ceil(ceil(axis / 4) / 32)
  const int threads = looped ? 1024 : 32 * (((axis + 3) / 4 + 31) / 32);
  e.dispatch(size3(rows, 1, 1), size3(threads, 1, 1));
}

void PrepOps::attention(Encoder &e, View q, View k, View v, View o, int L, int heads, float scale,
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
  e.pipeline("prep_attention");
  e.bind(q, 0);
  e.bind(k, 1);
  e.bind(v, 2);
  e.bind(o, 3);
  e.bytes(p, 4);
  e.dispatch(size3(p.NQ, heads, 1), size3(32, 4, 1));
}

void PrepOps::siluF32(Encoder &e, View x, View y, int n) {
  const PrepElemParams p{n, 0, 0, 0};
  e.pipeline("prep_silu_f32");
  e.bind(x, 0);
  e.bind(y, 1);
  e.bytes(p, 2);
  e.dispatch(grid1(n), size3(256, 1, 1));
}

void PrepOps::swiglu(Encoder &e, View fused, View y, int rows, int ffn) {
  const PrepElemParams p{rows * ffn, ffn, 0, 0};
  e.pipeline("prep_swiglu_bf16");
  e.bind(fused, 0);
  e.bind(y, 1);
  e.bytes(p, 2);
  e.dispatch(size3((ffn + 255) / 256, rows, 1), size3(256, 1, 1));
}

void PrepOps::add(Encoder &e, View a, View b, View y, int n) {
  const PrepElemParams p{n, 0, 0, 0};
  e.pipeline("prep_add_bf16");
  e.bind(a, 0);
  e.bind(b, 1);
  e.bind(y, 2);
  e.bytes(p, 3);
  e.dispatch(grid1(n), size3(256, 1, 1));
}

void PrepOps::castBf16(Encoder &e, View x, View y, int n) {
  const PrepElemParams p{n, 0, 0, 0};
  e.pipeline("prep_cast_bf16");
  e.bind(x, 0);
  e.bind(y, 1);
  e.bytes(p, 2);
  e.dispatch(grid1(n), size3(256, 1, 1));
}

void PrepOps::ropeInvFreq(Encoder &e, View inv) {
  e.pipeline("prep_rope_inv_freq");
  e.bind(inv, 0);
  e.dispatch(size3(1, 1, 1), size3(16, 1, 1));
}

void PrepOps::ropeTrig(Encoder &e, View pos, View inv, View cos, View sin, int rows) {
  const PrepRopeParams p{rows, 0, 0, 0};
  e.pipeline("prep_rope_trig");
  e.bind(pos, 0);
  e.bind(inv, 1);
  e.bind(cos, 2);
  e.bind(sin, 3);
  e.bytes(p, 4);
  e.dispatch(size3(1, (rows + 3) / 4, 1), size3(64, 4, 1));
}

void PrepOps::tembSinusoid(Encoder &e, View t, View y, int T) {
  const PrepTembParams p{T, 0, 0, 0};
  e.pipeline("prep_temb_sinusoid");
  e.bind(t, 0);
  e.bind(y, 1);
  e.bytes(p, 2);
  e.dispatch(size3(1, T, 1), size3(128, 1, 1));
}

void PrepOps::normal(Encoder &e, View bits, View y, int n) {
  const float lo = -0x1.fffffep-1f;  // nextafter(-1, 0)
  const float hi = 1.0f;
  const PrepNormalParams p{n, 0, lo, hi - lo};
  e.pipeline("prep_normal");
  e.bind(bits, 0);
  e.bind(y, 1);
  e.bytes(p, 2);
  e.dispatch(grid1(n), size3(256, 1, 1));
}

} // namespace h3::ops
