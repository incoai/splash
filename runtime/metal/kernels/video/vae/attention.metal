// Video VAE attention: dense non-causal fp16 attention per tile-clip, head dim 64. One simdgroup owns 16 query rows;
// Q held in a left-input cooperative tensor (loaded once), register-slot formulas (common/window.h), O / l rescaled
// only when a row max increases, per-lane partial row sums reduced once at the end, keys >= S masked on the last
// tile; P = fp32 exp2 rounded to fp16. Q / K / V fp16 [tile][32][Spad][64] -> O fp16 [tile S + row][ldo] at h 64.
#include "metal/abi/video/VaeABI.h"
#include "metal/kernels/video/common/window.h"

using namespace h3;

template <int NSG>
kernel void vae_attention(device half *Qb [[buffer(0)]], device half *Kb [[buffer(1)]], device half *Vb [[buffer(2)]],
                          device half *O [[buffer(3)]], constant VaeParams &p [[buffer(4)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 16, BK = 32, D = 64, SCAP = 16, OCAP = 32;
  const int h = tg.y, b = tg.z, q0 = (int(tg.x) * NSG + sg) * BQ;
  if (q0 >= p.Spad) return;
  const ulong hs = (ulong(b) * p.H + h) * ulong(p.Spad) * D;
  auto Q = mk<half>(Qb + hs, D, p.Spad);
  auto K = mk<half>(Kb + hs, D, p.Spad);
  auto V = mk<half>(Vb + hs, D, p.Spad);
  auto mQ = Q.template slice<D, BQ>(0, q0);
  constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto qL = opqk.template get_left_input_cooperative_tensor<half, half, float>();
  qL.load(mQ);
  auto pL = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V0 = V.template slice<D, BK>(0, 0);
  auto oT = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL)>, decltype(V0), float>();
#pragma unroll
  for (ushort i = 0; i < OCAP; ++i) oT[i] = 0.0f;
  const ushort r0 = lane_r0(lane), cb = lane_cb(lane);
  float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;
  const float ks = p.qk_scale_log2;
  const int ntiles = (p.S + BK - 1) / BK;
  for (int kt = 0; kt < ntiles; ++kt) {
    const int k0 = kt * BK, lim = p.S - k0;
    auto mK = K.template slice<D, BK>(0, k0);
    auto sT = opqk.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    opqk.run(qL, mK, sT);
    float s[SCAP];
#pragma unroll
    for (ushort i = 0; i < SCAP; ++i) s[i] = float(sT[i]) * ks;
    if (lim < BK) {
#pragma unroll
      for (ushort i = 0; i < SCAP; ++i)
        if (int(16 * (i >> 3) + cb + (i & 3)) >= lim) s[i] = -INFINITY;
    }
    float xA = s[0], xB = s[4];
#pragma unroll
    for (ushort i = 1; i < SCAP; ++i) {
      if (((i >> 2) & 1) == 0) xA = max(xA, s[i]);
      else xB = max(xB, s[i]);
    }
    xA = max(xA, simd_shuffle_xor(xA, ushort(1)));
    xA = max(xA, simd_shuffle_xor(xA, ushort(8)));
    xB = max(xB, simd_shuffle_xor(xB, ushort(1)));
    xB = max(xB, simd_shuffle_xor(xB, ushort(8)));
    if (xA > mA || xB > mB) {
      const float nA = max(mA, xA), nB = max(mB, xB);
      const float cA = exp2(mA - nA), cB = exp2(mB - nB);
      mA = nA;
      mB = nB;
      lA *= cA;
      lB *= cB;
#pragma unroll
      for (ushort i = 0; i < OCAP; ++i) oT[i] *= ((i >> 2) & 1) ? cB : cA;
    }
    float sumA = 0.0f, sumB = 0.0f;
#pragma unroll
    for (ushort i = 0; i < SCAP; i += 4) {
      const bool bb = (i >> 2) & 1;
      const float m = bb ? mB : mA;
      float4 s4 = float4(s[i], s[i + 1], s[i + 2], s[i + 3]) - m;
      half4 ph = half4(exp2(s4));
      if (lim < BK) {
        const int c0 = 16 * (i >> 3) + cb;
        ph = select(ph, half4(0.0h), int4(c0, c0 + 1, c0 + 2, c0 + 3) >= lim);
      }
      pL[i] = ph.x;
      pL[i + 1] = ph.y;
      pL[i + 2] = ph.z;
      pL[i + 3] = ph.w;
      const float t = float(ph.x) + float(ph.y) + float(ph.z) + float(ph.w);
      if (bb) sumB += t;
      else sumA += t;
    }
    lA += sumA;
    lB += sumB;
    auto mV = V.template slice<D, BK>(0, k0);
    oppv.run(pL, mV, oT);
  }
  lA += simd_shuffle_xor(lA, ushort(1));
  lA += simd_shuffle_xor(lA, ushort(8));
  lB += simd_shuffle_xor(lB, ushort(1));
  lB += simd_shuffle_xor(lB, ushort(8));
  const float iA = 1.0f / lA, iB = 1.0f / lB;
  device half *Oh = O + ulong(b) * p.S * p.ldo + h * D;
#pragma unroll
  for (ushort i = 0; i < OCAP; i += 4) {
    const bool bb = (i >> 2) & 1;
    const int row = q0 + r0 + (bb ? 8 : 0);
    const int col = 16 * (i >> 3) + cb;
    if (row >= p.S) continue;
    float4 v = float4(oT[i], oT[i + 1], oT[i + 2], oT[i + 3]) * (bb ? iB : iA);
    *(device half4 *)(Oh + ulong(row) * p.ldo + col) = half4(v);
  }
}
template [[host_name("vae_attention")]] [[kernel]] void vae_attention<H3_VAE_ATTENTION_NSG>(
    device half *, device half *, device half *, device half *, constant VaeParams &, uint3, ushort, ushort);
