// VDN linear branch (OpenVDN vdn-minimax-h3 hybrid attention): the per-frame linear-attention state of the inner
// video frames (anchor frames 0 and F - 1 excluded), combined outside each frame's softmax window.
//   vdn_gates          W8A8 [beta 56 | softmax gate 56 | pad 16 | og_down 128] on the QKV input (MPP matmul2d i8)
//   vdn_frame_mean     frame mean of the dequantized (rotated) int8 x
//   vdn_alpha          alpha = exp(-exp(A_log) softplus(W_up (W_down' mean) + dt_bias)), fp32
//   vdn_text_features  SiLU, [L2], * sqrt(beta) of the text rows (state start)
//   vdn_conv_features  depthwise 5x5 spatial + 5-tap temporal conv, SiLU, [L2], * sqrt(beta)
//   vdn_stats          A = K~^T K~, B = V~^T K~ per (frame, head), fp16 -> fp32 (MPP matmul2d)
//   vdn_inverse        (I + sym(A))^-1 by Gauss-Jordan (SPD, no pivoting), fp32
//   vdn_scan           S_t = (S_{t-1} Diag(alpha_t) + B_t) (I + A_t)^-1, forward and reverse (simdgroup 8x8 f32)
//   vdn_gather         state outside the window, decayed to t, fp16 + power-of-two scale
//   vdn_readout        q' = L2(SiLU(q_raw)); o = q' LS^T; RMSNorm(128) * w * sigmoid(og_h W_up^T + b) -> bf16
// Layouts: raw q / k / v fp16 head-major [H][rows][128]; features fp16 [H][Fp * S][128]; beta, sg fp32 [rows][H];
// og_h fp16 [rows][128]; A, B, inv, banks fp32 [Fp][H][128][128]; alpha fp32 [Fp][H][128]; LS fp16 [Fp][H][128][128]
// + fp32 scale [Fp][H]; readout bf16 [rows][ldo] at column h * 128.
#include "metal/abi/video/VdnABI.h"
#include "metal/kernels/video/common/quant.h"

using namespace h3;

// ---------------------------------------------------------------------------------------------------- gates, alpha
// [M, 256] tile grid (2, M / 64): beta[row][h] = sigmoid(v), sg[row][h] = sigmoid(v + b_sg[h]), og_h[row][j] = half(v)
kernel void vdn_gates(device int8_t *A [[buffer(0)]], device int8_t *W [[buffer(1)]], device const float *sa [[buffer(2)]],
                      device const float *sw [[buffer(3)]], device float *beta [[buffer(4)]],
                      device float *sgate [[buffer(5)]], device half *ogh [[buffer(6)]],
                      device const float *bsg [[buffer(7)]], constant BranchGateParams &p [[buffer(8)]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
  auto tA = mk<int8_t>(A, p.K, p.M);
  auto tW = mk<int8_t>(W, p.K, p.N);
  constexpr auto desc = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int m0 = tg.y * 64, n0 = tg.x * 128;
  auto a = tA.slice(0, m0);
  auto b = tW.slice(0, n0);
  auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), int>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0;
  op.run(a, b, c);
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto idx = c.get_multidimensional_index(i);
    int row = m0 + idx[1], col = n0 + idx[0];
    if (row >= p.M) continue;
    float v = float(c[i]) * sa[row] * sw[col];
    if (col < 56) beta[ulong(row) * 56 + col] = sigm(v);
    else if (col < 112) sgate[ulong(row) * 56 + col - 56] = sigm(v + bsg[col - 56]);
    else if (col >= 128) ogh[ulong(row) * 128 + col - 128] = half(v);
  }
}

// fmean[f][c] over the S rows of each inner frame
kernel void vdn_frame_mean(device const char *X [[buffer(0)]], device const float *sa [[buffer(1)]],
                           device float *fm [[buffer(2)]], constant BranchFrameParams &p [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  const int c = (tg.x * 256 + tid) * 4, f = tg.y;
  if (c >= p.K) return;
  float4 acc = 0.0f;
  const ulong r0 = ulong(p.row0) + ulong(f) * p.S;
  for (int r = 0; r < p.S; ++r) acc += float4(*(device const char4 *)(X + (r0 + r) * p.K + c)) * sa[r0 + r];
  *(device float4 *)(fm + ulong(f) * p.K + c) = acc / float(p.S);
}

// W_down' (fp32 [128][K]) carries the same Hadamard rotation as x. One threadgroup (1024) per frame.
kernel void vdn_alpha(device const float *fm [[buffer(0)]], device const float *Wd [[buffer(1)]],
                      device const float *Wu [[buffer(2)]], device const float *dtb [[buffer(3)]],
                      device const float *alog [[buffer(4)]], device float *alpha [[buffer(5)]],
                      constant BranchFrameParams &p [[buffer(6)]], uint f [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float hid[128];
  device const float *x = fm + ulong(f) * p.K;
  for (int j = sg; j < 128; j += 32) {
    float s = 0.0f;
    for (int c = lane * 4; c < p.K; c += 128) s += dot(*(device const float4 *)(x + c), *(device const float4 *)(Wd + ulong(j) * p.K + c));
    s = simd_sum(s);
    if (lane == 0) hid[j] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int C = p.H * 128;
  for (int o = tid; o < C; o += 1024) {
    float s = dtb[o];
    device const float *w = Wu + ulong(o) * 128;
    for (int j = 0; j < 128; j += 4) s += dot(*(device const float4 *)(w + j), *(threadgroup float4 *)(&hid[j]));
    float sp = s > 20.0f ? s : log(1.0f + exp(s));
    alpha[ulong(f) * C + o] = exp(-exp(alog[o >> 7]) * sp);
  }
}

// ---------------------------------------------------------------------------------------------------- features
// SiLU + [L2] + [sqrt(beta)] for nrows rows (text). raw at row0 of [H][Spad][128]; out [H][nrows][128].
kernel void vdn_text_features(device const half *raw [[buffer(0)]], device half *out [[buffer(1)]],
                              device const float *beta [[buffer(2)]], constant BranchParams &p [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort nsg [[simdgroups_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const int h = tg.y, r = int(tg.x) * nsg + sg;
  if (r >= p.nrows) return;
  float4 o = silu4(float4(*(device const half4 *)(raw + (ulong(h) * p.Spad + p.row0 + r) * 128 + 4 * lane)));
  float sc = 1.0f;
  if (p.l2) sc = 1.0f / max(sqrt(simd_sum(dot(o, o))), 1e-6f);
  if (beta) sc *= sqrt(beta[ulong(r) * 56 + h]);
  *(device half4 *)(out + (ulong(h) * p.nrows + r) * 128 + 4 * lane) = half4(o * sc);
}

// Each simdgroup computes PY vertically adjacent positions (shares 4 rows of taps); a threadgroup is TY / PY x TX
// simdgroups for a TY x TX tile of the token grid, the frames are walked with a 5-frame temporal register window.
template <int TY, int TX, int PY>
kernel void vdn_conv_features(device const half *raw [[buffer(0)]], device half *out [[buffer(1)]],
                              device const half *w_sp [[buffer(2)]], device const half *w_tm [[buffer(3)]],
                              device const float *beta [[buffer(4)]], constant BranchParams &p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int HY = TY + 4, HX = TX + 4, NT = HY * HX * 32, NTH = (TY / PY) * TX * 32;
  constexpr int NL = (NT + NTH - 1) / NTH;
  threadgroup half4 tile[HY * HX * 32];
  const int h = tg.y;
  const int ntx = (p.gw + TX - 1) / TX;
  const int y0 = int(tg.x) / ntx * TY, x0 = int(tg.x) % ntx * TX;
  const int sy = (sg / TX) * PY, sx = sg % TX;
  const int C = p.H * 128, c = h * 128 + 4 * lane;
  half4 ws[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) ws[i] = *(device const half4 *)(w_sp + i * C + c);
  float4 wt[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) wt[i] = float4(*(device const half4 *)(w_tm + i * C + c));
  device const half4 *src = (device const half4 *)(raw + (ulong(h) * p.Spad + p.row0) * 128);
  device half *dst = out + ulong(h) * ulong(p.Fp) * p.S * 128 + 4 * lane;
  int soff[NL];
#pragma unroll
  for (int j = 0; j < NL; ++j) {
    int e = tid + j * NTH;
    int tok = e >> 5, l = e & 31;
    int gy = y0 - 2 + tok / HX, gx = x0 - 2 + tok % HX;
    soff[j] = (e < NT && gy >= 0 && gy < p.gh && gx >= 0 && gx < p.gw) ? (gy * p.gw + gx) * 32 + l : -1;
  }
  half4 pf[NL];
#pragma unroll
  for (int j = 0; j < NL; ++j) pf[j] = soff[j] >= 0 ? src[soff[j]] : half4(0);
#pragma unroll
  for (int j = 0; j < NL; ++j) {
    int e = tid + j * NTH;
    if (e < NT) tile[e] = pf[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 r0[PY], r1[PY], r2[PY], r3[PY];
#pragma unroll
  for (int q = 0; q < PY; ++q) r0[q] = r1[q] = r2[q] = r3[q] = 0;
  for (int t = 0; t < p.Fp + 2; ++t) {
    if (t + 1 < p.Fp) {
      const ulong fo = ulong(t + 1) * p.S * 32;
#pragma unroll
      for (int j = 0; j < NL; ++j) pf[j] = soff[j] >= 0 ? src[fo + soff[j]] : half4(0);
    }
    float4 s[PY];
#pragma unroll
    for (int q = 0; q < PY; ++q) s[q] = 0;
    if (t < p.Fp) {
#pragma unroll
      for (int ry = 0; ry < PY + 4; ++ry)
#pragma unroll
        for (int dx = 0; dx < 5; ++dx) {
          float4 v = float4(tile[((sy + ry) * HX + sx + dx) * 32 + lane]);
#pragma unroll
          for (int q = 0; q < PY; ++q) {
            const int dy = ry - q;
            if (dy >= 0 && dy < 5) s[q] += v * float4(ws[dy * 5 + dx]);
          }
        }
    }
    const int to = t - 2;
#pragma unroll
    for (int q = 0; q < PY; ++q) {
      const int y = y0 + sy + q, x = x0 + sx;
      if (to >= 0 && y < p.gh && x < p.gw) {
        float4 o = wt[0] * r0[q] + wt[1] * r1[q] + wt[2] * r2[q] + wt[3] * r3[q] + wt[4] * s[q];
        o = silu4(o);
        float sc = 1.0f;
        if (p.l2) sc = 1.0f / max(sqrt(simd_sum(dot(o, o))), 1e-6f);
        const ulong row = ulong(to) * p.S + y * p.gw + x;
        if (beta) sc *= sqrt(beta[row * 56 + h]);
        *(device half4 *)(dst + row * 128) = half4(o * sc);
      }
      r0[q] = r1[q]; r1[q] = r2[q]; r2[q] = r3[q]; r3[q] = s[q];
    }
    if (t + 1 < p.Fp) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        int e = tid + j * NTH;
        if (e < NT) tile[e] = pf[j];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
template [[host_name("vdn_conv_features")]] [[kernel]] void vdn_conv_features<6, 6, 2>(
    device const half *, device half *, device const half *, device const half *, device const float *,
    constant BranchParams &, uint2, ushort, ushort, ushort);

// ---------------------------------------------------------------------------------------------------- statistics
// A = K~^T K~ and B = V~^T K~ per (frame, head): M = 64 (half of the 128 rows), N = 128, K = S (dynamic).
// grid: x = 4 tiles (A top, A bottom, B top, B bottom), y = head, z = frame. 4 simdgroups.
kernel void vdn_stats(device half *Kf [[buffer(0)]], device half *Vf [[buffer(1)]], device float *A [[buffer(2)]],
                      device float *B [[buffer(3)]], constant BranchParams &p [[buffer(4)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
  const int which = tg.x >> 1, half_i = tg.x & 1, h = tg.y, f = tg.z;
  const int nf = p.Fp;
  device half *kb = Kf + (ulong(h) * nf + f) * ulong(p.S) * 128;
  device half *lb = (which == 0 ? Kf : Vf) + (ulong(h) * nf + f) * ulong(p.S) * 128;
  auto tL = mk<half>(lb, 128, p.S);  // storage [S][128]; used transposed as the left operand
  auto tR = mk<half>(kb, 128, p.S);  // right operand K~ [S][128] (K = S, N = 128)
  constexpr auto desc = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent), true, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto a = tL.template slice<64, dynamic_extent>(half_i * 64, 0);
  auto b = tR.template slice<128, dynamic_extent>(0, 0);
  auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  op.run(a, b, c);
  device float *dst = (which == 0 ? A : B) + (ulong(f) * p.H + h) * 128 * 128 + half_i * 64 * 128;
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto idx = c.get_multidimensional_index(i);
    dst[idx[1] * 128 + idx[0]] = c[i];
  }
}

// (I + (A + A^T)/2)^-1 by in-place Gauss-Jordan without pivoting (I + A is SPD, pivots >= 1 in exact arithmetic).
// One threadgroup (1024 threads = 32 x 32) per matrix; thread (ty, tx) owns the 4x4 block at (4ty, 4tx).
// minpiv[m] = smallest pivot of matrix m.
kernel void vdn_inverse(device const float *A [[buffer(0)]], device float *Inv [[buffer(1)]],
                        device float *minpiv [[buffer(2)]], uint m [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float rowb[2][128], colb[2][128];
  const int ty = tid >> 5, tx = tid & 31;
  device const float *a = A + ulong(m) * 16384;
  float4 e[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int r = 4 * ty + i;
    float4 v = *(device const float4 *)(a + r * 128 + 4 * tx);
    float4 vt = float4(a[(4 * tx + 0) * 128 + r], a[(4 * tx + 1) * 128 + r], a[(4 * tx + 2) * 128 + r],
                       a[(4 * tx + 3) * 128 + r]);
    e[i] = 0.5f * (v + vt);
    if (ty == tx) e[i][i] += 1.0f;
  }
  float pmin = INFINITY;
  for (int k = 0; k < 128; ++k) {
    const int bsel = k & 1, kb = k >> 2, kr = k & 3;
    if (ty == kb) *(threadgroup float4 *)(&rowb[bsel][4 * tx]) = e[kr];
    if (tx == kb) {
#pragma unroll
      for (int i = 0; i < 4; ++i) colb[bsel][4 * ty + i] = e[i][kr];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float piv = rowb[bsel][k];
    const float ip = 1.0f / piv;
    pmin = min(pmin, piv);
    const float4 rk = *(threadgroup float4 *)(&rowb[bsel][4 * tx]) * ip;  // a_kj / p
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = 4 * ty + i;
      const float aik = colb[bsel][r];
      if (r == k) {
        e[i] = rk;
      } else {
        e[i] -= aik * rk;
      }
      if (tx == kb) e[i][kr] = (r == k) ? ip : -aik * ip;
    }
  }
  device float *o = Inv + ulong(m) * 16384;
#pragma unroll
  for (int i = 0; i < 4; ++i) *(device float4 *)(o + (4 * ty + i) * 128 + 4 * tx) = e[i];
  if (tid == 0) minpiv[m] = pmin;
}

// ---------------------------------------------------------------------------------------------------- scan, gather
// One threadgroup (1024 threads, 32 simdgroups) per (head, direction); frames sequential.
//   start: S0 = 0.5 (B_text inv_text)   (computed as one step from a zero state)
//   step : M = S_prev (.) alpha_t[col] + B_t (elementwise, scratch), S_t = M inv_t (simdgroup 8x8 f32)
// banks: pre / suf fp32 [Fp][H][128][128]; start state st[H]; scratch fp32 [2 H][128][128].
static inline void scan_mm(device const float *Mx, device const float *Iv, device float *Out, ushort sg) {
  // Out = Mx @ Iv; simdgroup sg owns rows 32 (sg >> 3) .. +31, cols 16 (sg & 7) .. +15
  const int r0 = 32 * (sg >> 3), c0 = 16 * (sg & 7);
  simdgroup_float8x8 acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int kk = 0; kk < 128; kk += 8) {
    simdgroup_float8x8 am[4], bm[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) simdgroup_load(am[i], Mx + (r0 + 8 * i) * 128 + kk, 128);
#pragma unroll
    for (int j = 0; j < 2; ++j) simdgroup_load(bm[j], Iv + kk * 128 + c0 + 8 * j, 128);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Out + (r0 + 8 * i) * 128 + c0 + 8 * j, 128);
}

kernel void vdn_scan(device const float *Inv [[buffer(0)]], device const float *B [[buffer(1)]],
                     device const float *alpha [[buffer(2)]], device float *pre [[buffer(3)]],
                     device float *suf [[buffer(4)]], device const float *inv_text [[buffer(5)]],
                     device const float *B_text [[buffer(6)]], device float *st [[buffer(7)]],
                     device float *scratch [[buffer(8)]], constant BranchParams &p [[buffer(9)]],
                     uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
  const int h = tg.x, dir = tg.y, H = p.H, Fp = p.Fp;
  device float *Mx = scratch + ulong(h * 2 + dir) * 16384;
  device float *bank = dir == 0 ? pre : suf;
  // start state: S0 = 0.5 * B_text @ inv_text  (text rows as one chunk; alpha plays no part)
  device float *S0 = st + ulong(h * 2 + dir) * 16384;
  scan_mm(B_text + ulong(h) * 16384, inv_text + ulong(h) * 16384, S0, sg);
  threadgroup_barrier(mem_flags::mem_device);
  for (int i = tid; i < 16384; i += 1024) S0[i] *= 0.5f;
  threadgroup_barrier(mem_flags::mem_device);
  device const float *prev = S0;
  for (int s = 0; s < Fp; ++s) {
    const int f = dir == 0 ? s : Fp - 1 - s;
    const ulong fh = (ulong(f) * H + h) * 16384;
    device const float *al = alpha + (ulong(f) * H + h) * 128;
    // M = prev (.) alpha[col] + B_f
    for (int i = tid * 4; i < 16384; i += 4096) {
      float4 v = *(device const float4 *)(prev + i) * *(device const float4 *)(al + (i & 127)) +
                 *(device const float4 *)(B + fh + i);
      *(device float4 *)(Mx + i) = v;
    }
    threadgroup_barrier(mem_flags::mem_device);
    scan_mm(Mx, Inv + fh, bank + fh, sg);
    threadgroup_barrier(mem_flags::mem_device);
    prev = bank + fh;
  }
}

// LS[t][h] = S_before (.) db + S_after (.) da  (decays per key channel = column),
//   S_before = pre[lo - 1] (or the start state), db = prod alpha_u, u = max(lo, 0)..t
//   S_after  = suf[hi + 1] (or the start state), da = prod alpha_u, u = t..min(hi, Fp - 1)
// (lo, hi) = rebased window of inner frame t. Output fp16 LS / sc with sc a power of two (max |LS / sc| <= 2^14)
// and sc in lsc[t][h]. One threadgroup (256 threads) per (t, h).
kernel void vdn_gather(device const float *pre [[buffer(0)]], device const float *suf [[buffer(1)]],
                       device const float *alpha [[buffer(2)]], device const float *st [[buffer(3)]],
                       device const int2 *bounds [[buffer(4)]], device half *LS [[buffer(5)]],
                       device float *lsc [[buffer(6)]], constant BranchParams &p [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float dec[2][128];
  threadgroup float red[8];
  const int t = tg.x, h = tg.y, H = p.H, Fp = p.Fp;
  const int2 bd = bounds[t];
  const int b = bd.x - 1, a = bd.y + 1;
  {
    const int k = tid & 127, side = tid >> 7;
    float lsum = 0.0f;
    int u0 = side == 0 ? max(b + 1, 0) : t, u1 = side == 0 ? t : min(a, Fp) - 1;
    for (int u = u0; u <= u1; ++u) lsum += log(max(alpha[(ulong(u) * H + h) * 128 + k], 1e-12f));
    dec[side][k] = exp(lsum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const float *Sb = b >= 0 ? pre + (ulong(b) * H + h) * 16384 : st + ulong(h * 2) * 16384;
  device const float *Sa = a < Fp ? suf + (ulong(a) * H + h) * 16384 : st + ulong(h * 2) * 16384;
  float4 v[16];
  float mx = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    int i = (tid + 256 * j) * 4;
    int col = i & 127;
    float4 db = *(threadgroup float4 *)(&dec[0][col]), da = *(threadgroup float4 *)(&dec[1][col]);
    v[j] = *(device const float4 *)(Sb + i) * db + *(device const float4 *)(Sa + i) * da;
    float4 av = fabs(v[j]);
    mx = max(mx, max(max(av.x, av.y), max(av.z, av.w)));
  }
  mx = simd_max(mx);
  if (lane == 0) red[sg] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = red[0];
  for (int i = 1; i < 8; ++i) mx = max(mx, red[i]);
  const float sc = mx > 0.0f ? exp2(ceil(log2(mx)) - 14.0f) : 1.0f;
  const float isc = 1.0f / sc;
  device half *o = LS + (ulong(t) * H + h) * 16384;
#pragma unroll
  for (int j = 0; j < 16; ++j) *(device half4 *)(o + (tid + 256 * j) * 4) = half4(v[j] * isc);
  if (tid == 0) lsc[t * H + h] = sc;
}

// ---------------------------------------------------------------------------------------------------- readout
// o = q' (LS / sc)^T and g = og_h W_up[h]^T, both MPP matmul2d 16 x 128 x 128 per simdgroup (same descriptor, so the
// two destination cooperative tensors share one slot layout); RMSNorm row sums in registers (rows spread over lanes
// {L, L^1, L^8, L^9}):
//   y = o sc rsqrt(mean((o sc)^2) + eps) * w * sigmoid(g + b) -> bf16 out[orow0 + t S + p][h 128 + c]
// ogh: fp16 [rows][128] at row nrows + t S + p; wup fp16 [H 128][128]; bup fp32 [H 128].
template <int NSG>
kernel void vdn_readout(device const half *raw [[buffer(0)]], device half *LS [[buffer(1)]],
                        device const float *lsc [[buffer(2)]], device half *ogh [[buffer(3)]],
                        device const float *nw [[buffer(4)]], device bfloat *out [[buffer(5)]],
                        constant BranchParams &p [[buffer(6)]], device half *wup [[buffer(7)]],
                        device const float *bup [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup half qbuf[NSG][16 * 128];
  const int h = tg.y, t = tg.z, H = p.H;
  const int p0 = (int(tg.x) * NSG + sg) * 16;
  if (p0 >= p.S) return;
  threadgroup half *qb = qbuf[sg];
  device const half *src = raw + (ulong(h) * p.Spad + p.row0 + ulong(t) * p.S + p0) * 128 + 4 * lane;
  for (int r = 0; r < 16; ++r) {
    float4 q = silu4(float4(*(device const half4 *)(src + r * 128)));
    q *= 1.0f / max(sqrt(simd_sum(dot(q, q))), 1e-6f);
    *(threadgroup half4 *)(qb + r * 128 + 4 * lane) = half4(q);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  using TQ = tensor<threadgroup half, extents<int, 128, 16>, tensor_inline>;
  TQ tq((typename TQ::data_handle_type)qb, extents<int, 128, 16>(), array<int, 2>{1, 128});
  auto tL = mk<half>(LS + (ulong(t) * H + h) * 16384, 128, 128);
  auto tH = mk<half>(ogh + (ulong(p.nrows) + ulong(t) * p.S + p0) * 128, 128, 16);
  auto tW = mk<half>(wup + ulong(h) * 16384, 128, 128);
  constexpr auto desc = matmul2d_descriptor(16, 128, 128, false, true, false);
  matmul2d<desc, execution_simdgroup> op;
  auto c = op.template get_destination_cooperative_tensor<decltype(tq), decltype(tL), float>();
  auto g = op.template get_destination_cooperative_tensor<decltype(tH), decltype(tW), float>();
  const ushort cap = c.get_capacity();
#pragma unroll
  for (ushort i = 0; i < cap; ++i) {
    c[i] = 0.0f;
    g[i] = 0.0f;
  }
  op.run(tq, tL, c);
  op.run(tH, tW, g);
  const int r0 = c.get_multidimensional_index(ushort(0))[1];
  float sA = 0.0f, sB = 0.0f;
#pragma unroll
  for (ushort i = 0; i < cap; ++i) {
    float v = c[i];
    if (c.get_multidimensional_index(i)[1] == r0) sA += v * v; else sB += v * v;
  }
  sA += simd_shuffle_xor(sA, 1); sA += simd_shuffle_xor(sA, 8);
  sB += simd_shuffle_xor(sB, 1); sB += simd_shuffle_xor(sB, 8);
  const float sc = lsc[t * H + h];
  const float fA = sc * rsqrt(sA * (sc * sc / 128.0f) + p.eps), fB = sc * rsqrt(sB * (sc * sc / 128.0f) + p.eps);
  device const float *bh = bup + h * 128;
#pragma unroll
  for (ushort i0 = 0; i0 < cap; i0 += 4) {
    auto a0 = c.get_multidimensional_index(i0);
    auto a3 = c.get_multidimensional_index(ushort(i0 + 3));
    const int row = a0[1], col = a0[0];
    const float f = row == r0 ? fA : fB;
    if (a3[1] == row && a3[0] == col + 3 && (col & 3) == 0) {
      float4 o = float4(c[i0], c[i0 + 1], c[i0 + 2], c[i0 + 3]) * f * *(device const float4 *)(nw + col);
      float4 gg = float4(g[i0], g[i0 + 1], g[i0 + 2], g[i0 + 3]) + *(device const float4 *)(bh + col);
      o *= 1.0f / (1.0f + exp(-gg));
      *(device vec<bfloat, 4> *)(out + (ulong(p.orow0) + ulong(t) * p.S + p0 + row) * p.ldo + h * 128 + col) =
          vec<bfloat, 4>(o);
    } else {
      for (ushort i = i0; i < i0 + 4; ++i) {
        auto ix = c.get_multidimensional_index(i);
        float fr = ix[1] == r0 ? fA : fB;
        float o = c[i] * fr * nw[ix[0]] * sigm(g[i] + bh[ix[0]]);
        out[(ulong(p.orow0) + ulong(t) * p.S + p0 + ix[1]) * p.ldo + h * 128 + ix[0]] = bfloat(o);
      }
    }
  }
}
template [[host_name("vdn_readout")]] [[kernel]] void vdn_readout<2>(
    device const half *, device half *, device const float *, device half *, device const float *, device bfloat *,
    constant BranchParams &, device half *, device const float *, uint3, ushort, ushort);
