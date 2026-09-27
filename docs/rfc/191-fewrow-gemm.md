# RFC 191 — Few-row QMM (M16 block) + wide-QMV + prompt split-K

Upstream issue: https://github.com/incoai/splash/issues/191

Source: bonsai `6acc822` (`quantized_utils`: `qmm_m16_block BM32/BN64`,
K-split pairs, idle-SG MMA skip; `qmv_wide_rr`; `Qwen35SmallNMatmul` /
`WideNMatmul b|a KC=512`).

## Problem

Splash affine Linear (`Linear.hpp:58`, `kernels/decode/linear_q4*.metal`,
`common/q4_mpp_tiles.h`, K-splits ≤8) lacks an M16 few-row block and wide
row-reuse tuned for verify-8/16 widths. Related: #168, #169, #154.

## Proposal

1. `runtime/ops/Rfc191FewRowGemm.hpp` declares the flag + env var
   (`SPLASH_FEWROW_GEMM`) and candidate geometry (M16 block, N64 columns,
   split-K ≤8).
2. Follow-up adds `M16`/N64 + wide row-reuse candidates to
   `Linear::decodeCandidates()` / `MoeTuning.cpp` (fp32 partials,
   fixed-order reduce, single bf16 rounding) plus a prompt split-K `b|a`
   candidate for prefill; `make tune-kernels` picks per device.

## Contracts

- Bit-exact per tile geometry (existing four-simdgroup rule); candidates
  additive; shipped default unchanged until gate passes.

## Kernel change (landed)

`runtime/metal/kernels/decode/linear_q4.metal` now instantiates N64
narrow-column candidates from the existing `q4_mpp_tiles.h` template
(no template changes):

- `decode_linear_q4_n64` (+`_f32`): `q4_mpp_tile<64,false,false,256>`,
  Sums=64 — M8 base over StorageN=256 slabs (4 column groups per slab).
- `decode_linear_q4_n64_m16` (+`_f32`):
  `q4_mpp_tile_batched<16,64,false,false,256>`, Sums=128.
- `decode_linear_q4_n64_m32` (+`_f32`):
  `q4_mpp_tile_batched<32,64,false,false,256>`, Sums=256.

Same fp32 accumulation order and single bf16 rounding as N128/N256, so
bit-identical per output element. Compiled into the metallib
(`xcrun metal` clean) but undispatched: no host `Linear` wiring yet,
shipped default unchanged. Pipeline names are pinned in
`Rfc191FewRowGemm.hpp` (`kN64Plain/kN64M16/kN64M32`) for the tuner
follow-up (`Linear::candidates()` + `make tune-kernels` per device).

## Validation

- `benchmark-gguf-projection` / `benchmark-gguf-moe` rows 8/16/24/32 +
  `benchmark-decode-profile`; metal fp64 bounds; ABBA.
