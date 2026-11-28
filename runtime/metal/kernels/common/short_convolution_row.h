#pragma once

#include "metal/abi/ShortConvolution.h"
#include "metal/kernels/common/verify_tree.h"

#include <metal_stdlib>

// A row's output of the short convolution (metal/abi/ShortConvolution.h) at
// channel c, as short_convolution and moe_combine_convolution write it: its
// static taps (and with p.heads the row's head's dynamic taps, `dynamic`)
// times its inputs k rows back, a verify block's (p.tree) along its path
// (`path`, `depth`), each read by `at` (a row's input by its place in the
// sequence, before its first row its state's), plus its own input `own`,
// over the divisor, rounded to bf16, then onto the residual where p.residual
// adds one, into the row's output (`packed`, its place in the command's rows).
// Its sums keep their order (reassociate(off)) and each tap's product and
// sum are one fma, which the compiler would otherwise contract or not by the
// kernel around it, so that every kernel that runs it, short_convolution's
// and the kernels it runs inside (moe_combine_convolution,
// block_residual_mix_convolved), writes the same bits.
template <class Out, class Residual, class At>
inline void short_convolution_output(At at, device const float *taps, device const bfloat *dynamic,
                                     device const Residual *residual, device Out *output,
                                     constant ShortConvolutionParams &p, uint c, uint packed, uint path, int depth,
                                     bfloat own) {
#pragma clang fp reassociate(off)
  constexpr uint kTaps = SHORT_CONVOLUTION_TAPS;
  float sum = 0.0f;
  // Lag k's static tap is column kTaps - 1 - k of the channel's (the stored
  // order), its dynamic tap the head's k.
  for (uint k = 0; k < kTaps; ++k) {
    const int back = depth - int(k);
    const float weight = taps[ulong(c) * kTaps + (kTaps - 1 - k)] + (p.heads ? float(dynamic[k]) : 0.0f);
    sum = fma(weight, float(at(back >= 0 && p.tree ? int(splash_path_row(path, uint(back))) : back)), sum);
  }
  const float y = float(bfloat((sum + float(own)) / p.divisor));
  const ulong i = ulong(packed) * p.channels + c;
  output[i] = Out((p.residual ? float(residual[i]) : 0.0f) + y);
}
