#pragma once

// Fused projections (qkv|z|ab, q|k|v; ops::Linear): up to three column segments of any formats in one dispatch, so
// the small segments do not run as dispatches of their own, on the staged and prefill tiles
// (kernels/shared/gguf_linear.metal) and the column tile (kernels/decode/linear_gguf_column.metal). The segments'
// planes bind from buffer 1, three each (GGUF_SEGMENTS); the dispatch's threadgroups cover their columns in order,
// `width` columns a threadgroup, and each threadgroup decodes its segment's planes and format.
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_SEGMENTS GGUF_SEGMENT(1, w0a, w1a, ma), GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc)
// The segment s of threadgroup `tile` among the segments' threadgroups of `width` columns (cols, 0 past the last) and
// the threadgroup's index `local` within it; GGUF_SEGMENT_PLANE the segment's `plane` of x##a, x##b and x##c;
// GGUF_SEGMENT_OF both, with the segment's planes w0, w1 and meta.
#define GGUF_SEGMENT_INDEX(tile, cols, width)                                                                      \
  const uint t0 = (cols)[0] / (width), t1 = t0 + (cols)[1] / (width);                                              \
  const uint s = (tile) < t0 ? 0 : (tile) < t1 ? 1 : 2, local = (tile) - (s == 0 ? 0 : s == 1 ? t0 : t1)
#define GGUF_SEGMENT_PLANE(plane, x) device uchar *plane = s == 0 ? x##a : s == 1 ? x##b : x##c
#define GGUF_SEGMENT_OF(tile, cols, width)                                                                         \
  GGUF_SEGMENT_INDEX(tile, cols, width);                                                                           \
  GGUF_SEGMENT_PLANE(w0, w0);                                                                                      \
  GGUF_SEGMENT_PLANE(w1, w1);                                                                                      \
  GGUF_SEGMENT_PLANE(meta, m)
