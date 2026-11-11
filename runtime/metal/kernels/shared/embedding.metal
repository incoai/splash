#include "metal/abi/Gguf.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gguf_embedding_formats.h"
#include "metal/kernels/common/quant_formats.h"

// The gather of rows as stored (ops::StoredRows) of a format of common/gguf_embedding_formats.h, each
// element its decode as T: bf16, the token gathers', or fp32, the F32 values load-time widening writes
// (model/GgufPreparation.cpp). Inlined, with the format's decode, so every gather stays one function (the compiler
// otherwise keeps Q6_K's as a call).
template <class F, class T>
__attribute__((always_inline)) inline void gguf_embedding(device const uint *tokens, device const uchar *table,
                                                         device T *output, constant GgufEmbedParams &p, uint index) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *block = table + (ulong(token) * (p.hidden / F::Weights) + dim / F::Weights) * F::Bytes;
  output[index] = T(F::decode(block, dim));
}

#define GGUF_GATHER(Name, F, T) \
  kernel void Name(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], \
                   device T *output [[buffer(2)]], constant GgufEmbedParams &p [[buffer(3)]], \
                   uint index [[thread_position_in_grid]]) { \
    gguf_embedding<F>(tokens, table, output, p, index); \
  }
// Two gathers per GGUF format (metal/abi/QuantFormat.h), named by its kQuantFormats token: gguf_embed_<token>
// into bf16 and gguf_embed_f32_<token> into fp32.
#define GGUF_EMBEDDING_FORMAT(f, F) GGUF_GATHER(gguf_embed_##f, F, bfloat) GGUF_GATHER(gguf_embed_f32_##f, F, float)
GGUF_EMBEDDING_FORMATS(GGUF_EMBEDDING_FORMAT)
#undef GGUF_EMBEDDING_FORMAT
#undef GGUF_GATHER

// The gather of a tied head's token rows from its projection's planes (ops::Embedding::gathersPlanes): one thread
// per (row, group of 32, chunk), each value decoded as the gather of its native rows decodes it (common/
// gguf_embedding_formats.h), so both write the same bf16.
template <class F>
__attribute__((always_inline)) inline void gguf_embedding_planes(device const uint *tokens, device uchar *plane0,
                                                                device uchar *plane1, device uchar *meta,
                                                                device bfloat *output, constant GgufEmbedParams &p,
                                                                uint index) {
#pragma clang fp reassociate(off)
  const uint groups = p.hidden / 32;
  if (index >= p.rows * groups * 4) return;
  const uint row = index / (groups * 4), group = index / 4 % groups;
  const ushort c = index % 4;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  const ulong at = quant_tile_index(token, group, groups);
  const typename F::Chunk q = F::loadChunk(plane0 + at * F::P0, plane1 + at * F::P1, c);
  const QuantCoef k = F::coef(F::loadMeta(meta + quant_tile_index(token, group / F::MetaGroups,
                                                                   groups / F::MetaGroups) * F::MetaBytes),
                              group % F::MetaGroups);
  // Chunk c holds pairs 0, 1 at elements 4c..4c+3 of the group and pairs 2, 3 at 16+4c..16+4c+3.
  device bfloat *out = output + ulong(row) * p.hidden + group * 32 + 4 * c;
  if constexpr (F::Kind == QuantLinear) {
    const uint4 codes = F::codes(q);
    for (ushort pair = 0; pair < 4; ++pair) {
      const float s = pair < 2 ? k.s.x : k.s.y, m = pair < 2 ? k.m.x : k.m.y;
      for (ushort i = 0; i < 2; ++i) {
        const uint code = (codes[pair] >> (16 * i)) & 0xFFFF;
        const float value = F::Zero ? s * float(int(code) - int(F::Zero)) : fma(float(code), s, m);
        out[16 * (pair >> 1) + 2 * (pair & 1) + i] = bfloat(value);
      }
    }
  } else {
    static_assert(F::Kind == QuantInt8, "a tied head's planes gather linear codes or int8 values");
    const char4 lo = as_type<char4>(F::values(q).x), hi = as_type<char4>(F::values(q).y);
    for (ushort i = 0; i < 4; ++i) {
      out[i] = bfloat(k.s.x * float(lo[i]));
      out[16 + i] = bfloat(k.s.y * float(hi[i]));
    }
  }
}

#define GGUF_PLANES_GATHER(F, f)                                                                                      \
  kernel void gguf_embed_planes_##f(device const uint *tokens [[buffer(0)]], device uchar *plane0 [[buffer(1)]],    \
                                    device uchar *plane1 [[buffer(2)]], device uchar *meta [[buffer(3)]],            \
                                    device bfloat *output [[buffer(4)]], constant GgufEmbedParams &p [[buffer(5)]], \
                                    uint index [[thread_position_in_grid]]) {                                       \
    gguf_embedding_planes<F>(tokens, plane0, plane1, meta, output, p, index);                                       \
  }
// The formats ops::Embedding::gathersPlanes names.
#define GGUF_PLANES_AFFINE(B) GGUF_PLANES_GATHER(FmtAF##B##G32, af##B##g32) GGUF_PLANES_GATHER(FmtAF##B##G64, af##B##g64) \
  GGUF_PLANES_GATHER(FmtAF##B##G128, af##B##g128)
GGUF_PLANES_GATHER(FmtQ4K, q4k) GGUF_PLANES_GATHER(FmtQ5K, q5k) GGUF_PLANES_GATHER(FmtQ6K, q6k)
GGUF_PLANES_GATHER(FmtQ80, q80)
GGUF_PLANES_AFFINE(2) GGUF_PLANES_AFFINE(3) GGUF_PLANES_AFFINE(4) GGUF_PLANES_AFFINE(5) GGUF_PLANES_AFFINE(6)
GGUF_PLANES_AFFINE(8)
#undef GGUF_PLANES_AFFINE
#undef GGUF_PLANES_GATHER

// A verify step's input tokens, SPLASH_TARGET_VERIFY_ROWS per lane: the
// lane's anchor, row 0 of its draft input, then the draft's proposals, each
// clamped into the vocabulary.
kernel void verify_input_tokens(
    device const uint *draft_input [[buffer(0)]],
    device const uint *draft_tokens [[buffer(1)]],
    device uint *verify_input [[buffer(2)]],
    constant VerifyInputBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  uint batch = index / SPLASH_TARGET_VERIFY_ROWS;
  uint row = index % SPLASH_TARGET_VERIFY_ROWS;
  uint token = row == 0
                   ? draft_input[batch * SPLASH_TARGET_VERIFY_ROWS]
                   : draft_tokens[batch * SPLASH_DRAFT_PROPOSAL_TOKENS +
                                  row - 1];
  verify_input[index] = min(token, params.vocabulary - 1u);
}

// Each lane's mask rows of its draft input, rows 1 to SPLASH_TARGET_VERIFY_ROWS - 1 (the
// rows after its anchor), set to the mask embedding the draft's checkpoint ships
// (ops/Embedding.cpp addDraftMaskRows): x a span of 256 columns, y the mask row, z the lane.
kernel void draft_mask_rows(
    device const bfloat *mask [[buffer(0)]],
    device bfloat *rows [[buffer(1)]],
    constant DraftMaskRowsParams &params [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const uint column = group.x * 256 + thread_index;
  if (column >= params.width)
    return;
  rows[(ulong(group.z) * SPLASH_TARGET_VERIFY_ROWS + 1 + group.y) * params.width + column] = mask[column];
}
