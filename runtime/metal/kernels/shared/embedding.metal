#include "metal/abi/Gguf.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gguf_embedding_formats.h"

// The gather of rows of native blocks (ops/Embedding.cpp) of a format of common/gguf_embedding_formats.h, each
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
// Two gathers per gguf_embedding_format (metal/abi/Gguf.h), named by its kQuantFormats token: gguf_embed_<token>
// into bf16 and gguf_embed_f32_<token> into fp32.
#define GGUF_EMBEDDING_FORMAT(f, F) GGUF_GATHER(gguf_embed_##f, F, bfloat) GGUF_GATHER(gguf_embed_f32_##f, F, float)
GGUF_EMBEDDING_FORMAT(q4k, GgufEmbedQ4K)
GGUF_EMBEDDING_FORMAT(q6k, GgufEmbedQ6K)
GGUF_EMBEDDING_FORMAT(q80, GgufEmbedQ80)
GGUF_EMBEDDING_FORMAT(q5k, GgufEmbedQ5K)
GGUF_EMBEDDING_FORMAT(q3k, GgufEmbedQ3K)
GGUF_EMBEDDING_FORMAT(q2k, GgufEmbedQ2K)
GGUF_EMBEDDING_FORMAT(q40, GgufEmbedQ40)
GGUF_EMBEDDING_FORMAT(q41, GgufEmbedQ41)
GGUF_EMBEDDING_FORMAT(q50, GgufEmbedQ50)
GGUF_EMBEDDING_FORMAT(q51, GgufEmbedQ51)
GGUF_EMBEDDING_FORMAT(pq20, GgufEmbedPQ20)
GGUF_EMBEDDING_FORMAT(iq4nl, GgufEmbedIQ4NL)
GGUF_EMBEDDING_FORMAT(iq4xs, GgufEmbedIQ4XS)
GGUF_EMBEDDING_FORMAT(iq3s, GgufEmbedIQ3S)
GGUF_EMBEDDING_FORMAT(iq3xxs, GgufEmbedIQ3XXS)
GGUF_EMBEDDING_FORMAT(iq2xxs, GgufEmbedIQ2XXS)
GGUF_EMBEDDING_FORMAT(iq2xs, GgufEmbedIQ2XS)
GGUF_EMBEDDING_FORMAT(iq2s, GgufEmbedIQ2S)
GGUF_EMBEDDING_FORMAT(iq1s, GgufEmbedIQ1S)
GGUF_EMBEDDING_FORMAT(iq1m, GgufEmbedIQ1M)
GGUF_EMBEDDING_FORMAT(mxfp4, GgufEmbedMXFP4)
GGUF_EMBEDDING_FORMAT(nvfp4, GgufEmbedNVFP4)
GGUF_EMBEDDING_FORMAT(fp8, GgufEmbedFP8)
GGUF_EMBEDDING_FORMAT(bf16, GgufEmbedBF16)
GGUF_EMBEDDING_FORMAT(f16, GgufEmbedF16)
#define GGUF_EMBEDDING_AFFINE(B)                                                                \
  GGUF_EMBEDDING_FORMAT(af##B##g32, GgufEmbedAF##B##G32)                                        \
  GGUF_EMBEDDING_FORMAT(af##B##g64, GgufEmbedAF##B##G64)                                        \
  GGUF_EMBEDDING_FORMAT(af##B##g128, GgufEmbedAF##B##G128)
GGUF_EMBEDDING_AFFINE(2) GGUF_EMBEDDING_AFFINE(3) GGUF_EMBEDDING_AFFINE(4) GGUF_EMBEDDING_AFFINE(5)
GGUF_EMBEDDING_AFFINE(6) GGUF_EMBEDDING_AFFINE(8)
#undef GGUF_EMBEDDING_AFFINE
#undef GGUF_EMBEDDING_FORMAT
#undef GGUF_GATHER

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
