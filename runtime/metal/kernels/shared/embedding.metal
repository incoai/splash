#include "metal/abi/Gguf.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gguf_embedding_formats.h"

// The token gather from native GGUF rows (ops/Embedding.cpp) of a format of
// common/gguf_embedding_formats.h. Inlined, with the format's value, so every
// gather stays one function (the compiler otherwise keeps Q6_K's as a call).
template <class F>
__attribute__((always_inline)) inline void gguf_embedding(device const uint *tokens, device const uchar *table,
                                                         device bfloat *output, constant GgufEmbedParams &p,
                                                         uint index) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *block = table + (ulong(token) * (p.hidden / F::Weights) + dim / F::Weights) * F::Bytes;
  output[index] = F::value(block, dim);
}

#define GGUF_EMBEDDING_ENTRY(Name, F) \
  kernel void Name(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], \
                   device bfloat *output [[buffer(2)]], constant GgufEmbedParams &p [[buffer(3)]], \
                   uint index [[thread_position_in_grid]]) { \
    gguf_embedding<F>(tokens, table, output, p, index); \
  }
// One gather per gguf_embedding_format (metal/abi/Gguf.h), named by its kQuantFormats token.
GGUF_EMBEDDING_ENTRY(gguf_embed_q4k, GgufEmbedQ4K)
GGUF_EMBEDDING_ENTRY(gguf_embed_q6k, GgufEmbedQ6K)
GGUF_EMBEDDING_ENTRY(gguf_embed_q80, GgufEmbedQ80)
GGUF_EMBEDDING_ENTRY(gguf_embed_q5k, GgufEmbedQ5K)
GGUF_EMBEDDING_ENTRY(gguf_embed_q3k, GgufEmbedQ3K)
GGUF_EMBEDDING_ENTRY(gguf_embed_q2k, GgufEmbedQ2K)
GGUF_EMBEDDING_ENTRY(gguf_embed_q40, GgufEmbedQ40)
GGUF_EMBEDDING_ENTRY(gguf_embed_q41, GgufEmbedQ41)
GGUF_EMBEDDING_ENTRY(gguf_embed_pq20, GgufEmbedPQ20)
GGUF_EMBEDDING_ENTRY(gguf_embed_iq4nl, GgufEmbedIQ4NL)
GGUF_EMBEDDING_ENTRY(gguf_embed_iq4xs, GgufEmbedIQ4XS)
GGUF_EMBEDDING_ENTRY(gguf_embed_iq3s, GgufEmbedIQ3S)
GGUF_EMBEDDING_ENTRY(gguf_embed_mxfp4, GgufEmbedMXFP4)
#define GGUF_EMBEDDING_AFFINE(B)                                                                \
  GGUF_EMBEDDING_ENTRY(gguf_embed_af##B##g32, GgufEmbedAF##B##G32)                              \
  GGUF_EMBEDDING_ENTRY(gguf_embed_af##B##g64, GgufEmbedAF##B##G64)                              \
  GGUF_EMBEDDING_ENTRY(gguf_embed_af##B##g128, GgufEmbedAF##B##G128)
GGUF_EMBEDDING_AFFINE(2) GGUF_EMBEDDING_AFFINE(3) GGUF_EMBEDDING_AFFINE(4) GGUF_EMBEDDING_AFFINE(5)
GGUF_EMBEDDING_AFFINE(6) GGUF_EMBEDDING_AFFINE(8)
#undef GGUF_EMBEDDING_AFFINE
#undef GGUF_EMBEDDING_ENTRY

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
