#include "metal/abi/Gguf.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gguf_embedding_formats.h"

// The gather of rows as stored (ops::StoredRows) of a format of common/gguf_embedding_formats.h, each
// element its decode as T: bf16, the token gathers', or fp32, the F32 values load-time widening writes
// (model/GgufPreparation.cpp). Inlined, with the format's decode, so every gather stays one function (the compiler
// otherwise keeps Q6_K's as a call).
template <bool Scaled, class F, class T>
__attribute__((always_inline)) inline void gguf_embedding(device const uint *tokens, device const uchar *table,
                                                         device T *output, constant GgufEmbedParams &p,
                                                         uint index) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *block = table + (ulong(token) * (p.hidden / F::Weights) + dim / F::Weights) * F::Bytes;
  // The unscaled gather writes the decode as it is: a multiply flushes the
  // subnormals fp32 keeps (Metal's fast math), which a bf16 table stores.
  if constexpr (Scaled) output[index] = T(F::decode(block, dim) * p.scale);
  else output[index] = T(F::decode(block, dim));
}

#define GGUF_GATHER(Name, F, T) \
  kernel void Name(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], \
                   device T *output [[buffer(2)]], constant GgufEmbedParams &p [[buffer(3)]], \
                   uint index [[thread_position_in_grid]]) { \
    gguf_embedding<false, F>(tokens, table, output, p, index); \
  } \
  kernel void Name##_scaled(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], \
                            device T *output [[buffer(2)]], constant GgufEmbedParams &p [[buffer(3)]], \
                            uint index [[thread_position_in_grid]]) { \
    gguf_embedding<true, F>(tokens, table, output, p, index); \
  }
// Two gathers per GGUF format (metal/abi/QuantFormat.h), named by its
// kQuantFormats token: gguf_embed_<token> into bf16 and
// gguf_embed_f32_<token> into fp32, and their -scaled forms, which multiply
// each gathered row by the params' scale (a family whose embeddings the
// hidden size's root scales).
#define GGUF_EMBEDDING_FORMAT(f, F) GGUF_GATHER(gguf_embed_##f, F, bfloat) GGUF_GATHER(gguf_embed_f32_##f, F, float)
GGUF_EMBEDDING_FORMATS(GGUF_EMBEDDING_FORMAT)
#undef GGUF_EMBEDDING_FORMAT
#undef GGUF_GATHER

// A verify step's input tokens, `params.rows` per lane (the served draft's
// block: the shipped drafts' 8, a 16-row draft's 16): the lane's anchor, row
// 0 of its draft input, then the draft's proposals, each clamped into the
// vocabulary.
kernel void verify_input_tokens(
    device const uint *draft_input [[buffer(0)]],
    device const uint *draft_tokens [[buffer(1)]],
    device uint *verify_input [[buffer(2)]],
    constant VerifyInputBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  uint batch = index / params.rows;
  uint row = index % params.rows;
  uint token = row == 0
                   ? draft_input[batch * params.rows]
                   : draft_tokens[batch * (params.rows - 1u) + row - 1];
  verify_input[index] = min(token, params.vocabulary - 1u);
}

// Each lane's mask rows of its draft input, rows 1 to params.rows - 1 (the
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
  rows[(ulong(group.z) * params.rows + 1 + group.y) * params.width + column] = mask[column];
}
