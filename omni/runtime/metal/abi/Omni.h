#pragma once

// Parameter layouts shared by the Omni (TTS/ASR) kernels and their host
// dispatch code (ops/Omni*.hpp). Every struct is plain data with fixed-width
// scalars; a static_assert pins its size on both sides.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The decoder attention the kernels are compiled for: Qwen3's 16 query
// heads over 8 KV heads of 128 dimensions, shared by the TTS talker, the TTS
// code predictor and the ASR text decoder.
#define OMNI_DECODER_QUERY_HEADS 16u
#define OMNI_DECODER_KV_HEADS 8u
#define OMNI_DECODER_HEAD_DIM 128u
#define OMNI_DECODER_QKV_WIDTH                                                 \
  ((OMNI_DECODER_QUERY_HEADS + 2u * OMNI_DECODER_KV_HEADS) * OMNI_DECODER_HEAD_DIM)
#define OMNI_DECODER_ATTENTION_WIDTH (OMNI_DECODER_QUERY_HEADS * OMNI_DECODER_HEAD_DIM)

// A KV page holds 32 tokens of one layer for every KV head: keys
// [kv head][token][dim] bf16, then values [kv head][dim][token] bf16.
#define OMNI_PAGE_TOKENS 32u
#define OMNI_PAGE_KEY_BYTES                                                    \
  (OMNI_DECODER_KV_HEADS * OMNI_PAGE_TOKENS * OMNI_DECODER_HEAD_DIM * 2u)
#define OMNI_PAGE_BYTES (2u * OMNI_PAGE_KEY_BYTES)

// Prefill attention tiles of 8 query rows; a tile never straddles sequences.
#define OMNI_PREFILL_TILE_ROWS 8u
#define OMNI_PREFILL_MAXIMUM_SPLITS 16u
#define OMNI_DECODE_MAXIMUM_SPLITS 32u
#define OMNI_MAXIMUM_LANES 32u
// A lone decode row over more than this many pages runs the long attention
// kernel; other decode steps run the fused kernel (ops/Omni.cpp).
#define OMNI_FUSED_MAXIMUM_PAGES 16u
// Verify forwards of at most this many rows in all (half that past 8
// pages), each sequence over at most this many pages, run the fused verify
// attention (kernels/omni/attention_verify.metal).
#define OMNI_VERIFY_MAXIMUM_ROWS 12u
#define OMNI_VERIFY_MAXIMUM_PAGES 16u

// The encoder attention of the dense kernels: heads of 64 dimensions, all
// heads keys, at most 16 heads (the ASR audio towers: 16, or 14 for
// Qwen3-ASR-0.6B's).
#define OMNI_ENCODER_HEADS 16u
#define OMNI_ENCODER_HEAD_DIM 64u
#define OMNI_DENSE_TILE_ROWS 16u
#define OMNI_DENSE_KEY_BLOCK 32u

// One sequence of an attention dispatch (prefill or decode): where its page
// table starts in the page table buffer (entries [layer * page_table_stride +
// block]), the tokens its KV already held before this dispatch's rows, its
// rows in the dispatch and their first row, and the position of its first
// row. Rows of one sequence are contiguous in the dispatch.
struct OmniSequenceParams {
  uint32_t page_table_offset;
  uint32_t page_table_stride;
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t row_offset;
  uint32_t position_offset;
  uint32_t splits;
  uint32_t padding;
};
static_assert(sizeof(OmniSequenceParams) == 32, "sequence parameters are 32 bytes on both sides");

// One prefill attention tile: its sequence, its first row in the dispatch
// and the rows it holds (1..8).
struct OmniPrefillTile {
  uint32_t sequence;
  uint32_t first_row;
  uint32_t rows;
  uint32_t padding;
};
static_assert(sizeof(OmniPrefillTile) == 16, "prefill tile parameters are 16 bytes on both sides");

// How the fused decode attention gets each row's rotations, the cosines and
// sines of its position times the 64 inverse frequencies: it computes them,
// computes and stores them in the step's rotation table (a forward's first
// layer), or loads them from it (the later layers), the same values.
#define OMNI_ROTATIONS_COMPUTE 0u
#define OMNI_ROTATIONS_STORE 1u
#define OMNI_ROTATIONS_LOAD 2u

// The QKV prepare of `rows` rows of a dispatch: which layer's pages the keys
// and values go to, the RMS epsilon of the q/k norms and, for the fused
// decode attention, its OMNI_ROTATIONS_* mode.
struct OmniQkvParams {
  uint32_t rows;
  uint32_t layer;
  uint32_t sequences;
  float rms_epsilon;
  uint32_t rotations;
};
static_assert(sizeof(OmniQkvParams) == 20, "QKV parameters are 20 bytes on both sides");

struct OmniAttentionParams {
  uint32_t layer;
  uint32_t tiles;        // prefill: tiles; decode: unused
  uint32_t sequences;
  uint32_t padded_rows;  // the dispatch's rows padded to whole tiles of 8
};
static_assert(sizeof(OmniAttentionParams) == 16, "attention parameters are 16 bytes on both sides");

// Dense (non-paged) attention over one or more sequences of an encoder: for
// every query row its sequence's key range [key_begin, key_end) plus the
// row's own index (for causal and window masks), as the kernel reads them
// from the row table. Mask kinds: 0 none (bidirectional within the range),
// 1 causal with a sliding window (key index > query index - window).
struct OmniDenseRow {
  uint32_t key_begin;
  uint32_t key_end;
  uint32_t query_index;
  uint32_t padding;
};
static_assert(sizeof(OmniDenseRow) == 16, "dense row parameters are 16 bytes on both sides");

struct OmniDenseParams {
  uint32_t rows;
  uint32_t qkv_stride;   // elements between rows of the Q, K and V buffers
  uint32_t mask;         // 0 none, 1 causal window
  uint32_t window;       // causal window (0: unbounded)
  float scale;
  uint32_t heads;        // heads of OMNI_ENCODER_HEAD_DIM, at most OMNI_ENCODER_HEADS
  uint32_t padding[2];
};
static_assert(sizeof(OmniDenseParams) == 32, "dense attention parameters are 32 bytes on both sides");

// Dense bf16 projections: output[m][n] = act(sum_k input[m][k] weight[n][k]
// + bias[n]) (+ residual[m][n]) over `rows` rows (input and output buffers
// hold whole tiles of rows). Activation: 0 none, 1 GELU (erf), 2 SiLU.
// Flags: bit 0 bias, bit 1 residual, bit 2 gate/up (a second weight stream
// `gate`, with its own bias when bit 0: out = silu(gate x + bias_g) * (up x
// + bias_u)).
#define OMNI_GEMM_FLAG_BIAS 1u
#define OMNI_GEMM_FLAG_RESIDUAL 2u
#define OMNI_GEMM_FLAG_GATE_UP 4u
struct OmniGemmParams {
  uint32_t rows;
  uint32_t output_size;
  uint32_t input_size;
  uint32_t flags;
  uint32_t activation;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t padding;
};
static_assert(sizeof(OmniGemmParams) == 32, "GEMM parameters are 32 bytes on both sides");
// Threads a threadgroup of the MMA decode tiles (omni_mma_*): four
// simdgroups, 32 output columns (64 from nine rows).
#define OMNI_MMA_THREADS 128u

// A packed bf16 weight (omni_gemv_packed_*): the bits of [N][K] bf16 split
// into planes, in one buffer, losslessly. Blocks are 64 weights along K
// (block b = n * K / 64 + k / 64); a lane's group of eight weights is
// stored in the order 0 2 1 3 4 6 5 7.
//   [0, NK)                sign << 7 | mantissa, a byte a weight
//   [NK, 3NK/2)            exponent codes, 32 bytes a block, 4 bytes a group:
//                          weights 0 2 4 6 in the low 16 bits' nibbles, then
//                          1 3 5 7; exponent = base + code
//   [3NK/2, 3NK/2 + NK/16) uint32 block headers: base exponent in bits 7:0,
//                          OMNI_PACKED_ESCAPE for an escape block, whose
//                          record index is in bits 31:8
//   then escape records    64 bytes each: the block's exponents a byte a
//                          weight, groups ordered as the sign/mantissa bytes
// A block is an escape when its exponents span more than 16 values or one
// is 255 (base = the least, so blocks of zeros and subnormals beside small
// exponents need none); its codes are unused.
#define OMNI_PACKED_ESCAPE 255u
#define OMNI_PACKED_ESCAPE_BYTES 64u

// Token gathers from a bf16 table of `width` columns.
struct OmniEmbedParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t width;
  uint32_t output_row_stride; // output row i at row i * stride (0 or 1: contiguous)
  uint32_t token_stride;      // token of row i at tokens[i * stride] (0 or 1: contiguous)
  uint32_t padding[3];
};
static_assert(sizeof(OmniEmbedParams) == 32, "embedding parameters are 32 bytes on both sides");

// The talker's input row of a decode step: the codec embedding of code 0
// plus the code predictor's embedding of each of the 15 residual codes plus
// the row's trailing text row (an index into the trailing pool).
#define OMNI_CODE_GROUPS 16u
struct OmniTtsStepParams {
  uint32_t rows;
  uint32_t width;
  uint32_t codec_vocabulary;
  uint32_t predictor_vocabulary;
};
static_assert(sizeof(OmniTtsStepParams) == 16, "TTS step parameters are 16 bytes on both sides");

// Prompt assembly: per position a text row (into the projected text rows) or
// -1, a codec id or -1, a reference frame index (into the frame codes,
// 16 per frame) or -1, and a speaker flag.
struct OmniPromptPosition {
  int32_t text_row;
  int32_t codec;
  int32_t frame;
  int32_t speaker;
};
static_assert(sizeof(OmniPromptPosition) == 16, "prompt position is 16 bytes on both sides");

// Sampling of codec codes: per row, logits [vocabulary] fp32 are
// penalized, suppressed and drawn. A row's history holds its past codes
// (history_count of them) for the repetition penalty. suppress_from marks
// the first suppressed id (ids >= suppress_from except `eos` get -inf);
// vocabulary 0 suppresses nothing. Bit r of mask_eos masks row r's EOS id
// (rows 0..31).
struct OmniSampleParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t suppress_from;
  uint32_t eos;
  uint32_t top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  uint32_t history_stride;
  uint32_t mask_eos;
  uint32_t output_stride;  // u32 elements between rows of the output (1: contiguous)
  uint32_t uniform_stride; // floats between rows of the uniforms (1: contiguous)
};
static_assert(sizeof(OmniSampleParams) == 48, "sampling parameters are 48 bytes on both sides");
#define OMNI_SAMPLE_MAXIMUM_VOCABULARY 4096u
#define OMNI_SAMPLE_MAXIMUM_TOP_K 256u

// Row r's token goes to output[r * output_stride] (0: contiguous).
struct OmniArgmaxParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t stride;
  uint32_t output_stride;
};
static_assert(sizeof(OmniArgmaxParams) == 16, "argmax parameters are 16 bytes on both sides");

// Row-wise norms with a run-time epsilon: RMSNorm (weight) and LayerNorm
// (weight and bias). Input and output element types are instance names.
struct OmniNormParams {
  uint32_t rows;
  uint32_t width;
  float epsilon;
  uint32_t padding;
};
static_assert(sizeof(OmniNormParams) == 16, "norm parameters are 16 bytes on both sides");

// Elementwise: out[i] = f(in[i]) over count elements; 1 GELU (erf), 2 SiLU.
// out[r][n] = act(in[r][n] + bias[n]) (omni_bias_activate).
struct OmniBiasParams {
  uint32_t rows;
  uint32_t width;
  uint32_t activation;
  uint32_t padding;
};
static_assert(sizeof(OmniBiasParams) == 16, "bias parameters are 16 bytes on both sides");

struct OmniElementwiseParams {
  uint32_t count;
  uint32_t activation;
};
static_assert(sizeof(OmniElementwiseParams) == 8, "elementwise parameters are 8 bytes on both sides");
