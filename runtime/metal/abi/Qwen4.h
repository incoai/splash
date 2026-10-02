#pragma once

// Parameters of Qwen3.8-Flash-Next's own kernels (kernels/shared/qwen4.metal,
// ops/Qwen4.cpp): its hyper-connections and PLE n-gram embedding.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The RMS epsilon of every Flash-Next norm (attention.layer_norm_rms_epsilon).
#define QWEN4_RMS_EPSILON 1e-6f
// Threads of the per-row kernels: one threadgroup per row.
#define QWEN4_ROW_THREADS 256u
// The most streams a mix reads.
#define QWEN4_MAXIMUM_STREAMS 4u

// A mix over `streams` streams of `hidden` values per row: its norm and
// injection (hyper_norm), its low-rank gate (hyper_low, `rank` values) and
// its average (hyper_mix); or the injection of a block's output (combine) and
// the streams' start (init).
struct Qwen4HyperParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t rank;
  uint32_t inject; // hyper_norm: whether to compute injection weights
};

// The PLE gate (ple_gate) and dilated convolution (ple_conv) of `rows` rows
// of one sequence: `taps` taps `dilation` rows apart, whose history is the
// (taps - 1) * dilation rows before the first. ple_conv writes the history
// after its rows when write_history is set; ple_commit writes, per lane, the
// history after a verify step's retained rows.
struct Qwen4PleParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t taps;
  uint32_t dilation;
  uint32_t write_history;
  uint32_t lanes;      // ple_commit
  uint32_t lane_rows;  // ple_commit: rows of a lane's step
};

// QSA (Qwen Sparse Attention) of a full-attention layer: the indexer's
// `heads` query heads and one key head of `dimension` values; blocks of
// `block_tokens` tokens; `top_blocks` blocks per query row past which the
// row attends sparsely. Each KV page of `page_tokens` tokens also holds, per
// layer, the raw indexer keys (bf16 [page_tokens][dimension]) and the pooled
// keys of its blocks (fp32 [page_tokens / block_tokens][dimension]) in its
// extent's index regions (ops/PageStorage indexPlacement), which kernels
// reach through the page's table entry and the regions' byte offsets.
#define QWEN4_QSA_DIMENSION 128u
#define QWEN4_QSA_HEADS 4u
#define QWEN4_QSA_BLOCK 4u
#define QWEN4_QSA_SELECT_THREADS 1024u
struct Qwen4QsaParams {
  uint32_t rows;          // query rows of the dispatch
  uint32_t first_position; // logical position of row 0 (one sequence or lane)
  uint32_t top_blocks;
  uint32_t mask_words;    // 32-bit words of a row's block bitmap
  uint32_t score_stride;  // scores per row of the score scratch
  uint32_t blocks;        // pool: blocks to pool
  uint32_t page_entries;  // entries of the page table
  uint32_t row_offset;    // first row of the dispatch in the bitmap and score rows
  uint32_t keys_offset;   // the layer's raw-key region in every extent, in bytes
  uint32_t pooled_offset; // the layer's pooled-key region in every extent, in bytes
};
// One block to pool: its first token's logical position and RoPE position
// (t, h, w; text repeats one), to which its pooled key is rotated.
struct Qwen4QsaBlock {
  uint32_t position;
  uint32_t rope[3];
};

// The QSA bitmaps the GQA-12 attention kernels read: words per row (0 for
// none: every row attends causally) and the bitmap row of the dispatch's
// first query row.
struct Qwen4QsaMaskParams {
  uint32_t mask_words;
  uint32_t row0;
};

// The MTP head (model/Qwen4Exp.hpp Qwen4MtpWeights) over a lane's verify
// rows. Row r of a lane pairs the residual streams of position r - 1 (row 0:
// the lane's carried streams, else history row r - 1) with the embedding of
// its token; rows from `limit` on repeat row limit - 1. qwen4_mtp_tokens
// builds a draft step's tokens: row r the anchor (r = 0) or proposal r - 1.
// qwen4_mtp_argmax writes the argmax of each lane's row `row` as its
// proposal `row` (and, on the last step, every later one) and its first
// sparse candidate. qwen4_mtp_carry keeps one row of streams: row
// retained - 1 (a GPU count) or, without one, row limit - 1.
struct Qwen4MtpParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t lane_rows;
  uint32_t limit;
  uint32_t vocabulary;
  uint32_t proposals;  // proposals per lane
  uint32_t candidates; // sparse candidates per proposal
  uint32_t row;        // argmax
  uint32_t last;       // argmax: whether to fill the later proposals
  uint32_t counted;    // carry: whether `retained` is bound
  uint32_t pad;
};

// The PLE rows (llama.cpp qwen4exp build_ple hash) of each lane's verify
// rows on the GPU, from the rows' tokens and the two tokens before a lane's
// first row (QWEN4_PLE_NONE for none).
#define QWEN4_PLE_NONE 0xffffffffu
#define QWEN4_PLE_MAXIMUM_HEADS 16u
struct Qwen4PleHashParams {
  uint32_t rows;
  uint32_t lane_rows;
  uint32_t ngram;
  uint32_t heads_per_ngram;
  uint32_t eos;
  uint32_t pad[3];
  uint64_t multipliers[4];
  uint32_t vocabularies[QWEN4_PLE_MAXIMUM_HEADS];
  uint32_t offsets[QWEN4_PLE_MAXIMUM_HEADS];
};
