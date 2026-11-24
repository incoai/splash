// Token gathers of the Omni models (ops::OmniEmbedding): bf16 tables as the
// checkpoints store them, the talker's per-step input (its codec embedding
// of code 0, the code predictor's embedding of each residual code and the
// step's trailing text row), and the TTS prompt assembly.

#include "metal/abi/Omni.h"
#include <metal_stdlib>

using namespace metal;

// One thread per output element.
kernel void omni_embed_bf16(device const uint *tokens [[buffer(0)]], device const bfloat *table [[buffer(1)]],
                            device bfloat *output [[buffer(2)]], constant OmniEmbedParams &p [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.width;
  if (index >= elements) return;
  const uint row = index / p.width, column = index % p.width;
  uint token = tokens[row * max(p.token_stride, 1u)];
  token = token < p.vocabulary ? token : 0;
  const uint stride = max(p.output_row_stride, 1u);
  output[ulong(row) * stride * p.width + column] = table[ulong(token) * p.width + column];
}

// codes: [rows][16] u32; predictor_tables: 15 tables of [predictor
// vocabulary][width] bf16, one after another; trailing_rows: per row, the row
// of the trailing pool to add. Sums in fp32, rounded once.
kernel void omni_tts_step_input(device const uint *codes [[buffer(0)]], device const bfloat *codec_table [[buffer(1)]],
                                device const bfloat *predictor_tables [[buffer(2)]],
                                device const uint *trailing_rows [[buffer(3)]],
                                device const bfloat *trailing_pool [[buffer(4)]], device bfloat *output [[buffer(5)]],
                                constant OmniTtsStepParams &p [[buffer(6)]], uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.width;
  if (index >= elements) return;
  const uint row = index / p.width, column = index % p.width;
  device const uint *frame = codes + row * OMNI_CODE_GROUPS;
  const uint code0 = frame[0] < p.codec_vocabulary ? frame[0] : 0;
  float sum = float(codec_table[ulong(code0) * p.width + column]);
  const ulong table_stride = ulong(p.predictor_vocabulary) * p.width;
  for (uint group = 1; group < OMNI_CODE_GROUPS; ++group) {
    const uint code = frame[group] < p.predictor_vocabulary ? frame[group] : 0;
    sum += float(predictor_tables[(group - 1) * table_stride + ulong(code) * p.width + column]);
  }
  sum += float(trailing_pool[ulong(trailing_rows[row]) * p.width + column]);
  output[index] = bfloat(sum);
}

// Prompt rows: per position, the projected text row, the codec embedding,
// the speaker row and a reference frame's 16 embeddings, each when its
// index is not negative, summed in fp32.
kernel void omni_prompt_assemble(constant OmniPromptPosition *positions [[buffer(0)]],
                                 device const bfloat *text_rows [[buffer(1)]],
                                 device const bfloat *codec_table [[buffer(2)]],
                                 device const bfloat *predictor_tables [[buffer(3)]],
                                 device const bfloat *speaker [[buffer(4)]], device const uint *frames [[buffer(5)]],
                                 device bfloat *output [[buffer(6)]], constant OmniTtsStepParams &p [[buffer(7)]],
                                 uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.width;
  if (index >= elements) return;
  const uint row = index / p.width, column = index % p.width;
  constant OmniPromptPosition &position = positions[row];
  float sum = 0.0f;
  if (position.text_row >= 0) sum += float(text_rows[ulong(position.text_row) * p.width + column]);
  if (position.codec >= 0 && uint(position.codec) < p.codec_vocabulary)
    sum += float(codec_table[ulong(position.codec) * p.width + column]);
  if (position.speaker > 0) sum += float(speaker[column]);
  if (position.frame >= 0) {
    device const uint *frame = frames + uint(position.frame) * OMNI_CODE_GROUPS;
    const uint code0 = frame[0] < p.codec_vocabulary ? frame[0] : 0;
    sum += float(codec_table[ulong(code0) * p.width + column]);
    const ulong table_stride = ulong(p.predictor_vocabulary) * p.width;
    for (uint group = 1; group < OMNI_CODE_GROUPS; ++group) {
      const uint code = frame[group] < p.predictor_vocabulary ? frame[group] : 0;
      sum += float(predictor_tables[(group - 1) * table_stride + ulong(code) * p.width + column]);
    }
  }
  output[index] = bfloat(sum);
}

// Rows [row_index[i]] of `source` (stride `width`) copied to rows i of the
// output: the last row of each prefill sequence for its logits, a lane's
// step row into its state, and so on.
kernel void omni_gather_rows_bf16(device const uint *row_index [[buffer(0)]], device const bfloat *source [[buffer(1)]],
                                  device bfloat *output [[buffer(2)]], constant OmniEmbedParams &p [[buffer(3)]],
                                  uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.width;
  if (index >= elements) return;
  const uint row = index / p.width, column = index % p.width;
  const uint stride = max(p.output_row_stride, 1u);
  output[ulong(row) * stride * p.width + column] = source[ulong(row_index[row]) * p.width + column];
}
