#pragma once

// Parameters of the logit finish (kernels/shared/logit_finish.metal,
// ops/LogitFinish.cpp): what a model's head does to its fp32 logits after its
// projection, before anything reads them.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define LOGIT_FINISH_THREADS 256u

// `rows` rows of `vocabulary` logits: those of the first `tokens` tokens
// become softcap * tanh(logit / softcap) (unchanged with a softcap of 0),
// those past them, rows a checkpoint's head was padded with to whole tiles,
// -infinity, which no selection or sampling takes.
struct LogitFinishParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t tokens;
  float softcap;
};

static_assert(sizeof(LogitFinishParams) == 16, "logit finish parameters are 16 bytes on both sides");
