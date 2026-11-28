#pragma once

// The finish of a head's fp32 logits (kernels/shared/logit_finish.metal),
// which a model whose head states one runs after its vocabulary projection,
// before anything selects, samples or scores them: a softcap of the tokens'
// logits, and -infinity for the rows past them that a checkpoint's head was
// padded with to whole tiles. The sampling kernels weigh a -infinity logit
// nothing (kernels/decode/sampling.metal, shard_mass) and greedy selection
// never takes it over a finite one.

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

// A head's finish by its values: softcap c (0: none) and the `tokens`
// tokens the head projects, of the vocabulary rows it computes.
struct LogitFinishValues final {
  float softcap = 0.0f;
  uint32_t tokens = 0;
};

struct LogitFinish final {
  // logits[r][t] = c tanh(logits[r][t] / c) for t < tokens, -infinity past
  // them, of `rows` rows of `vocabulary` fp32 logits, in place.
  static void add(metal::CommandGraph &graph, metal::MetalBuffer logits, uint32_t vocabulary, uint32_t rows,
                  LogitFinishValues values);
};

} // namespace splash::ops
