#include "ops/LogitFinish.hpp"

#include "metal/abi/LogitFinish.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops {

void LogitFinish::add(metal::CommandGraph &graph, metal::MetalBuffer logits, uint32_t vocabulary, uint32_t rows,
                      LogitFinishValues values) {
  if (!rows || !values.tokens || values.tokens > vocabulary || values.softcap < 0.0f)
    throw std::invalid_argument("invalid logit finish");
  requireBytes(logits, uint64_t{rows} * vocabulary * sizeof(float), "logit finish logits");
  graph.add("logit_finish", {std::move(logits)}, LogitFinishParams{rows, vocabulary, values.tokens, values.softcap},
            {(vocabulary + LOGIT_FINISH_THREADS - 1) / LOGIT_FINISH_THREADS, rows, 1}, {LOGIT_FINISH_THREADS, 1, 1});
}

} // namespace splash::ops
