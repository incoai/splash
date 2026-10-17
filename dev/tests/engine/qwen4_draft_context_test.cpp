// A request's draft context beside the MTP head's draft vocabulary
// (model/Qwen4DraftContext.hpp): which rows a state lane holds across
// activations, the context's bound, and the fallback to the whole head by
// the recent tokens outside the vocabulary, with and without the warm start.
#include "TestChecks.hpp"
#include "model/Qwen4DraftContext.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace splash::model;
using splash::test::require;

namespace {

constexpr float kGate = 0.03F;

void contextLifetime() {
  Qwen4DraftContextCache<4> cache;
  // Completion, cancellation and suspension all release the state lane.
  for (const uint32_t next : {1u, 2u, 3u}) {
    require(cache.bind(0, 0, 2) == 0, "context: a fresh request copies every row");
    require(cache.bind(0, 0, 2) == 2, "context: one activation reuses copied rows");
    cache.release(0);
    require(cache.bind(0, 0, next) == 0, "context: the same id after a release copies its rows again");
    cache.release(0);
  }
  require(cache.bind(0, 5, 3) == 0, "context: first activation");
  cache.release(0); // suspended, keeping the request's context
  require(cache.bind(2, 5, 3) == 0, "context: a resume on another state lane copies every row");
  require(cache.bind(0, 6, 1) == 0, "context: a released lane belongs to another request");
  require(cache.bind(2, 5, 4) == 3, "context: a resumed request copies only the rows it appended");
  bool refused = false;
  try {
    static_cast<void>(cache.bind(2, 5, 2));
  } catch (const std::logic_error &) {
    refused = true;
  }
  require(refused, "context: a count below the copied rows is refused");
}

void contextBounds() {
  std::vector<uint32_t> outside;
  for (uint32_t t = 0; outside.size() < 4100; ++t)
    if (!qwen4DraftContains(t, kQwen4DraftVocabularyTokens)) outside.push_back(t);
  for (const uint32_t capacity : {0u, 1u, 4095u, 4096u}) {
    Qwen4DraftContext context;
    auto history = outside;
    history.insert(history.end(), outside.begin(), outside.end());
    history.push_back(kQwen4DraftVocabularySource); // out of the vocabulary: ignored
    context.scan(history, kQwen4DraftVocabularySource, capacity);
    require(context.tokens.size() == capacity, "context: the capacity bounds it and repeats are kept once");
  }
  Qwen4DraftContext context;
  const std::array<uint32_t, 1> repeated{outside[0]};
  context.record(repeated);
  require(!context.useVocabulary(false, kGate), "context: a token outside falls back to the whole head");
  context.scan(repeated, kQwen4DraftVocabularySource);
  require(context.contains(outside[0]), "context: a scan takes the token into the context");
  for (uint32_t i = 0; i < 200; ++i) context.record(repeated);
  require(context.useVocabulary(false, kGate), "context: repeated topic tokens recover the draft vocabulary");
}

void warmStart() {
  for (const uint32_t size : {kQwen4DraftVocabularyTokens, kQwen4DraftVocabularyLargeTokens}) {
    uint32_t outside = 0, inside = 0;
    while (qwen4DraftContains(outside, size)) ++outside;
    while (!qwen4DraftContains(inside, size)) ++inside;
    Qwen4DraftContext c;
    c.scan({}, kQwen4DraftVocabularySource, kQwen4DraftContextTokens, size);
    require(c.useVocabulary(false, kGate) && c.useVocabulary(true, kGate), "a fresh request drafts from the vocabulary");
    c.record(std::array{outside});
    require(!c.useVocabulary(false, kGate), "without the warm start one token outside falls back");
    require(c.useVocabulary(true, kGate), "the warm start tolerates one early token outside");
    require(std::fabs(c.coveredPrior - 49.0F) < 1e-4F, "the prior decays per token");
    c.record(std::array{outside});
    require(!c.useVocabulary(true, kGate), "two tokens outside in a row fall back even from a warm start");
    c.scan(std::array{outside}, kQwen4DraftVocabularySource, kQwen4DraftContextTokens, size);
    for (int i = 0; i < 200; ++i) c.record(std::array{outside});
    require(c.useVocabulary(false, kGate) && c.useVocabulary(true, kGate), "the context's tokens recover the vocabulary");
    require(c.coveredPrior < 1.0F, "the prior vanishes over a request's output");
    Qwen4DraftContext adversarial;
    adversarial.scan({}, kQwen4DraftVocabularySource, 0, size);
    for (int i = 0; i < 200; ++i) adversarial.record(std::array{outside});
    require(!adversarial.useVocabulary(true, kGate), "the prior cannot hide sustained misses");
    // Without the warm start, the decision is the plain decayed rate's.
    Qwen4DraftContext plain;
    float misses = 0, observations = 0;
    for (int i = 0; i < 500; ++i) {
      const bool miss = i % 37 == 0;
      plain.record(std::array{miss ? outside : inside});
      misses = Qwen4DraftContext::kDecay * misses + (miss ? 1.0F : 0.0F);
      observations = Qwen4DraftContext::kDecay * observations + 1.0F;
      require(plain.useVocabulary(false, kGate) == (misses <= kGate * observations),
              "without the warm start every decision is the decayed rate's");
    }
  }
}

} // namespace

int main() {
  try {
    contextLifetime();
    contextBounds();
    warmStart();
  } catch (const std::exception &error) {
    std::cerr << "Qwen4 draft context tests FAILED: " << error.what() << '\n';
    return 1;
  }
  std::cout << "Qwen4 draft context tests passed\n";
}
