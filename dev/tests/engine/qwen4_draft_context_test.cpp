// Request-local draft context and state-lane row ownership.
#include "TestChecks.hpp"
#include "model/Qwen4DraftContext.hpp"

#include <iostream>

using namespace splash::model;
using splash::test::require;

namespace {
void contextLifetime() {
  Qwen4DraftContextCache<4> cache;
  // end, cancellation and suspension all release the runtime's state lane.
  for (const auto reason : {"completion", "cancel", "suspend"})
    for (const uint32_t next : {1u, 2u, 3u}) {
      require(cache.bind(0, 0, 2) == 0, "context: fresh request ID zero copies every row");
      require(cache.bind(0, 0, 2) == 2, "context: one activation reuses copied rows");
      cache.release(0);
      require(cache.bind(0, 0, next) == 0, std::string("context: same-ID ") + reason + " copies smaller/equal/larger sets");
      cache.release(0);
    }
  require(cache.bind(0, 5, 3) == 0, "context: first activation");
  cache.release(0); // suspend, retaining the request's host context
  require(cache.bind(2, 5, 3) == 0, "context: resume on another state lane copies all rows");
  require(cache.bind(0, 6, 1) == 0, "context: released lane belongs to another request");
  require(cache.bind(2, 5, 4) == 3, "context: resume keeps its own newly appended row");
  bool refused = false;
  try { static_cast<void>(cache.bind(2, 5, 2)); } catch (const std::logic_error &) { refused = true; }
  require(refused, "context: synced count cannot underflow a copy count");
}

void contextRecovery() {
  std::vector<uint32_t> outside;
  for (uint32_t t = 0; outside.size() < 4100; ++t)
    if (!inQwen4DraftVocabulary(t)) outside.push_back(t);
  for (const uint32_t capacity : {0u, 1u, 4095u, 4096u}) {
    Qwen4DraftContext context;
    auto history = outside;
    history.insert(history.end(), outside.begin(), outside.end());
    history.push_back(kQwen4DraftVocabularySource); // invalid token ignored
    context.scan(history, kQwen4DraftVocabularySource, capacity);
    require(context.tokens.size() == capacity, "context: capacity and duplicates stay bounded");
  }
  Qwen4DraftContext context;
  const std::array<uint32_t, 1> repeated{outside[0]};
  context.record(repeated);
  require(!context.useVocabulary(), "context: novel output falls back to the whole head");
  context.scan(repeated, kQwen4DraftVocabularySource);
  require(context.contains(outside[0]), "context: fallback scans new context without GPU copies");
  for (uint32_t i = 0; i < 200; ++i) context.record(repeated);
  require(context.useVocabulary(), "context: repeated topic tokens recover the small head");
}


void requestLocalContextRepeats() {
  std::vector<uint32_t> outside;
  for (uint32_t t = 0; outside.size() < 8; ++t)
    if (!inQwen4DraftVocabulary(t)) outside.push_back(t);
  struct Snapshot {
    std::vector<uint32_t> tokens;
    float misses, observations;
    bool smallHead;
    bool operator==(const Snapshot &) const = default;
  };
  const auto run = [&](bool reused) {
    Qwen4DraftContextCache<4> cache;
    if (reused) {
      static_cast<void>(cache.bind(2, 7, 6));
      cache.release(2); // completed request, same ID reused by this run
    }
    Qwen4DraftContext context;
    std::vector<uint32_t> history{outside[0], outside[1], outside[0]};
    std::vector<Snapshot> decisions;
    uint32_t stateLane = reused ? 2 : 0;
    for (uint32_t cycle = 0; cycle < 12; ++cycle) {
      context.scan(history, kQwen4DraftVocabularySource);
      const uint32_t count = static_cast<uint32_t>(context.tokens.size());
      const uint32_t copied = cache.bind(stateLane, 7, count);
      require(copied <= count, "context: physical copy count stays bounded");
      decisions.push_back({context.tokens, context.misses, context.observations, context.useVocabulary()});
      const std::array selected{outside[2 + cycle % 6]};
      context.record(selected);
      history.push_back(selected[0]);
      if (reused) {
        static_cast<void>(cache.bind(1, 90 + cycle, 1)); // another request's rows
        cache.release(1);
        if (cycle == 5) {
          cache.release(stateLane); // suspend and resume on a different lane
          stateLane = 3;
        }
      }
    }
    return decisions;
  };
  require(run(false) == run(true),
          "context: lone request's head decisions repeat across reuse, unrelated requests and lane migration");
}

} // namespace

int main() {
  try {
    contextLifetime();
    contextRecovery();
    requestLocalContextRepeats();
  } catch (const std::exception &error) {
    std::cerr << "Qwen4 draft context tests FAILED: " << error.what() << '\n';
    return 1;
  }
  std::cout << "Qwen4 draft context tests passed\n";
}
