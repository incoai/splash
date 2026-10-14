#pragma once

#include <cstdint>
#include <span>

namespace splash::model {

// Qwen3.8's draft vocabulary: the kQwen4DraftVocabularyTokens tokens of its
// tokenizer that text decodes to most often, ranked offline over a mixed
// corpus of reasoning, chat, code, English and 16 other languages' web text
// and math (dev/tools/draft_vocabulary.py), the chat template's, thinking and
// tool-call tokens among them. Qwen3.8-Flash-Next's MTP head drafts its
// argmax over them alone (Qwen4ExpLayout::mtpDraftTokens): it reads their rows
// of the vocabulary projection, an eighth of it, at each draft step. A draft
// is a proposal the verify judges with the whole vocabulary. Greedy output
// remains target-controlled and sampled acceptance preserves the target
// distribution. Request-local context and threshold depth make lone seeded
// runs repeat within a build with the same cached prefixes/configuration;
// changing proposals need not preserve the base build's seeded stream.
// A token outside this set and the context is never proposed by this head.
inline constexpr uint32_t kQwen4DraftVocabularyTokens = 32768;
// The vocabulary it is of: Qwen3.8's tokenizer.
inline constexpr uint32_t kQwen4DraftVocabularySource = 248320;
// Beside it each request drafts from up to kQwen4DraftContextTokens more:
// the tokens outside it that its own context holds (its prompt and output,
// in the order they first appear), whose rows of the vocabulary projection
// its state lane keeps. Context coverage depends on the request's tokens
// and the tokenizer; vocabulary size alone is not a tokenizer identity.
inline constexpr uint32_t kQwen4DraftContextTokens = 4096;

// The draft vocabulary's tokens, ascending, when `tokens` is its size; none
// for 0. Throws for any other size.
[[nodiscard]] std::span<const uint32_t> qwen4DraftVocabulary(uint32_t tokens);
// Whether `token` is one of the draft vocabulary's.
[[nodiscard]] bool inQwen4DraftVocabulary(uint32_t token) noexcept;

// The draft vocabulary size for a head with `vocabulary` rows: 32768
// for 248320 rows unless SPLASH_MTP_DRAFT_VOCAB=0; otherwise 0. This size
// check assumes Qwen3.8 token IDs and does not validate tokenizer identity.
[[nodiscard]] uint32_t qwen4DraftTokensFor(uint32_t vocabulary);

namespace detail {
extern const uint64_t kQwen4DraftVocabularyWords[kQwen4DraftVocabularySource / 64];
} // namespace detail

} // namespace splash::model
