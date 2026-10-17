#pragma once

#include <cstdint>
#include <span>

namespace splash::model {

// Qwen3.8's draft vocabulary: the tokens of its tokenizer that text decodes
// to most often, ranked offline over a mixed corpus of reasoning, chat, code,
// English and 16 other languages' web text and math
// (dev/tools/draft_vocabulary.py), the chat template's, thinking and
// tool-call tokens among them, in nested sets of kQwen4DraftVocabularyTokens
// (the ranking's first) and kQwen4DraftVocabularyLargeTokens tokens.
// Qwen3.8-Flash-Next's MTP head drafts its argmax over one of them alone
// (Qwen4ExpLayout::mtpDraftTokens): it reads their rows of the vocabulary
// projection, a quarter of it at most, at each draft step. A draft is a
// proposal the verify judges with the whole vocabulary, so greedy output is
// the target's and sampled acceptance keeps its distribution; a token
// outside the set and the request's context is never proposed.
inline constexpr uint32_t kQwen4DraftVocabularyTokens = 32768;
inline constexpr uint32_t kQwen4DraftVocabularyLargeTokens = 65536;
// The vocabulary they are of: Qwen3.8's tokenizer, which its size names
// (a performance assumption, not a tokenizer identity check).
inline constexpr uint32_t kQwen4DraftVocabularySource = 248320;
// Beside the set each request drafts from up to kQwen4DraftContextTokens
// more: the tokens outside it that its own context holds (its prompt and
// output, in the order they first appear), whose rows of the vocabulary
// projection its state lane keeps. A request's context names its topic's
// words, which no fixed list ranks.
inline constexpr uint32_t kQwen4DraftContextTokens = 4096;

// The draft vocabulary of `tokens` tokens, ascending; none for 0. Throws for
// a size that is not one of the two.
[[nodiscard]] std::span<const uint32_t> qwen4DraftVocabulary(uint32_t tokens);
// Whether `token` is one of the `tokens`-token draft vocabulary's.
[[nodiscard]] bool qwen4DraftContains(uint32_t token, uint32_t tokens) noexcept;

// The draft vocabulary an MTP head over `vocabulary` tokens drafts from: for
// Qwen3.8's tokenizer kQwen4DraftVocabularyLargeTokens, or what
// SPLASH_MTP_DRAFT_VOCAB names (0 drafts from every token, 32768 or 65536 a
// set); 0 for another vocabulary.
[[nodiscard]] uint32_t qwen4DraftTokensFor(uint32_t vocabulary);

namespace detail {
extern const uint64_t kQwen4DraftVocabularyWords[kQwen4DraftVocabularySource / 64];
extern const uint64_t kQwen4DraftVocabularyWords65536[kQwen4DraftVocabularySource / 64];
} // namespace detail

} // namespace splash::model
