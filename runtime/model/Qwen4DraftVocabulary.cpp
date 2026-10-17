#include "model/Qwen4DraftVocabulary.hpp"

#include <bit>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::model {
namespace {

static_assert(kQwen4DraftVocabularySource % 64 == 0, "the bitmap's words cover the vocabulary");

const uint64_t *wordsOf(uint32_t tokens) noexcept {
  if (tokens == kQwen4DraftVocabularyTokens) return detail::kQwen4DraftVocabularyWords;
  if (tokens == kQwen4DraftVocabularyLargeTokens) return detail::kQwen4DraftVocabularyWords65536;
  return nullptr;
}

std::vector<uint32_t> draftTokens(uint32_t size, const uint64_t *words) {
  std::vector<uint32_t> result;
  result.reserve(size);
  for (uint32_t word = 0; word < kQwen4DraftVocabularySource / 64; ++word)
    for (uint64_t bits = words[word]; bits; bits &= bits - 1)
      result.push_back(word * 64 + static_cast<uint32_t>(std::countr_zero(bits)));
  if (result.size() != size) throw std::logic_error("the draft vocabulary's bitmap does not hold its tokens");
  return result;
}

} // namespace

std::span<const uint32_t> qwen4DraftVocabulary(uint32_t tokens) {
  if (!tokens) return {};
  if (tokens == kQwen4DraftVocabularyTokens) {
    static const auto set = draftTokens(kQwen4DraftVocabularyTokens, detail::kQwen4DraftVocabularyWords);
    return set;
  }
  if (tokens == kQwen4DraftVocabularyLargeTokens) {
    static const auto set = draftTokens(kQwen4DraftVocabularyLargeTokens, detail::kQwen4DraftVocabularyWords65536);
    return set;
  }
  throw std::invalid_argument("no draft vocabulary has that size");
}

bool qwen4DraftContains(uint32_t token, uint32_t tokens) noexcept {
  const uint64_t *words = wordsOf(tokens);
  return words && token < kQwen4DraftVocabularySource && ((words[token / 64] >> (token % 64)) & 1);
}

uint32_t qwen4DraftTokensFor(uint32_t vocabulary) {
  if (vocabulary != kQwen4DraftVocabularySource) return 0;
  const char *setting = std::getenv("SPLASH_MTP_DRAFT_VOCAB");
  if (!setting || !*setting) return kQwen4DraftVocabularyLargeTokens;
  const std::string_view size(setting);
  if (size == "0") return 0;
  if (size == "32768") return kQwen4DraftVocabularyTokens;
  if (size == "65536") return kQwen4DraftVocabularyLargeTokens;
  throw std::invalid_argument("SPLASH_MTP_DRAFT_VOCAB must be 0, 32768 or 65536");
}

} // namespace splash::model
