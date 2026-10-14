#include "model/Qwen4DraftVocabulary.hpp"

#include <bit>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::model {
namespace {

static_assert(kQwen4DraftVocabularySource % 64 == 0, "the bitmap's words cover the vocabulary");

const std::vector<uint32_t> &draftTokens() {
  static const std::vector<uint32_t> tokens = [] {
    std::vector<uint32_t> result;
    result.reserve(kQwen4DraftVocabularyTokens);
    for (uint32_t word = 0; word < kQwen4DraftVocabularySource / 64; ++word)
      for (uint64_t bits = detail::kQwen4DraftVocabularyWords[word]; bits; bits &= bits - 1)
        result.push_back(word * 64 + static_cast<uint32_t>(std::countr_zero(bits)));
    if (result.size() != kQwen4DraftVocabularyTokens)
      throw std::logic_error("the draft vocabulary's bitmap does not hold its tokens");
    return result;
  }();
  return tokens;
}

} // namespace

std::span<const uint32_t> qwen4DraftVocabulary(uint32_t tokens) {
  if (!tokens) return {};
  if (tokens != kQwen4DraftVocabularyTokens) throw std::invalid_argument("no draft vocabulary has that size");
  return draftTokens();
}

bool inQwen4DraftVocabulary(uint32_t token) noexcept {
  return token < kQwen4DraftVocabularySource &&
         ((detail::kQwen4DraftVocabularyWords[token / 64] >> (token % 64)) & 1);
}

uint32_t qwen4DraftTokensFor(uint32_t vocabulary) {
  const char *setting = std::getenv("SPLASH_MTP_DRAFT_VOCAB");
  if (setting && std::string_view(setting) == "0") return 0;
  return vocabulary == kQwen4DraftVocabularySource ? kQwen4DraftVocabularyTokens : 0;
}

} // namespace splash::model
