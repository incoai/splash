// The vocabulary a DFlash draft step proposes from (model::DFlashDraft::vocabulary,
// ops::DraftVocabulary), on the CPU: the subset while each lane's last
// kVocabularyWindow tokens, its anchor the last, lie in it, its trailing
// tokens too; the whole vocabulary from a token outside it until that token
// has left every lane's window; always the whole vocabulary for a family with
// no subset.
#include "TestChecks.hpp"
#include "model/DFlashDraft.hpp"
#include "ops/DraftSelector.hpp"

#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <span>
#include <vector>

namespace {

using splash::model::DFlashDraft;
using splash::ops::DraftVocabulary;
using splash::test::rejects;
using splash::test::require;

void stepVocabulary() {
  constexpr DraftVocabulary subset{512, 960}, whole{};
  constexpr uint32_t window = DFlashDraft::kVocabularyWindow;
  const std::vector<uint32_t> inside(200, 7), trailing(200, 1000);
  std::vector<uint32_t> oldest = inside, past = inside;
  oldest[inside.size() - (window - 1)] = 600;  // the first token of the window before the anchor
  past[inside.size() - window] = 600;          // the token before that
  const auto step = [](DraftVocabulary vocabulary, std::initializer_list<const std::vector<uint32_t> *> histories,
                       std::initializer_list<uint32_t> anchors) {
    std::vector<std::span<const uint32_t>> tokens;
    for (const std::vector<uint32_t> *history : histories) tokens.emplace_back(*history);
    return DFlashDraft::vocabulary(vocabulary, tokens, std::vector<uint32_t>(anchors));
  };
  require(step(subset, {&inside}, {7}) == subset && step(subset, {&trailing}, {1000}) == subset &&
              step(subset, {&past}, {7}) == subset,
          "a lane whose window lies in the subset drafted over the whole vocabulary");
  require(step(subset, {&oldest}, {7}) == whole && step(subset, {&inside}, {600}) == whole,
          "a lane whose window holds a token outside the subset drafted over it");
  // The subset's edges: its last leading token and its first trailing one lie
  // in it, the tokens next to them between its parts do not.
  require(step(subset, {&inside}, {511}) == subset && step(subset, {&inside}, {960}) == subset,
          "a lane whose anchor is an edge token of the subset drafted over the whole vocabulary");
  require(step(subset, {&inside}, {512}) == whole && step(subset, {&inside}, {959}) == whole,
          "a lane whose anchor lies next to the subset's edge drafted over the subset");
  require(step(subset, {&inside, &oldest}, {7, 7}) == whole,
          "a step drafted over the subset while one of its lanes' windows leaves it");
  const std::vector<uint32_t> none;
  require(step(subset, {&none}, {7}) == subset,
          "a lane that read nothing but its anchor drafted over the whole vocabulary");
  require(step(whole, {&inside}, {7}) == whole, "a family without a subset drafted over one");
  rejects([&] { (void)step(subset, {&inside}, {7, 7}); }, "a draft step's lanes each have tokens and an anchor",
          "a step of more anchors than token histories was accepted");
}

} // namespace

int main() {
  try {
    stepVocabulary();
    std::cout << "draft vocabulary: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft vocabulary: FAIL: " << error.what() << '\n';
    return 1;
  }
}
