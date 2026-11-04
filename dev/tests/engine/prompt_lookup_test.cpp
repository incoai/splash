#include "TestChecks.hpp"
#include "model/PromptLookup.hpp"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using splash::model::PromptLookup;
using splash::test::require;

namespace {

// Independent scalar specification, also checking the exact bounded window.
std::optional<PromptLookup::Proposal> reference(const std::vector<uint32_t> &history,
                                               uint32_t anchor) {
  if (history.size() < 23) return {};
  const size_t first = history.size() > 65536 ? history.size() - 65536 : 0;
  std::optional<PromptLookup::Proposal> result;
  for (size_t start = first; start + 23 <= history.size(); ++start) {
    bool match = true;
    for (size_t i = 0; i < 16; ++i)
      match = match && history[start + i] == (i == 15 ? anchor : history[history.size() - 15 + i]);
    if (!match) continue;
    PromptLookup::Proposal next;
    for (size_t i = 0; i < 7; ++i) next[i] = history[start + 16 + i];
    if (result && *result != next) return {};
    result = next;
  }
  return result;
}

void cases() {
  std::vector<uint32_t> source(23);
  std::iota(source.begin(), source.end(), 100);
  auto history = source;
  history.push_back(900);
  history.insert(history.end(), source.begin(), source.begin() + 15);
  PromptLookup::Proposal expected{116, 117, 118, 119, 120, 121, 122};
  require(PromptLookup::find(history, 115) == expected, "exact suffix did not recover seven known tokens");
  require(!PromptLookup::find(history, 114), "lookup ignored the pending anchor");
  history.back() = 999;
  require(!PromptLookup::find(history, 115), "lookup did not compare every suffix token");
  history.back() = 114;
  auto repeated = source;
  repeated.insert(repeated.end(), history.begin(), history.end());
  require(PromptLookup::find(repeated, 115) == expected, "agreeing multiple occurrences were rejected");
  repeated[22] = 999;
  require(!PromptLookup::find(repeated, 115), "different seventh continuation token was not ambiguous");
  for (uint32_t count = 0; count < 23; ++count)
    require(!PromptLookup::find(std::vector<uint32_t>(count, 7), 7), "short history supplied future tokens");
  require(PromptLookup::find(std::vector<uint32_t>(23, 7), 7) == PromptLookup::Proposal{7,7,7,7,7,7,7},
          "oldest fully observed overlapping occurrence was omitted");
  std::vector<uint32_t> bounded(65536, 9999);
  std::copy(source.begin(), source.end(), bounded.begin());
  std::copy_n(source.begin(), 15, bounded.end() - 15);
  require(PromptLookup::find(bounded, 115) == expected, "search excluded its exact oldest boundary");
  bounded.insert(bounded.begin(), 9999);
  bounded[1 + 22] = 8888;
  // The old source still starts at the first admitted token.
  require(PromptLookup::find(bounded, 115) == PromptLookup::Proposal{116,117,118,119,120,121,8888},
          "search window moved relative to the current history");
  bounded.erase(bounded.begin());
  bounded.insert(bounded.end() - 15, 9999);
  require(!PromptLookup::find(bounded, 115), "lookup searched beyond the 65536-token bound");

  std::mt19937 rng(401);
  for (uint32_t iteration = 0; iteration < 250; ++iteration) {
    const size_t n = iteration == 249 ? 65570 : rng() % 300;
    std::vector<uint32_t> values(n);
    for (auto &token : values) token = rng() % 17;
    const uint32_t anchor = rng() % 17;
    if (n >= 48 && iteration % 2 == 0) {
      const size_t at = rng() % (n - 31);
      std::copy_n(values.end() - 15, 15, values.begin() + static_cast<std::ptrdiff_t>(at));
      values[at + 15] = anchor;
    }
    require(PromptLookup::find(values, anchor) == reference(values, anchor),
            "lookup differs from bounded scalar specification");
  }
}
} // namespace

int main() {
  try { cases(); }
  catch (const std::exception &error) {
    std::cerr << "prompt_lookup_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "prompt_lookup_test: PASS\n";
}
