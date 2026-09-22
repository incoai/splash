#include "flash/FlashMTPWindow.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {
using namespace splash::flash;
uint64_t checks = 0;
void require(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

constexpr auto geometryOne = flashMTPConvolutionPrefix(1);
static_assert(geometryOne && geometryOne->oldRows == 2 &&
              geometryOne->oldBegin == 1 && geometryOne->inputRows == 1 &&
              geometryOne->inputBegin == 0 &&
              geometryOne->destinationInputBegin == 2);
constexpr auto geometryEight = flashMTPConvolutionPrefix(8);
static_assert(geometryEight && geometryEight->oldRows == 0 &&
              geometryEight->inputRows == 3 && geometryEight->inputBegin == 5 &&
              geometryEight->destinationInputBegin == 0);
constexpr auto geometrySixteen = flashMTPConvolutionPrefix(16);
static_assert(geometrySixteen && geometrySixteen->oldRows == 0 &&
              geometrySixteen->inputRows == 3 && geometrySixteen->inputBegin == 13 &&
              geometrySixteen->destinationInputBegin == 0);
static_assert(kFlashSingletonMaximumMTPDepth == 15 &&
              kFlashSingletonMaximumVerifyRows == 16);
static_assert(!flashMTPConvolutionPrefix(0) && !flashMTPConvolutionPrefix(17));
static_assert(flashParseSingletonMTPDepth("1") == 1 &&
              flashParseSingletonMTPDepth("9") == 9 &&
              flashParseSingletonMTPDepth("10") == 10 &&
              flashParseSingletonMTPDepth("15") == 15);
static_assert(!flashParseSingletonMTPDepth("") &&
              !flashParseSingletonMTPDepth("0") &&
              !flashParseSingletonMTPDepth("16") &&
              !flashParseSingletonMTPDepth("03"));
static_assert(flashMTPCommittedFoldChunkRows(1) == 1 &&
              flashMTPCommittedFoldChunkRows(8) == 8 &&
              flashMTPCommittedFoldChunkRows(9) == 8 &&
              flashMTPCommittedFoldChunkRows(16) == 8);
static_assert(!flashMTPCommittedFoldChunkRows(0) &&
              !flashMTPCommittedFoldChunkRows(17));
constexpr std::array<uint32_t, 3> compileInputs{10, 11, 12};
constexpr std::array<uint32_t, 3> compilePredictions{11, 12, 13};
constexpr auto compileAccepted =
    flashMTPAcceptGreedyPrefix(compileInputs, compilePredictions, 3);
static_assert(compileAccepted && compileAccepted->matchedDrafts == 2 &&
              compileAccepted->retainedRows == 3 &&
              compileAccepted->output[2] == 13 &&
              compileAccepted->finish == FlashMTPPrefixFinish::Length);

void convolutionHistory() {
  // Distinct channels expose row-offset mistakes as well as copy lengths.
  constexpr size_t channels = 5;
  using Row = std::array<int32_t, channels>;
  const std::array<Row, 3> old{{{-31, -32, -33, -34, -35},
                              {-21, -22, -23, -24, -25},
                              {-11, -12, -13, -14, -15}}};
  std::array<Row, kFlashSingletonMaximumVerifyRows> incoming{};
  for (size_t row = 0; row < incoming.size(); ++row)
    for (size_t channel = 0; channel < channels; ++channel)
      incoming[row][channel] = static_cast<int32_t>(100 * (row + 1) + channel);
  auto reference = old;
  for (uint32_t kept = 1; kept <= incoming.size(); ++kept) {
    reference[0] = reference[1]; reference[1] = reference[2];
    reference[2] = incoming[kept - 1];
    const auto geometry = flashMTPConvolutionPrefix(kept);
    require(geometry.has_value(), "valid kept prefix must produce copy ranges");
    require(geometry->oldBegin + geometry->oldRows <= old.size(),
            "old copy must remain within three-row history");
    require(geometry->inputBegin + geometry->inputRows == kept,
            "input copy must end at the retained prefix");
    require(geometry->oldRows + geometry->inputRows == old.size() &&
            geometry->destinationInputBegin == geometry->oldRows,
            "copies must exactly cover the destination history");
    std::array<Row, 3> reconstructed{};
    std::copy_n(old.begin() + geometry->oldBegin, geometry->oldRows,
                reconstructed.begin());
    std::copy_n(incoming.begin() + geometry->inputBegin, geometry->inputRows,
                reconstructed.begin() + geometry->destinationInputBegin);
    require(reconstructed == reference,
            "retained history must equal streaming three-row shift reference");
  }
  require(!flashMTPConvolutionPrefix(0), "zero retained rows must reject");
  require(!flashMTPConvolutionPrefix(17), "seventeen retained rows must reject");
  require(!flashMTPConvolutionPrefix(std::numeric_limits<uint32_t>::max()),
          "overflow-sized retained rows must reject");
}

void depthParsing() {
  // Scan every possible byte so control, locale and non-ASCII characters
  // cannot accidentally become accepted by a permissive numeric conversion.
  for (uint32_t byte = 0; byte <= 255; ++byte) {
    const char character = static_cast<char>(byte);
    const auto depth = flashParseSingletonMTPDepth(std::string_view(&character, 1));
    const bool valid = byte >= static_cast<uint32_t>('1') &&
                       byte <= static_cast<uint32_t>('9');
    require(depth.has_value() == valid, "only ASCII digits 1 through 9 may parse alone");
    if (valid)
      require(*depth == byte - static_cast<uint32_t>('0'),
              "parsed singleton depth must equal the ASCII digit");
  }
  for (uint32_t first = 0; first <= 255; ++first) {
    for (uint32_t second = 0; second <= 255; ++second) {
      const std::array<char, 2> characters{
          static_cast<char>(first), static_cast<char>(second)};
      const auto depth = flashParseSingletonMTPDepth(
          std::string_view(characters.data(), characters.size()));
      const bool valid = first == static_cast<uint32_t>('1') &&
                         second >= static_cast<uint32_t>('0') &&
                         second <= static_cast<uint32_t>('5');
      require(depth.has_value() == valid,
              "only canonical ASCII decimal 10 through 15 may parse as two bytes");
      if (valid)
        require(*depth == 10 + second - static_cast<uint32_t>('0'),
                "parsed two-digit singleton depth must match its decimal value");
    }
  }
  constexpr std::array<std::string_view, 28> invalid{
      "", "01", "07", "00", "16", "17", "77", "999999999999999999999",
      " 3", "3 ", "\t3", "3\n", "\n3", "+3", "-3", "3.0", ".3", "3e0",
      "0x3", "3_", "NaN", "inf", "three", "\xD9\xA3", "\xEF\xBC\x93",
      "015", "100", "15 "};
  for (const std::string_view value : invalid)
    require(!flashParseSingletonMTPDepth(value),
            "noncanonical and out-of-range overrides must reject");
  constexpr std::array<char, 3> embeddedNull{'3', '\0', '4'};
  require(!flashParseSingletonMTPDepth(std::string_view(embeddedNull.data(), embeddedNull.size())),
          "embedded NUL must not truncate a counted override");
}

void committedFoldChunks() {
  std::array<uint32_t, kFlashSingletonMaximumVerifyRows> orderedTokens{};
  for (uint32_t index = 0; index < orderedTokens.size(); ++index)
    orderedTokens[index] = 100 + index;
  for (uint32_t total = 1; total <= orderedTokens.size(); ++total) {
    std::array<uint32_t, kFlashSingletonMaximumVerifyRows> foldedTokens{};
    uint32_t processed = 0, chunks = 0;
    while (processed < total) {
      const auto rows = flashMTPCommittedFoldChunkRows(total - processed);
      require(rows && *rows > 0 && *rows <= 8 && *rows <= total - processed,
              "fold chunks must remain nonempty and within the raw-coefficient row boundary");
      std::copy_n(orderedTokens.begin() + processed, *rows,
                  foldedTokens.begin() + processed);
      processed += *rows; ++chunks;
    }
    require(processed == total, "fold chunks must cover the whole committed prefix");
    require(chunks == (total <= 8 ? 1 : 2),
            "prefixes beyond eight rows must use exactly two ordered folds");
    for (uint32_t index = 0; index < total; ++index)
      require(foldedTokens[index] == orderedTokens[index],
              "fold chunking must preserve pair token and hidden row order");
  }
  require(!flashMTPCommittedFoldChunkRows(0), "empty fold must reject");
  require(!flashMTPCommittedFoldChunkRows(17), "oversized fold must reject");
  require(!flashMTPCommittedFoldChunkRows(std::numeric_limits<uint32_t>::max()),
          "overflow-sized fold must reject");
}

void matchingAndBudget() {
  std::array<uint32_t, kFlashSingletonMaximumVerifyRows> inputs{};
  for (uint32_t index = 0; index < inputs.size(); ++index) inputs[index] = 100 + index;
  for (uint32_t rows = 1; rows <= inputs.size(); ++rows) {
    for (uint32_t matches = 0; matches < rows; ++matches) {
      std::array<uint32_t, kFlashSingletonMaximumVerifyRows> predictions{};
      for (uint32_t index = 0; index < rows; ++index)
        predictions[index] = index < matches ? inputs[index + 1] : 900 + index;
      for (uint32_t budget = 1; budget <= kFlashSingletonMaximumVerifyRows + 1; ++budget) {
        const auto accepted = flashMTPAcceptGreedyPrefix(
            std::span(inputs).first(rows), std::span(predictions).first(rows), budget);
        require(accepted.has_value(), "valid acceptance window must succeed");
        require(accepted->matchedDrafts == matches,
                "acceptance must stop at the first draft mismatch");
        const uint32_t expectedRows = std::min(matches + 1, budget);
        require(accepted->retainedRows == expectedRows,
                "commit rows must equal emitted outputs after quota clamp");
        for (uint32_t index = 0; index < expectedRows; ++index)
          require(accepted->output[index] ==
                      (index < matches ? inputs[index + 1] : predictions[index]),
                  "output must preserve matched drafts and correction or bonus");
        require(accepted->finish == (expectedRows == budget
                    ? FlashMTPPrefixFinish::Length : FlashMTPPrefixFinish::None),
                "length must occur exactly when quota is consumed");
      }
    }
  }
  std::array<uint32_t, kFlashSingletonMaximumVerifyRows + 1> tooLarge{};
  require(!flashMTPAcceptGreedyPrefix({}, {}, 1), "empty window must reject");
  require(!flashMTPAcceptGreedyPrefix(inputs, inputs, 0), "zero budget must reject");
  require(!flashMTPAcceptGreedyPrefix(tooLarge, tooLarge, 1), "seventeen-row window must reject");
  require(!flashMTPAcceptGreedyPrefix(inputs, std::span(inputs).first(15), 1),
          "prediction count mismatch must reject");
}

void stoppingInsidePrefix() {
  constexpr std::array<uint32_t, 2> stops{248044, 248046};
  // Exercise each EOS position, each stop token, and every quota boundary.
  for (const uint32_t stop : stops) {
    for (uint32_t position = 0; position < kFlashSingletonMaximumVerifyRows; ++position) {
      std::array<uint32_t, kFlashSingletonMaximumVerifyRows> inputs{};
      std::array<uint32_t, kFlashSingletonMaximumVerifyRows> predictions{};
      for (uint32_t index = 0; index < inputs.size(); ++index) {
        inputs[index] = 100 + index; predictions[index] = 101 + index;
      }
      predictions[position] = stop;
      if (position < kFlashSingletonMaximumMTPDepth) inputs[position + 1] = stop;
      for (uint32_t budget = 1; budget <= kFlashSingletonMaximumVerifyRows + 1; ++budget) {
        const auto accepted = flashMTPAcceptGreedyPrefix(inputs, predictions, budget, stops);
        require(accepted && accepted->matchedDrafts == kFlashSingletonMaximumMTPDepth,
                "matched count must be observed before EOS and quota truncation");
        require(accepted->retainedRows == std::min(position + 1, budget),
                "commit rows must stop at EOS or earlier quota");
        require(accepted->finish == (budget <= position
                    ? FlashMTPPrefixFinish::Length : FlashMTPPrefixFinish::Stop),
                "EOS must take precedence at its quota boundary");
        if (budget > position)
          require(accepted->output[accepted->retainedRows - 1] == stop,
                  "EOS must remain in the committed output prefix");
      }
      // Also exercise EOS as the correction at the first mismatch.
      if (position < kFlashSingletonMaximumMTPDepth) {
        inputs[position + 1] = 700 + position;
        const auto corrected = flashMTPAcceptGreedyPrefix(
            inputs, predictions, kFlashSingletonMaximumVerifyRows + 1, stops);
        require(corrected && corrected->matchedDrafts == position &&
                corrected->retainedRows == position + 1 &&
                corrected->output[position] == stop &&
                corrected->finish == FlashMTPPrefixFinish::Stop,
                "EOS correction must terminate at the first mismatch");
      }
    }
  }
}
} // namespace

int main() {
  try {
    convolutionHistory(); depthParsing(); committedFoldChunks();
    matchingAndBudget(); stoppingInsidePrefix();
    std::cout << "flash_mtp_window_test: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_mtp_window_test failed: " << error.what() << '\n';
    return 1;
  }
}
