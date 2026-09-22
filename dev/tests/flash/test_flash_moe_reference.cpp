#include "FlashMoEReference.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ref = splash::flash::reference;
using ref::Bits;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Exception = std::invalid_argument, class Function>
void rejects(Function function, const char *message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

std::vector<Bits> bf16(std::initializer_list<float> values) {
  std::vector<Bits> result;
  for (float value : values)
    result.push_back(ref::toBF16(value));
  return result;
}

void conversionTests() {
  require(ref::toBF16(1.0F) == 0x3f80 && ref::toBF16(-1.0F) == 0xbf80,
          "BF16 canonical one conversion changed");
  require(ref::toBF16(std::bit_cast<float>(0x3f808000U)) == 0x3f80,
          "BF16 even lower halfway tie was not rounded down");
  require(ref::toBF16(std::bit_cast<float>(0x3f818000U)) == 0x3f82,
          "BF16 odd lower halfway tie was not rounded up");
  require(ref::toBF16(std::bit_cast<float>(0xbf808000U)) == 0xbf80,
          "negative BF16 halfway tie changed");
  require(ref::toBF16(0.0F) == 0 && ref::toBF16(-0.0F) == 0x8000,
          "BF16 signed zero was lost");
  require(std::signbit(ref::fromBF16(0x8000)),
          "BF16 signed zero did not decode");
  require(ref::toBF16(std::numeric_limits<float>::infinity()) == 0x7f80 &&
              ref::toBF16(-std::numeric_limits<float>::infinity()) == 0xff80 &&
              std::isnan(ref::fromBF16(ref::toBF16(
                  std::bit_cast<float>(0x7f800001U)))),
          "BF16 infinity/NaN conversion changed");
  for (uint32_t word = 0; word <= 0xffff; ++word) {
    const auto bits = static_cast<Bits>(word);
    if (std::isfinite(ref::fromBF16(bits)))
      require(ref::toBF16(ref::fromBF16(bits)) == bits,
              "a finite BF16 value did not round-trip");
  }
}

void routeFixtureTests() {
  // Evaluated on installed MLX 0.32.2, explicitly on mx.cpu. Probability and
  // normalization values are independent of the native kernel implementation.
  const auto logits = bf16({-3.1F, -1.0F, 0.2F, 1.0F, 3.1F});
  const auto selected = ref::route(logits, 1, 5, 3);
  require(selected.probabilities ==
              bf16({0.00171661376953125F, 0.013916015625F, 0.046142578125F,
                    0.10302734375F, 0.8359375F}),
          "MLX full probability fixture changed");
  require(selected.indices == std::vector<int64_t>({4, 3, 2}),
          "canonical route order changed");
  require(selected.scores == bf16({0.84765625F, 0.1044921875F, 0.046875F}),
          "MLX normalized score fixture changed");

  const auto unnormalized = ref::route(logits, 1, 5, 3, false);
  require(unnormalized.indices == selected.indices &&
              unnormalized.scores ==
                  bf16({0.8359375F, 0.10302734375F, 0.046142578125F}),
          "disabled normalization did not retain selected probabilities");

  const auto uniform = ref::route(bf16({0, 0, 0, 0, 0}), 1, 5, 3);
  require(uniform.probabilities == std::vector<Bits>(5, ref::toBF16(0.2001953125F)) &&
              uniform.scores == std::vector<Bits>(3, ref::toBF16(0.33203125F)),
          "BF16 selected sum storage boundary was removed");
  require(uniform.indices == std::vector<int64_t>({0, 1, 2}),
          "uniform probability ties did not prefer low expert IDs");
}

void fullExpertTests() {
  std::vector<Bits> logits(512, ref::toBF16(0.0F));
  auto selected = ref::route(logits, 1);
  for (uint32_t slot = 0; slot < 10; ++slot)
    require(selected.indices[slot] == slot &&
                selected.scores[slot] == ref::toBF16(0.10009765625F),
            "512-expert top-10 uniform fixture changed");
  require(selected.probabilities ==
              std::vector<Bits>(512, ref::toBF16(1.0F / 512.0F)),
          "512-expert softmax is not uniform");

  // Distinct input logits may become equal BF16 probabilities. Selection must
  // sort those probabilities, not recover the raw-logit ranking after rounding.
  logits.back() = ref::toBF16(0.0001F);
  selected = ref::route(logits, 1);
  require(selected.probabilities.front() == selected.probabilities.back(),
          "probability-collapse fixture no longer forms a BF16 tie");
  require(selected.indices == std::vector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          "router selected a raw-logit winner through a BF16 probability tie");

  for (uint32_t expert = 0; expert < 512; ++expert)
    logits[expert] = ref::toBF16((float(expert) - 256.0F) / 64.0F);
  selected = ref::route(logits, 1);
  for (uint32_t slot = 0; slot < 10; ++slot)
    require(selected.indices[slot] == 511 - slot,
            "full-512 expert search missed a high-index winner");

  const auto single = ref::route(logits, 1, 512, 1);
  require(single.indices.front() == 511 && single.scores.front() == ref::toBF16(1.0F),
          "single-selection normalization changed");
}

void extremeRouteTests() {
  // Finite BF16 extremes overflow a subtraction to -inf; exp(-inf)=0 is an
  // admissible softmax intermediate, and max subtraction keeps its sum finite.
  std::vector<Bits> logits(512, 0xff7f);
  logits[493] = 0x7f7f;
  const auto selected = ref::route(logits, 1);
  require(selected.indices.front() == 493 && selected.scores.front() == 0x3f80,
          "finite extreme logits overflowed the stable softmax");
  for (uint32_t slot = 1; slot < 10; ++slot)
    require(selected.indices[slot] == slot - 1 && selected.scores[slot] == 0,
            "underflowed probability ties were not deterministic");

  logits.assign(512, 0x7f7f);
  const auto uniform = ref::route(logits, 1);
  require(uniform.scores == std::vector<Bits>(10, ref::toBF16(0.10009765625F)),
          "all-large finite logits did not produce uniform scores");
}

void activationFixtureTests() {
  // Root's actual MLX 0.32.2 Metal export: moe-mlx-golden, width640_rows1
  // and adversarial fixtures. These ordinary values match precise CPU libm
  // exp and compiled GPU fast exp; not every BF16 rounding threshold does.
  const auto gate = bf16({-7.96875F, -1.0F, 0.0F, 1.0F, 0.75F, 0.5F, -0.5F});
  const auto up = bf16({-4.625F, 1.0F, 7.0F, 1.0F, 2.0F, 1.0F, 1.0F});
  const auto output = ref::swiglu(gate, up, 1, 7);
  require(output == bf16({0.0126953125F, -0.26953125F, 0.0F, 0.73046875F,
                         1.015625F, 0.3125F, -0.1884765625F}),
          "actual MLX Metal ordinary SwiGLU fixture changed");
  const float g = -7.96875F;
  const Bits floatSilu = ref::toBF16(g / (1.0F + std::exp(-g)));
  const Bits missingSigmoidBoundary =
      ref::toBF16(ref::fromBF16(floatSilu) * -4.625F);
  require(output.front() != missingSigmoidBoundary,
          "activation fixture does not catch missing BF16 sigmoid storage");
  require(ref::sigmoidBF16(ref::toBF16(0.75F)) == ref::toBF16(0.6796875F),
          "shared gate sigmoid fixture changed");
  require(ref::sigmoidBF16(ref::toBF16(0.5F)) == ref::toBF16(0.625F) &&
              ref::sigmoidBF16(ref::toBF16(-0.5F)) == ref::toBF16(0.376953125F),
          "actual MLX Metal sigmoid internal BF16 boundaries changed");
  const float oldTail = 1.0F / (1.0F + std::exp(0.5F));
  require(ref::toBF16(1.0F - oldTail) == ref::toBF16(0.62109375F) &&
              ref::sigmoidBF16(ref::toBF16(0.5F)) != ref::toBF16(1.0F - oldTail),
          "sigmoid fixture does not distinguish the old F32-tail implementation");
  // Actual uncompiled/unary GPU sigmoid uses precise exp. Compiled SwiGLU
  // uses fast exp; its gate=-6.84375, up=.125 golden output is BF16 0xba70.
  // The full GPU golden test validates compiled behavior at this threshold.
  require(ref::sigmoidBF16(ref::toBF16(-6.84375F)) == Bits{0x3a8b},
          "actual precise unary sigmoid threshold fixture changed");
  require(ref::sigmoidBF16(0xff7f) == 0 && ref::sigmoidBF16(0x7f7f) == 0x3f80,
          "finite extreme sigmoid overflow handling changed");
  const auto signedZero = ref::swiglu(std::array<Bits, 1>{0x8000},
                                    std::array<Bits, 1>{0x3f80}, 1, 1);
  require(signedZero.front() == 0x8000, "SwiGLU lost negative zero");
}

void combineFixtureTests() {
  const auto routed = bf16({1.25F, -2.5F, 3.5F, -4.75F, -2.0F, 8.0F});
  const std::array<int64_t, 3> ids{4, 3, 2};
  const auto scores = bf16({0.25F, 0.25F, 0.5F});
  const auto shared = bf16({2.0F, -3.0F});
  const auto gate = bf16({0.75F});
  require(ref::combine(routed, ids, scores, shared, gate, 1, 2, 5, 3) ==
              bf16({1.546875F, 0.15625F}),
          "canonical expert/shared arithmetic fixture changed");

  // Root's actual MLX GPU discriminatory_rows4 golden. The qualified column
  // reduction returns 1.015625, versus serial BF16=1 and F32=1.03125.
  std::vector<Bits> terms(10, ref::toBF16(1.0F / 256.0F));
  terms.front() = ref::toBF16(1.0F);
  std::vector<int64_t> tenIDs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<Bits> oneScores(10, ref::toBF16(1.0F));
  require(ref::combine(terms, tenIDs, oneScores, bf16({0}), bf16({0}), 1, 1) ==
              bf16({1.015625F}),
          "qualified eight-partial BF16 expert reduction changed");
  const auto cancellation = bf16({256, 1, -256, 0, 0, 0, 0, 0, 1, 0});
  require(ref::combine(cancellation, tenIDs, oneScores, bf16({0}), bf16({0}), 1, 1) == bf16({0}),
          "actual MLX GPU cancellation fixture changed");
  float floatSum = 0.0F;
  Bits serialSum = 0;
  for (Bits value : cancellation) {
    floatSum += ref::fromBF16(value);
    serialSum = ref::toBF16(ref::fromBF16(serialSum) + ref::fromBF16(value));
  }
  require(floatSum == 2.0F && ref::fromBF16(serialSum) == 1.0F,
          "cancellation fixture does not distinguish all three accumulation orders");

  // Expert IDs describe already-gathered rows. Relabeling valid IDs must not
  // accidentally index a full expert bank during the combine operation.
  const std::array<int64_t, 3> relabeled{0, 1, 4};
  require(ref::combine(routed, ids, scores, shared, gate, 1, 2, 5, 3) ==
              ref::combine(routed, relabeled, scores, shared, gate, 1, 2, 5, 3),
          "combine re-gathered slot-ordered expert rows using expert IDs");
}

void selectedScoreReductionTests() {
  // Root's actual MLX GPU selected_sum_expected.bin stores 1.0 and .03125
  // for these two rows, versus F32 sums 1.03125 and .0322265625 respectively.
  auto selected = bf16({1, 1.0F / 256, 1.0F / 256, 1.0F / 256, 1.0F / 256,
                       1.0F / 256, 1.0F / 256, 1.0F / 256, 1.0F / 256, 1.0F / 256});
  require(ref::normalizeSelectedScores(selected) == selected,
          "selected scores did not use the qualified BF16 serial denominator");
  selected = bf16({.03125F, .0001F, .0001F, .0001F, .0001F,
                   .0001F, .0001F, .0001F, .0001F, .0001F});
  auto expected = std::vector<Bits>(10, ref::toBF16(.003204345703125F));
  expected[0] = ref::toBF16(1.0F);
  require(ref::normalizeSelectedScores(selected) == expected,
          "small selected scores did not round at each BF16 addition");
  const auto unchanged = ref::normalizeSelectedScores(bf16({1}));
  require(unchanged == bf16({1}), "one selected score did not normalize to one");
  rejects([&] { ref::normalizeSelectedScores({}); }, "empty selected scores were accepted");
  rejects([&] { ref::normalizeSelectedScores(std::vector<Bits>(11, ref::toBF16(.1F))); },
          "oversized selected-score row was accepted");
  rejects([&] { ref::normalizeSelectedScores(bf16({-.5F})); },
          "negative selected score was accepted");
  rejects<std::overflow_error>([&] { ref::normalizeSelectedScores(bf16({0, 0})); },
                               "zero selected-score denominator was accepted");
}

template <class T>
std::vector<T> permuteRows(std::span<const T> input, uint32_t rowSize,
                         std::span<const uint32_t> permutation) {
  std::vector<T> result;
  result.reserve(input.size());
  for (uint32_t row : permutation)
    result.insert(result.end(), input.begin() + uint64_t{row} * rowSize,
                  input.begin() + uint64_t{row + 1} * rowSize);
  return result;
}

void rowPermutationTests() {
  constexpr uint32_t rows = 5;
  constexpr uint32_t experts = 512;
  constexpr uint32_t selections = 10;
  const std::array<uint32_t, rows> permutation{3, 1, 4, 0, 2};
  std::vector<Bits> logits(rows * experts);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t expert = 0; expert < experts; ++expert)
      logits[row * experts + expert] =
          ref::toBF16(float((expert * 73 + row * 11) % 997) / 97.0F - 5.0F);
  const auto first = ref::route(logits, rows);
  const auto reordered = ref::route(permuteRows<Bits>(logits, experts, permutation), rows);
  require(reordered.indices == permuteRows<int64_t>(first.indices, selections, permutation) &&
              reordered.scores == permuteRows<Bits>(first.scores, selections, permutation) &&
              reordered.probabilities == permuteRows<Bits>(first.probabilities, experts, permutation),
          "routing changed under row permutation");
  // Both actual routed-expert and shared-expert widths are exercised.
  for (const uint32_t width : {640U, 2560U}) {
    const uint32_t activationSelections = width == 640 ? selections : 1;
    std::vector<Bits> gate(uint64_t{rows} * activationSelections * width);
    std::vector<Bits> up(gate.size());
    for (uint64_t index = 0; index < gate.size(); ++index) {
      gate[index] = ref::toBF16(float((index * 23) % 991) / 131.0F - 3.0F);
      up[index] = ref::toBF16(float((index * 41) % 601) / 79.0F - 2.0F);
    }
    const auto activated = ref::swiglu(gate, up, rows, width, activationSelections);
    const auto reorderedActivation = ref::swiglu(
        permuteRows<Bits>(gate, activationSelections * width, permutation),
        permuteRows<Bits>(up, activationSelections * width, permutation),
        rows, width, activationSelections);
    require(reorderedActivation == permuteRows<Bits>(
                activated, activationSelections * width, permutation),
            "SwiGLU changed under row permutation");
  }

  constexpr uint32_t width = 2560;
  std::vector<Bits> routed(rows * selections * width);
  std::vector<Bits> shared(rows * width);
  std::vector<Bits> sharedGate(rows);
  for (uint64_t index = 0; index < routed.size(); ++index)
    routed[index] = ref::toBF16(float((index * 37) % 977) / 61.0F - 8.0F);
  for (uint64_t index = 0; index < shared.size(); ++index)
    shared[index] = ref::toBF16(float((index * 59) % 997) / 71.0F - 6.0F);
  for (uint32_t row = 0; row < rows; ++row)
    sharedGate[row] = ref::toBF16(float(row) - 2.0F);
  const auto output = ref::combine(routed, first.indices, first.scores, shared,
                                   sharedGate, rows, width);
  const auto reorderedOutput = ref::combine(
      permuteRows<Bits>(routed, selections * width, permutation), reordered.indices,
      reordered.scores, permuteRows<Bits>(shared, width, permutation),
      permuteRows<Bits>(sharedGate, 1, permutation), rows, width);
  require(reorderedOutput == permuteRows<Bits>(output, width, permutation),
          "expert combination changed under row permutation");
}

void malformedInputTests() {
  const std::vector<Bits> good(512, 0);
  for (const auto [rows, experts, selections] :
       {std::array<uint32_t, 3>{0, 512, 10}, {2049, 512, 10}, {1, 0, 10},
        {1, 513, 10}, {1, 512, 0}, {1, 512, 11}, {1, 5, 10}})
    rejects([&] { ref::route(good, rows, experts, selections); },
            "out-of-bounds routing parameters were accepted");
  rejects([&] { ref::route(std::span<const Bits>(good).first(511), 1); },
          "truncated logit rows were accepted");
  for (Bits invalid : {Bits{0x7f80}, Bits{0xff80}, Bits{0x7fc1}}) {
    auto logits = good;
    logits[511] = invalid;
    rejects([&] { ref::route(logits, 1); },
            "nonfinite logit outside top-10 search was accepted");
    rejects([&] { ref::sigmoidBF16(invalid); }, "nonfinite sigmoid input was accepted");
    rejects([&] { ref::swiglu(std::array{invalid}, std::array<Bits, 1>{0}, 1, 1); },
            "nonfinite activation gate was accepted");
    rejects([&] { ref::swiglu(std::array<Bits, 1>{0}, std::array{invalid}, 1, 1); },
            "nonfinite activation up was accepted");
  }
  rejects([&] { ref::swiglu(bf16({0}), bf16({0}), 1, 2561); },
          "oversized activation width was accepted");
  rejects([&] { ref::swiglu(bf16({0}), bf16({0, 0}), 1, 1); },
          "activation input size mismatch was accepted");
  rejects<std::overflow_error>(
      [&] { ref::swiglu(std::array<Bits, 1>{0x7f7f}, bf16({2}), 1, 1); },
      "activation output overflow was accepted");

  const auto terms = bf16({1, 2});
  const auto scores = bf16({0.5F, 0.5F});
  const auto shared = bf16({0});
  const auto gate = bf16({0});
  const std::array<int64_t, 2> ids{0, 1};
  for (int64_t invalid : {-1LL, 512LL, std::numeric_limits<int64_t>::max()}) {
    auto badIDs = ids;
    badIDs[1] = invalid;
    rejects([&] { ref::combine(terms, badIDs, scores, shared, gate, 1, 1, 512, 2); },
            "malformed expert ID was accepted");
  }
  rejects([&] { ref::combine(terms, std::span(ids).first(1), scores, shared, gate, 1, 1, 512, 2); },
          "truncated expert IDs were accepted");
  const std::array<int64_t, 2> repeatedIDs{0, 0};
  rejects([&] { ref::combine(terms, repeatedIDs, scores, shared, gate, 1, 1, 512, 2); },
          "repeated expert IDs within one row were accepted");
  for (float outOfRange : {-0.00390625F, 1.0078125F}) {
    auto badScores = scores;
    badScores[1] = ref::toBF16(outOfRange);
    rejects([&] { ref::combine(terms, ids, badScores, shared, gate, 1, 1, 512, 2); },
            "route score outside [0,1] was accepted");
  }
  for (uint32_t plane = 0; plane < 4; ++plane) {
    auto badTerms = terms;
    auto badScores = scores;
    auto badShared = shared;
    auto badGate = gate;
    if (plane == 0) badTerms[1] = 0x7fc0;
    if (plane == 1) badScores[1] = 0x7f80;
    if (plane == 2) badShared[0] = 0xff80;
    if (plane == 3) badGate[0] = 0x7fc1;
    rejects([&] { ref::combine(badTerms, ids, badScores, badShared, badGate, 1, 1, 512, 2); },
            "nonfinite combine plane was accepted");
  }
  rejects<std::overflow_error>(
      [&] { ref::combine(std::array<Bits, 2>{0x7f7f, 0x7f7f}, ids,
                        bf16({1, 1}), shared, gate, 1, 1, 512, 2); },
      "expert accumulation overflow was accepted");
}

void maximumRowTests() {
  const std::vector<Bits> logits(uint64_t{2048} * 512, 0);
  const auto selected = ref::route(logits, 2048);
  require(selected.indices.size() == uint64_t{2048} * 10 &&
              selected.probabilities.size() == uint64_t{2048} * 512,
          "maximum-row output geometry changed");
  for (uint32_t row = 0; row < 2048; ++row)
    for (uint32_t slot = 0; slot < 10; ++slot)
      require(selected.indices[uint64_t{row} * 10 + slot] == slot &&
                  selected.scores[uint64_t{row} * 10 + slot] == ref::toBF16(0.10009765625F),
              "maximum-row routing crossed a row boundary");
}

} // namespace

int main() {
  try {
    conversionTests();
    routeFixtureTests();
    fullExpertTests();
    extremeRouteTests();
    activationFixtureTests();
    combineFixtureTests();
    selectedScoreReductionTests();
    rowPermutationTests();
    malformedInputTests();
    maximumRowTests();
    std::cout << "Flash MoE CPU reference tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Flash MoE CPU reference tests failed: " << error.what() << '\n';
    return 1;
  }
}
