// Inkling 2 Mini's DSpark draft (model/DSparkDraft.hpp) on its real weights
// against a llama.cpp reference: the target features of each position of a
// prompt and its greedy continuation, saved by the reference, are fed to the
// draft's context; at every anchor of the continuation (teacher forcing, as
// greedy speculative decoding sees it) the draft's block and Markov chain
// propose seven tokens, whose chain, candidates and logits are compared with
// the reference's, and whose accepted tokens (those the target's greedy text
// continues with) with the reference's. The draft GGUF, the target GGUF
// (its head, embedding and embedding norm) and the reference directory are
// arguments: the test runs where they are, and checks
//  - each lane's f_rnn row against the reference's of its last position;
//  - the first anchor's final normed hidden rows against the reference's
//    tensors, and its base logits' top 16 against the reference's;
//  - the first proposal of every anchor, and its top 16 candidates and their
//    logits along the chain while it agrees with the reference's;
//  - the mean accepted tokens per anchor within 15% of the reference's.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/DSpark.h"
#include "model/DSparkDraft.hpp"
#include "model/GgufPreparation.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightStore.hpp"
#include "ops/ExecutionPlans.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace splash;
using metal::MetalBackend;
using metal::MetalBuffer;
using test::require;

constexpr uint32_t kHidden = 2304;
constexpr uint32_t kVocabulary = 201024;
constexpr uint32_t kTargetLayers = 48;
constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kSteps = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kPairs = SPLASH_DRAFT_HEAD_DIMENSION / 2;
constexpr uint32_t kPrefillRows = SPLASH_PREFILL_TOKEN_BUDGET;

uint16_t bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float widen(uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); }

std::vector<float> readFloats(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open " + path.string());
  file.seekg(0, std::ios::end);
  const auto bytes = static_cast<size_t>(file.tellg());
  std::vector<float> values(bytes / sizeof(float));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
  return values;
}

MetalBuffer buffer(MetalBackend &backend, uint64_t bytes) {
  return test::sharedBuffer(backend, std::max<uint64_t>(bytes, 16));
}

ops::LinearScratch scratchOf(MetalBackend &backend, const ops::LinearScratchSize &size) {
  return {buffer(backend, size.input), buffer(backend, size.sums), buffer(backend, size.partials),
          buffer(backend, size.counters), buffer(backend, size.rotated)};
}

// The RoPE rows of `rows` positions from `first` on, NeoX pairs of the
// draft's base.
void ropeRows(float *cosines, float *sines, uint32_t first, uint32_t rows, float theta) {
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t pair = 0; pair < kPairs; ++pair) {
      const double angle = double(first + row) * std::pow(double(theta), -2.0 * pair / (2.0 * kPairs));
      cosines[row * kPairs + pair] = static_cast<float>(std::cos(angle));
      sines[row * kPairs + pair] = static_cast<float>(std::sin(angle));
    }
}

void run(MetalBackend &backend, const metal::CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// The target's pieces the draft reads: its head and embedding norm (as the
// target loads them, model/Inkling.cpp) and its Q8_0 token rows, which the
// test decodes on the host.
struct Target {
  std::unique_ptr<model::OpenGguf> gguf;
  model::WeightImages images;
  ops::Projection head;
  ops::NormWeights embeddingNorm;
  const model::GgufTensor *table = nullptr;

  Target(MetalBackend &backend, const std::filesystem::path &path)
      : gguf(std::make_unique<model::OpenGguf>(std::vector<std::filesystem::path>{path})),
        images(backend, "dspark-reference-target") {
    const model::GgufFile &file = gguf->file();
    model::gguf::ImageBuilder builder("head.bin", kTargetLayers, 2);
    const model::GgufTensor &output = file.require("output.weight");
    const model::GgufTensor &norm = file.require("token_embd_norm.weight");
    table = &file.require("token_embd.weight");
    require(output.type == model::ggml::kQ8_0 && table->type == model::ggml::kQ8_0 &&
                norm.type == model::ggml::kF32,
            "the reference target is the Q8_0 GGUF");
    builder.copy({norm.name, norm.type, norm.offset, 1, norm.bytes, {}, &file.source(norm.file)});
    const uint64_t rowBytes = model::ggufRowBytes(kQuantFormats[GGUF_FMT_Q80], kHidden);
    model::gguf::Repack repack = builder.planes(GGUF_FMT_Q80, kVocabulary, kHidden, output.name);
    repack.sources.push_back({output.name, output.type, output.offset, kVocabulary, rowBytes, {}, &file.source(output.file)});
    builder.repack(std::move(repack));
    auto plan = std::make_shared<const model::gguf::Image>(builder.finish());
    model::WeightFile image = model::loadImage(backend, images, "target/", plan, [] {});
    embeddingNorm = model::readNorm(image, kHidden, true, "embedding-norm");
    head = model::readBlockProjection(image, kVocabulary, kHidden, "logits");
    head.destination = ops::FloatOutput::Float32;
    image.finish();
  }

  // Token `token`'s embedding row, Q8_0 decoded.
  std::vector<float> embedding(uint32_t token) const {
    const uint64_t blocks = kHidden / 32, rowBytes = blocks * 34;
    std::vector<uint8_t> row(rowBytes);
    gguf->file().source(table->file).readData(table->offset + uint64_t{token} * rowBytes, row);
    std::vector<float> values(kHidden);
    for (uint64_t block = 0; block < blocks; ++block) {
      uint16_t half;
      std::memcpy(&half, &row[block * 34], 2);
      const float scale = static_cast<float>(std::bit_cast<_Float16>(half));
      for (uint32_t i = 0; i < 32; ++i)
        values[block * 32 + i] = scale * static_cast<float>(static_cast<int8_t>(row[block * 34 + 2 + i]));
    }
    return values;
  }
};

struct Prompt {
  std::string name;
  uint32_t prompt = 0;
  std::vector<uint32_t> tokens;
  NSArray *anchors = nil;
  std::vector<float> features, frnn;
};

Prompt readPrompt(const std::filesystem::path &directory, const std::string &name) {
  Prompt result;
  result.name = name;
  NSData *data = [NSData dataWithContentsOfFile:@((directory / (name + ".json")).c_str())];
  if (!data) throw std::runtime_error("no reference " + name);
  NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
  result.prompt = [json[@"n_prompt"] unsignedIntValue];
  for (NSNumber *token in json[@"tokens"]) result.tokens.push_back(token.unsignedIntValue);
  result.anchors = json[@"anchors"];
  result.features = readFloats(directory / (name + ".features.f32"));
  result.frnn = readFloats(directory / (name + ".f_rnn.f32"));
  return result;
}

double relativeError(std::span<const float> got, std::span<const float> want) {
  double difference = 0.0, norm = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    difference += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
    norm += double(want[i]) * want[i];
  }
  return std::sqrt(difference / std::max(norm, 1e-30));
}

struct Totals {
  uint32_t anchors = 0, firstEqual = 0, chainsEqual = 0;
  double accepted = 0.0, referenceAccepted = 0.0;
  uint32_t overlap = 0, compared = 0;
  double logitError = 0.0;
};

void runPrompt(MetalBackend &backend, const ops::ExecutionPlans &operators, const model::DSparkDraft &draft,
               const model::DFlashDraftWeights &weights, const Target &target, const Prompt &prompt,
               const std::filesystem::path &reference, Totals &totals) {
  const model::DFlashDraftLayout &layout = weights.layout;
  const auto tracker = std::make_shared<model::StateAllocationTracker>();
  const model::DraftStateLayout state = layout.stateLayout();
  model::DFlashDraftRing ring(backend, tracker, state, "dspark-test-ring");
  model::DFlashContextWindow window(backend, tracker, state, "dspark-test-window");
  const uint64_t positions = prompt.features.size() / (uint64_t{SPLASH_DSPARK_TAPS} * kHidden);

  // Prefill buffers.
  const ops::Linear &linear = operators.draftLinear();
  const ops::LinearScratch prefillScratch =
      scratchOf(backend, linear.scratchSize({layout.contextKvSize(), kHidden}, kPrefillRows));
  model::DFlashPrefillBuffers prefill{
      buffer(backend, uint64_t{kPrefillRows} * SPLASH_DSPARK_TAPS * kHidden * 2), prefillScratch,
      buffer(backend, uint64_t{kPrefillRows} * kHidden * 2), buffer(backend, uint64_t{kPrefillRows} * 3 * kHidden * 2),
      buffer(backend, uint64_t{kPrefillRows} * layout.contextKvSize() * 2),
      buffer(backend, uint64_t{kPrefillRows} * kPairs * 4), buffer(backend, uint64_t{kPrefillRows} * kPairs * 4)};

  // Decode buffers, one lane.
  const auto plan = operators.draftAttention(layout.attentionShape(), 1);
  const ops::DraftAttentionWorkspace attention = plan.workspace();
  ops::LinearScratchSize decodeSize;
  for (const ops::ProjectionShape shape :
       {ops::ProjectionShape{layout.qkvSize, kHidden}, {kHidden, layout.attentionSize},
        {layout.intermediateSize, kHidden}, {kHidden, layout.intermediateSize}, {layout.contextKvSize(), kHidden}})
    decodeSize.include(linear.scratchSize(shape, kRows));
  decodeSize.include(operators.decodeLinear().scratchSize({kVocabulary, kHidden}, kRows));
  model::DFlashDecodeBuffers decode;
  decode.linearScratch = scratchOf(backend, decodeSize);
  decode.hidden = {buffer(backend, kRows * kHidden * 2), buffer(backend, kRows * kHidden * 2)};
  decode.normalized = buffer(backend, kRows * kHidden * 2);
  decode.proposalQkv = buffer(backend, attention.qkvBytes);
  decode.attention = buffer(backend, attention.groupedQueriesBytes);
  decode.residual = buffer(backend, kRows * kHidden * 2);
  decode.intermediate = buffer(backend, uint64_t{kRows} * layout.intermediateSize * 2);
  decode.finalHidden = buffer(backend, kRows * kHidden * 2);
  decode.logits = buffer(backend, uint64_t{kRows} * kVocabulary * 4);
  decode.candidates = buffer(backend, uint64_t{kSteps} * kCandidates * 4);
  decode.unary = buffer(backend, uint64_t{kSteps} * kCandidates * 4);
  decode.queryKeys = buffer(backend, attention.queryKeysBytes);
  decode.queryValues = buffer(backend, attention.queryValuesBytes);
  decode.ropeCos = buffer(backend, SPLASH_TARGET_VERIFY_ROWS * kPairs * 4);
  decode.ropeSin = buffer(backend, SPLASH_TARGET_VERIFY_ROWS * kPairs * 4);
  decode.gateScratch = buffer(backend, operators.gateUpWorkspace({layout.intermediateSize, kHidden}, kRows));
  decode.persistentKeys.resize(layout.layers);
  decode.persistentValues.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    decode.persistentKeys[layer].fill(ring.layers()[layer].keys);
    decode.persistentValues[layer].fill(ring.layers()[layer].values);
  }
  decode.laneAuxiliary.fill(ring.layers().front().auxiliary);

  uint32_t committed = 0;
  bool first = true;
  for (NSDictionary *entry in prompt.anchors) {
    const uint32_t anchorPosition = [entry[@"pos"] unsignedIntValue];
    if (anchorPosition > positions) break;
    // The context of the positions before the anchor.
    if (anchorPosition > committed) {
      const uint32_t rows = anchorPosition - committed;
      require(rows <= kPrefillRows, "a reference prompt fits one prefill chunk");
      auto *captured = static_cast<uint16_t *>(prefill.capturedTargetHidden.contents());
      const float *source = prompt.features.data() + uint64_t{committed} * SPLASH_DSPARK_TAPS * kHidden;
      for (uint64_t i = 0; i < uint64_t{rows} * SPLASH_DSPARK_TAPS * kHidden; ++i) captured[i] = bf16(source[i]);
      ropeRows(static_cast<float *>(prefill.ropeCos.contents()), static_cast<float *>(prefill.ropeSin.contents()),
               committed, rows, layout.rotaryTheta);
      const model::DFlashPrefillSpan span{0, rows, committed, ring.layers(), window.buffer()};
      metal::CommandGraph graph;
      draft.addContextPrefill(graph, prefill, rows, {&span, 1}, {});
      run(backend, graph);
      committed = anchorPosition;
    }
    // f_rnn of the last position.
    {
      const auto *row = static_cast<const uint16_t *>(ring.layers().front().auxiliary.contents());
      std::vector<float> got(kHidden);
      for (uint32_t i = 0; i < kHidden; ++i) got[i] = widen(row[i]);
      const double error =
          relativeError(got, std::span(prompt.frnn).subspan(uint64_t{anchorPosition - 1} * kHidden, kHidden));
      require(error < 0.02, prompt.name + ": f_rnn differs from the reference's by " + std::to_string(error));
    }
    // The block at the anchor.
    const uint32_t anchor = [entry[@"anchor"] unsignedIntValue];
    const std::vector<float> embedding = target.embedding(anchor);
    auto *input = static_cast<uint16_t *>(decode.hidden[0].contents());
    for (uint32_t i = 0; i < kHidden; ++i) input[i] = bf16(embedding[i]);
    ropeRows(static_cast<float *>(decode.ropeCos.contents()), static_cast<float *>(decode.ropeSin.contents()),
             anchorPosition, SPLASH_TARGET_VERIFY_ROWS, layout.rotaryTheta);
    const uint32_t cacheLength = anchorPosition;
    const ops::DraftTreeLane lane{.anchor = anchor, .depth = kSteps};
    const model::DFlashDraftStep step = draft.step({&cacheLength, 1}, {}, {&lane, 1});
    metal::CommandGraph graph;
    draft.addDecode(graph, decode, step);
    run(backend, graph);

    if (first) {
      // The final normed rows against the reference's tensors.
      const std::filesystem::path tensors = reference / (prompt.name + "-tensors");
      const std::vector<float> hidden = readFloats(tensors / "0078_node_131.f32");
      const auto *rows = static_cast<const uint16_t *>(decode.finalHidden.contents());
      std::vector<float> got(kRows * kHidden);
      for (uint32_t i = 0; i < got.size(); ++i) got[i] = widen(rows[i]);
      const double error = relativeError(got, hidden);
      std::cout << "  " << prompt.name << ": final hidden relative error " << error << '\n';
      require(error < 0.03, prompt.name + ": the draft's final rows differ from the reference's");
      // The base logits' top 16 of each proposal row.
      NSArray *base = entry[@"base_top"];
      const auto *logits = static_cast<const float *>(decode.logits.contents());
      uint32_t overlap = 0;
      for (uint32_t row = 1; row < kRows; ++row) {
        std::vector<std::pair<float, uint32_t>> order;
        for (uint32_t token = 0; token < layout.proposedVocabulary; ++token)
          order.emplace_back(-logits[uint64_t{row} * kVocabulary + token], token);
        std::partial_sort(order.begin(), order.begin() + kCandidates, order.end());
        std::set<uint32_t> want;
        for (NSArray *pair in base[row - 1]) want.insert([pair[0] unsignedIntValue]);
        for (uint32_t i = 0; i < kCandidates; ++i) overlap += want.contains(order[i].second);
      }
      std::cout << "  " << prompt.name << ": base top-16 overlap " << overlap << "/" << (kRows - 1) * kCandidates
                << '\n';
      require(overlap >= (kRows - 1) * kCandidates * 3 / 4, prompt.name + ": base logits differ");
      first = false;
    }

    // The chain against the reference's.
    const auto *candidates = static_cast<const uint32_t *>(decode.candidates.contents());
    const auto *unary = static_cast<const float *>(decode.unary.contents());
    NSArray *proposals = entry[@"draft"], *targets = entry[@"target"], *tops = entry[@"top"];
    bool agrees = true;
    uint32_t accepted = 0;
    bool accepting = true;
    for (uint32_t position = 0; position < kSteps; ++position) {
      const uint32_t token = candidates[position * kCandidates];
      if (agrees) {
        NSArray *top = tops[position];
        std::set<uint32_t> want;
        for (NSArray *pair in top) want.insert([pair[0] unsignedIntValue]);
        for (uint32_t i = 0; i < kCandidates; ++i) totals.overlap += want.contains(candidates[position * kCandidates + i]);
        totals.compared += kCandidates;
        totals.logitError = std::max(totals.logitError,
                                     std::fabs(double(unary[position * kCandidates]) - [top[0][1] doubleValue]));
      }
      if (token != [proposals[position] unsignedIntValue]) agrees = false;
      if (position == 0) totals.firstEqual += token == [proposals[0] unsignedIntValue];
      if (accepting && position < targets.count && token == [targets[position] unsignedIntValue]) ++accepted;
      else accepting = false;
    }
    totals.chainsEqual += agrees;
    totals.accepted += accepted;
    totals.referenceAccepted += [entry[@"accepted"] doubleValue];
    ++totals.anchors;
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 5)
        throw std::invalid_argument("usage: dspark-reference METALLIB DRAFT_DIRECTORY TARGET_Q8_0_GGUF REFERENCE_DIRECTORY");
      MetalBackend backend(argv[1]);
      const std::filesystem::path draftDirectory = argv[2], reference = argv[4];
      const ops::ExecutionPlans operators(backend.capabilities(), ops::Int8Activations::On);
      Target target(backend, argv[3]);
      const model::DSparkHeader header = model::readDSparkHeader(draftDirectory);
      const model::DFlashDraftLayout layout = model::dsparkLayout(kHidden, kVocabulary, kTargetLayers, header);
      require(layout.targetLayerCount == 4 && layout.targetLayers[0] == 4 && layout.targetLayers[1] == 22 &&
                  layout.targetLayers[2] == 40 && layout.targetLayers[3] == 48 && layout.slidingLayers == 0b101 &&
                  layout.slidingWindow == 1024 && layout.proposedVocabulary == 200058,
              "the DSpark layout is Inkling 2 Mini's draft's");
      model::WeightImages images(backend, "dspark-reference-draft");
      const model::DFlashDraftWeights weights = model::loadDSparkDraftWeights(backend, images, draftDirectory, layout);
      const model::DSparkDraft draft(weights, backend, operators, target.head, target.embeddingNorm);
      {
        const MetalBuffer rows = buffer(backend, uint64_t{SPLASH_MAXIMUM_BATCH_WIDTH} * SPLASH_TARGET_VERIFY_ROWS *
                                                     kVocabulary * sizeof(float));
        ops::LinearScratchSize size = operators.decodeLinear().scratchSize(
            {kVocabulary, kHidden}, SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS);
        metal::CommandGraph graph;
        draft.addSetup(graph, scratchOf(backend, size), rows,
                       SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS);
        run(backend, graph);
      }
      Totals totals;
      for (const char *name : {"a_short_en", "c_code", "d_tool_chat"}) {
        const Prompt prompt = readPrompt(reference, name);
        Totals one;
        runPrompt(backend, operators, draft, weights, target, prompt, reference, one);
        std::cout << "  " << name << ": anchors " << one.anchors << ", first proposals equal " << one.firstEqual
                  << ", chains equal " << one.chainsEqual << ", accepted per anchor " << one.accepted / one.anchors
                  << " (reference " << one.referenceAccepted / one.anchors << "), top-16 overlap " << one.overlap
                  << "/" << one.compared << ", top logit error " << one.logitError << '\n';
        totals.anchors += one.anchors;
        totals.firstEqual += one.firstEqual;
        totals.chainsEqual += one.chainsEqual;
        totals.accepted += one.accepted;
        totals.referenceAccepted += one.referenceAccepted;
        totals.overlap += one.overlap;
        totals.compared += one.compared;
        totals.logitError = std::max(totals.logitError, one.logitError);
      }
      require(totals.firstEqual * 10 >= totals.anchors * 9, "the first proposals differ from the reference's");
      require(totals.overlap * 10 >= totals.compared * 9, "the candidates differ from the reference's");
      require(std::fabs(totals.accepted - totals.referenceAccepted) <= 0.15 * totals.referenceAccepted,
              "the accepted tokens differ from the reference's");
      std::cout << "dspark_reference_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "dspark_reference_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
