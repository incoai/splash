// Inkling-2-Mini's target (model/Inkling2LoopTarget.cpp) on a small GGUF of
// the family in the engine's shapes (Inkling2Fixtures.hpp
// inkling2EngineLayout: 12 weight layers in blocks of 3, layers 3-8 looped
// three times, 24 applications), its values seeded, with a DFlash2 draft of
// zero weights, against the fp32 reference forward of the same file
// (dev/tools/inkling2_reference.py), whose outputs the fixture directory
// holds. The draft's captures read the rows passed into each application
// (the family's QwenTargetDimensions::draftReadsLayerInputs), the output of
// the application before it, so a run whose draft reads those ids checks
// those outputs. Within bounds, as under shader validation too:
// - the output rows of every application another reads (all but each loop
//   pass's last before a restart and the last application) and the final
//   norm's rows, which the draft reads at the id past the applications, at
//   the sampled rows of a 1100-token prompt, past the window (1024 keys) and
//   the relative extent, and the logits of the prompt's last row;
// - the same prompt in two commands, split inside a page, the second reading
//   the state and the K/V the first left;
// - four lanes at once, a prefill command of four sequences and a decode step
//   of the four, which run;
// - the prompt and the four lanes with the routed experts streamed through
//   the smallest store (--expert-memory), whose every load evicts another
//   block's experts and whose decode step predicts the next application's;
// - a draft that passes its rows through to the head: its candidates,
//   unary scores and first position's probabilities against the fixture's
//   tables (its token rows through the table's norm, its logits' finish).
// What only an uninstrumented run holds (`bitwise`), as shader validation
// compiles the kernels apart (qwen4_kernels_test.mm):
// - bit for bit: the prompt run in passes of one application each, a command
//   stopping between every two applications, as the whole command; a lane
//   beside other prompts of the same lengths; four lanes of one prompt alike;
//   the prompt and the four lanes with streamed experts, as with resident
//   ones;
// - each lane of the four against its prompt alone, within bounds.
//
//   inkling2-target METALLIB FIXTURE_DIR [bitwise]
//   inkling2-target METALLIB --write-fixture DIR
//
// FIXTURE_DIR holds expected.f32 and expected.json (the reference's run, the
// commands that regenerate it and the file's hash, dev/tools/
// inkling2_target_fixture.py); --write-fixture writes the GGUF and the
// prompt's ids (prompt.u32) the reference reads, to DIR.

#include "Inkling2Fixtures.hpp"
#include "TestCheckpoint.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "metal/MetalBackend.hpp"
#include "model/Bfloat16.hpp"
#include "model/ModelFactory.hpp"
#include "model/ModelSizing.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "model/SsdReaders.hpp"
#include "ops/PageStorage.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

// The fixture's values and prompt, which the reference's run read: a seed,
// and 1100 ids by a formula, past the 1024 keys a sliding row reads.
constexpr uint32_t kSeed = 2611;
constexpr uint32_t kPromptRows = 1100;
// The rows compared: the first, either side of a page and of the window's
// and extent's edge, and the last.
constexpr std::array<uint32_t, 10> kSampledRows{0, 1, 2, 3, 511, 512, 1022, 1023, 1024, 1099};
// The most ids a run captures (a draft's target layers).
constexpr uint32_t kCaptured = model::ExecutionLimits::maximumDraftTargetLayers;
// Where the two-command run splits the prompt: inside a page.
constexpr uint32_t kSplitRow = 600;
// The lanes of the several-lane case: a decode step of them stages each lane's
// rows in whole KV blocks, which a step of one lane never shows.
constexpr uint32_t kLanes = 4;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<uint32_t> prompt(uint32_t tokens) {
  std::vector<uint32_t> ids(kPromptRows);
  for (uint32_t i = 0; i < kPromptRows; ++i) ids[i] = (i * 37 + 11) % tokens;
  return ids;
}

// The fixture's draft, of one layer, reading the target layer ids `ids`.
model::DFlashDraftLayout draftReading(const model::Inkling2LoopLayout &layout, std::span<const uint32_t> ids) {
  return model::draftLayout(layout.hiddenSize, layout.vocabularySize, 1, 256, 10'000.0F, ids);
}

// The kCaptured ids from `first` on.
std::vector<uint32_t> captureIds(uint32_t first) {
  std::vector<uint32_t> ids(kCaptured);
  std::iota(ids.begin(), ids.end(), first);
  return ids;
}

// A model root of the fixture: its GGUF in target/ and its draft's weights,
// all zero, in draft/, as an installation holds them; a run's descriptor
// names the ids its draft reads (descriptor).
struct Fixture final {
  std::filesystem::path root;
  model::Inkling2LoopLayout layout = gguf_fixtures::inkling2EngineLayout();

  // Without vision, its layout's output width is the target's, as a model
  // root's descriptor states it (ModelDescriptor.mm describeSourceModel).
  [[nodiscard]] model::ModelDescriptor descriptor(std::span<const uint32_t> ids) const {
    return model::makeModelDescriptor("inkling2-target-test", layout, draftReading(layout, ids),
                                      {.outputHiddenSize = layout.hiddenSize}, model::TargetSource::Gguf,
                                      model::VisionSource::None);
  }
};

Fixture writeFixture(const std::filesystem::path &root) {
  Fixture fixture;
  fixture.root = root;
  std::filesystem::create_directories(root / "target");
  gguf_fixtures::writeInkling2Gguf(root / "target" / "inkling2-fixture.gguf", fixture.layout, kSeed);
  // Every run's draft reads kCaptured ids, whose weights are of one shape.
  test::writeSyntheticShard(root / "draft" / "model.safetensors",
                            test::draftTensors(draftReading(fixture.layout, captureIds(1))));
  return fixture;
}

// A model root of the fixture whose draft passes its own rows through to the
// head: all its weights zero but its final norm's, `finalNorm` (bf16 bits),
// so that its layers add nothing to its rows and its head reads its rows of
// its tokens (the token table's rows through the table's norm, as the
// family's drafts read them) through that norm.
std::filesystem::path writePassThroughRoot(const Fixture &fixture, const std::filesystem::path &root,
                                           std::span<const uint16_t> finalNorm) {
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "target");
  std::filesystem::copy_file(fixture.root / "target" / "inkling2-fixture.gguf",
                             root / "target" / "inkling2-fixture.gguf");
  std::vector<test::SyntheticTensor> tensors = test::draftTensors(draftReading(fixture.layout, captureIds(1)));
  for (test::SyntheticTensor &tensor : tensors)
    if (tensor.name == "norm.weight") {
      tensor.data.resize(finalNorm.size_bytes());
      std::memcpy(tensor.data.data(), finalNorm.data(), finalNorm.size_bytes());
    }
  test::writeSyntheticShard(root / "draft" / "model.safetensors", tensors);
  return root;
}

// What a prompt's run leaves: the output rows of the applications it captured
// (each prefill row's, the applications' hidden values in their order) and
// its last row's logits.
struct Run final {
  std::vector<float> captured;
  std::vector<float> logits;
};

// A runtime over the loaded model, its KV and state for one request.
class Harness final {
public:
  Harness(metal::MetalBackend &backend, model::LoadedModel &model)
      : backend_(backend), model_(model), operators_(backend.capabilities(), ops::Int8Activations::Off),
        governor_(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1, queryHostMemory, 0),
        claim_(governor_, governor_.ledger().open({.cls = ClaimClass::Floor})),
        vocabulary_(std::visit([](const auto &layout) { return layout.vocabularySize; }, model.descriptor.target)) {
    const kv::Layout layout = model.targetKvLayout(kv::Format::BFloat16);
    const uint32_t extent = layout.maximumExtentPages();
    lanePages_ = (kPromptRows + 2 * kv::kPageTokens - 1) / kv::kPageTokens;
    const uint32_t lanePages = lanePages_;
    const uint32_t pageCount = (kLanes * lanePages + extent - 1) / extent * extent;
    pages_ = std::make_unique<kv::PageStorage>(backend, governor_.allocationAdmission(), layout, pageCount, extent);
    for (uint32_t index = 0; index < pageCount / extent; ++index)
      require(static_cast<bool>(pages_->allocateExtent(index)), "could not allocate the KV extents");
    states_ = std::make_unique<model::QwenStateStorage>(backend, governor_.allocationAdmission(),
                                                        model.stateLayout(ops::GdnStateFormat::Float32), nullptr);
    pageTable_.resize(lanePages);
    std::iota(pageTable_.begin(), pageTable_.end(), 0u);
  }

  // Each prompt on its own lane, their rows in one prefill command (a chunk of
  // several sequences), then one decode step of every lane: each lane's
  // logits after its prompt and after the token its prompt selected.
  std::vector<std::array<std::vector<float>, 2>> runLanes(const std::vector<std::vector<uint32_t>> &prompts) {
    model::RuntimeContext context{backend_, model_, *pages_, *states_, operators_};
    model::Runtime executor(context);
    const auto lanes = static_cast<uint32_t>(prompts.size());
    std::vector<std::vector<uint32_t>> tables(lanes);
    BatchPlan prefill{.kind = WorkKind::Prefill, .decodeStage = DecodeStage::Regular};
    std::vector<ModelBatchItem> items;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      EngineRequest request;
      request.id = ++id_;
      request.prompt = prompts[lane];
      request.maxNewTokens = 2;
      executor.beginColdRequest(request.modelView(), lane);
      tables[lane].resize(lanePages_);
      std::iota(tables[lane].begin(), tables[lane].end(), lane * lanePages_);
      const auto rows = static_cast<uint32_t>(prompts[lane].size());
      prefill.items.push_back({request.id, rows, 0});
      ModelBatchItem &item = items.emplace_back(ModelBatchItem{request.id, 0, rows, tables[lane], 1});
      item.inputTokens = prompts[lane];
    }
    std::vector<std::array<std::vector<float>, 2>> result(lanes);
    const std::vector<ModelStepResult> prefilled = executor.prefill(prefill, items);
    require(prefilled.size() == lanes, "a prefill of several sequences returned another count of results");
    BatchPlan decode{.kind = WorkKind::Decode, .decodeStage = DecodeStage::Regular};
    std::vector<ModelBatchItem> steps;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(prefilled[lane].failure.empty() && prefilled[lane].firstToken, "a lane's prefill failed");
      const std::span<const float> logits = model::RuntimeInstrumentation::lastLogits(executor, lane);
      result[lane][0].assign(logits.begin(), logits.begin() + vocabulary_);
      decode.items.push_back({items[lane].requestId, 0});
      steps.push_back(ModelBatchItem{items[lane].requestId, prompts[lane].size(), 0, tables[lane], 1});
    }
    const std::vector<ModelStepResult> decoded = executor.decode(decode, steps);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(decoded[lane].failure.empty(), "a lane's decode step failed: " + decoded[lane].failure);
      const std::span<const float> logits = model::RuntimeInstrumentation::lastLogits(executor, lane);
      result[lane][1].assign(logits.begin(), logits.begin() + vocabulary_);
      result[lane][0].push_back(static_cast<float>(*prefilled[lane].firstToken));
      executor.end(items[lane].requestId);
    }
    return result;
  }

  // The draft's lattice of the first decode step after the prompt `ids`, in
  // one command (RuntimeInstrumentation::lastDraftLattice).
  model::RuntimeInstrumentation::DraftLattice latticeAfter(std::span<const uint32_t> ids) {
    model::RuntimeContext context{backend_, model_, *pages_, *states_, operators_};
    model::Runtime executor(context);
    EngineRequest request;
    request.id = ++id_;
    request.prompt.assign(ids.begin(), ids.end());
    request.maxNewTokens = 8;
    executor.beginColdRequest(request.modelView(), 0);
    const auto rows = static_cast<uint32_t>(ids.size());
    BatchPlan prefill{.kind = WorkKind::Prefill, .items = {{request.id, rows, 0}}, .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{request.id, 0, rows, pageTable_, 1};
    item.inputTokens = ids;
    const std::vector<ModelStepResult> prefilled = executor.prefill(prefill, std::span(&item, 1));
    require(prefilled.size() == 1 && prefilled[0].failure.empty() && prefilled[0].firstToken,
            "the prompt's prefill failed");
    BatchPlan decode{.kind = WorkKind::Decode, .items = {{request.id, 0}}, .decodeStage = DecodeStage::Regular};
    ModelBatchItem step{request.id, rows, 0, pageTable_, 1};
    const std::vector<ModelStepResult> decoded = executor.decode(decode, std::span(&step, 1));
    require(decoded.size() == 1 && decoded[0].failure.empty(), "the decode step failed");
    model::RuntimeInstrumentation::DraftLattice lattice = model::RuntimeInstrumentation::lastDraftLattice(executor, 0, 1);
    executor.end(request.id);
    return lattice;
  }

  // The prompt in commands of `commands` rows each (their sum the prompt's),
  // each in `passes` passes over the applications (BatchPlan::passes).
  Run run(std::span<const uint32_t> ids, std::span<const uint32_t> commands, uint32_t passes) {
    model::RuntimeContext context{backend_, model_, *pages_, *states_, operators_};
    model::Runtime executor(context);
    EngineRequest request;
    request.id = ++id_;
    request.prompt.assign(ids.begin(), ids.end());
    request.maxNewTokens = 1;
    executor.beginColdRequest(request.modelView(), 0);
    uint32_t offset = 0;
    for (const uint32_t count : commands) {
      BatchPlan plan{.kind = WorkKind::Prefill, .items = {{request.id, count, offset}},
                            .decodeStage = DecodeStage::Regular, .passes = passes};
      ModelBatchItem item{request.id, offset, count, pageTable_, 1};
      item.inputTokens = ids.subspan(offset, count);
      for (uint32_t pass = 0; pass < passes; ++pass) {
        const std::vector<ModelStepResult> results = executor.prefill(plan, std::span(&item, 1));
        if (results.size() != 1 || !results[0].failure.empty())
          throw std::runtime_error("prefill failed: " +
                                   (results.empty() ? std::string("no result") : results[0].failure));
        if (pass + 1 < passes) plan = followingPass(plan, executor.commandLimits().layers);
      }
      offset += count;
    }
    Run result;
    const std::span<const uint16_t> captured = model::RuntimeInstrumentation::lastCapturedRows(executor, 0);
    result.captured.resize(captured.size());
    std::ranges::transform(captured, result.captured.begin(), model::widenBfloat16);
    const std::span<const float> logits = model::RuntimeInstrumentation::lastLogits(executor, 0);
    result.logits.assign(logits.begin(), logits.end());
    executor.end(request.id);
    return result;
  }

private:
  metal::MetalBackend &backend_;
  model::LoadedModel &model_;
  ops::ExecutionPlans operators_;
  MemoryGovernor governor_;
  MemoryGovernor::Scope claim_;
  std::unique_ptr<kv::PageStorage> pages_;
  std::unique_ptr<model::QwenStateStorage> states_;
  std::vector<uint32_t> pageTable_;
  uint32_t vocabulary_ = 0;
  uint32_t lanePages_ = 0;
  uint64_t id_ = 0;
};

// The reference's outputs: each application's output at the sampled rows,
// the last row's logits over the tokens, and the final norm's rows (the
// head's input, which the draft captures at the id past the applications) at
// the sampled rows.
struct Expected final {
  std::vector<float> outputs;
  std::vector<float> logits;
  std::vector<float> finalNorm;

  // An application's output rows, or the final norm's past the applications.
  [[nodiscard]] std::span<const float> rows(uint32_t application, uint32_t applications, uint32_t hidden) const {
    const uint64_t count = uint64_t{kSampledRows.size()} * hidden;
    return application == applications ? std::span<const float>(finalNorm)
                                       : std::span<const float>(outputs).subspan(application * count, count);
  }
};

Expected readExpected(const std::filesystem::path &directory, const model::Inkling2LoopLayout &layout) {
  std::ifstream file(directory / "expected.f32", std::ios::binary);
  const uint64_t rows = uint64_t{kSampledRows.size()} * layout.hiddenSize;
  Expected expected{std::vector<float>(layout.layers * rows), std::vector<float>(layout.tokens()),
                    std::vector<float>(rows)};
  for (std::vector<float> *values : {&expected.outputs, &expected.logits, &expected.finalNorm})
    file.read(reinterpret_cast<char *>(values->data()), static_cast<std::streamsize>(values->size() * 4));
  require(file && file.peek() == EOF, "expected.f32 is not the fixture's outputs (" + directory.string() + ")");
  return expected;
}

// Each application's output against the fp32 reference, the engine's most
// exact path (bf16 KV and activations): mean |engine - reference| over mean
// |reference| at the sampled rows, at most its bound, twice what an M5 Pro
// measured (2026-11-21); an M6, and both under shader validation, stay within
// them. The references' own distance (the source's bf16 rounding against
// fp32) grows on this fixture as the engine's does, 0.008 at application 0 to
// 0.18 at 23, as the loop's later passes take the earlier ones' rounding
// through their messages; a wrong plane or gate moves an application by far
// more.
constexpr std::array<double, 24> kOutputBounds{
    0.0161, 0.0205, 0.0223, 0.0281, 0.0296, 0.0321, 0.0404, 0.0456, 0.0459, 0.0770, 0.0775, 0.0961,
    0.1665, 0.1846, 0.1844, 0.3472, 0.3059, 0.3280, 0.3765, 0.4390, 0.4584, 0.6012, 0.6439, 0.6661};
// The final norm's rows, the final mix of the stack and the last
// application's output through output_norm: twice what an M5 Pro measured
// (0.216, 2026-11-22).
constexpr double kFinalNormBound = 0.4324;
// The last row's logits against the reference's, and a run's against another
// run of the same rows in other commands (KL). Rounding moved them by at most
// 0.0028 (an M5 Pro and an M6, with and without shader validation: a
// command's rows choose its tiles, and validation compiles the kernels
// apart). A wrong stack plane or message gate moved the logits by 0.04 and
// more; a second command that read nothing of the first by 0.19, and one
// after other tokens in the three rows before the split by 0.06.
constexpr double kLogitsBound = 0.02;
// A lane of a several-lane command against its prompt alone, uninstrumented:
// its prefill row measured at most 0.00049 and its decode row 0.109 (a
// one-token lane's), on an M5 Pro and an M6 alike; twice those. The lanes'
// joint prefill takes other tiles than a prompt's own command, and its 1,000
// rows run the message gates' float projections where a prompt of at most
// MessageGate::kFusedRows rows alone, and every decode step, runs the fused
// gates (ops/MessageGate.hpp): other summation orders of the same roundings,
// which this fixture carries through its 24 applications into the lane's KV
// and states, which its decode step reads.
constexpr double kLanePrefillBound = 0.001, kLaneDecodeBound = 0.22;

// A captured slot's rows against the reference's (`want`, the sampled
// rows'): the mean absolute difference over the mean absolute value.
double relativeError(const Run &run, uint32_t slot, std::span<const float> want, uint32_t hidden) {
  double difference = 0.0, magnitude = 0.0;
  for (uint32_t sample = 0; sample < kSampledRows.size(); ++sample)
    for (uint32_t column = 0; column < hidden; ++column) {
      const float got = run.captured[(uint64_t{kSampledRows[sample]} * kCaptured + slot) * hidden + column];
      difference += std::abs(got - want[uint64_t{sample} * hidden + column]);
      magnitude += std::abs(want[uint64_t{sample} * hidden + column]);
    }
  return difference / magnitude;
}

double klDivergence(std::span<const float> reference, std::span<const float> engine) {
  const auto logSoftmax = [](std::span<const float> logits) {
    const double top = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float logit : logits) sum += std::exp(logit - top);
    std::vector<double> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) result[i] = logits[i] - top - std::log(sum);
    return result;
  };
  const std::vector<double> p = logSoftmax(reference), q = logSoftmax(engine);
  double kl = 0.0;
  for (size_t i = 0; i < p.size(); ++i) kl += std::exp(p[i]) * (p[i] - q[i]);
  return kl;
}

// The fixture loaded, with its prompt and the reference's outputs, its draft
// reading the ids from 1 on (`model`); and loaded again with its routed
// experts streamed through the smallest store, which holds a block's
// (`streamed`). A run that captures other ids loads its own (capturing).
struct Loaded final {
  Loaded(const char *metallib, const std::filesystem::path &fixtureDirectory)
      : root(std::filesystem::temp_directory_path() / "inkling2-target-test"), fixture(written(root)),
        expected(readExpected(fixtureDirectory, fixture.layout)), ids(prompt(fixture.layout.tokens())),
        backend(metallib), model(capturing(1)),
        streamed(model::loadModel(
            backend, root, fixture.descriptor(captureIds(1)),
            {.expertMemoryBytes = model::streamableWeights(root, fixture.descriptor(captureIds(1)),
                                                           {.streamExperts = true})
                                      .expertMinimumBytes,
             .readers = std::make_shared<model::SsdReaders>(model::kSsdReaders)})) {
    require(streamed.expertStore() != nullptr, "the fixture's routed experts did not stream");
  }
  ~Loaded() { std::filesystem::remove_all(root); }

  static Fixture written(const std::filesystem::path &root) {
    std::filesystem::remove_all(root);
    return writeFixture(root);
  }

  // The fixture whose draft reads the kCaptured ids from `first` on.
  [[nodiscard]] model::LoadedModel capturing(uint32_t first) {
    return model::loadModel(backend, root, fixture.descriptor(captureIds(first)));
  }

  std::filesystem::path root;
  Fixture fixture;
  Expected expected;
  std::vector<uint32_t> ids;
  metal::MetalBackend backend;
  model::LoadedModel model;
  model::LoadedModel streamed;
};

// The application whose output rows application `application` reads: the one
// before it, or, where a loop pass restarts, the one before the loop, whose
// output the loop's first stack plane holds.
uint32_t writer(const model::Inkling2LoopSchedule &schedule, uint32_t application) {
  uint32_t source = application - 1;
  while (schedule.output(source) != schedule.input(application)) --source;
  return source;
}

// The lanes' prompts: three of a row and the rest of the prompt's first 1000.
std::vector<std::vector<uint32_t>> lanePrompts(const std::vector<uint32_t> &ids) {
  return {{ids[0]}, {ids[1]}, {ids[2]}, std::vector<uint32_t>(ids.begin() + 3, ids.begin() + 1000)};
}

// The draft's rows and logits finish against the fixture's tables, through
// a draft that passes its rows through (writePassThroughRoot): its token
// rows reach its head as the token table's rows through the table's norm
// (bf16), then through its final norm, so every proposal position's
// candidates are the 16 tokens of the most logits of the mask token's row,
// none past the head's tokens; its unary scores those logits; its edges none,
// its codebooks and selector projection zero; and its first position's
// probabilities the softmax of its scores softcapped at the head's softcap.
// Its final norm's weights favour the head's first padding row, the row's
// values times that row's, so that the row would lead the logits it leaves
// out. A lost norm moves the logits (the table's norm weights are not
// uniform), a lost finish proposes the padding row or leaves the scores
// uncapped.
int runDraftRows(Loaded &loaded) {
  const model::Inkling2LoopLayout &layout = loaded.fixture.layout;
  const uint32_t hidden = layout.hiddenSize, tokens = layout.tokens(), candidates = SPLASH_DRAFT_CANDIDATES;
  const std::vector<gguf_fixtures::Tensor> tensors = gguf_fixtures::inkling2Tensors(layout, kSeed);
  const auto tensor = [&](std::string_view name) -> const gguf_fixtures::Tensor & {
    const auto found = std::ranges::find_if(tensors, [&](const gguf_fixtures::Tensor &t) { return t.name == name; });
    require(found != tensors.end(), "the fixture has no " + std::string(name));
    return *found;
  };
  const auto tableRow = [&](const gguf_fixtures::Tensor &table, uint32_t row) {
    std::vector<float> values(hidden);
    const uint32_t bytes = gguf_reference::rowBytes(gguf_reference::Q80, hidden);
    gguf_reference::rowValues(gguf_reference::Q80, table.data.data() + uint64_t{row} * bytes, hidden, values.data());
    return values;
  };
  const auto bf16 = [](float value) { return gguf_reference::bf2f(gguf_reference::f2bf(value)); };
  // An RMS norm in fp32 into bf16 rows, as the engine's.
  const auto normed = [&](std::vector<float> rows, const auto &weight) {
    double squares = 0.0;
    for (const float value : rows) squares += double(value) * value;
    const float scale = 1.0F / std::sqrt(float(squares / hidden) + SPLASH_RMS_EPSILON);
    for (uint32_t i = 0; i < hidden; ++i) rows[i] = bf16(rows[i] * scale * weight(i));
    return rows;
  };
  const gguf_fixtures::Tensor &norm = tensor("token_embd_norm.weight"), &head = tensor("output.weight");
  std::vector<float> row = tableRow(tensor("token_embd.weight"), layout.maskToken);
  for (float &value : row) value = bf16(value);
  row = normed(row, [&](uint32_t i) { return gguf_fixtures::loadFloat(norm.data, i); });
  // The final norm's weights: the row's values times the first padding
  // row's, of a root mean square of 1, in bf16.
  const std::vector<float> padding = tableRow(head, tokens);
  std::vector<float> weights(hidden);
  double squares = 0.0;
  for (uint32_t i = 0; i < hidden; ++i) {
    weights[i] = row[i] * padding[i];
    squares += double(weights[i]) * weights[i];
  }
  std::vector<uint16_t> finalNorm(hidden);
  for (uint32_t i = 0; i < hidden; ++i) {
    finalNorm[i] = gguf_reference::f2bf(float(weights[i] / std::sqrt(squares / hidden)));
    weights[i] = gguf_reference::bf2f(finalNorm[i]);
  }
  row = normed(row, [&](uint32_t i) { return weights[i]; });
  std::vector<float> logits(layout.vocabularySize);
  for (uint32_t token = 0; token < layout.vocabularySize; ++token) {
    const std::vector<float> values = tableRow(head, token);
    double sum = 0.0;
    for (uint32_t i = 0; i < hidden; ++i) sum += double(row[i]) * values[i];
    logits[token] = float(sum);
  }
  std::vector<uint32_t> order(tokens);
  std::iota(order.begin(), order.end(), 0u);
  std::ranges::sort(order, [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
  require(logits[tokens] > logits[order[0]], "the first padding row does not lead the pass-through draft's logits");
  // Splash's lattice after a prompt of 100 of the fixture's ids.
  const std::filesystem::path root = loaded.root.string() + "-pass-through";
  writePassThroughRoot(loaded.fixture, root, finalNorm);
  model::LoadedModel model = model::loadModel(loaded.backend, root, loaded.fixture.descriptor(captureIds(1)));
  Harness harness(loaded.backend, model);
  const model::RuntimeInstrumentation::DraftLattice lattice =
      harness.latticeAfter(std::span(loaded.ids).first(100));
  std::filesystem::remove_all(root);
  const uint32_t positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  require(lattice.candidates.size() == positions * candidates && lattice.first.size() == candidates,
          "the draft's lattice is not a cycle's");
  // A candidate past the 16th logit only within the rounding of the 16th.
  double unaryDifference = 0.0;
  bool sameCandidates = true, noPadding = true;
  for (uint32_t position = 0; position < positions; ++position)
    for (uint32_t i = 0; i < candidates; ++i) {
      const uint32_t token = lattice.candidates[position * candidates + i];
      noPadding = noPadding && token < tokens;
      if (token >= tokens) continue;
      sameCandidates = sameCandidates && logits[token] >= logits[order[candidates - 1]] - 1e-3F;
      unaryDifference = std::max(unaryDifference, double(std::abs(lattice.unary[position * candidates + i] - logits[token])));
    }
  const bool noEdges = std::ranges::all_of(lattice.edges, [](float edge) { return edge == 0.0F; });
  // The first position's probabilities from its own unary scores, softcapped
  // and not.
  const auto softmax = [&](float softcap) {
    std::vector<double> p(candidates);
    double top = -INFINITY, sum = 0.0;
    for (uint32_t i = 0; i < candidates; ++i) {
      const double unary = lattice.unary[i];
      p[i] = softcap > 0.0F ? softcap * std::tanh(unary / softcap) : unary;
      top = std::max(top, p[i]);
    }
    for (double &value : p) sum += value = std::exp(value - top);
    for (double &value : p) value /= sum;
    return p;
  };
  const auto distance = [&](const std::vector<double> &p) {
    double most = 0.0;
    for (uint32_t i = 0; i < candidates; ++i) most = std::max(most, std::abs(p[i] - lattice.first[i]));
    return most;
  };
  const double capped = distance(softmax(layout.logitSoftcap)), uncapped = distance(softmax(0.0F));
  const bool within = sameCandidates && noPadding && noEdges && unaryDifference <= 5e-3 && capped <= 1e-5;
  std::printf("%sa pass-through draft: candidates the mask row's 16 most logits %s, none past %u tokens %s, "
              "unary within %.2g of them, no edges %s, first position's probabilities %.2g from its softcapped "
              "scores (%.2g uncapped)\n",
              within ? "" : "FAIL: ", sameCandidates ? "yes" : "no", tokens, noPadding ? "yes" : "no", unaryDifference,
              noEdges ? "yes" : "no", capped, uncapped);
  return within ? 0 : 1;
}

// The comparisons within bounds.
int runComparisons(Loaded &loaded) {
  const model::Inkling2LoopLayout &layout = loaded.fixture.layout;
  const std::vector<uint32_t> &ids = loaded.ids;
  const uint32_t hidden = layout.hiddenSize, applications = layout.layers, tokens = layout.tokens();
  const std::array<uint32_t, 1> whole{kPromptRows};
  const model::Inkling2LoopSchedule schedule = layout.schedule();
  int failures = 0;
  std::optional<Run> wholeRun;
  // The ids 1 to the applications: the rows passed into each application, the
  // output of the application before it; the last, the final norm's.
  require(applications % kCaptured == 0, "the fixture's ids are not whole runs of a draft's");
  for (uint32_t first = 1; first + kCaptured - 1 <= applications; first += kCaptured) {
    model::LoadedModel model = loaded.capturing(first);
    Harness harness(loaded.backend, model);
    const Run run = harness.run(ids, whole, 1);
    for (uint32_t slot = 0; slot < kCaptured; ++slot) {
      const uint32_t id = first + slot, application = id == applications ? id : writer(schedule, id);
      const double error =
          relativeError(run, slot, loaded.expected.rows(application, applications, hidden), hidden);
      const double bound = id == applications ? kFinalNormBound : kOutputBounds[application];
      const bool within = error <= bound;
      if (id == applications)
        std::printf("%sid %2u, the final norm's rows: mean|d|/mean|ref| %.5f (bound %.4f)\n", within ? "" : "FAIL: ",
                    id, error, bound);
      else
        std::printf("%sid %2u, application %2u's output: mean|d|/mean|ref| %.5f (bound %.4f)\n",
                    within ? "" : "FAIL: ", id, application, error, bound);
      failures += !within;
    }
    if (first == 1) wholeRun = run;
  }
  const std::span<const float> wholeLogits = std::span(wholeRun->logits).first(tokens);
  const double kl = klDivergence(loaded.expected.logits, wholeLogits);
  std::printf("%slast row's logits: KL %.3g (bound %.3g)\n", kl <= kLogitsBound ? "" : "FAIL: ", kl, kLogitsBound);
  failures += kl > kLogitsBound;
  // The prompt in two commands: the second's rows read the convolutions'
  // state and the K/V the first left.
  Harness harness(loaded.backend, loaded.model);
  const std::array<uint32_t, 2> split{kSplitRow, kPromptRows - kSplitRow};
  const Run two = harness.run(ids, split, 1);
  const double splitKl = klDivergence(wholeLogits, std::span(two.logits).first(tokens));
  std::printf("%stwo commands against one: KL %.3g (bound %.3g)\n", splitKl <= kLogitsBound ? "" : "FAIL: ", splitKl,
              kLogitsBound);
  failures += splitKl > kLogitsBound;
  // Several lanes at once: a prefill command of four sequences, then a
  // decode step of the four. Under validation they run, within their
  // buffers; the uninstrumented run compares them.
  const std::vector<std::array<std::vector<float>, 2>> lanes = harness.runLanes(lanePrompts(ids));
  bool finite = true;
  for (const std::array<std::vector<float>, 2> &lane : lanes)
    for (const std::vector<float> &logits : lane)
      for (uint32_t token = 0; token < tokens; ++token) finite = finite && std::isfinite(logits[token]);
  std::printf("%sfour lanes: a prefill of four sequences and a decode step of the four ran\n", finite ? "" : "FAIL: ");
  failures += !finite;
  // The same with streamed experts: the prompt's chunk reads every block's
  // experts ahead of its next application's, and the lanes' decode step
  // predicts that block's routes with its selection bias.
  Harness streamed(loaded.backend, loaded.streamed);
  const Run streamedRun = streamed.run(ids, whole, 1);
  const double streamedKl = klDivergence(loaded.expected.logits, std::span(streamedRun.logits).first(tokens));
  std::printf("%sstreamed experts, last row's logits: KL %.3g (bound %.3g)\n",
              streamedKl <= kLogitsBound ? "" : "FAIL: ", streamedKl, kLogitsBound);
  failures += streamedKl > kLogitsBound;
  finite = true;
  for (const std::array<std::vector<float>, 2> &lane : streamed.runLanes(lanePrompts(ids)))
    for (const std::vector<float> &logits : lane)
      for (uint32_t token = 0; token < tokens; ++token) finite = finite && std::isfinite(logits[token]);
  std::printf("%sstreamed experts, four lanes: a prefill of four sequences and a decode step of the four ran\n",
              finite ? "" : "FAIL: ");
  failures += !finite;
  return failures + runDraftRows(loaded);
}

// What an uninstrumented run holds: the runs that compute the same bits, and
// the lanes against their prompts alone.
int runBitwise(Loaded &loaded) {
  const std::vector<uint32_t> &ids = loaded.ids;
  const uint32_t applications = loaded.fixture.layout.layers, tokens = loaded.fixture.layout.tokens();
  const std::array<uint32_t, 1> whole{kPromptRows};
  int failures = 0;
  const auto check = [&](bool same, const std::string &what) {
    std::printf("%s%s\n", same ? "" : "FAIL: ", what.c_str());
    failures += !same;
  };
  // Stopped between every two applications, the command computes what it
  // computes whole: the same rows through the same kernels, the final norm's
  // among them.
  for (uint32_t first = 1; first + kCaptured - 1 <= applications; first += kCaptured) {
    model::LoadedModel model = loaded.capturing(first);
    Harness harness(loaded.backend, model);
    const Run run = harness.run(ids, whole, 1), cut = harness.run(ids, whole, applications);
    check(cut.captured == run.captured && cut.logits == run.logits,
          "the rows ids " + std::to_string(first) + "-" + std::to_string(first + kCaptured - 1) +
              " read and the logits in passes of one application: the whole command's bits");
  }
  Harness harness(loaded.backend, loaded.model);
  const std::vector<std::vector<uint32_t>> prompts = lanePrompts(ids);
  const std::vector<std::array<std::vector<float>, 2>> lanes = harness.runLanes(prompts);
  // Other prompts of the same lengths beside lane 0's: a lane's rows read its
  // own state, pages and rows alone, and a command of the same rows runs the
  // same tiles, so lane 0's logits are the same bits whatever the lanes beside
  // it hold.
  std::vector<std::vector<uint32_t>> others = prompts;
  for (uint32_t lane = 1; lane < kLanes; ++lane)
    for (uint32_t &id : others[lane]) id = (id + 101) % tokens;
  check(harness.runLanes(others)[0] == lanes[0], "lane 0 beside other prompts of the same lengths: the same bits");
  // The same prompt on every lane: every lane computes the same logits.
  const std::vector<std::array<std::vector<float>, 2>> same =
      harness.runLanes(std::vector<std::vector<uint32_t>>(kLanes, prompts[1]));
  bool alike = true;
  for (uint32_t lane = 1; lane < kLanes; ++lane) alike = alike && same[lane] == same[0];
  check(alike, "four lanes of one prompt: the same bits on every lane");
  // Each lane's logits against its own prompt's run alone, after its prompt
  // and after the token it selected.
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    std::vector<uint32_t> extended = prompts[lane];
    const std::array<uint32_t, 1> alone{static_cast<uint32_t>(extended.size())};
    const Run single = harness.run(extended, alone, 1);
    extended.push_back(static_cast<uint32_t>(lanes[lane][0].back()));
    const std::array<uint32_t, 1> longer{static_cast<uint32_t>(extended.size())};
    const Run next = harness.run(extended, longer, 1);
    const double prefillKl =
        klDivergence(std::span(single.logits).first(tokens), std::span(lanes[lane][0]).first(tokens));
    const double decodeKl = klDivergence(std::span(next.logits).first(tokens), std::span(lanes[lane][1]).first(tokens));
    const bool within = prefillKl <= kLanePrefillBound && decodeKl <= kLaneDecodeBound;
    std::printf("%slane %u of %u against its own prompt alone: prefill KL %.3g (bound %.3g), decode step KL %.3g "
                "(bound %.3g)\n",
                within ? "" : "FAIL: ", lane, kLanes, prefillKl, kLanePrefillBound, decodeKl, kLaneDecodeBound);
    failures += !within;
  }
  // Streamed experts compute the resident ones' bits (expert_store_test):
  // the prompt in one command, and the lanes' prefill and decode step.
  Harness streamed(loaded.backend, loaded.streamed);
  check(streamed.run(ids, whole, 1).logits == harness.run(ids, whole, 1).logits,
        "streamed experts, the prompt: the resident experts' bits");
  check(streamed.runLanes(prompts) == lanes, "streamed experts, four lanes: the resident experts' bits");
  return failures;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 4 && std::string_view(argv[2]) == "--write-fixture") {
        const std::filesystem::path directory(argv[3]);
        const Fixture fixture = writeFixture(directory);
        const std::vector<uint32_t> ids = prompt(fixture.layout.tokens());
        std::ofstream file(directory / "prompt.u32", std::ios::binary);
        file.write(reinterpret_cast<const char *>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
        require(static_cast<bool>(file), "cannot write the prompt");
        std::printf("wrote %s/target/inkling2-fixture.gguf and %s/prompt.u32\n", argv[3], argv[3]);
        return 0;
      }
      const bool bitwise = argc == 4 && std::string_view(argv[3]) == "bitwise";
      if (argc != 3 && !bitwise) {
        std::cerr << "usage: inkling2-target METALLIB FIXTURE_DIR [bitwise] | METALLIB --write-fixture DIR\n";
        return 2;
      }
      // Shader validation compiles the kernels apart, which can move a value
      // by its last bit (qwen4_kernels_test.mm).
      if (bitwise && std::getenv("MTL_SHADER_VALIDATION"))
        throw std::runtime_error("bitwise runs without shader validation");
      Loaded loaded(argv[1], argv[2]);
      const int failures = bitwise ? runBitwise(loaded) : runComparisons(loaded);
      std::printf(failures ? "inkling2-target: %d failures\n" : "inkling2-target: passed\n", failures);
      return failures ? 1 : 0;
    } catch (const std::exception &error) {
      std::cerr << "inkling2-target: " << error.what() << '\n';
      return 1;
    }
  }
}
