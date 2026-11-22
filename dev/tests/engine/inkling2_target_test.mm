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
//   pass's last before a restart and the last application) at the sampled
//   rows of a 1100-token prompt, past the window (1024 keys) and the
//   relative extent, and the logits of the prompt's last row;
// - the same prompt in two commands, split inside a page, the second reading
//   the state and the K/V the first left;
// - four lanes at once, a prefill command of four sequences and a decode step
//   of the four, which run;
// - the prompt and the four lanes with the routed experts streamed through
//   the smallest store (--expert-memory), whose every load evicts another
//   block's experts and whose decode step predicts the next application's.
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
// then the last row's logits over the tokens.
struct Expected final {
  std::vector<float> outputs;
  std::vector<float> logits;
};

Expected readExpected(const std::filesystem::path &directory, const model::Inkling2LoopLayout &layout) {
  std::ifstream file(directory / "expected.f32", std::ios::binary);
  const uint64_t outputs = uint64_t{layout.layers} * kSampledRows.size() * layout.hiddenSize;
  Expected expected{std::vector<float>(outputs), std::vector<float>(layout.tokens())};
  file.read(reinterpret_cast<char *>(expected.outputs.data()), static_cast<std::streamsize>(outputs * 4));
  file.read(reinterpret_cast<char *>(expected.logits.data()),
            static_cast<std::streamsize>(expected.logits.size() * 4));
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
// The last row's logits against the reference's, and a run's against another
// run of the same rows in other commands (KL). Rounding moved them by at most
// 0.0028 (an M5 Pro and an M6, with and without shader validation: a
// command's rows choose its tiles, and validation compiles the kernels
// apart). A wrong stack plane or message gate moved the logits by 0.04 and
// more; a second command that read nothing of the first by 0.19, and one
// after other tokens in the three rows before the split by 0.06.
constexpr double kLogitsBound = 0.02;
// A lane of a several-lane command against its prompt alone, uninstrumented:
// its prefill row measured at most 0.00049 and its decode row 0.108 (a
// one-token lane's), on an M5 Pro and an M6 alike; twice those. The lanes'
// joint prefill takes other tiles than a prompt's own command, whose other
// summation order this fixture carries through its 24 applications into the
// lane's KV and states, which its decode step reads.
constexpr double kLanePrefillBound = 0.001, kLaneDecodeBound = 0.22;

// An application's output against the reference: the mean absolute
// difference over the mean absolute value at the sampled rows.
double relativeError(const Run &run, uint32_t slot, const Expected &expected, uint32_t application,
                     uint32_t hidden) {
  double difference = 0.0, magnitude = 0.0;
  for (uint32_t sample = 0; sample < kSampledRows.size(); ++sample)
    for (uint32_t column = 0; column < hidden; ++column) {
      const float got = run.captured[(uint64_t{kSampledRows[sample]} * kCaptured + slot) * hidden + column];
      const float want = expected.outputs[(uint64_t{application} * kSampledRows.size() + sample) * hidden + column];
      difference += std::abs(got - want);
      magnitude += std::abs(want);
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
  // output of the application before it; the last, the final norm's, compares
  // with nothing the reference wrote but the logits it makes.
  require(applications % kCaptured == 0, "the fixture's ids are not whole runs of a draft's");
  for (uint32_t first = 1; first + kCaptured - 1 <= applications; first += kCaptured) {
    model::LoadedModel model = loaded.capturing(first);
    Harness harness(loaded.backend, model);
    const Run run = harness.run(ids, whole, 1);
    for (uint32_t slot = 0; slot < kCaptured && first + slot < applications; ++slot) {
      const uint32_t application = writer(schedule, first + slot);
      const double error = relativeError(run, slot, loaded.expected, application, hidden);
      const bool within = error <= kOutputBounds[application];
      std::printf("%sid %2u, application %2u's output: mean|d|/mean|ref| %.5f (bound %.4f)\n",
                  within ? "" : "FAIL: ", first + slot, application, error, kOutputBounds[application]);
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
  return failures;
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
