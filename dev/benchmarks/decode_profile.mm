// Per-kernel GPU time attribution for the production executor.
//
//   decode-profile METALLIB MODEL_ROOT [--prompt-tokens N] [--cycles K] [--kv-format int8|bf16]
//                  [--state-format fp32|bf16] [--weights original|auto] [--expert-memory SIZE] [--ple-memory SIZE]
//                  [--dispatch-list FILE] [--verify-rows auto|all] [--step-rows T] [--attribute on|off]
//                  [--census on|off] [--census-cycles N] [--plain R] [--experts on|off] [--chat-tokens FILE]
//                  [--warmups on|off] [--warmup-widths N] [--serving-check on|off]
//   decode-profile METALLIB MODEL_ROOT --images HxW[,HxW...] [--cycles K]
//
// Drives the real model runtime with Metal dispatch profiling enabled, so
// every dispatch of a prefill command and B1 through B4 DFlash cycles
// is replayed as its own command and attributed to its pipeline.
// The fused (unprofiled) GPU time of the same work is reported alongside, so
// the gap between the sum of parts and the fused command shows how much a
// cycle pays in dispatch boundaries rather than kernel work. --dispatch-list
// writes the attributed cycles' dispatches in order, one line each: index,
// pipeline and GPU microseconds, B1's to FILE and Bn's to FILE.Bn. A lane
// whose answer ends begins it again between cycles, untimed. --prompt FILE[,FILE...] takes each lane's prompt
// from a file of token ids, lane i the (i mod n)th of n files, as a served
// width's lanes read different text; --widths N profiles B1 through BN.
// --step-rows has every decode step take T
// rows (at most 16 and at least one a lane: RuntimeInstrumentation::
// forceStepRows), for the cost of a step by its rows; --lane-rows R has
// every width's steps take R rows a lane; --attribute off times
// the fused cycles alone and the prompt's fused prefill. --answer-tokens
// sizes each lane's pages too. --census on adds each width's fused census:
// --census-cycles N (8) cycles are captured, with their routed experts, and
// each pipeline's dispatches of a cycle, in order and nothing else, run as
// one command (3 times, the median), cycle by cycle: what that pipeline's
// dispatches take back to back inside a command, where a dispatch waits for
// the one before it, as in the cycle, against their replay as commands of
// their own, which adds a command's start to every dispatch; each whole
// captured cycle runs too, against the fused cycle (printCensus). Its
// commands write the lanes' rows and states again (the same dispatches over
// the same buffers), so the widths after it run on disturbed states: give
// it one width a run. --plain R makes every decode step a plain step
// (RuntimeInstrumentation::skipDraft): no draft, each lane R rows that all
// hold its anchor, its row a step (R = 1, another engine's plain decode
// step) or copied down a chain (more, another engine's verify of copied
// tokens); --step-rows and --lane-rows give way to it. --experts on runs
// one more cycle a width that copies each MoE visit's selection and prints
// the mean distinct routed experts a visit. --serving-check on (with
// --warmups on) begins serving after the start's warmups, as a start that is
// Ready does (MetalBackend::beginServing), then runs a cycle of every step's
// rows of every width up to --widths, a row a lane to a verify block a lane,
// and prints each step that created pipelines while serving (the backend
// logs each with its creation time): the kernels a cycle runs that the
// start's warmups did not create, any of which fails the check (exit 1). --warmup-widths N warms widths 1 to N
// alone, as a start whose memory affords N widths does (Bootstrap), and
// creates the rest's pipelines with no step run, as that start does
// (Runtime::prepareDecodeBatch), the rest joining while serving. --block-plans on has a streamed-expert
// target's cycles plan a draft's block a lane at their floor
// (RuntimeInstrumentation::planBlocks), for what the larger class's kernels
// cost a cycle.
// Its chat and counting requests are built in in the Qwen families' tokens and Flint's; --chat-tokens FILE gives
// them in a model's own, as `python -m dev.benchmarks.chat_prompts decode-profile MODEL_ROOT FILE` renders them
// with its tokenizer and chat template (make benchmark-decode-profile passes it where they are not the built-in
// ones). Another family's model without it (Inkling's), or a prompt token outside the model's vocabulary, is
// refused before Metal opens.
// With --images it profiles the vision encoder instead, loading only the
// model's vision role: K encodes of a synthetic image per patch grid (height
// x width in patches), fused and then attributed.

#include "engine/Types.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "ops/PageStorage.hpp"
#include "ops/Vision.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/ExpertStore.hpp"
#include "model/SparseRowStore.hpp"
#include "model/SsdReaders.hpp"
#include "model/VisionLoader.hpp"
#include "model/WeightImages.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"
#include "tools/LoadOptions.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using namespace splash;
using namespace splash::engine;
using metal::BackendInstrumentation;

namespace {

struct Attribution final {
  uint64_t dispatches = 0;
  double gpuSeconds = 0.0;
};

using Table = std::map<std::string, Attribution>;

void accumulate(Table &table, std::span<const metal::DispatchTiming> timings) {
  for (const metal::DispatchTiming &timing : timings) {
    Attribution &entry = table[timing.pipelineName];
    ++entry.dispatches;
    entry.gpuSeconds += timing.gpuSeconds;
  }
}

// `fusedGpuSeconds` is already per unit of work; the table accumulated
// `divisor` units.
void print(const std::string &title, const Table &table, double divisor,
           double fusedGpuSeconds) {
  std::vector<std::pair<std::string, Attribution>> rows(table.begin(),
                                                        table.end());
  std::sort(rows.begin(), rows.end(), [](const auto &left, const auto &right) {
    return left.second.gpuSeconds > right.second.gpuSeconds;
  });
  double total = 0.0;
  uint64_t dispatches = 0;
  for (const auto &[_, entry] : rows) {
    total += entry.gpuSeconds;
    dispatches += entry.dispatches;
  }
  std::printf("\n== %s: %.2f ms fused, %.2f ms as %llu separate dispatches ==\n",
              title.c_str(), fusedGpuSeconds * 1e3, total * 1e3 / divisor,
              static_cast<unsigned long long>(dispatches / divisor));
  std::printf("%-44s %9s %10s %6s\n", "pipeline", "count", "gpu_ms", "share");
  for (const auto &[name, entry] : rows) {
    std::printf("%-44s %9.1f %10.3f %5.1f%%\n", name.c_str(),
                entry.dispatches / divisor, entry.gpuSeconds * 1e3 / divisor,
                entry.gpuSeconds / total * 100.0);
  }
}

// A cycle's routed experts, from a copy of each MoE visit's selection
// (moe_route_select's ids, `top_k` a row of its `rows` rows, MoeRouteParams
// first in its parameters): its visits, their rows and routes a row, the
// mean distinct experts a visit, and the mean experts a visit by the routes
// each takes, 1, 2-4, 5-8, 9-16 and more (past 8, an 8-route tile's second
// pass).
constexpr std::array<uint32_t, 4> kRouteBins{1, 4, 8, 16};
struct Routing final {
  size_t visits = 0;
  uint32_t rows = 0, topK = 0;
  double distinct = 0.0;
  std::array<double, kRouteBins.size() + 1> byRoutes{};
};
Routing routingOf(const std::vector<metal::DispatchSnapshot> &snapshots) {
  Routing routing;
  double distinct = 0.0;
  for (const metal::DispatchSnapshot &snapshot : snapshots) {
    uint32_t params[4] = {};
    if (snapshot.parameters.size() < sizeof(params)) throw std::runtime_error("short MoE routing parameters");
    std::memcpy(params, snapshot.parameters.data(), sizeof(params));
    routing.rows = params[0];
    routing.topK = params[3];
    const uint64_t ids = uint64_t{routing.rows} * routing.topK;
    if (snapshot.buffer.size() < ids * sizeof(uint32_t)) throw std::runtime_error("short MoE selection");
    std::vector<uint32_t> selected(ids);
    std::memcpy(selected.data(), snapshot.buffer.data(), ids * sizeof(uint32_t));
    std::sort(selected.begin(), selected.end());
    for (size_t first = 0; first < selected.size();) {
      size_t last = first;
      while (last < selected.size() && selected[last] == selected[first]) ++last;
      const auto routes = static_cast<uint32_t>(last - first);
      routing.byRoutes[std::ranges::lower_bound(kRouteBins, routes) - kRouteBins.begin()] += 1.0;
      distinct += 1.0;
      first = last;
    }
  }
  routing.visits = snapshots.size();
  if (routing.visits) {
    routing.distinct = distinct / double(routing.visits);
    for (double &bin : routing.byRoutes) bin /= double(routing.visits);
  }
  return routing;
}

// "e1 e2-4 e5-8 e9-16 e17+": a routing's mean experts a visit by their routes.
std::string routesHistogram(const Routing &routing) {
  char text[128];
  std::snprintf(text, sizeof text, "%.2f/%.2f/%.2f/%.2f/%.2f", routing.byRoutes[0], routing.byRoutes[1],
                routing.byRoutes[2], routing.byRoutes[3], routing.byRoutes[4]);
  return text;
}

// The routed experts of a cycle (--experts on).
void printExperts(const std::string &title, const Routing &routing) {
  if (!routing.visits) {
    std::printf("%s routed experts: no MoE visit\n", title.c_str());
    return;
  }
  std::printf("%s routed experts: %zu visits of %u rows, %u a row: %.2f distinct experts a visit, %.3f of its "
              "routes; experts a visit of 1/2-4/5-8/9-16/17+ routes %s\n",
              title.c_str(), routing.visits, routing.rows, routing.topK, routing.distinct,
              routing.distinct / (double(routing.rows) * routing.topK), routesHistogram(routing).c_str());
}

// The fused census of `cycles` captured cycles (--census on, --census-cycles):
// each pipeline's dispatches of a cycle, in order and alone, run as one
// command kCensusRounds times, the median, cycle by cycle; the whole cycle
// alike. A line a pipeline, the most time first: its dispatches a cycle, us a
// dispatch and ms a cycle, the median over the cycles and their least and
// most, and, where cycles were attributed (`attributed` of them in
// `replayed`), its replayed us a dispatch and how many times the fused us
// that is. A pipeline of an indirect grid (~), such as a MoE expert pass,
// reads the grid and the tables of the cycle's last dispatch that wrote them
// when its dispatches run alone, so its line is its last visit's routing, not
// each visit's. Each cycle's routed experts head the lines (routingOf).
constexpr uint32_t kCensusRounds = 3;
void printCensus(metal::MetalBackend &backend, const std::string &title,
                 const std::vector<std::vector<metal::CapturedCommand>> &cycles,
                 const std::vector<Routing> &routing, const Table &replayed, uint32_t attributed,
                 double fusedGpuSeconds) {
  const auto median = [&](std::span<const metal::ComputeDispatch> dispatches) {
    std::vector<double> seconds;
    for (uint32_t round = 0; round < kCensusRounds; ++round)
      seconds.push_back(backend.submitCommandAsync(dispatches).wait().gpuSeconds);
    std::sort(seconds.begin(), seconds.end());
    return seconds[kCensusRounds / 2];
  };
  // The median and the least and most of values a cycle.
  struct Spread {
    double median = 0.0, least = 0.0, most = 0.0;
  };
  const auto spread = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return Spread{values[values.size() / 2], values.front(), values.back()};
  };
  struct Line {
    std::vector<double> seconds;
    std::vector<double> dispatches;
    bool indirect = false;
  };
  std::map<std::string, Line> lines;
  std::vector<double> wholes, sums;
  size_t dispatches = 0;
  for (const std::vector<metal::CapturedCommand> &commands : cycles) {
    std::vector<metal::ComputeDispatch> all;
    for (const metal::CapturedCommand &command : commands)
      all.insert(all.end(), command.dispatches.begin(), command.dispatches.end());
    dispatches = all.size();
    std::map<std::string, std::vector<metal::ComputeDispatch>> byPipeline;
    for (const metal::ComputeDispatch &dispatch : all) byPipeline[dispatch.pipelineName].push_back(dispatch);
    double sum = 0.0;
    for (const auto &[name, own] : byPipeline) {
      Line &line = lines[name];
      line.seconds.push_back(median(own));
      line.dispatches.push_back(double(own.size()));
      line.indirect = line.indirect || std::ranges::any_of(own, [](const metal::ComputeDispatch &dispatch) {
                        return static_cast<bool>(dispatch.indirectThreadgroups.buffer);
                      });
      sum += line.seconds.back();
    }
    sums.push_back(sum);
    wholes.push_back(median(all));
  }
  const Spread whole = spread(wholes), sum = spread(sums);
  std::printf("\n== %s fused census: %zu cycles of %zu dispatches; the cycle %.2f ms fused, %.2f ms as a captured "
              "cycle (%.2f-%.2f), %.2f ms as each pipeline alone ==\n",
              title.c_str(), cycles.size(), dispatches, fusedGpuSeconds * 1e3, whole.median * 1e3, whole.least * 1e3,
              whole.most * 1e3, sum.median * 1e3);
  if (!routing.empty() && routing.front().visits) {
    std::printf("routed experts a visit, cycle by cycle (%zu visits of %u rows, %u a row), distinct and by their "
                "routes 1/2-4/5-8/9-16/17+:",
                routing.front().visits, routing.front().rows, routing.front().topK);
    for (const Routing &cycle : routing) std::printf(" %.2f (%s)", cycle.distinct, routesHistogram(cycle).c_str());
    std::printf("\n");
  }
  std::vector<std::pair<std::string, Spread>> order;
  for (const auto &[name, line] : lines)
    if (line.seconds.size() == cycles.size()) order.emplace_back(name, spread(line.seconds));
  std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a.second.median > b.second.median; });
  std::printf("%-44s %6s %10s %9s %9s %9s %11s %7s\n", "pipeline", "count", "fused_us", "fused_ms", "least_ms",
              "most_ms", "replayed_us", "ratio");
  for (const auto &[name, time] : order) {
    const Line &line = lines.at(name);
    const double count = spread(line.dispatches).median, each = time.median * 1e6 / count;
    const std::string label = name + (line.indirect ? " ~" : "");
    std::printf("%-44s %6.0f %10.2f %9.3f %9.3f %9.3f", label.c_str(), count, each, time.median * 1e3,
                time.least * 1e3, time.most * 1e3);
    const auto found = replayed.find(name);
    if (attributed && found != replayed.end() && found->second.dispatches) {
      const double replayedEach = found->second.gpuSeconds * 1e6 / double(found->second.dispatches);
      std::printf(" %11.2f %7.2f", replayedEach, replayedEach / each);
    }
    std::printf("\n");
  }
}

std::vector<uint32_t> pageRange(uint32_t first, uint32_t count) {
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index)
    result[index] = first + index;
  return result;
}

// The two requests a profile sends, in a model's tokens. The chat request asks for a long answer, so decode cycles
// keep producing tokens instead of a stop token right after prefill: its user text repeats one sentence, "Write a
// detailed, multi-paragraph essay about the history of the Roman Empire. ", until the prompt reaches the requested
// length, and the answer starts without thinking. The counting request, "Count upward from 1, writing each number
// and its English name on a new line. Continue until you reach 200." answered without thinking, is an answer of
// near-certain chains, where a lookup has nothing to copy at first. Built in are the Qwen families' tokens (one
// tokenizer) and Flint's; --chat-tokens gives any model's (dev/benchmarks/chat_prompts.py renders them).
struct ChatTokens final {
  std::vector<uint32_t> userHeader;
  std::vector<uint32_t> sentence;
  std::vector<uint32_t> assistantHeader;
  std::vector<uint32_t> counting;
};
const ChatTokens kQwenChat{
    // <|im_start|>user\n
    .userHeader = {248045, 846, 198},
    .sentence = {7734, 264, 11346, 11, 7072, 12, 26829, 8627, 883, 279, 3712, 314, 279, 12386, 19825, 13, 220},
    // <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    .assistantHeader = {248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271},
    .counting = {248045, 846, 198, 2427, 38453, 494, 220, 16, 11, 4237, 1754, 1324, 321, 1141,
                 6163, 803, 383, 264, 491, 1500, 13, 14569, 2980, 488, 5372, 220, 17, 15,
                 15, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271}};
const ChatTokens kFlintChat{
    // <|user|>
    .userHeader = {150003},
    .sentence = {7421, 260, 7426, 11, 4519, 12, 41088, 6369, 762, 269, 3476, 297, 269, 10831, 18381, 13, 220},
    // <|assistant|><think></think>
    .assistantHeader = {150004, 150006, 150007},
    .counting = {150003, 3201, 29915, 497, 220, 16, 11, 3758, 1218, 1067, 299, 869, 5095, 1060, 358,
                 260, 618, 1507, 13, 27110, 2843, 361, 4495, 220, 905, 13, 150004, 150006, 150007}};

// --chat-tokens FILE: ChatTokens' four parts, one line of whitespace-separated token ids each, in its order.
ChatTokens readChatTokens(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::invalid_argument("cannot read --chat-tokens " + path.string());
  std::vector<std::vector<uint32_t>> lines;
  for (std::string line; std::getline(input, line);) {
    std::istringstream words(line);
    std::vector<uint32_t> &ids = lines.emplace_back();
    for (uint32_t id = 0; words >> id;)
      ids.push_back(id);
    if (!words.eof())
      throw std::invalid_argument("--chat-tokens reads lines of token ids");
  }
  if (lines.size() != 4 || std::any_of(lines.begin(), lines.end(), [](const auto &ids) { return ids.empty(); }))
    throw std::invalid_argument("--chat-tokens takes 4 lines of tokens: the chat request's user header, its "
                                "sentence and its assistant header, and the counting request");
  return {std::move(lines[0]), std::move(lines[1]), std::move(lines[2]), std::move(lines[3])};
}

std::vector<uint32_t> chatPrompt(const ChatTokens &chat, uint32_t tokens) {
  if (tokens < chat.userHeader.size() + chat.assistantHeader.size() + 1)
    throw std::invalid_argument("prompt is too short for a chat request");
  std::vector<uint32_t> prompt = chat.userHeader;
  const uint32_t body = tokens - chat.userHeader.size() - chat.assistantHeader.size();
  for (uint32_t index = 0; index < body; ++index)
    prompt.push_back(chat.sentence[index % chat.sentence.size()]);
  prompt.insert(prompt.end(), chat.assistantHeader.begin(), chat.assistantHeader.end());
  return prompt;
}

// The Qwen families' targets, which read one tokenizer's tokens (kQwenChat's).
bool qwenTarget(const model::TargetLayout &target) {
  return std::holds_alternative<model::Qwen3_8Layout>(target) ||
         std::holds_alternative<model::Qwen3_6MoeLayout>(target) ||
         std::holds_alternative<model::Qwen4ExpLayout>(target);
}

// The prompts a text profile sends, in the model's tokens: each --prompt file's, else the chat request of
// `promptTokens` tokens or the counting request, by --chat-tokens or the tokens built in for the Qwen families and
// Flint; another family's (Inkling's) needs --chat-tokens. Refused, before Metal opens, where a token lies outside
// the model's vocabulary.
std::vector<std::vector<uint32_t>> profilePrompts(const model::ModelDescriptor &descriptor,
                                                  const std::optional<std::filesystem::path> &chatTokens,
                                                  bool counting, uint32_t promptTokens,
                                                  const std::vector<std::filesystem::path> &promptFiles) {
  const uint32_t vocabulary = descriptor.capabilities.vocabularySize;
  const auto inVocabulary = [&](const std::vector<uint32_t> &tokens, std::string_view source) {
    for (const uint32_t token : tokens)
      if (token >= vocabulary)
        throw std::invalid_argument(std::string(source) + " token " + std::to_string(token) +
                                    " lies outside the model's vocabulary of " + std::to_string(vocabulary) +
                                    " tokens");
  };
  std::vector<std::vector<uint32_t>> prompts;
  for (const std::filesystem::path &file : promptFiles) {
    std::ifstream input(file);
    std::vector<uint32_t> &ids = prompts.emplace_back();
    for (uint32_t id = 0; input >> id;)
      ids.push_back(id);
    if (ids.empty() || !input.eof())
      throw std::invalid_argument("--prompt reads files of token ids");
    if (ids.size() > promptTokens)
      throw std::invalid_argument("--prompt-tokens holds no lane's prompt of " + file.string());
    inVocabulary(ids, "--prompt");
  }
  if (!prompts.empty())
    return prompts;
  ChatTokens chat;
  if (chatTokens)
    chat = readChatTokens(*chatTokens);
  else if (qwenTarget(descriptor.target))
    chat = kQwenChat;
  else if (std::holds_alternative<model::FlintLayout>(descriptor.target))
    chat = kFlintChat;
  else
    throw std::invalid_argument("the built-in prompts are the Qwen families' and Flint's tokens, not " +
                                descriptor.name + "'s: --chat-tokens FILE gives its own, as `python -m "
                                "dev.benchmarks.chat_prompts decode-profile MODEL_ROOT FILE` renders them");
  for (const std::vector<uint32_t> *part : {&chat.userHeader, &chat.sentence, &chat.assistantHeader, &chat.counting})
    inVocabulary(*part, chatTokens ? "--chat-tokens" : "built-in prompt");
  prompts.push_back(counting ? chat.counting : chatPrompt(chat, promptTokens));
  return prompts;
}

uint32_t parseCount(std::string_view text, std::string_view label) {
  uint32_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !value) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return value;
}

// HxW[,HxW...]: patch grids of images the encoder accepts.
std::vector<ops::ImageGrid> parseGrids(std::string_view text) {
  std::vector<ops::ImageGrid> grids;
  while (true) {
    const std::string_view item = text.substr(0, text.find(','));
    const size_t cross = item.find('x');
    if (cross == std::string_view::npos)
      throw std::invalid_argument("--images takes HxW[,HxW...] patch grids");
    const ops::ImageGrid grid{parseCount(item.substr(0, cross), "grid height"),
                              parseCount(item.substr(cross + 1), "grid width")};
    if (!grid.valid() || grid.patches() > ops::kMaximumImagePatches) {
      throw std::invalid_argument(
          std::string(item) + " is not a grid of even sides and at most " +
          std::to_string(ops::kMaximumImagePatches) + " patches");
    }
    grids.push_back(grid);
    if (item.size() == text.size())
      return grids;
    text.remove_prefix(item.size() + 1);
  }
}

// The vision encoder as the runtime runs it for the largest of the images:
// an arena sized for that image's patches, and each image's dispatches in
// one command. Every image is encoded once to warm up, then `cycles` times
// fused for the median GPU time and `cycles` times attributed.
void profileImages(metal::MetalBackend &backend,
                   const std::filesystem::path &root,
                   std::span<const ops::ImageGrid> grids, uint32_t cycles) {
  const model::ModelDescriptor descriptor = model::inspectModelRoot(root);
  if (!descriptor.hasVision())
    throw std::invalid_argument("the model has no vision role");
  model::WeightImages images(backend, descriptor.sourceIdentity);
  const auto loader = model::planVisionLoader(root, descriptor);
  const model::VisionTowerWeights vision =
      model::loadVisionWeights(backend, images, loader.get());
  uint64_t largest = 0;
  for (const ops::ImageGrid &grid : grids)
    largest = std::max(largest, grid.patches());
  const auto encoder = ops::ImageEncoder::make(backend, vision.tensors, static_cast<uint32_t>(largest));
  const uint32_t patch = vision.tensors.layout.patchSize;

  for (const ops::ImageGrid &grid : grids) {
    // Gradients along both axes and a checkerboard of patch-sized squares.
    const metal::MetalBuffer pixels = backend.allocateBuffer(
        grid.pixelBytes(patch), metal::BufferStorage::Shared, "image pixels");
    const uint32_t height = grid.height * patch;
    const uint32_t width = grid.width * patch;
    auto *pixel = static_cast<uint8_t *>(pixels.contents());
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x, pixel += ops::kImageChannels) {
        pixel[0] = static_cast<uint8_t>(x * 255 / width);
        pixel[1] = static_cast<uint8_t>(y * 255 / height);
        pixel[2] = static_cast<uint8_t>(
            (x / patch + y / patch) % 2 * 255);
      }
    }
    const metal::MetalBuffer embeddings = backend.allocateBuffer(
        uint64_t{grid.mergedTokens()} *
            vision.tensors.layout.outputHiddenSize * sizeof(uint16_t),
        metal::BufferStorage::Private, "image embeddings");
    const auto encode = [&] {
      metal::CommandGraph graph;
      encoder->encode(graph, grid, pixels, embeddings);
      return backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
    };

    static_cast<void>(encode());
    std::vector<double> fused;
    for (uint32_t cycle = 0; cycle < cycles; ++cycle)
      fused.push_back(encode());
    std::sort(fused.begin(), fused.end());
    BackendInstrumentation::setDispatchProfiling(backend, true);
    for (uint32_t cycle = 0; cycle < cycles; ++cycle)
      static_cast<void>(encode());
    BackendInstrumentation::setDispatchProfiling(backend, false);
    Table table;
    accumulate(table, BackendInstrumentation::takeDispatchProfile(backend));
    print("image " + std::to_string(grid.height) + "x" +
              std::to_string(grid.width) + " (" +
              std::to_string(grid.patches()) + " patches) encode",
          table, cycles, fused[fused.size() / 2]);
  }
}

// A lane's output budget, unless --answer-tokens gives another.
constexpr uint32_t kAnswerTokens = 256;

struct Lane final {
  uint64_t id = 0;
  uint32_t stateLane = 0;
  uint64_t position = 0;
  std::vector<uint32_t> pages;
  // Its blocks' window unit pages, a table a page (kv::Layout::blockPages),
  // as the engine's lanes hold them (engine::Cache::cutPageTable): absent
  // before its window, windowPages blocks before its committed ones end,
  // each made absent as the window passes it; `units` keeps them all for a
  // prompt that begins again.
  std::vector<std::vector<uint32_t>> windows;
  std::vector<std::vector<uint32_t>> units;
  uint32_t windowFrom = 0;
  // A revision for each change of its tables, from the first changed page.
  uint64_t pageTableRevision = 1;
  uint32_t firstChanged = 0;
  // The tokens of its answer so far, its cycles, and its output budget.
  uint32_t answered = 0;
  uint32_t answerCycles = 0;
  uint32_t budget = kAnswerTokens;
  // Its answer ended in the last cycle, at a stop token or its output
  // budget: it begins again before the next.
  bool finished = false;
};

// The window pages of the layout (kv::Layout::windowPages), none without one.
uint32_t windowPages = 0;

// A command of the lane at its position: the units its window has passed
// leave its tables first, as an engine lane's cut makes them absent.
void cutWindow(Lane &lane) {
  if (!windowPages)
    return;
  const auto blocks = static_cast<uint32_t>(lane.position / kv::kPageTokens);
  const uint32_t start = blocks - std::min(blocks, windowPages);
  if (start <= lane.windowFrom)
    return;
  for (std::vector<uint32_t> &table : lane.windows)
    std::fill(table.begin() + lane.windowFrom, table.begin() + start, kv::kAbsentPage);
  ++lane.pageTableRevision;
  lane.firstChanged = lane.windowFrom;
  lane.windowFrom = start;
}

// How each lane's request answers: greedy or sampled, under a thinking
// budget or none, and within its output budget.
struct Answer final {
  SamplingParameters sampling;
  std::optional<ThinkingBudget> thinking;
  uint32_t tokens = kAnswerTokens;
};

void prefill(model::Runtime &executor, Lane &lane,
             std::span<const uint32_t> prompt, const Answer &answer) {
  EngineRequest request;
  request.id = lane.id;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = answer.tokens;
  lane.budget = answer.tokens;
  request.sampling = answer.sampling;
  request.thinkingBudget = answer.thinking;
  lane.answered = 0;
  lane.answerCycles = 0;
  // A prompt that begins again holds its units from its first page.
  lane.position = 0;
  if (lane.windowFrom) {
    lane.windows = lane.units;
    lane.windowFrom = 0;
    ++lane.pageTableRevision;
    lane.firstChanged = 0;
  }
  executor.beginColdRequest(request.modelView(), lane.stateLane);
  uint32_t offset = 0;
  while (offset < prompt.size()) {
    const uint32_t count = std::min<uint32_t>(
        model::ExecutionLimits::prefillTokenBudget,
        static_cast<uint32_t>(prompt.size()) - offset);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{lane.id, count, offset}},
                   .decodeStage = DecodeStage::Regular};
    lane.position = offset;
    cutWindow(lane);
    ModelBatchItem item{lane.id, offset, count, lane.pages,
                        lane.pageTableRevision};
    item.pageTableFirstChanged = lane.firstChanged;
    item.windowTables = lane.windows;
    item.inputTokens = prompt.subspan(offset, count);
    auto results =
        executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    if (results.size() != 1 || results[0].consumedPromptTokens != count)
      throw std::runtime_error("prefill consumed the wrong row count");
    if (!results[0].failure.empty())
      throw std::runtime_error(results[0].failure);
    offset += count;
  }
  lane.position = prompt.size();
}

struct CycleTiming final {
  double gpuSeconds = 0.0;
  double wallSeconds = 0.0;
  uint64_t commands = 0;
  uint64_t tokens = 0;
};

// Each answer's first two cycles, a lane's: its place among the active lanes,
// the cycle of its answer, the rows it verified and the tokens it retained
// (decode-profile --step-log on), until profileWidth prints them.
struct AnswerCycle final {
  uint32_t lane = 0, cycle = 0, rows = 0, tokens = 0;
};
std::vector<AnswerCycle> answerCycles;
// An FNV-1a digest of every token the timed cycles of a width retained, each
// lane's in its place, until profileWidth prints it: two builds that lay the
// same rows (--step-rows) and compute them alike print the same digest.
uint64_t outputDigest = 0;

CycleTiming decodeCycle(metal::MetalBackend &backend,
                        model::Runtime &executor,
                        std::span<Lane> lanes) {
  const uint64_t submissionsBefore =
      BackendInstrumentation::submittedCommands(backend);
  const auto started = std::chrono::steady_clock::now();
  BatchPlan plan;
  plan.kind = WorkKind::Decode;
  std::vector<ModelBatchItem> items;
  for (Lane &lane : lanes) {
    cutWindow(lane);
    plan.items.push_back({lane.id, 0, 0});
    items.push_back({lane.id, lane.position, 0, lane.pages,
                     lane.pageTableRevision});
    items.back().pageTableFirstChanged = lane.firstChanged;
    items.back().windowTables = lane.windows;
  }
  auto results = executor.decode(plan, items);
  if (results.size() != lanes.size())
    throw std::runtime_error("decode width changed");
  uint64_t tokens = 0;
  for (size_t index = 0; index < lanes.size(); ++index) {
    tokens += results[index].outputTokens.size();
    for (const uint32_t token : results[index].outputTokens)
      for (const uint64_t word : {uint64_t{index}, uint64_t{token}})
        outputDigest = (outputDigest ^ word) * 0x100000001b3ULL;
    if (!results[index].failure.empty())
      throw std::runtime_error(results[index].failure);
    lanes[index].answered += static_cast<uint32_t>(results[index].outputTokens.size());
    if (++lanes[index].answerCycles <= 2)
      answerCycles.push_back({static_cast<uint32_t>(index), lanes[index].answerCycles,
                              results[index].draftedTokens + 1,
                              static_cast<uint32_t>(results[index].outputTokens.size())});
    lanes[index].finished = results[index].finished || lanes[index].answered == lanes[index].budget;
    lanes[index].position += results[index].outputTokens.size() -
                             results[index].outputTokensWithoutKv;
  }
  const auto finished = std::chrono::steady_clock::now();
  return {executor.telemetry().lastDecodeGpuSeconds,
          std::chrono::duration<double>(finished - started).count(),
          BackendInstrumentation::submittedCommands(backend) - submissionsBefore, tokens};
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) {
        std::cerr << "usage: decode-profile METALLIB MODEL_ROOT "
                     "[--prompt-tokens N] [--cycles K] [--kv-format int8|bf16] "
                     "[--state-format fp32|bf16] [--weights original|auto] [--expert-memory SIZE] [--ple-memory SIZE] "
                     "[--dispatch-list FILE] [--verify-rows auto|all] [--step-rows T] [--lane-rows R] [--activations auto|bf16] [--attribute on|off] [--census on|off] [--census-cycles N] [--plain R] [--experts on|off] "
                     "[--widths N] "
                     "[--warmups on|off] [--warmup-widths N] [--serving-check on|off] "
                     "[--prompt chat|counting|FILE[,FILE...]] [--chat-tokens FILE] "
                     "[--step-log on|off] "
                     "[--block-plans on|off] "
                     "[--temperature T [--seed S]] [--thinking-budget TOKENS] [--answer-tokens N]\n"
                     "       decode-profile METALLIB MODEL_ROOT "
                     "--images HxW[,HxW...] [--cycles K]\n";
        return 2;
      }
      uint32_t promptTokens = 512;
      uint32_t cycles = 4;
      // The widest width profiled, B1 through it.
      uint32_t widths = 4;
      kv::Format format = kv::Format::Int8;
      ops::GdnStateFormat stateFormat = ops::GdnStateFormat::Float32;
      std::vector<ops::ImageGrid> grids;
      // Streamed routed experts and PLE table (tools/LoadOptions.hpp).
      model::ModelLoadOptions loadOptions;
      bool textOptions = false;
      std::string dispatchList;
      model::VerifyRows verifyRows = model::VerifyRows::Auto;
      std::optional<uint32_t> stepRows;
      // Rows a lane every step takes at every width (--lane-rows), in place
      // of one count of rows for every width (--step-rows).
      std::optional<uint32_t> laneRows;
      // The engine's --activations: auto (int8 where the plans take it) or bf16 (none: a decode step's attention
      // queries stay bf16); not given, the model's default (ModelDescriptor::activations).
      std::optional<ops::Int8Activations> int8Activations;
      bool attribute = true, census = false, routedExperts = false;
      uint32_t censusCycles = 8;
      // Rows a lane of plain steps (--plain): no draft, each lane's anchor copied down a chain.
      std::optional<uint32_t> plainRows;
      // The start's warmups (its prefill of the most rows and of each row class, with the pipelines they create,
      // and its decode warmups, which measure the step costs, StepCosts);
      // the counting prompt, or a file's token ids; and each cycle's choice
      // of its rows.
      bool warmups = false, counting = false, stepLog = false, blockPlans = false;
      // Serving after the warmups, a cycle of every step's rows of every width (--serving-check), and the
      // widths the warmups warm (--warmup-widths; every lane's by default).
      bool servingCheck = false;
      std::optional<uint32_t> warmupWidths;
      // A file of token ids a lane, lane i the (i mod n)th of n.
      std::vector<std::filesystem::path> promptFiles;
      // The chat request's and the counting request's tokens in the model's own (--chat-tokens).
      std::optional<std::filesystem::path> chatTokens;
      // Sampled answers at Qwen's recommended top-k 20 and top-p 0.95, and a
      // thinking budget the answers think under from an open thinking block.
      Answer answer;
      answer.sampling.seed = 1000;
      std::optional<uint32_t> thinkingTokens;
      for (int index = 3; index < argc; index += 2) {
        const std::string_view option(argv[index]);
        if (index + 1 >= argc)
          throw std::invalid_argument(std::string(option) + " requires a value");
        if (option == "--prompt-tokens") {
          promptTokens = parseCount(argv[index + 1], "--prompt-tokens");
          textOptions = true;
        } else if (option == "--cycles")
          cycles = parseCount(argv[index + 1], "--cycles");
        else if (option == "--widths") {
          widths = parseCount(argv[index + 1], "--widths");
          if (widths > model::ExecutionLimits::maximumBatchWidth)
            throw std::invalid_argument("--widths takes 1 to the most lanes of a step");
          textOptions = true;
        }
        else if (option == "--kv-format") {
          const std::string_view value(argv[index + 1]);
          if (value != "int8" && value != "bf16")
            throw std::invalid_argument("--kv-format takes int8 or bf16");
          format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
          textOptions = true;
        } else if (option == "--state-format") {
          const std::string_view value(argv[index + 1]);
          if (value != "fp32" && value != "bf16")
            throw std::invalid_argument("--state-format takes fp32 or bf16");
          stateFormat = value == "fp32" ? ops::GdnStateFormat::Float32 : ops::GdnStateFormat::BFloat16;
          textOptions = true;
        } else if (tools::takeLoadOption(option, argv[index + 1], loadOptions)) {
          textOptions = true;
        } else if (option == "--verify-rows") {
          const std::string_view value(argv[index + 1]);
          if (value != "auto" && value != "all")
            throw std::invalid_argument("--verify-rows takes auto or all");
          verifyRows = value == "all" ? model::VerifyRows::All : model::VerifyRows::Auto;
          textOptions = true;
        } else if (option == "--warmups" || option == "--step-log" || option == "--block-plans" ||
                   option == "--serving-check") {
          const std::string_view value(argv[index + 1]);
          if (value != "on" && value != "off")
            throw std::invalid_argument(std::string(option) + " takes on or off");
          (option == "--warmups"         ? warmups
           : option == "--step-log"      ? stepLog
           : option == "--serving-check" ? servingCheck
                                         : blockPlans) = value == "on";
          textOptions = true;
        } else if (option == "--prompt") {
          const std::string_view value(argv[index + 1]);
          counting = value == "counting";
          if (value != "chat" && !counting)
            for (size_t start = 0; start <= value.size();) {
              const size_t comma = std::min(value.find(',', start), value.size());
              promptFiles.emplace_back(std::string(value.substr(start, comma - start)));
              start = comma + 1;
            }
          textOptions = true;
        } else if (option == "--chat-tokens") {
          chatTokens = std::filesystem::path(argv[index + 1]);
          textOptions = true;
        } else if (option == "--temperature") {
          answer.sampling.temperature = std::stof(argv[index + 1]);
          if (!(answer.sampling.temperature > 0.0F))
            throw std::invalid_argument("--temperature takes a positive temperature");
          answer.sampling.topK = 20;
          answer.sampling.topP = 0.95F;
          textOptions = true;
        } else if (option == "--seed") {
          answer.sampling.seed = std::stoull(argv[index + 1]);
          textOptions = true;
        } else if (option == "--answer-tokens") {
          answer.tokens = parseCount(argv[index + 1], "--answer-tokens");
          textOptions = true;
        } else if (option == "--thinking-budget") {
          thinkingTokens = parseCount(argv[index + 1], "--thinking-budget");
          textOptions = true;
        } else if (option == "--attribute") {
          const std::string_view value(argv[index + 1]);
          if (value != "on" && value != "off")
            throw std::invalid_argument("--attribute takes on or off");
          attribute = value == "on";
          textOptions = true;
        } else if (option == "--census-cycles") {
          censusCycles = parseCount(argv[index + 1], "--census-cycles");
          if (!censusCycles) throw std::invalid_argument("--census-cycles takes at least one cycle");
          textOptions = true;
        } else if (option == "--experts") {
          const std::string_view value(argv[index + 1]);
          if (value != "on" && value != "off")
            throw std::invalid_argument("--experts takes on or off");
          routedExperts = value == "on";
          textOptions = true;
        } else if (option == "--plain") {
          plainRows = parseCount(argv[index + 1], "--plain");
          if (!*plainRows || *plainRows > model::ExecutionLimits::targetVerifyRows)
            throw std::invalid_argument("--plain takes 1 to a verify block's rows a lane");
          textOptions = true;
        } else if (option == "--census") {
          const std::string_view value(argv[index + 1]);
          if (value != "on" && value != "off")
            throw std::invalid_argument("--census takes on or off");
          census = value == "on";
          textOptions = true;
        } else if (option == "--warmup-widths") {
          warmupWidths = parseCount(argv[index + 1], "--warmup-widths");
          if (!*warmupWidths || *warmupWidths > model::ExecutionLimits::maximumBatchWidth)
            throw std::invalid_argument("--warmup-widths takes 1 to the most lanes of a step");
          textOptions = true;
        } else if (option == "--step-rows") {
          stepRows = parseCount(argv[index + 1], "--step-rows");
          textOptions = true;
        } else if (option == "--activations") {
          const std::string_view value(argv[index + 1]);
          if (value != "auto" && value != "bf16")
            throw std::invalid_argument("--activations takes auto or bf16");
          int8Activations = value == "auto" ? ops::Int8Activations::On : ops::Int8Activations::Off;
          textOptions = true;
        } else if (option == "--lane-rows") {
          laneRows = parseCount(argv[index + 1], "--lane-rows");
          if (*laneRows > model::ExecutionLimits::targetVerifyRows)
            throw std::invalid_argument("--lane-rows takes 1 to a verify block's rows");
          textOptions = true;
        } else if (option == "--dispatch-list") {
          dispatchList = argv[index + 1];
          textOptions = true;

        } else if (option == "--images")
          grids = parseGrids(argv[index + 1]);
        else
          throw std::invalid_argument("unknown option");
      }
      if (!grids.empty() && textOptions)
        throw std::invalid_argument("--images takes no other option than --cycles");
      if (servingCheck && (!warmups || plainRows || stepRows || laneRows))
        throw std::invalid_argument("--serving-check takes --warmups on and forces its own steps");

      const std::filesystem::path root(argv[2]);
      if (!grids.empty()) {
        metal::MetalBackend backend(argv[1]);
        std::printf("device %s, %u encodes per image\n",
                    backend.capabilities().deviceName.c_str(), cycles);
        profileImages(backend, root, grids, cycles);
        return 0;
      }
      // The model's record and its prompts, refused before Metal opens where they do not fit it.
      const model::ModelDescriptor descriptor = model::inspectModelRoot(root);
      std::vector<std::vector<uint32_t>> prompts =
          profilePrompts(descriptor, chatTokens, counting, promptTokens, promptFiles);
      metal::MetalBackend backend(argv[1]);
      // Streamed weights read on a pool of SSD readers, as the engine's do.
      loadOptions.readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
      model::LoadedModel model =
          model::loadModel(backend, root, descriptor, loadOptions);
      ops::ExecutionPlans operators(backend.capabilities(), int8Activations.value_or(descriptor.activations()));

      // Enough Page32 pages for each width's lanes of prompt plus generated rows.
      const uint32_t pagesPerLane =
          (promptTokens + std::max(answer.tokens, kAnswerTokens) + model::ExecutionLimits::targetVerifyRows) /
              kv::kPageTokens +
          2;
      // Whole extents of the largest size the pool rule picks, as a large
      // pool's would be.
      const kv::Layout kvLayout = model.targetKvLayout(format);
      const uint32_t extentPages = kvLayout.maximumExtentPages();
      const uint32_t pageCount =
          (pagesPerLane * kvLayout.blockPages() * widths + extentPages - 1) / extentPages * extentPages;
      MemoryGovernor governor(
          backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
          queryHostMemory, 0);
      // Every allocation through the governor is for a claim: the profile's
      // are a start's, one Floor claim for its whole run.
      const MemoryGovernor::Scope run(governor, governor.ledger().open({.cls = ClaimClass::Floor}));
      kv::PageStorage pages(backend, governor.allocationAdmission(), kvLayout,
                            pageCount, extentPages);
      for (uint32_t extent = 0; extent < pageCount / extentPages; ++extent) {
        if (!pages.allocateExtent(extent))
          throw std::runtime_error("could not allocate the KV extents");
      }
      model::QwenStateStorage states(backend,
                                      governor.allocationAdmission(),
                                      model.stateLayout(stateFormat), nullptr);
      model::RuntimeContext context{backend, model, pages, states, operators};
      context.verifyRows = verifyRows;
      // The lanes a start runs that profiles its widths: the default's, or
      // more as --max-concurrent-requests asks.
      context.lanes = std::max(widths, model::ExecutionLimits::defaultLanes);
      model::Runtime executor(context);
      if (warmups) {
        // A start's prefill warmups (Bootstrap): the prefill of the most rows,
        // then a chunk of each row class, and the pipelines they created.
        const auto prefillStarted = std::chrono::steady_clock::now();
        static_cast<void>(executor.warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
        const size_t mostPipelines = BackendInstrumentation::cachedPipelines(backend);
        const auto classesStarted = std::chrono::steady_clock::now();
        for (const uint32_t rows : executor.prefillRowClasses()) static_cast<void>(executor.warmupPrefill(rows));
        std::printf("prefill warmups: most rows %.2f s, %zu row classes %.2f s, pipelines %zu then %zu\n",
                    std::chrono::duration<double>(classesStarted - prefillStarted).count(),
                    executor.prefillRowClasses().size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - classesStarted).count(),
                    mostPipelines, BackendInstrumentation::cachedPipelines(backend));
        for (uint32_t width = 1; width <= std::min(context.lanes, warmupWidths.value_or(context.lanes)); ++width) {
          // A start's decode warmup of the lane count, which measures its
          // classes' costs, and the costs it measured.
          const auto started = std::chrono::steady_clock::now();
          static_cast<void>(executor.warmupDecodeBatch(width));
          std::printf("B%u decode warmup: %.2f s\n", width,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
          for (const model::StepCosts::Point &point : model::RuntimeInstrumentation::stepPoints(executor, width))
            std::printf("B%u step class %u-%u rows, lane cap %u: %.2f ms\n", width, point.first, point.last,
                        point.cap, point.milliseconds);
        }
        // The widths a start's memory does not afford to warm: their cycles'
        // pipelines, no step run (Bootstrap, Runtime::prepareDecodeBatch).
        for (uint32_t width = warmupWidths.value_or(context.lanes) + 1; width <= context.lanes; ++width) {
          const auto started = std::chrono::steady_clock::now();
          static_cast<void>(executor.prepareDecodeBatch(width));
          std::printf("B%u decode pipelines prepared, no warmup step: %.2f s\n", width,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        }
        // What the start's warmups and preparations took together, and the
        // pipelines it holds when it begins serving.
        std::printf("start: %zu pipelines after %.2f s of warmups\n", BackendInstrumentation::cachedPipelines(backend),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - prefillStarted).count());
        static_cast<void>(model::RuntimeInstrumentation::takeStepChoices(executor));
      }
      model::RuntimeInstrumentation::forceStepRows(executor, stepRows);
      if (plainRows) model::RuntimeInstrumentation::skipDraft(executor, plainRows);
      model::RuntimeInstrumentation::planBlocks(executor, blockPlans);

      std::printf("device %s, %u prompt tokens, %u cycles per width, %s state\n",
                  backend.capabilities().deviceName.c_str(), promptTokens,
                  cycles, std::string(ops::formatName(stateFormat)).c_str());

      std::array<Lane, model::ExecutionLimits::maximumBatchWidth> lanes;
      // Lane i's tables are the i-th run of blockPages tables of
      // pagesPerLane pages: its base pages, then each window unit page's.
      const uint32_t blockPages = kvLayout.blockPages();
      windowPages = kvLayout.windowUnitPages() ? kvLayout.windowPages() : 0;
      for (uint32_t index = 0; index < lanes.size(); ++index) {
        const uint32_t first = index * blockPages * pagesPerLane;
        std::vector<std::vector<uint32_t>> windows;
        for (uint32_t table = 1; table < blockPages; ++table)
          windows.push_back(pageRange(first + table * pagesPerLane, pagesPerLane));
        lanes[index] = {index + 1, index, 0, pageRange(first, pagesPerLane), windows, windows};
      }
      // Lane `index`'s prompt.
      const auto promptOf = [&](uint32_t index) -> std::vector<uint32_t> & { return prompts[index % prompts.size()]; };
      std::vector<uint32_t> &prompt = prompts.front();
      // Under a thinking budget a Qwen answer thinks: a prompt's empty
      // thinking block stays open, and the budget's end is "\n\n</think>".
      if (thinkingTokens) {
        if (!qwenTarget(descriptor.target))
          throw std::invalid_argument("--thinking-budget takes a Qwen target");
        constexpr uint32_t kThinkEnd = 248069, kBlankLines = 271;
        if (prompt.size() >= 2 && prompt[prompt.size() - 2] == kThinkEnd && prompt.back() == kBlankLines)
          prompt.resize(prompt.size() - 2);
        answer.thinking = ThinkingBudget{*thinkingTokens, {kBlankLines, kThinkEnd}, kThinkEnd};
      }

      if (servingCheck) {
        // As a start that is Ready: every pipeline from here on is one a
        // request waits for. Lane w joins before width w's steps.
        backend.beginServing();
        uint64_t created = 0;
        for (uint32_t width = 1; width <= widths; ++width) {
          prefill(executor, lanes[width - 1], promptOf(width - 1), answer);
          const std::span<Lane> active(lanes.data(), width);
          for (uint32_t rows = width; rows <= width * model::ExecutionLimits::targetVerifyRows; ++rows) {
            model::RuntimeInstrumentation::forceStepRows(executor, rows);
            const uint64_t before = backend.pipelinesCreatedWhileServing();
            const CycleTiming timing = decodeCycle(backend, executor, active);
            const uint64_t made = backend.pipelinesCreatedWhileServing() - before;
            if (made)
              std::printf("serving check B%u %u rows: %llu pipelines created while serving, the cycle %.2f ms\n",
                          width, rows, static_cast<unsigned long long>(made), timing.wallSeconds * 1e3);
            created += made;
            for (Lane &lane : active)
              if (lane.finished) {
                executor.end(lane.id);
                prefill(executor, lane, promptOf(lane.stateLane), answer);
                lane.finished = false;
              }
          }
        }
        std::printf("serving check: %llu pipelines created while serving over B1-B%u, every step's rows%s\n",
                    static_cast<unsigned long long>(created), widths, created ? ": FAIL" : "");
        for (uint32_t index = 0; index < widths; ++index) executor.end(lanes[index].id);
        return created ? 1 : 0;
      }

      // Warm prefill and B1 decode with real work before measuring.
      prefill(executor, lanes[0], prompt, answer);
      static_cast<void>(decodeCycle(backend, executor, std::span<Lane>(&lanes[0], 1)));
      executor.end(lanes[0].id);

      // Prefill: fused timing first, then the attributed replay. Both cover
      // every chunk of the prompt.
      const double prefillBefore = executor.telemetry().totalPrefillGpuSeconds;
      prefill(executor, lanes[0], prompt, answer);
      const double prefillFused =
          executor.telemetry().totalPrefillGpuSeconds - prefillBefore;
      if (attribute) {
        executor.end(lanes[0].id);
        BackendInstrumentation::setDispatchProfiling(backend, true);
        prefill(executor, lanes[0], prompt, answer);
        BackendInstrumentation::setDispatchProfiling(backend, false);
        Table prefillTable;
        accumulate(prefillTable,
                   BackendInstrumentation::takeDispatchProfile(backend));
        print("prefill " + std::to_string(prompt.size()) + " rows", prefillTable,
              1.0, prefillFused);
      } else {
        std::printf("== prefill %zu rows: %.2f ms fused ==\n", prompt.size(), prefillFused * 1e3);
      }

      // A cycle of the active lanes, then, untimed, the prompt again of each
      // lane whose answer it ended.
      const auto cycleOf = [&](std::span<Lane> active) {
        const CycleTiming timing = decodeCycle(backend, executor, active);
        for (Lane &lane : active)
          if (lane.finished) {
            executor.end(lane.id);
            prefill(executor, lane, promptOf(lane.stateLane), answer);
            lane.finished = false;
          }
        return timing;
      };
      auto profileWidth = [&](const char *title, std::span<Lane> active) {
        if (laneRows)
          model::RuntimeInstrumentation::forceStepRows(executor, *laneRows * static_cast<uint32_t>(active.size()));
        std::vector<CycleTiming> fused;
        static_cast<void>(model::RuntimeInstrumentation::takeStepChoices(executor));
        answerCycles.clear();
        outputDigest = 0xcbf29ce484222325ULL;
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          fused.push_back(cycleOf(active));
        std::printf("%s output digest %016llx\n", title, static_cast<unsigned long long>(outputDigest));
        if (stepLog)
          for (const AnswerCycle &answer : answerCycles)
            std::printf("%s answer cycle: lane %u, cycle %u, rows %u, tokens %u\n", title, answer.lane, answer.cycle,
                        answer.rows, answer.tokens);
        if (stepLog)
          for (const auto &choice : model::RuntimeInstrumentation::takeStepChoices(executor)) {
            std::printf("%s step: %u rows at lane cap %u (expected %.3f, cost %.2f ms), verified %u, "
                        "retained %u, wall %.2f ms, gpu %.2f ms, k %.4f, floor %.4f\n",
                        title, choice.rows, choice.laneCap, choice.expected, choice.cost, choice.verified,
                        choice.retained, choice.wallMilliseconds, choice.gpuMilliseconds, choice.contextCost,
                        choice.floor);
          }
        // The cycles in the order they ran, and their mean: a greedy prompt
        // runs the same cycle sequence every time, so two builds compare
        // cycle by cycle; the median alone may pick a different cycle.
        double fusedSum = 0.0;
        std::printf("\n%s: fused gpu ms in order:", title);
        for (const CycleTiming &timing : fused) {
          std::printf(" %.2f", timing.gpuSeconds * 1e3);
          fusedSum += timing.gpuSeconds;
        }
        std::printf("\n%s: mean fused gpu %.3f ms\n", title,
                    fusedSum * 1e3 / double(fused.size()));
        std::sort(fused.begin(), fused.end(),
                  [](const CycleTiming &left, const CycleTiming &right) {
                    return left.gpuSeconds < right.gpuSeconds;
                  });
        const CycleTiming median = fused[fused.size() / 2];
        double tokens = 0.0, wall = 0.0;
        for (const CycleTiming &cycle : fused) {
          tokens += cycle.tokens;
          wall += cycle.wallSeconds;
        }
        std::printf("\n%s: median fused gpu %.2f ms, wall %.2f ms, %llu "
                    "command(s) per cycle, %.2f tokens per cycle, %.2f wall ms per token\n",
                    title, median.gpuSeconds * 1e3, median.wallSeconds * 1e3,
                    static_cast<unsigned long long>(median.commands), tokens / fused.size(),
                    wall * 1e3 / tokens);
        Table replayed;
        if (attribute) {
          BackendInstrumentation::setDispatchProfiling(backend, true);
          for (uint32_t cycle = 0; cycle < cycles; ++cycle)
            static_cast<void>(cycleOf(active));
          BackendInstrumentation::setDispatchProfiling(backend, false);
          const std::vector<metal::DispatchTiming> timings =
              BackendInstrumentation::takeDispatchProfile(backend);
          if (!dispatchList.empty()) {
            const std::string path =
                active.size() == 1 ? dispatchList : dispatchList + ".B" + std::to_string(active.size());
            FILE *file = std::fopen(path.c_str(), "w");
            if (!file) throw std::runtime_error("cannot write " + path);
            for (size_t index = 0; index < timings.size(); ++index)
              std::fprintf(file, "%zu %s %.2f\n", index, timings[index].pipelineName.c_str(),
                           timings[index].gpuSeconds * 1e6);
            std::fclose(file);
          }
          accumulate(replayed, timings);
          print(title, replayed, cycles, median.gpuSeconds);
        }
        if (routedExperts) {
          // The routed experts of one more cycle, from each visit's selection.
          BackendInstrumentation::snapshotEvery(backend, "moe_route_select", 1);
          static_cast<void>(cycleOf(active));
          BackendInstrumentation::snapshotEvery(backend, {}, 0);
          printExperts(title, routingOf(BackendInstrumentation::takeSnapshots(backend)));
        }
        if (census) {
          // Each captured cycle runs as usual, its MoE selections copied.
          std::vector<std::vector<metal::CapturedCommand>> captured;
          std::vector<Routing> routing;
          for (uint32_t cycle = 0; cycle < censusCycles; ++cycle) {
            BackendInstrumentation::captureCommands(backend, true);
            BackendInstrumentation::snapshotEvery(backend, "moe_route_select", 1);
            static_cast<void>(cycleOf(active));
            BackendInstrumentation::snapshotEvery(backend, {}, 0);
            BackendInstrumentation::captureCommands(backend, false);
            captured.push_back(BackendInstrumentation::takeCapturedCommands(backend));
            routing.push_back(routingOf(BackendInstrumentation::takeSnapshots(backend)));
          }
          printCensus(backend, title, captured, routing, replayed, attribute ? cycles : 0, median.gpuSeconds);
        }
      };
      // The streamed experts the B1 cycles read (--expert-memory): per
      // cycle, the routed experts, those not resident, the bytes read, and
      // the loads' host steps (work: reading and staging them), which tell
      // what a cycle waits for the SSD.
      const model::ExpertStore *experts = model.expertStore();
      const model::ExpertStoreStats expertsBefore = experts ? experts->stats() : model::ExpertStoreStats{};
      profileWidth("B1 decode cycle", std::span<Lane>(&lanes[0], 1));
      if (experts) {
        const model::ExpertStoreStats after = experts->stats();
        const model::ExpertStoreStats::Use &u = after.uses[1], &b = expertsBefore.uses[1];
        const double perCycle = 1.0 / cycles;
        std::printf("experts B1: %llu of %llu resident in %llu MiB; per cycle %.1f block loads, %.1f routed, %.1f "
                    "not resident, %.1f read ahead, %.2f MiB read; host step work mean %.3f p50 %.3f p95 %.3f ms "
                    "over the run\n",
                    (unsigned long long)after.resident, (unsigned long long)after.experts,
                    (unsigned long long)(after.capBytes >> 20), (u.loads - b.loads) * perCycle,
                    (u.routed - b.routed) * perCycle, (u.read - b.read) * perCycle,
                    (u.readAhead - b.readAhead) * perCycle, (u.readBytes - b.readBytes) * perCycle / (1 << 20),
                    u.work.count ? u.work.totalMilliseconds / u.work.count : 0.0, u.work.p50Milliseconds,
                    u.work.p95Milliseconds);
      }
      // Add one lane at a time so M16 and M24 paths are measured too.
      for (uint32_t index = 1; index < widths; ++index) {
        prefill(executor, lanes[index], promptOf(index), answer);
        const std::string title = "B" + std::to_string(index + 1) + " decode cycle";
        profileWidth(title.c_str(), std::span<Lane>(lanes.data(), index + 1));
      }

      // A streamed table's stages and reads over the whole run (--ple-memory),
      // which tell what reading its rows from its source costs.
      if (const model::SparseRowStore *store = model.streamedTable()) {
        const model::SparseRowStats stats = store->stats();
        for (uint32_t use = 0; use < model::kSparseRowUses; ++use) {
          const model::SparseRowStats::Use &u = stats.uses[use];
          std::printf("table %s %s: %llu stages, %llu ids, %llu distinct, %llu hit, %llu read; stage mean %.3f "
                      "p50 %.3f p95 %.3f p99 %.3f max %.3f ms\n",
                      stats.table.c_str(), use == 0 ? "prefill" : "decode", (unsigned long long)u.stages,
                      (unsigned long long)u.requestedRows, (unsigned long long)u.uniqueRows,
                      (unsigned long long)u.hitRows, (unsigned long long)u.readRows,
                      u.stage.count ? u.stage.totalMilliseconds / u.stage.count : 0.0, u.stage.p50Milliseconds,
                      u.stage.p95Milliseconds, u.stage.p99Milliseconds, u.stage.maximumMilliseconds);
        }
        std::printf("table %s reads: %llu bytes, %llu prefetched rows; a row's read p50 %.3f p95 %.3f p99 %.3f ms\n",
                    stats.table.c_str(), (unsigned long long)stats.readBytes,
                    (unsigned long long)stats.prefetchReadRows, stats.read.p50Milliseconds, stats.read.p95Milliseconds,
                    stats.read.p99Milliseconds);
      }
      for (uint32_t index = 0; index < widths; ++index)
        executor.end(lanes[index].id);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "decode-profile: " << error.what() << '\n';
      return 1;
    }
  }
}
