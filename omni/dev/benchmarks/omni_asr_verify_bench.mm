// Times the forwards of ASR speculative decoding on real weights:
//   omni-asr-verify-bench METALLIB MODEL_DIR [--history N] [--lanes N,N...] [--rows N]
//                         [--decode-histories N,N... | --decode-mix H:N,H:N...] [--chain N] [--rounds N]
// For each lane count: a decode step (one row per lane, the decode arena),
// then forwards of 1..rows rows per lane through a verify arena (decode
// projections, ragged attention) and through a prefill arena, each
// selecting every row's logits and their argmax, over a history of N
// tokens per lane, and every layer's attention alone through each kernel
// that takes the rows; --head-rows prices a head over the vocabulary's
// leading rows. Run it on the target to price verification and on
// the draft to price its steps. GPU milliseconds, the median of 15 runs.
// --decode-histories times only decode steps, at each of the histories;
// --decode-mix one step of lanes at different histories.

#include "metal/MetalBackend.hpp"
#include "model/WeightImages.hpp"
#include "omni/KvPages.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/Qwen3Decoder.hpp"
#include "ops/Omni.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::omni;

namespace {

double medianGpu(metal::MetalBackend &backend, const std::function<void(metal::CommandGraph &)> &build) {
  std::vector<double> times;
  for (int i = 0; i < 17; ++i) {
    metal::CommandGraph graph;
    build(graph);
    const metal::CommandTiming timing = backend.submitCommandAsync(graph.command()).wait();
    if (i >= 2) times.push_back(timing.gpuSeconds * 1e3);
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: omni-asr-verify-bench METALLIB MODEL_DIR [--history N] [--lanes N,N...] [--rows N] "
                 "[--head-rows N] [--decode-histories N,N... | --decode-mix H:N,H:N...] [--chain N] [--rounds N]\n";
    return 2;
  }
  try {
    uint32_t history = 200, maxRows = 9, headRows = 0, chain = 8, rounds = 3;
    std::vector<uint32_t> laneCounts{1, 2, 4, 8}, histories, mix;
    for (int i = 3; i + 1 < argc; i += 2) {
      const std::string arg = argv[i];
      if (arg == "--history") history = std::stoul(argv[i + 1]);
      else if (arg == "--rows") maxRows = std::stoul(argv[i + 1]);
      else if (arg == "--head-rows") headRows = std::stoul(argv[i + 1]);
      else if (arg == "--chain") chain = std::stoul(argv[i + 1]);
      else if (arg == "--rounds") rounds = std::stoul(argv[i + 1]);
      else if (arg == "--decode-mix") {
        std::stringstream list(argv[i + 1]);
        for (std::string item; std::getline(list, item, ',');)
          mix.insert(mix.end(), std::stoul(item.substr(item.find(':') + 1)), std::stoul(item.substr(0, item.find(':'))));
      } else if (arg == "--decode-histories") {
        std::stringstream list(argv[i + 1]);
        for (std::string item; std::getline(list, item, ',');) histories.push_back(std::stoul(item));
      } else if (arg == "--lanes") {
        laneCounts.clear();
        std::stringstream list(argv[i + 1]);
        for (std::string item; std::getline(list, item, ',');) laneCounts.push_back(std::stoul(item));
      } else throw std::runtime_error("unknown option " + arg);
    }
    // --decode-mix H:N,...: one step of N lanes at history H, and so on.
    if (!mix.empty()) {
      laneCounts = {static_cast<uint32_t>(mix.size())};
      histories = {*std::max_element(mix.begin(), mix.end())};
    }
    if (!histories.empty()) history = *std::max_element(histories.begin(), histories.end());
    metal::MetalBackend backend(argv[1]);
    const OmniConfig config = readOmniConfig(argv[2]);
    if (config.kind != ModelKind::Asr) throw std::runtime_error("not an ASR model");
    const AsrConfig &c = *config.asr;
    model::WeightImages images(backend, "verify-bench");
    AsrTextWeights weights = loadAsrTextWeights(backend, images, argv[2], c, config.quantized);
    (void)packDenseWeights(backend, weights, UINT64_MAX);
    // --head-rows: the head over the vocabulary's leading rows only (an
    // affine head's whole tiles), as a shortlisted draft head would read.
    uint32_t vocabulary = c.vocabulary, logitStride = c.decoder.headRows;
    if (headRows) {
      Qwen3Weights &w = weights.decoder;
      w.head = std::get<ops::Projection>(*w.head).leadingRows(backend, headRows);
      w.dims.headRows = w.dims.vocabulary = vocabulary = logitStride = headRows;
    }
    Qwen3Decoder decoder(backend, std::move(weights.decoder));
    const uint32_t lanes = *std::max_element(laneCounts.begin(), laneCounts.end());
    const uint32_t blocks = (history + 64 + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS;
    KvPagePool pool(backend, lanes * c.decoder.layers * blocks);
    LanePageTables tables(backend, lanes, c.decoder.layers, blocks);
    std::vector<std::unique_ptr<LaneKv>> kv;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      kv.push_back(std::make_unique<LaneKv>(pool, tables, lane));
      if (!kv.back()->reserve(history + 64)) throw std::runtime_error("no KV pages");
    }
    const KvBinding binding{tables.buffer(), tables.entries(), pool.pages(), pool.pageCount()};
    Qwen3Arena decode(backend, decoder.linear(), decoder.weights(), lanes, lanes, lanes, true);
    Qwen3Arena verify(backend, decoder.linear(), decoder.weights(), OMNI_MAXIMUM_LANES, lanes, OMNI_MAXIMUM_LANES, true,
                      true);
    Qwen3Arena prefill(backend, decoder.linear(), decoder.weights(), OMNI_MAXIMUM_LANES, lanes, OMNI_MAXIMUM_LANES,
                       false);
    metal::MetalBuffer sampled = backend.allocateBuffer(OMNI_MAXIMUM_LANES * 4, metal::BufferStorage::Shared, "sampled");
    // A forward of `perLane` rows for each of `n` lanes through `arena`.
    const auto forward = [&](Qwen3Arena &arena, uint32_t n, uint32_t perLane) {
      std::vector<ops::omni::AttentionSequence> sequences;
      for (uint32_t lane = 0; lane < n; ++lane)
        sequences.push_back({tables.offset(lane), tables.stride(), history, perLane, lane * perLane, history});
      const ops::omni::AttentionTables attention = ops::omni::attentionTables(sequences, arena.decode() && !arena.verify());
      const uint32_t rows = n * perLane;
      return medianGpu(backend, [&](metal::CommandGraph &graph) {
        Qwen3Decoder::uploadTables(arena, attention);
        for (uint32_t r = 0; r < rows; ++r) static_cast<uint32_t *>(arena.logitRows.contents())[r] = r;
        decoder.addForward(graph, arena, binding, {&attention, rows});
        ops::omni::Sampling::addArgmax(graph, arena.logits, sampled, rows, vocabulary, logitStride);
      });
    };
    // Every layer's attention alone over the verify arena's buffers: the
    // prefill kernels (prepare, split, reduce), the fused verify kernel and,
    // for one or two rows per lane, the fused decode kernel.
    metal::MetalBuffer frequencies = backend.allocateBuffer(OMNI_DECODER_HEAD_DIM / 2 * 4, metal::BufferStorage::Shared,
                                                            "frequencies");
    for (uint32_t i = 0; i < OMNI_DECODER_HEAD_DIM / 2; ++i)
      static_cast<float *>(frequencies.contents())[i] = float(std::pow(1e6, -2.0 * i / OMNI_DECODER_HEAD_DIM));
    enum class Kernel { Prefill, Verify, Decode };
    const auto attentionOnly = [&](uint32_t n, uint32_t perLane, Kernel kernel) {
      std::vector<ops::omni::AttentionSequence> sequences;
      for (uint32_t lane = 0; lane < n; ++lane)
        sequences.push_back({tables.offset(lane), tables.stride(), history, perLane, lane * perLane, history});
      const ops::omni::AttentionTables attention = ops::omni::attentionTables(sequences, false);
      if ((kernel == Kernel::Verify && !ops::omni::Attention::verifyFusable(attention)) ||
          (kernel == Kernel::Decode && !ops::omni::Attention::fusable(attention)))
        return -1.0;
      return medianGpu(backend, [&](metal::CommandGraph &graph) {
        Qwen3Decoder::uploadTables(verify, attention);
        for (uint32_t layer = 0; layer < c.decoder.layers; ++layer) {
          const Qwen3LayerWeights &w = decoder.weights().layers[layer];
          ops::omni::AttentionBuffers b{verify.qkv, w.queryNorm, w.keyNorm, frequencies,
                                        verify.currentSlot().rowSequence, verify.currentSlot().sequences,
                                        verify.currentSlot().tiles, binding.pageTable, binding.pages, verify.queries,
                                        verify.partials, verify.statistics, verify.attention, verify.rotations, {}};
          const auto rotations = layer ? ops::omni::Rotations::Load : ops::omni::Rotations::Store;
          if (kernel == Kernel::Verify)
            ops::omni::Attention::addVerifyFused(graph, b, attention, layer, 1e-6F, binding.pageTableEntries,
                                                 binding.pageCount, rotations);
          else if (kernel == Kernel::Decode)
            ops::omni::Attention::addDecodeFused(graph, b, attention, layer, 1e-6F, binding.pageTableEntries,
                                                 binding.pageCount, rotations);
          else {
            ops::omni::Attention::addPrepare(graph, b, attention, layer, 1e-6F, binding.pageTableEntries,
                                             binding.pageCount);
            ops::omni::Attention::addPrefill(graph, b, attention, layer, binding.pageTableEntries, binding.pageCount);
          }
        }
      });
    };
    // --decode-histories: decode steps alone, `chain` forwards (and their
    // argmax) in one command so that no step starts from an idle GPU, at
    // each history and lane count; GPU ms a forward, the median and the
    // least of `rounds` medians of 15 commands.
    if (!histories.empty()) {
      std::printf("%s: decode steps of %u chained forwards; GPU ms a forward (median, least of %u rounds)\n", argv[2],
                  chain, rounds);
      for (uint32_t n : laneCounts)
        for (uint32_t h : histories) {
          std::vector<ops::omni::AttentionSequence> sequences;
          for (uint32_t lane = 0; lane < n; ++lane) {
            const uint32_t lh = mix.empty() ? h : mix[lane];
            sequences.push_back({tables.offset(lane), tables.stride(), lh, 1, lane, lh});
          }
          const ops::omni::AttentionTables attention = ops::omni::attentionTables(sequences, true);
          std::vector<double> times;
          for (uint32_t round = 0; round < rounds; ++round)
            times.push_back(medianGpu(backend, [&](metal::CommandGraph &graph) {
                              Qwen3Decoder::uploadTables(decode, attention);
                              for (uint32_t r = 0; r < n; ++r) static_cast<uint32_t *>(decode.logitRows.contents())[r] = r;
                              for (uint32_t f = 0; f < chain; ++f) {
                                decoder.addForward(graph, decode, binding, {&attention, n});
                                ops::omni::Sampling::addArgmax(graph, decode.logits, sampled, n, vocabulary, logitStride);
                              }
                            }) / chain);
          std::sort(times.begin(), times.end());
          std::printf("lanes %u history %5u: %.3f %.3f\n", n, h, times[times.size() / 2], times.front());
          std::fflush(stdout);
        }
      return 0;
    }
    std::printf("%s: %u layers, hidden %u, history %u tokens; GPU ms (median of 15)\n", argv[2], c.decoder.layers,
                c.decoder.hidden, history);
    for (uint32_t n : laneCounts) {
      std::printf("lanes %u: decode step %.3f\n", n, forward(decode, n, 1));
      for (uint32_t perLane = 1; perLane <= maxRows && n * perLane <= OMNI_MAXIMUM_LANES; ++perLane)
        std::printf("  %u rows/lane (%2u rows): verify %.3f  prefill %.3f;  attention alone: prefill kernels %.3f, "
                    "fused verify %.3f, fused decode %.3f\n",
                    perLane, n * perLane, forward(verify, n, perLane), forward(prefill, n, perLane),
                    attentionOnly(n, perLane, Kernel::Prefill), attentionOnly(n, perLane, Kernel::Verify),
                    attentionOnly(n, perLane, Kernel::Decode));
    }
  } catch (const std::exception &error) {
    std::cerr << "omni asr verify bench: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
