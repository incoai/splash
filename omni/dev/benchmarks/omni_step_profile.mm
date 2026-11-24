// Times the pieces of a TTS step on real weights:
//   omni-step-profile METALLIB MODEL_DIR [--rows N]
// Each piece is encoded alone into a command and timed over repetitions
// (GPU time from the ticket): the talker decode forward, one code
// predictor sub-step forward (decode arena), one predictor layer's
// projections, the predictor head, a code sampling, and an empty dispatch.

#include "metal/MetalBackend.hpp"
#include "model/WeightImages.hpp"
#include "omni/KvPages.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/TtsModel.hpp"
#include "ops/Omni.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::omni;

namespace {

double timeGraph(metal::MetalBackend &backend, const std::function<void(metal::CommandGraph &)> &build, int repetitions,
                 size_t &dispatches) {
  double best = 1e9;
  for (int i = 0; i < repetitions; ++i) {
    metal::CommandGraph graph;
    build(graph);
    dispatches = graph.dispatches().size();
    metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
    const metal::CommandTiming timing = ticket.wait();
    best = std::min(best, timing.gpuSeconds * 1e3);
  }
  return best;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: omni-step-profile METALLIB MODEL_DIR [--rows N]\n";
    return 2;
  }
  try {
    uint32_t rows = 1;
    for (int i = 3; i + 1 < argc; ++i)
      if (std::string(argv[i]) == "--rows") rows = std::stoul(argv[i + 1]);
    metal::MetalBackend backend(argv[1]);
    const OmniConfig config = readOmniConfig(argv[2]);
    const TtsConfig &c = *config.tts;
    model::WeightImages images(backend, "profile");
    TtsWeights weights = loadTtsWeights(backend, images, argv[2], c, config.quantized);
    (void)packDenseWeights(backend, weights, UINT64_MAX);
    TtsModel model(backend, std::move(weights), c, 64, 16);
    const Qwen3Decoder &talker = model.talker(), &predictor = model.predictor();
    KvPagePool pool(backend, 4096);
    LanePageTables talkerTables(backend, 16, c.talker.layers, 8), predictorTables(backend, 16, c.predictor.layers, 1);
    std::vector<std::unique_ptr<LaneKv>> talkerLanes, predictorLanes;
    for (uint32_t l = 0; l < rows; ++l) {
      talkerLanes.push_back(std::make_unique<LaneKv>(pool, talkerTables, l));
      predictorLanes.push_back(std::make_unique<LaneKv>(pool, predictorTables, l));
      (void)talkerLanes.back()->reserve(100);
      (void)predictorLanes.back()->reserve(16);
    }
    const KvBinding talkerKv{talkerTables.buffer(), talkerTables.entries(), pool.pages(), pool.pageCount()};
    const KvBinding predictorKv{predictorTables.buffer(), predictorTables.entries(), pool.pages(), pool.pageCount()};
    Qwen3Arena talkerDecode(backend, talker.linear(), talker.weights(), 16, 16, 16, true);
    Qwen3Arena predictorDecode(backend, predictor.linear(), predictor.weights(), 16, 16, 16, true);
    std::vector<ops::omni::AttentionSequence> talkerSeq, predictorSeq;
    for (uint32_t r = 0; r < rows; ++r) {
      talkerSeq.push_back({talkerTables.offset(r), talkerTables.stride(), 60, 1, r, 60});
      predictorSeq.push_back({predictorTables.offset(r), predictorTables.stride(), 8, 1, r, 8});
    }
    const ops::omni::AttentionTables talkerAttention = ops::omni::attentionTables(talkerSeq, true);
    const ops::omni::AttentionTables predictorAttention = ops::omni::attentionTables(predictorSeq, true);
    for (uint32_t r = 0; r < rows; ++r) {
      static_cast<uint32_t *>(talkerDecode.logitRows.contents())[r] = r;
      static_cast<uint32_t *>(predictorDecode.logitRows.contents())[r] = r;
    }
    metal::MetalBuffer logits = backend.allocateBuffer(16 * 2048 * 4, metal::BufferStorage::Shared, "logits");
    metal::MetalBuffer sampled = backend.allocateBuffer(16 * 16 * 4, metal::BufferStorage::Shared, "sampled");
    metal::MetalBuffer uniforms = backend.allocateBuffer(16 * 16 * 4, metal::BufferStorage::Shared, "uniforms");
    size_t dispatches = 0;
    std::cout << "rows " << rows << "\n";
    double ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      Qwen3Decoder::uploadTables(talkerDecode, talkerAttention);
      talker.addForward(g, talkerDecode, talkerKv, {&talkerAttention, rows});
    }, 8, dispatches);
    std::cout << "talker decode forward (28 layers + head): " << ms << " ms, " << dispatches << " dispatches\n";
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      Qwen3Decoder::uploadTables(predictorDecode, predictorAttention);
      predictor.addForward(g, predictorDecode, predictorKv, {&predictorAttention, rows});
    }, 8, dispatches);
    std::cout << "predictor decode forward (5 layers, final norm): " << ms << " ms, " << dispatches << " dispatches\n";
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      Qwen3Decoder::uploadTables(predictorDecode, predictorAttention);
      predictor.addForward(g, predictorDecode, predictorKv, {&predictorAttention, rows});
      model.addPredictorHead(g, predictorDecode, 3, rows, logits);
      OmniSampleParams p{};
      p.rows = rows;
      p.vocabulary = 2048;
      p.top_k = 50;
      p.top_p = 1.0F;
      p.temperature = 0.9F;
      p.repetition_penalty = 1.0F;
      p.output_stride = 16;
      p.uniform_stride = 16;
      ops::omni::Sampling::addSampleCodes(g, logits, {}, {}, uniforms, sampled, p);
      model.addPredictorInput(g, predictorDecode.hidden, predictorDecode.normalized, rows);
    }, 8, dispatches);
    std::cout << "predictor sub-step (forward + head + sampling + input): " << ms << " ms, " << dispatches << " dispatches\n";
    // The tail alone: forward + head, then with greedy and top-k sampling.
    for (const float temperature : {-1.0F, 0.0F, 0.9F}) {
      ms = timeGraph(backend, [&](metal::CommandGraph &g) {
        Qwen3Decoder::uploadTables(predictorDecode, predictorAttention);
        predictor.addForward(g, predictorDecode, predictorKv, {&predictorAttention, rows});
        model.addPredictorHead(g, predictorDecode, 3, rows, logits);
        if (temperature < 0.0F) return;
        OmniSampleParams p{};
        p.rows = rows;
        p.vocabulary = 2048;
        p.top_k = 50;
        p.top_p = 1.0F;
        p.temperature = temperature;
        p.repetition_penalty = 1.0F;
        p.output_stride = 16;
        p.uniform_stride = 16;
        ops::omni::Sampling::addSampleCodes(g, logits, {}, {}, uniforms, sampled, p);
      }, 8, dispatches);
      std::cout << "predictor forward + head" << (temperature < 0.0F ? "" : temperature == 0.0F ? " + greedy" : " + top-k 50")
                << ": " << ms << " ms\n";
    }
    // Individual kernels of the talker's decode path, 20 of each chained.
    const Qwen3LayerWeights &layer = talker.weights().layers[0];
    const KvBinding &kv = talkerKv;
    ops::omni::AttentionBuffers ab;
    ab.qkv = talkerDecode.qkv;
    ab.queryNorm = layer.queryNorm;
    ab.keyNorm = layer.keyNorm;
    {
      std::vector<float> inverse(64);
      for (uint32_t i = 0; i < 64; ++i) inverse[i] = float(std::pow(1e6, -2.0 * i / 128));
      ab.inverseFrequencies = backend.allocateBuffer(256, metal::BufferStorage::Shared, "inv");
      std::memcpy(ab.inverseFrequencies.contents(), inverse.data(), 256);
    }
    Qwen3Decoder::uploadTables(talkerDecode, talkerAttention);
    ab.rowSequence = talkerDecode.currentSlot().rowSequence;
    ab.sequences = talkerDecode.currentSlot().sequences;
    ab.tiles = talkerDecode.currentSlot().tiles;
    ab.pageTable = kv.pageTable;
    ab.pages = kv.pages;
    ab.queries = talkerDecode.queries;
    ab.partials = talkerDecode.partials;
    ab.statistics = talkerDecode.statistics;
    ab.output = talkerDecode.attention;
    ab.rotations = talkerDecode.rotations;
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      for (int i = 0; i < 20; ++i)
        ops::omni::Attention::addDecodeFused(g, ab, talkerAttention, 1, 1e-6F, kv.pageTableEntries, kv.pageCount,
                                             i ? ops::omni::Rotations::Load : ops::omni::Rotations::Store);
    }, 10, dispatches);
    std::cout << "talker fused attention (context 61): " << ms / 20 * 1000 << " us\n";
    {
      // The predictor's fused attention, one row at committed 8 (a middle
      // sub-step) and two rows at committed 0 (sub-step 0), 20 chained.
      const Qwen3LayerWeights &p0 = predictor.weights().layers[0];
      ops::omni::AttentionBuffers pb = ab;
      pb.queryNorm = p0.queryNorm;
      pb.keyNorm = p0.keyNorm;
      pb.pageTable = predictorKv.pageTable;
      pb.qkv = predictorDecode.qkv;
      pb.output = predictorDecode.attention;
      pb.rotations = predictorDecode.rotations;
      std::vector<ops::omni::AttentionSequence> pairSeq;
      for (uint32_t r = 0; r < rows; ++r) pairSeq.push_back({predictorTables.offset(r), predictorTables.stride(), 0, 2, 2 * r, 0});
      const ops::omni::AttentionTables pairAttention = ops::omni::attentionTables(pairSeq, true);
      for (const auto &[label, tables] : {std::pair{"one row, committed 8", &predictorAttention},
                                          std::pair{"two rows, committed 0", &pairAttention}}) {
        Qwen3Decoder::uploadTables(predictorDecode, *tables);
        pb.sequences = predictorDecode.currentSlot().sequences;
        ms = timeGraph(backend, [&](metal::CommandGraph &g) {
          for (int i = 0; i < 20; ++i)
            ops::omni::Attention::addDecodeFused(g, pb, *tables, 0, 1e-6F, predictorKv.pageTableEntries,
                                                 predictorKv.pageCount, ops::omni::Rotations::Load);
        }, 10, dispatches);
        std::cout << "predictor fused attention (" << label << "): " << ms / 20 * 1000 << " us\n";
      }
    }
    const ops::Linear &linear = talker.linear();
    const uint32_t lanes = (rows + 7) / 8;
    if (std::holds_alternative<ops::Projection>(layer.qkv)) {
      const ops::Projection &qkv = std::get<ops::Projection>(layer.qkv);
      const ops::Projection &o = std::get<ops::Projection>(layer.output);
      const ops::Projection &gate = std::get<ops::Projection>(layer.gate), &up = std::get<ops::Projection>(layer.up),
                            &down = std::get<ops::Projection>(layer.down);
      ms = timeGraph(backend, [&](metal::CommandGraph &g) {
        for (int i = 0; i < 20; ++i)
          linear.add(g, {.input = talkerDecode.normalized, .output = talkerDecode.qkv, .scratch = talkerDecode.scratch}, qkv,
                     linear.decodePlan(qkv, lanes));
      }, 10, dispatches);
      std::cout << "talker qkv projection (4096x2048 Q4): " << ms / 20 * 1000 << " us, plan tile " << int(linear.decodePlan(qkv, lanes).configuration().tile) << " splits " << linear.decodePlan(qkv, lanes).configuration().splits << " groups " << linear.decodePlan(qkv, lanes).groups() << "\n";
      ms = timeGraph(backend, [&](metal::CommandGraph &g) {
        for (int i = 0; i < 20; ++i)
          linear.add(g, {.input = talkerDecode.attention, .output = talkerDecode.residual, .residual = talkerDecode.hidden, .scratch = talkerDecode.scratch}, o,
                     linear.decodePlan(o, lanes, ops::LinearEpilogue::Residual));
      }, 10, dispatches);
      std::cout << "talker o projection (2048x2048 Q4 + residual): " << ms / 20 * 1000 << " us\n";
      ms = timeGraph(backend, [&](metal::CommandGraph &g) {
        for (int i = 0; i < 20; ++i) {
          const ops::LinearPlan plan = linear.decodePlan(up, lanes, ops::LinearEpilogue::GateUp, &gate);
          linear.add(g, {.input = talkerDecode.normalized, .output = talkerDecode.intermediate, .gateScratch = talkerDecode.gateScratch, .scratch = talkerDecode.scratch}, up, plan, &gate);
        }
      }, 10, dispatches);
      std::cout << "talker gate/up (2x 6144x2048 Q4): " << ms / 20 * 1000 << " us\n";
      ms = timeGraph(backend, [&](metal::CommandGraph &g) {
        for (int i = 0; i < 20; ++i)
          linear.add(g, {.input = talkerDecode.intermediate, .output = talkerDecode.hidden, .residual = talkerDecode.residual, .scratch = talkerDecode.scratch}, down,
                     linear.decodePlan(down, lanes, ops::LinearEpilogue::Residual));
      }, 10, dispatches);
      std::cout << "talker down (2048x6144 Q4 + residual): " << ms / 20 * 1000 << " us\n";
      const Qwen3LayerWeights &p0 = predictor.weights().layers[0];
      const ops::Projection &pqkv = std::get<ops::Projection>(p0.qkv);
      ms = timeGraph(backend, [&](metal::CommandGraph &g) {
        for (int i = 0; i < 20; ++i)
          linear.add(g, {.input = predictorDecode.normalized, .output = predictorDecode.qkv, .scratch = predictorDecode.scratch}, pqkv,
                     linear.decodePlan(pqkv, lanes));
      }, 10, dispatches);
      std::cout << "predictor qkv projection (4096x1024 Q4): " << ms / 20 * 1000 << " us\n";
    }
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      for (int i = 0; i < 20; ++i)
        ops::Normalization::addRms(g, talkerDecode.hidden, layer.inputNorm, talkerDecode.normalized, 2048, rows);
    }, 10, dispatches);
    std::cout << "talker rms norm (staged): " << ms / 20 * 1000 << " us\n";
    ms = timeGraph(backend, [&](metal::CommandGraph &g) { model.addPredictorHead(g, predictorDecode, 3, rows, logits); }, 20, dispatches);
    std::cout << "predictor head: " << ms << " ms, " << dispatches << " dispatches\n";
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      OmniSampleParams p{};
      p.rows = rows;
      p.vocabulary = 2048;
      p.top_k = 50;
      p.top_p = 1.0F;
      p.temperature = 0.9F;
      p.repetition_penalty = 1.0F;
      p.output_stride = 16;
      p.uniform_stride = 16;
      ops::omni::Sampling::addSampleCodes(g, logits, {}, {}, uniforms, sampled, p);
    }, 20, dispatches);
    std::cout << "code sampling (top-k 50): " << ms << " ms\n";
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      OmniSampleParams p{};
      p.rows = rows;
      p.vocabulary = 2048;
      p.top_k = 50;
      p.top_p = 1.0F;
      p.temperature = 0.9F;
      p.repetition_penalty = 1.0F;
      p.output_stride = 16;
      p.uniform_stride = 16;
      for (int i = 0; i < 20; ++i) ops::omni::Sampling::addSampleCodes(g, logits, {}, {}, uniforms, sampled, p);
    }, 10, dispatches);
    std::cout << "code sampling (top-k 50), 20 chained: " << ms / 20 * 1000 << " us each\n";
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      ops::omni::Norms::addRms(g, predictorDecode.hidden, model.weights().predictor.finalNorm.buffer, predictorDecode.normalized, rows, 1024, 1e-6F, false);
    }, 20, dispatches);
    std::cout << "one rms norm: " << ms << " ms\n";
    ms = timeGraph(backend, [&](metal::CommandGraph &g) {
      for (int i = 0; i < 50; ++i)
        ops::omni::Norms::addRms(g, predictorDecode.hidden, model.weights().predictor.finalNorm.buffer, predictorDecode.normalized, rows, 1024, 1e-6F, false);
    }, 10, dispatches);
    std::cout << "50 chained rms norms: " << ms << " ms (" << ms / 50 * 1000 << " us each)\n";
  } catch (const std::exception &error) {
    std::cerr << "omni step profile: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
