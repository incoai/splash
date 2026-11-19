// Times the decode projections of the Omni models' shapes with random Q4
// weights under ops::Linear's plans, each preceded by its norm dispatch
// where the model runs one.
//   omni-dq4-bench METALLIB [--rows N] [--shape SUBSTRING]
//   omni-dq4-bench METALLIB --sweep [--rows N] [--shape SUBSTRING] [--plan-only]
//   omni-dq4-bench METALLIB --qmv [--shape SUBSTRING]
//   omni-dq4-bench METALLIB --multirow [--row-counts N,N...] [--register] [--shape SUBSTRING]
// Without --sweep it times ops::Linear's plan for N live rows after its norm
// and, where the plan folds the norm (one live row), the folded projection,
// checking that the two give the same bits; --shape picks shapes there too.
// --sweep times one lane's projection alone (no norm) of N live rows (1 to
// 8; the lane's other rows are zero) under ops::Linear's plan, the Split128
// and LaneSplit128 tiles at every K split and, for gate/up, the N256 gate/up
// tile, in three interleaved rounds (best of each), and reports each
// output's largest difference from the plan's relative to the output's
// magnitude; --plan-only times the plan alone.
// --qmv times one live row under the one-lane tiles' plan, ops::Linear's
// one-row plan and the Qmv tile at four and two outputs per simdgroup,
// folded and after the norm dispatch (qmvSweep).
// --multirow times steps of several live rows (multiRow) under ops::Linear's
// plan, the Qmv tile (one row; two sharing its threadgroups) and, with
// --register, the register tile at every K split.
// Each configuration is timed as a chain of 16 dispatches in one command
// (best of 5), so the figure per dispatch includes the in-command boundary.

#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;

namespace {

std::mt19937 rng(7);

metal::MetalBuffer randomBuffer(metal::MetalBackend &backend, uint64_t bytes, const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  auto *bits = static_cast<uint8_t *>(buffer.contents());
  std::uniform_int_distribution<int> byte(0, 255);
  for (uint64_t i = 0; i < bytes; ++i) bits[i] = static_cast<uint8_t>(byte(rng));
  return buffer;
}

metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, uint64_t count, float low, float high, const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(count * 2, metal::BufferStorage::Shared, label);
  auto *bits = static_cast<uint16_t *>(buffer.contents());
  std::uniform_real_distribution<float> value(low, high);
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t u;
    const float f = value(rng);
    std::memcpy(&u, &f, 4);
    bits[i] = static_cast<uint16_t>(u >> 16);
  }
  return buffer;
}

ops::Projection randomProjection(metal::MetalBackend &backend, uint32_t n, uint32_t k) {
  const uint64_t groups = uint64_t{n} * k / 64;
  return ops::Projection(n, k,
                         ops::AffineWeights{randomBuffer(backend, uint64_t{n} * k / 2, "codes"),
                                            bf16Buffer(backend, groups, 0.001F, 0.02F, "scales"),
                                            bf16Buffer(backend, groups, -0.1F, 0.1F, "biases")});
}

// What follows the projection: nothing, a residual (or bias rows), or the
// SiLU product of a gate/up pair.
enum class Mode : uint8_t { Plain, Residual, GateUp };

struct Shape {
  const char *name;
  uint32_t n, k;
  Mode mode;
  bool f32;
};

// Best of eight chains of 16 after four warm-up chains: the GPU clock ramps
// over the first milliseconds of work, which the warm-up absorbs.
double timeChain(metal::MetalBackend &backend, const std::function<void(metal::CommandGraph &)> &build) {
  double best = 1e9;
  for (int i = 0; i < 12; ++i) {
    metal::CommandGraph graph;
    for (int j = 0; j < 16; ++j) build(graph);
    const metal::CommandTiming timing = backend.submitCommandAsync(graph.command()).wait();
    if (i >= 4) best = std::min(best, timing.gpuSeconds * 1e3 / 16);
  }
  return best;
}


// One lane's projection under the device's plan and under each one-lane
// split configuration ops::Linear can run (Split128 and LaneSplit128 at
// every K split), checked against the plan's output.
int sweep(metal::MetalBackend &backend, const std::vector<Shape> &shapes, uint32_t rows, const std::string &filter,
          bool planOnly) {
  const ops::Linear linear(backend.capabilities());
  if (!rows || rows > 8) throw std::invalid_argument("--sweep takes 1 to 8 rows");
  std::printf("rows %u (1 lane); projection alone, per-dispatch us and GB/s of weights, best of 3 rounds\n", rows);
  for (const Shape &shape : shapes) {
    if (!filter.empty() && std::string(shape.name).find(filter) == std::string::npos) continue;
    const bool gateUp = shape.mode == Mode::GateUp;
    const double bytes = double(shape.n) * shape.k * (0.5 + 4.0 / 64) * (gateUp ? 2 : 1);
    const uint32_t copies = std::max<uint32_t>(2, uint32_t(256e6 / bytes) + 1);
    std::vector<ops::Projection> ws, ups;
    for (uint32_t c = 0; c < copies; ++c) {
      ws.push_back(randomProjection(backend, shape.n, shape.k));
      ws.back().destination = shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16;
      if (gateUp) ups.push_back(randomProjection(backend, shape.n, shape.k));
    }
    uint32_t turn = 0;
    metal::MetalBuffer input = bf16Buffer(backend, uint64_t{8} * shape.k, -1.0F, 1.0F, "input");
    std::memset(static_cast<uint16_t *>(input.contents()) + uint64_t{rows} * shape.k, 0, uint64_t{8 - rows} * shape.k * 2);
    metal::MetalBuffer residual = bf16Buffer(backend, uint64_t{8} * shape.n, -1.0F, 1.0F, "residual");
    metal::MetalBuffer output = backend.allocateBuffer(uint64_t{8} * shape.n * 4, metal::BufferStorage::Shared, "output");
    metal::MetalBuffer gateScratch = backend.allocateBuffer(uint64_t{8} * shape.n * 2, metal::BufferStorage::Shared, "gate");
    const ops::LinearEpilogue epilogue = shape.mode == Mode::Residual ? ops::LinearEpilogue::Residual
                                         : gateUp ? ops::LinearEpilogue::GateUp : ops::LinearEpilogue::None;
    const ops::LinearWorkload workload{{shape.n, shape.k}, 8, ops::LinearPhase::Decode, epilogue};
    const ops::FloatOutput destination = shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16;
    std::vector<ops::LinearPlan> plans{linear.decodePlan(ws[0], 1, epilogue, gateUp ? &ups[0] : nullptr)};
    for (const ops::LinearTile tile : {ops::LinearTile::Split128, ops::LinearTile::LaneSplit128})
      for (const uint32_t splits : {1u, 2u, 4u, 8u}) {
        if (planOnly || splits > shape.k / 256 || (tile == ops::LinearTile::Split128 && splits == 1)) continue;
        plans.push_back(ops::Linear::plan(workload, {tile, 0, ops::LinearSimdgroups::Eight, splits}, destination));
      }
    if (gateUp && !planOnly)
      plans.push_back(ops::Linear::plan(workload, {ops::LinearTile::N256, shape.n / 256}, destination));
    ops::LinearScratchSize size;
    for (const ops::LinearPlan &plan : plans) size.include(plan.scratchSize());
    ops::LinearScratch scratch{backend.allocateBuffer(std::max<uint64_t>(size.input, 16), metal::BufferStorage::Shared, "t"),
                               backend.allocateBuffer(std::max<uint64_t>(size.sums, 16), metal::BufferStorage::Shared, "s"),
                               backend.allocateBuffer(std::max<uint64_t>(size.partials, 16), metal::BufferStorage::Shared, "p"),
                               backend.allocateBuffer(std::max<uint64_t>(size.counters, 16), metal::BufferStorage::Shared, "c")};
    std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
    struct Variant { std::string label; std::function<void(metal::CommandGraph &)> build; double best = 1e9; };
    std::vector<Variant> variants;
    for (size_t i = 0; i < plans.size(); ++i) {
      const ops::LinearPlan &plan = plans[i];
      variants.push_back({std::string(i ? "" : "plan ") + std::string(plan.pipeline()) +
                              (plan.secondPipeline().empty() ? "" : " + " + std::string(plan.secondPipeline())) + " s" +
                              std::to_string(plan.configuration().splits),
                          [&, plan](metal::CommandGraph &g) {
                            const uint32_t c = turn++ % copies;
                            linear.add(g, {.input = input, .output = output,
                                           .residual = shape.mode == Mode::Residual ? residual : metal::MetalBuffer{},
                                           .gateScratch = gateScratch, .scratch = scratch},
                                       ws[c], plan, gateUp ? &ups[c] : nullptr);
                          }});
    }
    // Each variant's output on the first copy against the plan's: the
    // largest difference relative to the plan output's largest magnitude.
    const uint64_t outputs = uint64_t{8} * shape.n;
    const auto run = [&](const Variant &v) {
      turn = 0;
      metal::CommandGraph g;
      v.build(g);
      (void)backend.submitCommandAsync(g.command()).wait();
      std::vector<float> values(outputs);
      for (uint64_t i = 0; i < outputs; ++i) {
        if (shape.f32) values[i] = static_cast<const float *>(output.contents())[i];
        else {
          const uint32_t bits = uint32_t{static_cast<const uint16_t *>(output.contents())[i]} << 16;
          std::memcpy(&values[i], &bits, 4);
        }
      }
      return values;
    };
    const std::vector<float> expected = run(variants[0]);
    float scale = 0;
    for (float value : expected) scale = std::max(scale, std::abs(value));
    std::vector<float> errors;
    for (const Variant &v : variants) {
      const std::vector<float> actual = run(v);
      float error = 0;
      for (uint64_t i = 0; i < outputs; ++i) error = std::max(error, std::abs(actual[i] - expected[i]));
      errors.push_back(std::isfinite(error) ? error / scale : 1e9F);
    }
    for (int round = 0; round < 3; ++round)
      for (Variant &v : variants) v.best = std::min(v.best, timeChain(backend, v.build));
    std::printf("%s (%u x %u, %.2f MB)\n", shape.name, shape.n, shape.k, bytes / 1e6);
    for (size_t i = 0; i < variants.size(); ++i)
      std::printf("  %-58s %7.1f us %6.0f GB/s  err %.1e\n", variants[i].label.c_str(), variants[i].best * 1e3,
                  bytes / variants[i].best / 1e6, errors[i]);
    std::fflush(stdout);
  }
  return 0;
}

// One live row of each shape through the one-lane plans: the device's plan
// for all eight rows of the lane (the tiles), its plan for one live row
// (Qmv on Apple10) and the Qmv tile at two and four outputs per simdgroup.
// Each norm shape runs the plan's folded norm where it folds and the norm
// dispatch + projection; each is checked against the first plan's outputs
// (the largest difference relative to their magnitude) and each folded run
// against its norm dispatch + projection, bit for bit.
int qmvSweep(metal::MetalBackend &backend, const std::vector<Shape> &shapes, const std::string &filter) {
  const ops::Linear linear(backend.capabilities());
  std::printf("one live row; per-dispatch us and GB/s of weights, best of 3 rounds\n");
  for (const Shape &shape : shapes) {
    if (!filter.empty() && std::string(shape.name).find(filter) == std::string::npos) continue;
    const bool gateUp = shape.mode == Mode::GateUp, residualMode = shape.mode == Mode::Residual;
    const bool norm = !residualMode;
    const double bytes = double(shape.n) * shape.k * (0.5 + 4.0 / 64) * (gateUp ? 2 : 1);
    const uint32_t copies = std::max<uint32_t>(2, uint32_t(256e6 / bytes) + 1);
    std::vector<ops::Projection> ws, ups;
    for (uint32_t c = 0; c < copies; ++c) {
      ws.push_back(randomProjection(backend, shape.n, shape.k));
      ws.back().destination = shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16;
      if (gateUp) ups.push_back(randomProjection(backend, shape.n, shape.k));
    }
    uint32_t turn = 0;
    metal::MetalBuffer input = bf16Buffer(backend, uint64_t{8} * shape.k, -1.0F, 1.0F, "input");
    metal::MetalBuffer normalized = bf16Buffer(backend, uint64_t{8} * shape.k, -1.0F, 1.0F, "normalized");
    metal::MetalBuffer normWeight = bf16Buffer(backend, shape.k, 0.5F, 1.5F, "norm");
    metal::MetalBuffer residual = bf16Buffer(backend, uint64_t{8} * shape.n, -1.0F, 1.0F, "residual");
    metal::MetalBuffer output = backend.allocateBuffer(uint64_t{8} * shape.n * 4, metal::BufferStorage::Shared, "output");
    metal::MetalBuffer gateScratch = backend.allocateBuffer(uint64_t{8} * shape.n * 2, metal::BufferStorage::Shared, "gate");
    const ops::LinearEpilogue epilogue = residualMode ? ops::LinearEpilogue::Residual
                                         : gateUp ? ops::LinearEpilogue::GateUp : ops::LinearEpilogue::None;
    const ops::FloatOutput destination = shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16;
    const ops::LinearWorkload oneRow{{shape.n, shape.k}, 8, ops::LinearPhase::Decode, epilogue,
                                     ops::WeightLayout::Affine64, 1};
    std::vector<ops::LinearPlan> plans{linear.decodePlan(ws[0], 1, epilogue, gateUp ? &ups[0] : nullptr),
                                       linear.decodePlan(ws[0], 1, epilogue, gateUp ? &ups[0] : nullptr, 1)};
    for (const uint32_t outputs : {4u, 2u})
      plans.push_back(ops::Linear::plan(oneRow, {ops::LinearTile::Qmv, 0, ops::LinearSimdgroups::Eight, 1, outputs},
                                        destination));
    const ops::LinearScratchSize size = linear.decodeScratchSize(ws[0].shape());
    ops::LinearScratch scratch{backend.allocateBuffer(std::max<uint64_t>(size.input, 16), metal::BufferStorage::Shared, "t"),
                               backend.allocateBuffer(std::max<uint64_t>(size.sums, 16), metal::BufferStorage::Shared, "s"),
                               backend.allocateBuffer(std::max<uint64_t>(size.partials, 16), metal::BufferStorage::Shared, "p"),
                               backend.allocateBuffer(std::max<uint64_t>(size.counters, 16), metal::BufferStorage::Shared, "c")};
    std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
    struct Variant { std::string label; std::function<void(metal::CommandGraph &)> build; double best = 1e9; };
    std::vector<Variant> variants;
    for (size_t i = 0; i < plans.size(); ++i) {
      const ops::LinearPlan plan = plans[i];
      const std::string name = std::string(i == 0 ? "lane " : i == 1 ? "plan " : "") + std::string(plan.pipeline());
      const auto run = [&, plan](metal::CommandGraph &g, bool fold) {
        const uint32_t c = turn++ % copies;
        ops::PreparedInput prepared;
        if (norm && !fold)
          prepared = ops::Normalization::addRms(g, input, ops::NormWeights{normWeight}, normalized, shape.k, 8,
                                                scratch, plan.input());
        linear.add(g, {.input = norm && !fold ? normalized : input, .output = output,
                       .residual = residualMode ? residual : metal::MetalBuffer{}, .gateScratch = gateScratch,
                       .scratch = scratch, .prepared = prepared,
                       .normWeight = fold ? normWeight : metal::MetalBuffer{}, .normRows = fold ? 1u : 0u},
                   ws[c], plan, gateUp ? &ups[c] : nullptr);
      };
      if (norm && plan.foldsNorm(1))
        variants.push_back({name + " folded", [run](metal::CommandGraph &g) { run(g, true); }});
      variants.push_back({name + (norm ? " + norm" : ""), [run](metal::CommandGraph &g) { run(g, false); }});
    }
    // Row 0 of each variant on the first copy.
    const auto row0 = [&](const Variant &v) {
      turn = 0;
      std::memset(output.contents(), 0xff, output.sizeBytes());
      metal::CommandGraph g;
      v.build(g);
      (void)backend.submitCommandAsync(g.command()).wait();
      std::vector<float> values(shape.n);
      for (uint32_t i = 0; i < shape.n; ++i) {
        if (shape.f32) values[i] = static_cast<const float *>(output.contents())[i];
        else {
          const uint32_t bits = uint32_t{static_cast<const uint16_t *>(output.contents())[i]} << 16;
          std::memcpy(&values[i], &bits, 4);
        }
      }
      return values;
    };
    std::vector<std::vector<float>> outputsOf;
    for (const Variant &v : variants) outputsOf.push_back(row0(v));
    float scale = 0;
    for (float value : outputsOf[0]) scale = std::max(scale, std::abs(value));
    for (int round = 0; round < 3; ++round)
      for (Variant &v : variants) v.best = std::min(v.best, timeChain(backend, v.build));
    std::printf("%s (%u x %u, %.2f MB)\n", shape.name, shape.n, shape.k, bytes / 1e6);
    for (size_t i = 0; i < variants.size(); ++i) {
      float error = 0;
      for (uint32_t j = 0; j < shape.n; ++j) error = std::max(error, std::abs(outputsOf[i][j] - outputsOf[0][j]));
      // A folded variant is followed by its norm + projection form.
      const bool folded = variants[i].label.ends_with(" folded");
      const bool same = folded && outputsOf[i] == outputsOf[i + 1];
      std::printf("  %-56s %7.1f us %6.0f GB/s  err %.1e%s\n", variants[i].label.c_str(), variants[i].best * 1e3,
                  bytes / variants[i].best / 1e6, std::isfinite(error) ? error / scale : 1e9F,
                  folded ? (same ? "  = norm chain" : "  DIFFERS from norm chain") : "");
    }
    std::fflush(stdout);
  }
  return 0;
}

// Steps of several live rows (--multirow: 2, 4, 8, 12 and 16) through
// ops::Linear's plan, the Qmv tile at one live row and at two sharing its
// threadgroups, and with --register the register tile (Q4Register) at every
// K split. Each norm shape runs as the decoder runs it: the plan folds the norm where
// it folds, else the norm dispatch writes the plan's input (Table64 for the
// register tile); each residual shape's register tile prepares its table
// from the plain rows ("+ prep") or, as if its producer wrote it, reads a
// table prepared beforehand ("alone"). Outputs of the live rows against
// the plan's, relative to their magnitude.
int multiRow(metal::MetalBackend &backend, const std::vector<Shape> &shapes, const std::string &filter,
             const std::vector<uint32_t> &rowCounts, bool withRegister) {
  const ops::Linear linear(backend.capabilities());
  std::printf("live rows of a step; per-dispatch us and GB/s of weights, best of 3 rounds\n");
  for (const Shape &shape : shapes) {
    if (!filter.empty() && std::string(shape.name).find(filter) == std::string::npos) continue;
    const bool gateUp = shape.mode == Mode::GateUp, residualMode = shape.mode == Mode::Residual;
    const bool norm = !residualMode;
    const double bytes = double(shape.n) * shape.k * (0.5 + 4.0 / 64) * (gateUp ? 2 : 1);
    const uint32_t copies = std::max<uint32_t>(2, uint32_t(256e6 / bytes) + 1);
    std::vector<ops::Projection> ws, ups;
    for (uint32_t c = 0; c < copies; ++c) {
      ws.push_back(randomProjection(backend, shape.n, shape.k));
      ws.back().destination = shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16;
      if (gateUp) ups.push_back(randomProjection(backend, shape.n, shape.k));
    }
    const uint32_t maxRows = 32;
    metal::MetalBuffer input = bf16Buffer(backend, uint64_t{maxRows} * shape.k, -1.0F, 1.0F, "input");
    metal::MetalBuffer normalized = bf16Buffer(backend, uint64_t{maxRows} * shape.k, -1.0F, 1.0F, "normalized");
    metal::MetalBuffer normWeight = bf16Buffer(backend, shape.k, 0.5F, 1.5F, "norm");
    metal::MetalBuffer residual = bf16Buffer(backend, uint64_t{maxRows} * shape.n, -1.0F, 1.0F, "residual");
    metal::MetalBuffer output = backend.allocateBuffer(uint64_t{maxRows} * shape.n * 4, metal::BufferStorage::Shared, "output");
    metal::MetalBuffer gateScratch = backend.allocateBuffer(uint64_t{maxRows} * shape.n * 2, metal::BufferStorage::Shared, "gate");
    const ops::LinearEpilogue epilogue = residualMode ? ops::LinearEpilogue::Residual
                                         : gateUp ? ops::LinearEpilogue::GateUp : ops::LinearEpilogue::None;
    const ops::FloatOutput destination = shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16;
    ops::LinearScratchSize size = linear.decodeScratchSize(ws[0].shape());
    for (uint32_t lanes = 1; lanes <= 4; ++lanes)
      for (const uint32_t splits : {1u, 2u, 4u, 8u}) {
        if ((shape.k / 64) % splits) continue;
        size.include(ops::Linear::plan({{shape.n, shape.k}, lanes * 8, ops::LinearPhase::Decode, epilogue},
                                       {ops::LinearTile::Q4Register, 0, ops::LinearSimdgroups::Four, splits},
                                       destination).scratchSize());
      }
    ops::LinearScratch scratch{backend.allocateBuffer(std::max<uint64_t>(size.input, 16), metal::BufferStorage::Shared, "t"),
                               backend.allocateBuffer(std::max<uint64_t>(size.sums, 16), metal::BufferStorage::Shared, "s"),
                               backend.allocateBuffer(std::max<uint64_t>(size.partials, 16), metal::BufferStorage::Shared, "p"),
                               backend.allocateBuffer(std::max<uint64_t>(size.counters, 16), metal::BufferStorage::Shared, "c")};
    std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
    std::printf("%s (%u x %u, %.2f MB)\n", shape.name, shape.n, shape.k, bytes / 1e6);
    for (const uint32_t rows : rowCounts) {
      const uint32_t lanes = (rows + 7) / 8, tileRows = lanes * 8;
      // The live rows hold the input, the others zeros.
      std::vector<uint8_t> original(static_cast<const uint8_t *>(input.contents()),
                                    static_cast<const uint8_t *>(input.contents()) + uint64_t{maxRows} * shape.k * 2);
      std::memset(static_cast<uint8_t *>(input.contents()) + uint64_t{rows} * shape.k * 2, 0,
                  uint64_t{maxRows - rows} * shape.k * 2);
      uint32_t turn = 0;
      const ops::LinearWorkload workload{{shape.n, shape.k}, tileRows, ops::LinearPhase::Decode, epilogue,
                                         ops::WeightLayout::Affine64, rows};
      struct Variant { std::string label; std::function<void(metal::CommandGraph &)> build; double best = 1e9; };
      std::vector<Variant> variants;
      const ops::LinearPlan plan = linear.decodePlan(ws[0], lanes, epilogue, gateUp ? &ups[0] : nullptr, rows);
      const bool fold = norm && plan.foldsNorm(rows);
      const auto addPlan = [&, fold](metal::CommandGraph &g, const ops::LinearPlan &p, bool prepare) {
        const uint32_t c = turn++ % copies;
        ops::PreparedInput prepared;
        const bool usesTable = p.input() == ops::LinearInput::Table64;
        if (norm && !(fold && &p == &plan))
          prepared = ops::Normalization::addRms(g, input, ops::NormWeights{normWeight}, normalized, shape.k, tileRows,
                                                scratch, p.input());
        else if (usesTable && !prepare)
          prepared = {input, ops::LinearInput::Table64};
        const bool folded = fold && &p == &plan;
        linear.add(g, {.input = norm && !folded ? normalized : input, .output = output,
                       .residual = residualMode ? residual : metal::MetalBuffer{}, .gateScratch = gateScratch,
                       .scratch = scratch, .prepared = prepared,
                       .normWeight = folded ? normWeight : metal::MetalBuffer{}, .normRows = folded ? rows : 0u},
                   ws[c], p, gateUp ? &ups[c] : nullptr);
      };
      std::vector<ops::LinearPlan> registers;
      for (const uint32_t splits : {1u, 2u, 4u, 8u})
        if ((shape.k / 64) % splits == 0 && (withRegister || splits == 1))
          registers.push_back(ops::Linear::plan(workload, {ops::LinearTile::Q4Register, 0, ops::LinearSimdgroups::Four, splits},
                                                destination));
      variants.push_back({"plan " + std::string(plan.pipeline()) + " s" + std::to_string(plan.configuration().splits) +
                              (fold ? " folded" : norm ? " + norm" : ""),
                          [&](metal::CommandGraph &g) { addPlan(g, plan, true); }});
      // The Qmv tile: one row at both outputs per simdgroup, folding the
      // norm, or two shared rows, folded and after the norm's dispatch.
      std::vector<ops::LinearPlan> qmvs;
      if (rows == 1)
        for (const uint32_t outputs : {4u, 2u})
          qmvs.push_back(ops::Linear::plan(workload, {ops::LinearTile::Qmv, 0, ops::LinearSimdgroups::Eight, 1, outputs},
                                           destination));
      if (rows == 2)
        qmvs.push_back(ops::Linear::plan(workload, {ops::LinearTile::Qmv, 0, ops::LinearSimdgroups::Eight, 1,
                                                    gateUp ? 2u : 4u, true},
                                         destination));
      for (const ops::LinearPlan &q : qmvs)
        for (const bool folded : {true, false}) {
          if (folded && !(norm && q.canFoldNorm())) continue;
          if (!folded && norm && q.canFoldNorm() && !q.configuration().sharedRows) continue;
          variants.push_back({std::string(q.configuration().sharedRows ? "qmv shared o" : "qmv o") +
                                  std::to_string(q.configuration().simdgroupOutputs) +
                                  (folded ? " folded" : norm ? " + norm" : ""),
                              [&, q, folded](metal::CommandGraph &g) {
                                const uint32_t c = turn++ % copies;
                                if (norm && !folded)
                                  ops::Normalization::addRms(g, input, ops::NormWeights{normWeight}, normalized,
                                                             shape.k, tileRows);
                                linear.add(g, {.input = norm && !folded ? normalized : input, .output = output,
                                               .residual = residualMode ? residual : metal::MetalBuffer{},
                                               .gateScratch = gateScratch, .scratch = scratch,
                                               .normWeight = folded ? normWeight : metal::MetalBuffer{},
                                               .normRows = folded ? rows : 0u},
                                           ws[c], q, gateUp ? &ups[c] : nullptr);
                              }});
        }
      for (const ops::LinearPlan &r : registers) {
        if (!withRegister) break;
        const std::string name = "register s" + std::to_string(r.configuration().splits);
        variants.push_back({name + (norm ? " + norm" : " + prep"), [&, r](metal::CommandGraph &g) { addPlan(g, r, true); }});
        if (!norm) variants.push_back({name + " alone", [&, r](metal::CommandGraph &g) { addPlan(g, r, false); }});
      }
      // The live rows of each variant's output on the first copy.
      const uint64_t outputs = uint64_t{rows} * shape.n;
      const auto run = [&](const Variant &v) {
        // A table read "alone" holds the input prepared beforehand.
        metal::CommandGraph setup;
        linear.add(setup, {.input = input, .output = output, .residual = residual, .scratch = scratch}, ws[0],
                   registers[0], gateUp ? &ups[0] : nullptr);
        (void)backend.submitCommandAsync(setup.command()).wait();
        turn = 0;
        std::memset(output.contents(), 0xff, output.sizeBytes());
        metal::CommandGraph g;
        v.build(g);
        (void)backend.submitCommandAsync(g.command()).wait();
        std::vector<float> values(outputs);
        for (uint64_t i = 0; i < outputs; ++i) {
          if (shape.f32) values[i] = static_cast<const float *>(output.contents())[i];
          else {
            const uint32_t bits = uint32_t{static_cast<const uint16_t *>(output.contents())[i]} << 16;
            std::memcpy(&values[i], &bits, 4);
          }
        }
        return values;
      };
      std::vector<std::vector<float>> outputsOf;
      for (const Variant &v : variants) outputsOf.push_back(run(v));
      float scale = 0;
      for (float value : outputsOf[0]) scale = std::max(scale, std::abs(value));
      for (int round = 0; round < 3; ++round)
        for (Variant &v : variants) v.best = std::min(v.best, timeChain(backend, v.build));
      for (size_t i = 0; i < variants.size(); ++i) {
        float error = 0;
        for (uint64_t j = 0; j < outputs; ++j) error = std::max(error, std::abs(outputsOf[i][j] - outputsOf[0][j]));
        std::printf("  rows %2u  %-56s %7.1f us %6.0f GB/s  err %.1e\n", rows, variants[i].label.c_str(),
                    variants[i].best * 1e3, bytes / variants[i].best / 1e6, std::isfinite(error) ? error / scale : 1e9F);
      }
      std::memcpy(input.contents(), original.data(), original.size());
      std::fflush(stdout);
    }
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: omni-dq4-bench METALLIB [--rows N]\n");
    return 2;
  }
  uint32_t rows = 8;
  bool sweepKernels = false, planOnly = false, qmv = false, multi = false, withRegister = false;
  std::vector<uint32_t> rowCounts{2, 4, 8, 12, 16};
  std::string filter;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rows" && i + 1 < argc) rows = std::stoul(argv[++i]);
    else if (arg == "--multirow") multi = true;
    else if (arg == "--register") withRegister = true;
    else if (arg == "--row-counts" && i + 1 < argc) {
      rowCounts.clear();
      for (std::string list = argv[++i]; !list.empty();) {
        const size_t comma = list.find(',');
        rowCounts.push_back(std::stoul(list.substr(0, comma)));
        list = comma == std::string::npos ? "" : list.substr(comma + 1);
      }
    }
    else if (arg == "--shape" && i + 1 < argc) filter = argv[++i];
    else if (arg == "--sweep") sweepKernels = true;
    else if (arg == "--plan-only") planOnly = true;
    else if (arg == "--qmv") qmv = true;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    const ops::Linear linear(backend.capabilities());
      const std::vector<Shape> shapes{
        {"talker qkv", 4096, 2048, Mode::Plain, false},
        {"talker o +res", 2048, 2048, Mode::Residual, false},
        {"talker gate/up", 6144, 2048, Mode::GateUp, false},
        {"talker down +res", 2048, 6144, Mode::Residual, false},
        {"talker head f32", 3072, 2048, Mode::Plain, true},
        {"predictor qkv", 4096, 1024, Mode::Plain, false},
        {"predictor o +res", 1024, 2048, Mode::Residual, false},
        {"predictor gate/up", 3072, 1024, Mode::GateUp, false},
        {"predictor down +res", 1024, 3072, Mode::Residual, false},
        {"predictor head f32", 2048, 1024, Mode::Plain, true},
        {"small_to_mtp +bias", 1024, 2048, Mode::Residual, false},
        {"asr head f32", 152064, 2048, Mode::Plain, true},
    };
    if (qmv) return qmvSweep(backend, shapes, filter);
    if (multi) return multiRow(backend, shapes, filter, rowCounts, withRegister);
    if (sweepKernels) return sweep(backend, shapes, rows, filter, planOnly);
    const uint32_t padded = (rows + 7) / 8 * 8, lanes = padded / 8;
    std::printf("rows %u (%u lanes); per-dispatch ms and GB/s of weights\n", rows, lanes);
    for (const Shape &shape : shapes) {
      if (!filter.empty() && std::string(shape.name).find(filter) == std::string::npos) continue;
      // Copies of the weights cycled through the chain, at least 256 MB in
      // all, so each dispatch streams from DRAM rather than the system cache.
      const double copyBytes = double(shape.n) * shape.k * (0.5 + 4.0 / 64) * (shape.mode == Mode::GateUp ? 2 : 1);
      const uint32_t copies = std::max<uint32_t>(2, uint32_t(256e6 / copyBytes) + 1);
      std::vector<ops::Projection> ws, ups;
      for (uint32_t c = 0; c < copies; ++c) {
        ws.push_back(randomProjection(backend, shape.n, shape.k));
        ws.back().destination = shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16;
        if (shape.mode == Mode::GateUp) ups.push_back(randomProjection(backend, shape.n, shape.k));
      }
      uint32_t turn = 0;
      ops::Projection &w = ws[0];
      ops::Projection up = shape.mode == Mode::GateUp ? ups[0] : ws[0];
      const bool norm = shape.mode != Mode::Residual;
      const double bytes = double(shape.n) * shape.k * (0.5 + 4.0 / 64) * (shape.mode == Mode::GateUp ? 2 : 1);
      metal::MetalBuffer input = bf16Buffer(backend, uint64_t{padded} * shape.k, -1.0F, 1.0F, "input");
      metal::MetalBuffer normalized = bf16Buffer(backend, uint64_t{padded} * shape.k, -1.0F, 1.0F, "normalized");
      metal::MetalBuffer normWeight = bf16Buffer(backend, shape.k, 0.5F, 1.5F, "norm");
      metal::MetalBuffer residual = bf16Buffer(backend, uint64_t{padded} * shape.n, -1.0F, 1.0F, "residual");
      metal::MetalBuffer output = backend.allocateBuffer(uint64_t{padded} * shape.n * 4, metal::BufferStorage::Shared, "output");
      metal::MetalBuffer gateScratch = backend.allocateBuffer(uint64_t{padded} * shape.n * 2, metal::BufferStorage::Shared, "gate");
      // ops::Linear baseline.
      ops::LinearScratchSize size = linear.decodeScratchSize(w.shape());
      ops::LinearScratch scratch{backend.allocateBuffer(std::max<uint64_t>(size.input, 16), metal::BufferStorage::Shared, "t"),
                                 backend.allocateBuffer(std::max<uint64_t>(size.sums, 16), metal::BufferStorage::Shared, "s"),
                                 backend.allocateBuffer(std::max<uint64_t>(size.partials, 16), metal::BufferStorage::Shared, "p"),
                                 backend.allocateBuffer(std::max<uint64_t>(size.counters, 16), metal::BufferStorage::Shared, "c")};
      std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
      const ops::LinearEpilogue epilogue = shape.mode == Mode::Residual ? ops::LinearEpilogue::Residual
                                           : shape.mode == Mode::GateUp ? ops::LinearEpilogue::GateUp
                                                                                  : ops::LinearEpilogue::None;
      const ops::LinearPlan plan = linear.decodePlan(w, lanes, epilogue, shape.mode == Mode::GateUp ? &up : nullptr, rows);
      const auto addPlan = [&](metal::CommandGraph &g, bool withNorm) {
        ops::PreparedInput prepared;
        if (withNorm) prepared = ops::Normalization::addRms(g, input, ops::NormWeights{normWeight}, normalized, shape.k, padded, scratch, plan.input());
        linear.add(g, {.input = norm ? normalized : input, .output = output, .residual = shape.mode == Mode::Residual ? residual : metal::MetalBuffer{},
                       .gateScratch = gateScratch, .scratch = scratch, .prepared = prepared},
                   ws[turn % copies], plan, shape.mode == Mode::GateUp ? &ups[turn % copies] : nullptr);
        ++turn;
      };
      // The norm folded into the projection, where the plan folds it (one
      // live row); its output must equal the norm + projection chain's bit
      // for bit.
      const bool fold = norm && plan.foldsNorm(rows);
      const auto addFolded = [&](metal::CommandGraph &g, uint32_t c) {
        linear.add(g, {.input = input, .output = output, .gateScratch = gateScratch, .scratch = scratch,
                       .normWeight = normWeight, .normRows = rows},
                   ws[c], plan, shape.mode == Mode::GateUp ? &ups[c] : nullptr);
      };
      bool identical = false;
      if (fold) {
        const uint64_t outputBytes = uint64_t{rows} * shape.n * (shape.f32 ? 4 : 2);
        metal::CommandGraph chain;
        turn = 0;
        addPlan(chain, true);
        (void)backend.submitCommandAsync(chain.command()).wait();
        const std::vector<uint8_t> expected(static_cast<const uint8_t *>(output.contents()),
                                            static_cast<const uint8_t *>(output.contents()) + outputBytes);
        metal::CommandGraph folded;
        addFolded(folded, 0);
        (void)backend.submitCommandAsync(folded.command()).wait();
        identical = !std::memcmp(expected.data(), output.contents(), outputBytes);
      }
      // Three interleaved rounds, the best of each.
      double baseline = 1e9, projectionOnly = 1e9, folded = 1e9;
      for (int round = 0; round < 3; ++round) {
        baseline = std::min(baseline, timeChain(backend, [&](metal::CommandGraph &g) { addPlan(g, norm); }));
        projectionOnly = std::min(projectionOnly, timeChain(backend, [&](metal::CommandGraph &g) { addPlan(g, false); }));
        if (fold) folded = std::min(folded, timeChain(backend, [&](metal::CommandGraph &g) { addFolded(g, turn++ % copies); }));
      }
      std::printf("%-20s linear %-28s %7.4f ms %6.0f GB/s (projection alone %7.4f ms)\n", shape.name, std::string(plan.pipeline()).c_str(), baseline, bytes / baseline / 1e6, projectionOnly);
      if (fold)
        std::printf("%-20s folded %-28s %7.4f ms %6.0f GB/s (%s)\n", "", std::string(plan.pipeline()).c_str(), folded,
                    bytes / folded / 1e6, identical ? "bit-identical" : "DIFFERS");
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "omni dq4 bench: %s\n", error.what());
    return 1;
  }
  return 0;
}
