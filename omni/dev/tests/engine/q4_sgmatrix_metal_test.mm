#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "tuning/LinearNumerics.hpp"

#include "LinearInputReference.hpp"
#include "NormReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;
using namespace splash::test;
namespace {
struct Guarded {
  metal::MetalBuffer backing, view;
  uint64_t size;
  Guarded(metal::MetalBackend &backend, uint64_t bytes) : size(bytes) {
    backing = test::sharedBuffer(backend, bytes + 256);
    std::memset(backing.contents(), 0xa5, bytes + 256);
    if (bytes) view = backend.view(backing, 0, bytes);
  }
  void check() const {
    const auto *p = static_cast<const uint8_t *>(backing.contents());
    for (uint64_t i = size; i < size + 256; ++i) require(p[i] == 0xa5, "out-of-view write");
  }
};
uint32_t hash(uint32_t v) { v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; return v * 0x846ca68b; }
Projection weights(metal::MetalBackend &backend, LinearMatrix shape, uint32_t seed, bool zero) {
  const uint64_t params = uint64_t(shape.outputSize) * shape.inputSize / 64;
  Projection p(shape.outputSize, shape.inputSize,
               AffineWeights{test::sharedBuffer(backend, params * 32), test::sharedBuffer(backend, params * 2),
                             test::sharedBuffer(backend, params * 2)});
  auto *q = static_cast<uint8_t *>(p.affine().weights.contents());
  auto *s = static_cast<uint16_t *>(p.affine().scales.contents());
  auto *b = static_cast<uint16_t *>(p.affine().biases.contents());
  for (uint64_t i = 0; i < params * 32; ++i) q[i] = zero ? 0 : hash(uint32_t(i) + seed);
  for (uint64_t i = 0; i < params; ++i) {
    s[i] = floatToBf16((int(hash(uint32_t(i) + seed + 7) % 17) - 8) / 2048.0f);
    b[i] = zero ? 0 : floatToBf16(-float((hash(uint32_t(i) + seed + 11) % 16)) * bf16ToFloat(s[i]));
  }
  return p;
}
// The fp64 projection of one output and the operand magnitudes its error
// bound scales with.
struct Exact { double value, quantMagnitude, magnitude; };
Exact exact(const Projection &p, const uint16_t *input, uint32_t row, uint32_t col) {
  const auto *q = static_cast<const uint8_t *>(p.affine().weights.contents());
  const auto *sc = static_cast<const uint16_t *>(p.affine().scales.contents());
  const auto *bi = static_cast<const uint16_t *>(p.affine().biases.contents());
  double value = 0, magnitude = 0, quantMagnitude = 0;
  const uint32_t groups = p.inputSize / 64;
  constexpr uint32_t tile = SPLASH_AFFINE_TILE_ROWS;
  for (uint32_t g = 0; g < groups; ++g) {
    const uint64_t at = (uint64_t(col / tile) * groups + g) * tile + col % tile;
    double dot = 0, sum = 0, absolute = 0;
    for (uint32_t k = 0; k < 64; ++k) {
      const double x = bf16ToFloat(input[uint64_t(row) * p.inputSize + g * 64 + k]);
      const uint32_t nibble = (q[at * 32 + k / 2] >> (4 * (k % 2))) & 15;
      dot += x * nibble; sum += x; absolute += std::abs(x);
    }
    const double scale = bf16ToFloat(sc[at]), bias = bf16ToFloat(bi[at]);
    value += scale * dot + bias * sum;
    quantMagnitude += absolute * std::abs(scale);
    magnitude += absolute * (15 * std::abs(scale) + std::abs(bias));
  }
  return {value, quantMagnitude, magnitude};
}
struct Reference { double value, error; };
// The kernel's bf16 projection and its error bound with K split `splits` ways.
Reference reference(const Exact &e, uint32_t groups, uint32_t splits) {
  // gamma_n bounds a chain of n fp32 roundings. Within a 64-element group,
  // operands are <=143|x|; budget 64 dot steps plus 8 for the row sum and
  // offset correction. Across groups budget two affine FMAs and S additions.
  // Absolute operand magnitudes make this valid even under cancellation.
  constexpr double u = 0x1p-24;
  const auto gamma = [&](double n) { return n * u / (1 - n * u); };
  const double error = gamma(72) * 143 * e.quantMagnitude + gamma(2 * groups + splits) * e.magnitude;
  return {double(bf16ToFloat(floatToBf16(float(e.value)))),
          error + ulpBf16(float(e.value))};
}
// The reference of an output after its epilogue: plus the residual, or
// times silu(gate) for GateUp.
Reference withEpilogue(Reference ref, LinearEpilogue epilogue, double residual, Reference gate) {
  if (epilogue == LinearEpilogue::Residual) ref.value += residual;
  if (epilogue == LinearEpilogue::GateUp) {
    const double activation = gate.value / (1 + std::exp(-gate.value));
    ref.error = 1.1 * gate.error * (std::abs(ref.value) + ref.error) + std::abs(activation) * ref.error;
    ref.value *= activation;
  }
  return ref;
}
// Whether a bf16 output is finite and within the reference's bound.
bool within(const Reference &ref, uint16_t actual) {
  const double expected = bf16ToFloat(floatToBf16(float(ref.value)));
  const double value = bf16ToFloat(actual);
  return std::isfinite(value) &&
         std::abs(value - expected) <= ref.error + ulpBf16(float(expected)) + ulpBf16(float(value));
}
// The tile's plan: the Q4 register tile (four simdgroups) or one lane's
// LaneSplit128 tile (eight).
LinearPlan tilePlan(LinearWorkload workload, LinearTile tile, uint32_t splits) {
  return Linear::plan(workload,
                      {tile, 0, tile == LinearTile::Q4Register ? LinearSimdgroups::Four : LinearSimdgroups::Eight,
                       splits},
                      FloatOutput::BFloat16);
}
void runCase(metal::MetalBackend &backend, uint32_t n, uint32_t k, uint32_t splits,
             LinearEpilogue epilogue, uint32_t fixture, uint32_t rows,
             LinearTile tile = LinearTile::Q4Register) {
  const LinearWorkload workload{{n, k}, rows, LinearPhase::Decode, epilogue};
  const auto plan = tilePlan(workload, tile, splits);
  const auto size = plan.scratchSize();
  Guarded input(backend, 2ULL * rows * k), output(backend, 2ULL * rows * n), residual(backend, 2ULL * rows * n);
  Guarded table(backend, size.input), sums(backend, size.sums), partials(backend, size.partials), counters(backend, size.counters);
  LinearScratch scratch{table.view, sums.view, partials.view, counters.view};
  if (size.counters) std::memset(counters.view.contents(), 0, size.counters);
  auto *x = static_cast<uint16_t *>(input.view.contents());
  auto *r = static_cast<uint16_t *>(residual.view.contents());
  for (uint32_t i = 0; i < rows * k; ++i) {
    float v = float(int(hash(i + 37) % 257) - 128) / 32;
    if (fixture == 1) v *= 262144; // bf16 values above half's range.
    if (fixture == 2) v = (i % 2 ? -1 : 1) * 131072.0f + float(i % 7) * 1024;
    if (fixture == 3) v = float(int(i % 5) - 2) * 524288;
    x[i] = floatToBf16(v);
  }
  for (uint32_t i = 0; i < rows * n; ++i) r[i] = floatToBf16(float(int(i % 31) - 15) / 8);
  const auto p = weights(backend, {n,k}, 31, fixture == 3);
  const auto gate = weights(backend, {n,k}, 177, fixture == 3);
  Linear linear(backend.capabilities());
  LinearBuffers b{input.view, output.view, {}, epilogue == LinearEpilogue::Residual ? residual.view : metal::MetalBuffer{}, {}, {}, scratch};
  metal::CommandGraph graph;
  // Reuse one workspace repeatedly in a single command to expose incomplete
  // publication, stale counters and dependencies between consecutive dispatches.
  for (uint32_t repeat = 0; repeat < 3; ++repeat)
    linear.add(graph, b, p, plan, epilogue == LinearEpilogue::GateUp ? &gate : nullptr);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const auto *prepared = static_cast<const uint16_t *>(table.view.contents());
  const auto *rowSums = static_cast<const float *>(sums.view.contents());
  // Only the register tile prepares a table.
  for (uint32_t g=0;tile==LinearTile::Q4Register&&g<k/64;++g) for (uint32_t row=0;row<rows;++row) {
    float sum=0;
    for (uint32_t z=0;z<64;++z) {
      const uint32_t j=(z%16/8)*4+z%4, kp=2*(z/16)+(z/4)%2;
      const uint32_t at=(row/8)*k*8+g*512+(((j/4)*8+kp)*4+(row%8)/2)*8+(j%4)*2+row%2;
      require(prepared[at]==x[row*k+g*64+z],"prepared input layout mismatch");
      sum+=bf16ToFloat(x[row*k+g*64+z]);
    }
    require(sum==rowSums[(row/8)*k/8+g*8+row%8],"prepared sum mismatch");
  }
  const auto *actual = static_cast<const uint16_t *>(output.view.contents());
  const std::vector<uint16_t> first(actual, actual + rows * n);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  require(std::memcmp(first.data(), actual, 2ULL * rows * n) == 0, "nondeterministic split reduction");
  const auto *counts = static_cast<const uint32_t *>(counters.view.contents());
  for (uint64_t i = 0; i < size.counters / 4; ++i) require(counts[i] == 0, "counter not reset");
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col : {0U, 7U, 8U, 31U, 32U, 63U, 64U, 127U, 128U, 255U, n-1}) {
      const auto ref = withEpilogue(reference(exact(p, x, row, col), k / 64, splits), epilogue,
                                    bf16ToFloat(r[row * n + col]),
                                    epilogue == LinearEpilogue::GateUp
                                        ? reference(exact(gate, x, row, col), k / 64, splits) : Reference{});
      const uint16_t value = actual[row * n + col];
      if (!within(ref, value)) {
        std::cerr << "tile=" << int(tile) << " M=" << rows << " N=" << n << " K=" << k << " S=" << splits << " epilogue=" << int(epilogue)
                  << " fixture=" << fixture << " row=" << row << " col=" << col
                  << " actual=" << bf16ToFloat(value) << " reference=" << ref.value << " error=" << ref.error << '\n';
        throw std::runtime_error("Q4 tile result exceeds independent fp64 error bound");
      }
      if (fixture == 3)
        require(bf16ToFloat(value) == bf16ToFloat(floatToBf16(float(ref.value))),
                "zero-weight offset cancellation is not exact");
    }
  }
  for (auto *guard : {&input,&output,&residual,&table,&sums,&partials,&counters}) guard->check();
}
// Two production projections that run back to back in a decode step share
// one LinearScratch, as every projection there does, at each pair of K
// splits the Apple9 policy picks for one core count. The partials are
// overwritten before each projection, with a large finite value and in
// another run with NaN, so reading a partial before its writer published it
// changes the output. Every output element must meet the fp64 bound, as the
// unsplit outputs must, and not depend on the poison or the run; every
// counter must return to zero.
struct SplitOperand {
  LinearWorkload workload;
  Projection weights, gate;
  metal::MetalBuffer input, residual, output;
  std::vector<Exact> exact, gateExact;
};
void requireFp64(const SplitOperand &o, uint32_t splits, const std::string &what) {
  const auto [n, k] = o.workload.matrix;
  const auto *actual = static_cast<const uint16_t *>(o.output.contents());
  const auto *r = static_cast<const uint16_t *>(o.residual.contents());
  const bool gateUp = o.workload.epilogue == LinearEpilogue::GateUp;
  for (uint64_t i = 0; i < uint64_t{o.workload.rows} * n; ++i) {
    const auto ref = withEpilogue(reference(o.exact[i], k / 64, splits), o.workload.epilogue, bf16ToFloat(r[i]),
                                  gateUp ? reference(o.gateExact[i], k / 64, splits) : Reference{});
    if (!within(ref, actual[i]))
      throw std::runtime_error(what + ": element " + std::to_string(i) + " exceeds the fp64 bound");
  }
}
// `tile` is the register tile (Apple9's policy) or LaneSplit128 (Apple10's,
// one lane).
void splitVisibility(metal::MetalBackend &backend,
                     std::array<std::pair<LinearMatrix, LinearEpilogue>, 2> pair, uint32_t lanes,
                     LinearTile tile = LinearTile::Q4Register) {
  const uint32_t rows = lanes * 8;
  std::vector<SplitOperand> operands;
  for (uint32_t i = 0; i < 2; ++i) {
    const auto [matrix, epilogue] = pair[i];
    const auto [n, k] = matrix;
    SplitOperand o{{matrix, rows, LinearPhase::Decode, epilogue},
                   weights(backend, matrix, 31 + 100 * i, false), weights(backend, matrix, 177 + 100 * i, false),
                   test::sharedBuffer(backend, 2ULL * rows * k), test::sharedBuffer(backend, 2ULL * rows * n),
                   test::sharedBuffer(backend, 2ULL * rows * n), {}, {}};
    auto *x = static_cast<uint16_t *>(o.input.contents());
    auto *r = static_cast<uint16_t *>(o.residual.contents());
    for (uint32_t j = 0; j < rows * k; ++j) x[j] = floatToBf16(float(int(hash(j + 37 + 1000 * i) % 257) - 128) / 32);
    for (uint32_t j = 0; j < rows * n; ++j) r[j] = floatToBf16(float(int(j % 31) - 15) / 8);
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t col = 0; col < n; ++col) {
        o.exact.push_back(exact(o.weights, x, row, col));
        if (epilogue == LinearEpilogue::GateUp) o.gateExact.push_back(exact(o.gate, x, row, col));
      }
    operands.push_back(std::move(o));
  }
  std::set<std::array<uint32_t, 2>> splitPairs;
  for (uint32_t cores = 8; cores <= 80; ++cores) {
    DeviceCapabilities device;
    device.appleGpuFamily = tile == LinearTile::Q4Register ? 9 : 10;
    device.gpuCoreCount = cores;
    const Linear policy(device);
    const auto a = policy.plan(operands[0].workload).configuration();
    const auto b = policy.plan(operands[1].workload).configuration();
    if (a.tile == tile && b.tile == tile && std::max(a.splits, b.splits) > 1)
      splitPairs.insert({a.splits, b.splits});
  }
  require(!splitPairs.empty(), "the policy splits neither projection");
  const auto plan = [&](uint32_t i, uint32_t splits) { return tilePlan(operands[i].workload, tile, splits); };
  LinearScratchSize size;
  const auto grow = [&](const LinearPlan &p) {
    const auto s = p.scratchSize();
    size = {std::max(size.input, s.input), std::max(size.sums, s.sums), std::max(size.partials, s.partials),
            std::max(size.counters, s.counters)};
  };
  for (uint32_t i = 0; i < 2; ++i) grow(plan(i, 1));
  for (const auto &splits : splitPairs) for (uint32_t i = 0; i < 2; ++i) grow(plan(i, splits[i]));
  Guarded table(backend, size.input), sums(backend, size.sums), partials(backend, size.partials), counters(backend, size.counters);
  if (size.counters) std::memset(counters.view.contents(), 0, size.counters);
  const LinearScratch scratch{table.view, sums.view, partials.view, counters.view};
  const auto poison = test::sharedBuffer(backend, size.partials);
  const Linear linear(backend.capabilities());
  const auto add = [&](metal::CommandGraph &graph, uint32_t i, uint32_t splits) {
    const SplitOperand &o = operands[i];
    const bool gateUp = o.workload.epilogue == LinearEpilogue::GateUp;
    linear.add(graph, {o.input, o.output, {}, o.workload.epilogue == LinearEpilogue::Residual ? o.residual : metal::MetalBuffer{},
                       {}, {}, scratch}, o.weights, plan(i, splits), gateUp ? &o.gate : nullptr);
  };
  const std::string shape = std::string(tile == LinearTile::Q4Register ? "register tile, " : "LaneSplit128, ") +
      std::to_string(lanes) + " lanes, " + std::to_string(pair[0].first.outputSize) + "x" +
      std::to_string(pair[0].first.inputSize) + " then " + std::to_string(pair[1].first.outputSize) + "x" +
      std::to_string(pair[1].first.inputSize);
  metal::CommandGraph unsplit;
  for (uint32_t i = 0; i < 2; ++i) add(unsplit, i, 1);
  (void)backend.submitCommandAsync(unsplit.dispatches()).wait();
  for (uint32_t i = 0; i < 2; ++i) requireFp64(operands[i], 1, shape + " unsplit");
  for (const auto &splits : splitPairs) {
    const std::string what = shape + " splits " + std::to_string(splits[0]) + "/" + std::to_string(splits[1]);
    metal::CommandGraph graph;
    for (uint32_t i = 0; i < 2; ++i) {
      // A test kernel's copy poisons the partials in order.
      graph.add("test_copy_u32", {poison, partials.view}, uint32_t(size.partials / 4),
                {uint32_t((size.partials / 4 + 255) / 256), 1, 1}, {256, 1, 1});
      add(graph, i, splits[i]);
    }
    std::vector<std::vector<uint8_t>> first;
    for (uint32_t bits : {0x7E800000U, 0x7FC00000U, 0x7E800000U}) {
      std::fill_n(static_cast<uint32_t *>(poison.contents()), size.partials / 4, bits);
      (void)backend.submitCommandAsync(graph.dispatches()).wait();
      const auto *counts = static_cast<const uint32_t *>(counters.view.contents());
      for (uint64_t c = 0; c < size.counters / 4; ++c) require(counts[c] == 0, "counter not reset");
      for (uint32_t i = 0; i < 2; ++i) {
        const auto *bytes = static_cast<const uint8_t *>(operands[i].output.contents());
        const std::vector<uint8_t> output(bytes, bytes + operands[i].output.sizeBytes());
        if (first.size() < 2) first.push_back(output);
        else if (output != first[i]) throw std::runtime_error(what + ": output depends on the poison or the run");
      }
    }
    for (uint32_t i = 0; i < 2; ++i) requireFp64(operands[i], splits[i], what);
  }
  for (auto *guard : {&table, &sums, &partials, &counters}) guard->check();
}
// Rows of widely spread values and a norm's bf16 weights.
struct NormCase {
  metal::MetalBuffer input;
  NormWeights weight;
};
NormCase normCase(metal::MetalBackend &backend, uint32_t k, uint32_t rows) {
  NormCase c{test::sharedBuffer(backend, k*rows*2),
             makeNormWeights(backend, k, [&](uint32_t i) { return float(int(i%17)-8)/4; })};
  auto *x=static_cast<uint16_t *>(c.input.contents());
  for (uint32_t i=0;i<k*rows;++i) x[i]=floatToBf16(float(int(hash(i)%257)-128)*8192);
  return c;
}
void requireNorm(const NormCase &c, const metal::MetalBuffer &output, uint32_t k, uint32_t rows,
                 const char *what) {
  const auto *x=static_cast<const uint16_t *>(c.input.contents());
  const auto *out=static_cast<const uint16_t *>(output.contents());
  for (uint32_t r=0;r<rows;++r) {
    const std::vector<double> exact=rmsNorm(x+uint64_t{r}*k,c.weight,k);
    for (uint32_t i=0;i<k;++i) require(roundedOnceToBf16(out[r*k+i],exact[i]),what);
  }
}
void fusedNorm(metal::MetalBackend &backend, uint32_t k, uint32_t rows, LinearInput layout) {
  const uint64_t sumsBytes = tableSumsBytes(layout, k, rows);
  const NormCase c=normCase(backend,k,rows);
  auto output=test::sharedBuffer(backend, k*rows*2), fused=test::sharedBuffer(backend, k*rows*2);
  auto a=test::sharedBuffer(backend, tableBytes(k,rows)), b=test::sharedBuffer(backend, tableBytes(k,rows));
  auto sa=test::sharedBuffer(backend, sumsBytes), sb=test::sharedBuffer(backend, sumsBytes);
  metal::CommandGraph graph;
  require(Normalization::addRms(graph,c.input,c.weight,output,k,rows).layout==LinearInput::Plain,
          "plain norm claimed a table");
  require(graph.dispatches().back().pipelineName.starts_with("norm_rms_staged")==(k<=SPLASH_STAGED_NORM_WIDTH),
          "plain norm staged the wrong widths");
  addReferencePreparation(graph,layout,output,a,sa,k,rows/8);
  const PreparedInput prepared=Normalization::addRms(graph,c.input,c.weight,fused,k,rows,{b,sb,{},{}},layout);
  require(prepared.layout==layout && prepared.source.sameView(fused),"fused norm did not report the table it wrote");
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  requireNorm(c,output,k,rows,"norm differs from the fp64 reference");
  require(!std::memcmp(output.contents(),fused.contents(),k*rows*2),"fused norm changed bf16 output");
  require(!std::memcmp(a.contents(),b.contents(),tableBytes(k,rows)),"fused operand permutation mismatch");
  require(!std::memcmp(sa.contents(),sb.contents(),sumsBytes),"fused input sums mismatch");
}
// The affine prefill's norm, with bf16 weights only, whose rows need not fill
// its 32-row sum tiles. Its rows equal the plain norm's bit for bit, which a
// prefill whose consumer reads no sums runs instead.
void prefillNorm(metal::MetalBackend &backend, uint32_t k, uint32_t rows) {
  const NormCase c=normCase(backend,k,rows);
  auto output=test::sharedBuffer(backend, k*rows*2), plain=test::sharedBuffer(backend, k*rows*2);
  auto sums=test::sharedBuffer(backend, (rows+31)/32*32*(k/64)*4);
  metal::CommandGraph graph;
  Normalization::addRmsWithQ4Sums(graph,c.input,c.weight,output,sums,k,rows);
  Normalization::addRms(graph,c.input,c.weight,plain,k,rows);
  require(graph.dispatches().back().pipelineName.starts_with("norm_rms_staged")==
              (rows<=SPLASH_STAGED_NORM_ROWS && k<=SPLASH_STAGED_NORM_WIDTH),
          "plain norm staged the wrong rows");
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  requireNorm(c,output,k,rows,"prefill norm differs from the fp64 reference");
  require(!std::memcmp(output.contents(),plain.contents(),k*rows*2),"prefill norm rows differ from the plain norm's");
}
// Each buffer the norms reach, at its extent and one element short: the rows
// of inputs and outputs, the weights, the table and sums a table layout
// writes, and the Q4-sum norm's sums, [32-row tile][64-input group][row of
// the tile], which end at the last row's sum of the last group. The Q4-sum
// norm refuses no rows or a width of partial groups before anything is
// encoded.
void normExtents(metal::MetalBackend &backend) {
  const auto requireNormExtents = [&](uint32_t width, uint32_t rows, const auto &encode) {
    const NormCase c = normCase(backend, width, rows);
    const metal::MetalBuffer output = test::sharedBuffer(backend, uint64_t{width} * rows * 2);
    const uint64_t weightElement = 2;
    requireExtent(backend, c.input, uint64_t{width} * rows * 2, 2, "norm input",
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &input) {
                    encode(graph, input, c.weight, output);
                  });
    requireExtent(backend, output, uint64_t{width} * rows * 2, 2, "norm output",
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    encode(graph, c.input, c.weight, view);
                  });
    requireExtent(backend, c.weight.buffer, width * weightElement, weightElement, "norm weight",
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &weights) {
                    encode(graph, c.input, NormWeights{weights}, output);
                  });
  };
  // The plain kernel, the staged one and the table layout.
  for (const auto [width, rows] : {std::pair{5120U, 37U}, std::pair{2048U, 8U}})
    requireNormExtents(width, rows,
                       [&](metal::CommandGraph &graph, const metal::MetalBuffer &input, const NormWeights &weight,
                           const metal::MetalBuffer &output) {
                         (void)Normalization::addRms(graph, input, weight, output, width, rows);
                       });
  constexpr uint32_t width = 5120;
  {
    constexpr LinearInput layout = LinearInput::Table64;
    constexpr uint32_t rows = 16;
    const metal::MetalBuffer table = test::sharedBuffer(backend, tableBytes(width, rows)),
                             sums = test::sharedBuffer(backend, tableSumsBytes(layout, width, rows));
    const auto encode = [&](metal::CommandGraph &graph, const metal::MetalBuffer &input, const NormWeights &weight,
                            const metal::MetalBuffer &output, const LinearScratch &scratch) {
      (void)Normalization::addRms(graph, input, weight, output, width, rows, scratch, layout);
    };
    requireNormExtents(width, rows,
                       [&](metal::CommandGraph &graph, const metal::MetalBuffer &input, const NormWeights &weight,
                           const metal::MetalBuffer &output) {
                         encode(graph, input, weight, output, {table, sums, {}, {}});
                       });
    const NormCase c = normCase(backend, width, rows);
    const metal::MetalBuffer output = test::sharedBuffer(backend, uint64_t{width} * rows * 2);
    requireExtent(backend, table, tableBytes(width, rows), 2, "linear table",
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    encode(graph, c.input, c.weight, output, {view, sums, {}, {}});
                  });
    requireExtent(backend, sums, tableSumsBytes(layout, width, rows), 4, "linear table sums",
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    encode(graph, c.input, c.weight, output, {table, view, {}, {}});
                  });
  }
  constexpr uint32_t rows = 37, groups = width / 64;
  // Row 36 is row 4 of the second tile.
  const uint64_t sumsBytes = ((uint64_t{1} * groups + groups - 1) * 32 + 4 + 1) * 4;
  const metal::MetalBuffer sums = test::sharedBuffer(backend, sumsBytes);
  requireNormExtents(width, rows,
                     [&](metal::CommandGraph &graph, const metal::MetalBuffer &input, const NormWeights &weight,
                         const metal::MetalBuffer &output) {
                       Normalization::addRmsWithQ4Sums(graph, input, weight, output, sums, width, rows);
                     });
  const NormCase c = normCase(backend, width, rows);
  const metal::MetalBuffer output = test::sharedBuffer(backend, uint64_t{width} * rows * 2);
  requireExtent(backend, sums, sumsBytes, 4, "norm sums", [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
    Normalization::addRmsWithQ4Sums(graph, c.input, c.weight, output, view, width, rows);
  });
  metal::CommandGraph graph;
  for (const auto &[normWidth, normRows] :
       {std::pair{width - 32, rows}, std::pair{0U, rows}, std::pair{width, 0U}})
    rejects([&] { Normalization::addRmsWithQ4Sums(graph, c.input, c.weight, output, sums, normWidth, normRows); },
            "the Q4-sum norm takes whole 64-input groups",
            "the Q4-sum norm took a partial group or no rows");
  require(graph.empty(), "a refused Q4-sum norm encoded a dispatch");
}

// The folded norm's plan and buffer checks: only a one-row Qmv plan's plain
// (or gate/up) projection folds a norm, of its live row; the split tiles and
// a residual projection take the norm's dispatch, and the norm's weights
// hold the input width.
void foldedNormRefusals(metal::MetalBackend &backend) {
  constexpr uint32_t n = 256, k = 512;
  const auto weight = weights(backend, {n, k}, 31, false);
  const NormCase c = normCase(backend, k, 8);
  const metal::MetalBuffer output = test::sharedBuffer(backend, 8ULL * n * 2),
                           residual = test::sharedBuffer(backend, 8ULL * n * 2),
                           partials = test::sharedBuffer(backend, 8ULL * 8 * n * 4),
                           counters = test::sharedBuffer(backend, 64);
  const LinearScratch scratch{{}, {}, partials, counters};
  const Linear linear(backend.capabilities());
  const LinearWorkload one{{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None, WeightLayout::Affine64, 1};
  LinearWorkload residualOne = one;
  residualOne.epilogue = LinearEpilogue::Residual;
  const LinearPlan plain = Linear::plan(one, {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, 4}, FloatOutput::BFloat16);
  const LinearPlan residualPlan =
      Linear::plan(residualOne, {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, 4}, FloatOutput::BFloat16);
  const LinearPlan split = Linear::plan({{n, k}, 16, LinearPhase::Decode, LinearEpilogue::None},
                                        {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}, FloatOutput::BFloat16);
  const LinearPlan lane = Linear::plan({{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                       {LinearTile::LaneSplit128, 0, LinearSimdgroups::Eight, 1}, FloatOutput::BFloat16);
  require(plain.canFoldNorm() && !residualPlan.canFoldNorm() && !split.canFoldNorm() && !lane.canFoldNorm(),
          "only a one-row Qmv plan's plain and gate/up projections fold a norm");
  require(plain.foldsNorm(1) && !plain.foldsNorm(0) && !plain.foldsNorm(2) && !residualPlan.foldsNorm(1) &&
              !lane.foldsNorm(1),
          "the folded norm's row policy");
  metal::CommandGraph graph;
  rejects([&] {
    linear.add(graph, {.input = c.input, .output = output, .residual = residual, .scratch = scratch,
                       .normWeight = c.weight.buffer, .normRows = 1}, weight, residualPlan);
  }, "the plan does not fold a norm", "a residual plan took norm weights");
  rejects([&] {
    linear.add(graph, {.input = c.input, .output = output, .scratch = scratch, .normWeight = c.weight.buffer,
                       .normRows = 1}, weight, lane);
  }, "the plan does not fold a norm", "a lane split plan took norm weights");
  for (const uint32_t rows : {0U, 9U})
    rejects([&] {
      linear.add(graph, {.input = c.input, .output = output, .scratch = scratch, .normWeight = c.weight.buffer,
                         .normRows = rows}, weight, plain);
    }, "a folded norm takes 1 to the lane's rows", "a folded norm took 0 or 9 rows");
  requireExtent(backend, c.input, uint64_t{k} * 2, 2, "projection input", [&](metal::CommandGraph &g, const metal::MetalBuffer &input) {
    linear.add(g, {.input = input, .output = output, .scratch = scratch, .normWeight = c.weight.buffer, .normRows = 1},
               weight, plain);
  });
  requireExtent(backend, c.weight.buffer, uint64_t{k} * 2, 2, "norm weight", [&](metal::CommandGraph &g, const metal::MetalBuffer &norm) {
    linear.add(g, {.input = c.input, .output = output, .scratch = scratch, .normWeight = norm, .normRows = 1},
               weight, plain);
  });
  require(graph.empty(), "a refused folded norm encoded a dispatch");
}

// The one-row Qmv tile (kernels/decode/linear_q4_qmv.metal). Its sums chain
// at most K / 512 steps per lane and then the 32 lanes, which the bound of
// 32 K splits covers.
constexpr uint32_t kQmvLanes = 32;
LinearPlan qmvPlan(LinearMatrix matrix, LinearEpilogue epilogue, uint32_t outputs, FloatOutput destination) {
  return Linear::plan({matrix, 8, LinearPhase::Decode, epilogue, WeightLayout::Affine64, 1},
                      {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, outputs}, destination);
}
// One live row of the lane through the Qmv tile at `outputs` outputs per
// simdgroup: within the fp64 bound (logits: unrounded fp32) at every output,
// the same bits through repeated dispatches, reading the live row only (a
// view of one row) and writing nothing past it in the lane's eight rows.
// Returns the output row, which no Results changes.
std::vector<uint8_t> qmvCase(metal::MetalBackend &backend, uint32_t n, uint32_t k, LinearEpilogue epilogue,
                             uint32_t fixture, uint32_t outputs, FloatOutput destination = FloatOutput::BFloat16) {
  const LinearPlan plan = qmvPlan({n, k}, epilogue, outputs, destination);
  const uint64_t element = elementBytes(destination), rowBytes = uint64_t{n} * element;
  Guarded input(backend, 2ULL * k), output(backend, 8 * rowBytes), residual(backend, 2ULL * n);
  std::memset(output.view.contents(), 0x5a, 8 * rowBytes);
  auto *x = static_cast<uint16_t *>(input.view.contents());
  auto *r = static_cast<uint16_t *>(residual.view.contents());
  for (uint32_t i = 0; i < k; ++i) {
    float v = float(int(hash(i + 37) % 257) - 128) / 32;
    if (fixture == 1) v *= 262144;
    if (fixture == 2) v = (i % 2 ? -1 : 1) * 131072.0f + float(i % 7) * 1024;
    if (fixture == 3) v = float(int(i % 5) - 2) * 524288;
    x[i] = floatToBf16(v);
  }
  for (uint32_t i = 0; i < n; ++i) r[i] = floatToBf16(float(int(i % 31) - 15) / 8);
  const auto p = weights(backend, {n, k}, 31, fixture == 3), gate = weights(backend, {n, k}, 177, fixture == 3);
  const bool gateUp = epilogue == LinearEpilogue::GateUp;
  const Linear linear(backend.capabilities());
  metal::CommandGraph graph;
  for (uint32_t repeat = 0; repeat < 3; ++repeat)
    linear.add(graph,
               {.input = input.view, .output = output.view,
                .residual = epilogue == LinearEpilogue::Residual ? residual.view : metal::MetalBuffer{}},
               p, plan, gateUp ? &gate : nullptr);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const auto *bytes = static_cast<const uint8_t *>(output.view.contents());
  const std::vector<uint8_t> first(bytes, bytes + rowBytes);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const std::string what = "Qmv N=" + std::to_string(n) + " K=" + std::to_string(k) + " outputs=" +
                           std::to_string(outputs) + " epilogue=" + std::to_string(int(epilogue)) + " fixture=" +
                           std::to_string(fixture) + (destination == FloatOutput::Float32 ? " fp32" : "");
  require(std::memcmp(first.data(), bytes, rowBytes) == 0, what + ": nondeterministic");
  for (uint64_t i = rowBytes; i < 8 * rowBytes; ++i) require(bytes[i] == 0x5a, what + ": wrote past the live row");
  for (uint32_t col = 0; col < n; ++col) {
    const Reference gateRef = gateUp ? reference(exact(gate, x, 0, col), k / 64, kQmvLanes) : Reference{};
    const Reference ref = withEpilogue(reference(exact(p, x, 0, col), k / 64, kQmvLanes), epilogue,
                                       bf16ToFloat(r[col]), gateRef);
    if (destination == FloatOutput::Float32) {
      const float value = reinterpret_cast<const float *>(bytes)[col];
      require(std::isfinite(value) && std::abs(value - ref.value) <= ref.error,
              what + ": column " + std::to_string(col) + " exceeds the fp64 bound");
    } else {
      const uint16_t value = reinterpret_cast<const uint16_t *>(bytes)[col];
      require(within(ref, value), what + ": column " + std::to_string(col) + " exceeds the fp64 bound");
      if (fixture == 3)
        require(bf16ToFloat(value) == bf16ToFloat(floatToBf16(float(ref.value))),
                what + ": zero-weight offset cancellation is not exact");
    }
  }
  for (auto *guard : {&input, &output, &residual}) guard->check();
  return first;
}
// Two live rows sharing the Qmv tile's threadgroups (LinearConfig::
// sharedRows): each row's output is the one-row plan's on that row bit for
// bit (whose fp64 bound qmvCase checks), through repeated dispatches,
// reading the two live rows only (a view of two rows) and writing nothing
// past them in the lane's eight rows.
void qmvSharedCase(metal::MetalBackend &backend, uint32_t n, uint32_t k, LinearEpilogue epilogue, uint32_t fixture,
                   FloatOutput destination = FloatOutput::BFloat16) {
  const bool gateUp = epilogue == LinearEpilogue::GateUp;
  const uint32_t outputs = gateUp ? 2 : 4;
  const LinearPlan shared = Linear::plan({{n, k}, 8, LinearPhase::Decode, epilogue, WeightLayout::Affine64, 2},
                                         {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, outputs, true}, destination);
  const LinearPlan one = qmvPlan({n, k}, epilogue, outputs, destination);
  require(!shared.canFoldNorm(), "shared Qmv rows take the norm's dispatch");
  const uint64_t element = elementBytes(destination), rowBytes = uint64_t{n} * element;
  Guarded input(backend, 4ULL * k), output(backend, 8 * rowBytes), residual(backend, 4ULL * n);
  const metal::MetalBuffer alone = test::sharedBuffer(backend, 2 * rowBytes);
  std::memset(output.view.contents(), 0x5a, 8 * rowBytes);
  auto *x = static_cast<uint16_t *>(input.view.contents());
  auto *r = static_cast<uint16_t *>(residual.view.contents());
  for (uint32_t row = 0; row < 2; ++row)
    for (uint32_t i = 0; i < k; ++i) {
      float v = float(int(hash(i * 3 + row * 101 + 37) % 257) - 128) / 32;
      if (fixture == 1) v *= 262144;
      if (fixture == 2) v = (i % 2 ? -1 : 1) * 131072.0f + float((i + row) % 7) * 1024;
      if (fixture == 3) v = float(int((i + row) % 5) - 2) * 524288;
      x[row * k + i] = floatToBf16(v);
    }
  for (uint32_t i = 0; i < 2 * n; ++i) r[i] = floatToBf16(float(int(i % 31) - 15) / 8);
  const auto p = weights(backend, {n, k}, 31, fixture == 3), gate = weights(backend, {n, k}, 177, fixture == 3);
  const Linear linear(backend.capabilities());
  const bool residualMode = epilogue == LinearEpilogue::Residual;
  metal::CommandGraph graph;
  for (uint32_t repeat = 0; repeat < 3; ++repeat)
    linear.add(graph, {.input = input.view, .output = output.view,
                       .residual = residualMode ? residual.view : metal::MetalBuffer{}},
               p, shared, gateUp ? &gate : nullptr);
  for (uint32_t row = 0; row < 2; ++row)
    linear.add(graph,
               {.input = backend.view(input.view, row * 2ULL * k, 2ULL * k),
                .output = backend.view(alone, row * rowBytes, rowBytes),
                .residual = residualMode ? backend.view(residual.view, row * 2ULL * n, 2ULL * n) : metal::MetalBuffer{}},
               p, one, gateUp ? &gate : nullptr);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const std::string what = "shared Qmv N=" + std::to_string(n) + " K=" + std::to_string(k) + " epilogue=" +
                           std::to_string(int(epilogue)) + " fixture=" + std::to_string(fixture) +
                           (destination == FloatOutput::Float32 ? " fp32" : "");
  const auto *bytes = static_cast<const uint8_t *>(output.view.contents());
  require(!std::memcmp(bytes, alone.contents(), 2 * rowBytes), what + ": differs from the one-row plan");
  for (uint64_t i = 2 * rowBytes; i < 8 * rowBytes; ++i) require(bytes[i] == 0x5a, what + ": wrote past the live rows");
  for (auto *guard : {&input, &output, &residual}) guard->check();
}
// The norm folded into a Qmv plan: the norm dispatch then the plain plan
// give the reference, within the fp64 bound of the projection of the norm's
// rows, and the folded plan, reading the hidden row (a view of one row) and
// the norm's weights, gives it bit for bit.
void qmvFoldedNorm(metal::MetalBackend &backend, uint32_t n, uint32_t k, LinearEpilogue epilogue, uint32_t outputs,
                   FloatOutput destination = FloatOutput::BFloat16) {
  const LinearPlan plan = qmvPlan({n, k}, epilogue, outputs, destination);
  require(plan.canFoldNorm() && plan.foldsNorm(1) && !plan.foldsNorm(2), "a Qmv plan folds its one row's norm");
  const bool gateUp = epilogue == LinearEpilogue::GateUp;
  const uint64_t rowBytes = uint64_t{n} * elementBytes(destination);
  const NormCase c = normCase(backend, k, 1);
  auto *x = static_cast<uint16_t *>(c.input.contents());
  for (uint32_t i = 0; i < k; ++i) x[i] = floatToBf16(bf16ToFloat(x[i]) / 65536);
  const auto weight = weights(backend, {n, k}, 31, false), gate = weights(backend, {n, k}, 177, false);
  Guarded folded(backend, rowBytes);
  const metal::MetalBuffer normalized = test::sharedBuffer(backend, 2ULL * k),
                           expected = test::sharedBuffer(backend, rowBytes);
  const Linear linear(backend.capabilities());
  metal::CommandGraph graph;
  (void)Normalization::addRms(graph, c.input, c.weight, normalized, k, 1);
  linear.add(graph, {.input = normalized, .output = expected}, weight, plan, gateUp ? &gate : nullptr);
  for (uint32_t repeat = 0; repeat < 3; ++repeat)
    linear.add(graph, {.input = c.input, .output = folded.view, .normWeight = c.weight.buffer, .normRows = 1}, weight,
               plan, gateUp ? &gate : nullptr);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const std::string what = "Qmv folded norm N=" + std::to_string(n) + " K=" + std::to_string(k) + " outputs=" +
                           std::to_string(outputs) + (gateUp ? " gate/up" : "") +
                           (destination == FloatOutput::Float32 ? " fp32" : "");
  requireNorm(c, normalized, k, 1, (what + ": norm differs from the fp64 reference").c_str());
  const auto *row = static_cast<const uint16_t *>(normalized.contents());
  for (uint32_t col = 0; col < n; ++col) {
    const Reference gateRef = gateUp ? reference(exact(gate, row, 0, col), k / 64, kQmvLanes) : Reference{};
    const Reference ref = withEpilogue(reference(exact(weight, row, 0, col), k / 64, kQmvLanes), epilogue, 0, gateRef);
    if (destination == FloatOutput::Float32) {
      const float value = static_cast<const float *>(expected.contents())[col];
      require(std::isfinite(value) && std::abs(value - ref.value) <= ref.error, what + ": exceeds the fp64 bound");
    } else {
      require(within(ref, static_cast<const uint16_t *>(expected.contents())[col]), what + ": exceeds the fp64 bound");
    }
  }
  require(!std::memcmp(expected.contents(), folded.view.contents(), rowBytes),
          what + ": differs from the norm dispatch and the projection");
  folded.check();
}
// Apple10's policy runs one or two live rows of a lane on the Qmv tile
// (four outputs per simdgroup, two for gate/up; two rows sharing its
// threadgroups) and every other row count, and Apple9, on the tiles; a
// one-row Qmv plan folds its row's norm up to 16384 outputs, a two-row one
// none, and both refuse other live rows, outputs per simdgroup and norm
// rows.
void qmvPolicy(metal::MetalBackend &backend) {
  DeviceCapabilities apple10;
  apple10.appleGpuFamily = 10;
  apple10.gpuCoreCount = 16;
  DeviceCapabilities apple9 = apple10;
  apple9.appleGpuFamily = 9;
  const Linear ten(apple10), nine(apple9);
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
    for (const LinearMatrix matrix : {LinearMatrix{4096, 2048}, LinearMatrix{2048, 6144}, LinearMatrix{152064, 2048}}) {
      const LinearWorkload one{matrix, 8, LinearPhase::Decode, epilogue, WeightLayout::Affine64, 1};
      const LinearConfig config = ten.plan(one).configuration();
      require(config.tile == LinearTile::Qmv &&
                  config.simdgroupOutputs == (epilogue == LinearEpilogue::GateUp ? 2u : 4u),
              "Apple10 runs one live row on the Qmv tile");
      require(!config.sharedRows, "one live row runs alone");
      LinearWorkload two = one;
      two.liveRows = 2;
      const LinearConfig pair = ten.plan(two).configuration();
      require(pair.tile == LinearTile::Qmv && pair.sharedRows &&
                  pair.simdgroupOutputs == (epilogue == LinearEpilogue::GateUp ? 2u : 4u) &&
                  !ten.plan(two).canFoldNorm(),
              "Apple10 runs two live rows on the Qmv tile's shared threadgroups, after the norm's dispatch");
      for (const uint32_t live : {0U, 3U, 8U}) {
        LinearWorkload more = one;
        more.liveRows = live;
        require(ten.plan(more).configuration().tile != LinearTile::Qmv, "only one or two live rows run the Qmv tile");
      }
      require(nine.plan(two).configuration().tile != LinearTile::Qmv, "Apple9 keeps its tiles");
      require(nine.plan(one).configuration().tile != LinearTile::Qmv, "Apple9 keeps its tiles");
      const LinearPlan plan = ten.plan(one);
      require(plan.canFoldNorm() ==
                  (epilogue != LinearEpilogue::Residual && matrix.outputSize <= 16384),
              "a Qmv plan folds the norms of its plain and gate/up projections up to 16384 outputs");
    }
  const LinearWorkload two{{256, 512}, 8, LinearPhase::Decode, LinearEpilogue::None, WeightLayout::Affine64, 2};
  rejects([&] { (void)Linear::plan(two, {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, 4}, FloatOutput::BFloat16); },
          "the Qmv tile projects one live row, or two sharing its threadgroups", "a Qmv plan took two live rows alone");
  LinearWorkload one = two;
  one.liveRows = 1;
  for (const uint32_t live : {1U, 3U}) {
    LinearWorkload other = two;
    other.liveRows = live;
    rejects([&] {
      (void)Linear::plan(other, {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, 4, true}, FloatOutput::BFloat16);
    }, "the Qmv tile projects one live row, or two sharing its threadgroups", "shared Qmv rows took 1 or 3 rows");
  }
  rejects([&] { (void)Linear::plan(two, {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, 2, true}, FloatOutput::BFloat16); },
          "shared Qmv rows take four outputs a simdgroup, two for gate/up", "shared Qmv rows took two outputs");
  rejects([&] {
    (void)Linear::plan(two, {LinearTile::LaneSplit128, 0, LinearSimdgroups::Eight, 1, 0, true}, FloatOutput::BFloat16);
  }, "only the Qmv tile shares its threadgroups between rows", "a tile took shared rows");
  for (const uint32_t outputs : {0U, 1U, 3U, 8U})
    rejects([&] {
      (void)Linear::plan(one, {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, outputs}, FloatOutput::BFloat16);
    }, "the Qmv tile takes 2 or 4 outputs per simdgroup, every other tile 0", "a Qmv plan took other outputs");
  rejects([&] {
    (void)Linear::plan(one, {LinearTile::LaneSplit128, 0, LinearSimdgroups::Eight, 1, 4}, FloatOutput::BFloat16);
  }, "the Qmv tile takes 2 or 4 outputs per simdgroup, every other tile 0", "a tile took outputs per simdgroup");
  rejects([&] { (void)Linear::plan({{256, 512}, 32, LinearPhase::Decode, LinearEpilogue::None, WeightLayout::Affine64, 1},
                                   {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, 4}, FloatOutput::BFloat16); },
          "paired Q4 tile requires one lane", "a Qmv plan took four lanes");
  const LinearPlan plan = Linear::plan(one, {LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, 4}, FloatOutput::BFloat16);
  const auto weight = weights(backend, {256, 512}, 31, false);
  const NormCase c = normCase(backend, 512, 2);
  const metal::MetalBuffer output = test::sharedBuffer(backend, 256 * 2);
  const Linear linear(backend.capabilities());
  metal::CommandGraph graph;
  rejects([&] {
    linear.add(graph, {.input = c.input, .output = output, .normWeight = c.weight.buffer, .normRows = 2}, weight, plan);
  }, "a folded norm of a plan of live rows takes those rows", "a Qmv plan folded two rows' norms");
  requireExtent(backend, c.input, 512ULL * 2, 2, "projection input", [&](metal::CommandGraph &g, const metal::MetalBuffer &in) {
    linear.add(g, {.input = in, .output = output}, weight, plan);
  });
  requireExtent(backend, output, 256ULL * 2, 2, "projection output", [&](metal::CommandGraph &g, const metal::MetalBuffer &out) {
    linear.add(g, {.input = c.input, .output = out}, weight, plan);
  });
  require(graph.empty(), "a refused Qmv plan encoded a dispatch");
}

}
int main(int argc,char **argv) {
  try {
    require(argc==2,"usage: q4-sgmatrix <production.metallib>");
    metal::MetalBackend backend(argv[1]);
    uint32_t cases=0;
    for (auto [n,k] : std::array<std::array<uint32_t,2>,4>{{{256,256},{768,768},{512,5120},{512,17408}}})
      for (uint32_t splits : {1U,2U,4U,8U}) {
        if ((k/64)%splits) continue;
        for (auto e : {LinearEpilogue::None,LinearEpilogue::Residual,LinearEpilogue::GateUp})
          for (uint32_t fixture=0;fixture<4;++fixture)
            for (uint32_t rows : {8U,16U,24U,32U}) { runCase(backend,n,k,splits,e,fixture,rows); ++cases; }
      }
    // One lane's LaneSplit128 tile at every K split of whole 256-input blocks.
    for (auto [n,k] : std::array<std::array<uint32_t,2>,5>{{{256,256},{768,768},{512,5120},{512,17408},{1024,3072}}})
      for (uint32_t splits : {1U,2U,4U,8U}) {
        if (k/256<splits) continue;
        for (auto e : {LinearEpilogue::None,LinearEpilogue::Residual,LinearEpilogue::GateUp})
          for (uint32_t fixture=0;fixture<4;++fixture) {
            runCase(backend,n,k,splits,e,fixture,8,LinearTile::LaneSplit128);
            ++cases;
          }
      }
    // The norm fused with the Table64 preparation (kernels/shared/normalization.metal).
    for (uint32_t width : {64U, 320U, 1984U, 2048U, 2112U, 5120U, 17408U})
      for (uint32_t rows : {8U,16U,24U,32U}) fusedNorm(backend, width, rows, LinearInput::Table64);
    for (uint32_t width : {64U, 2048U, 5120U, 17408U})
      for (uint32_t rows : {1U,37U,64U,65U}) prefillNorm(backend, width, rows);
    normExtents(backend);
    foldedNormRefusals(backend);
    // One live row on the Qmv tile: every epilogue and fixture at four and
    // two outputs per simdgroup, which give the same bits, at K of whole and
    // partial 512-input steps; logits in fp32; the folded norm; the policy.
    for (auto [n,k] : std::array<std::array<uint32_t,2>,5>{{{256,256},{512,768},{768,2048},{256,5120},{1024,3072}}}) {
      for (auto e : {LinearEpilogue::None,LinearEpilogue::Residual,LinearEpilogue::GateUp})
        for (uint32_t fixture=0;fixture<4;++fixture) {
          require(qmvCase(backend,n,k,e,fixture,4) == qmvCase(backend,n,k,e,fixture,2),
                  "Qmv outputs per simdgroup changed a sum");
          ++cases;
        }
      (void)qmvCase(backend,n,k,LinearEpilogue::None,0,4,FloatOutput::Float32);
      qmvFoldedNorm(backend,n,k,LinearEpilogue::None,4);
      qmvFoldedNorm(backend,n,k,LinearEpilogue::GateUp,2);
      qmvFoldedNorm(backend,n,k,LinearEpilogue::None,4,FloatOutput::Float32);
    }
    // Two live rows sharing the Qmv tile: each row the one-row plan's bits.
    for (auto [n,k] : std::array<std::array<uint32_t,2>,4>{{{256,256},{512,768},{768,2048},{1024,3072}}}) {
      for (auto e : {LinearEpilogue::None,LinearEpilogue::Residual,LinearEpilogue::GateUp})
        for (uint32_t fixture=0;fixture<4;++fixture) { qmvSharedCase(backend,n,k,e,fixture); ++cases; }
      qmvSharedCase(backend,n,k,LinearEpilogue::None,0,FloatOutput::Float32);
    }
    qmvPolicy(backend);
    // Wide projections sharing split scratch: K 6144, 17408 and 5120.
    for (uint32_t lanes : {1U, 4U}) {
      splitVisibility(backend, {{{{5120, 6144}, LinearEpilogue::Residual}, {{5120, 17408}, LinearEpilogue::Residual}}}, lanes);
      splitVisibility(backend, {{{{16640, 5120}, LinearEpilogue::None}, {{17408, 5120}, LinearEpilogue::GateUp}}}, lanes);
    }
    // Apple10's one-lane split tiles sharing split scratch: an output
    // projection then a gate/up pair, and a down projection then a plain one.
    splitVisibility(backend, {{{{2048, 2048}, LinearEpilogue::Residual}, {{3072, 1024}, LinearEpilogue::GateUp}}}, 1,
                    LinearTile::LaneSplit128);
    splitVisibility(backend, {{{{1024, 3072}, LinearEpilogue::Residual}, {{4096, 2048}, LinearEpilogue::None}}}, 1,
                    LinearTile::LaneSplit128);
    std::cout << "Q4 register, lane split and Qmv tiles: PASS cases=" << cases
              << " (fp64, range, cancellation, guards, repeated dispatch, fused and folded norms, shared split scratch)\n";
  } catch (const std::exception &e) { std::cerr << "Q4 register, lane split and Qmv tiles: FAIL: " << e.what() << '\n'; return 1; }
}
