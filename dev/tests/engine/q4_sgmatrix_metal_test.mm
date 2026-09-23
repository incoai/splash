#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;
namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
struct Guarded {
  metal::MetalBuffer backing, view;
  uint64_t size;
  Guarded(metal::MetalBackend &backend, uint64_t bytes) : size(bytes) {
    backing = backend.allocateBuffer(bytes + 256);
    std::memset(backing.contents(), 0xa5, bytes + 256);
    view = backend.view(backing, 0, bytes);
  }
  void check() const {
    const auto *p = static_cast<const uint8_t *>(backing.contents());
    for (uint64_t i = size; i < size + 256; ++i) require(p[i] == 0xa5, "out-of-view write");
  }
};
uint32_t hash(uint32_t v) { v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; return v * 0x846ca68b; }
Q4Projection weights(metal::MetalBackend &backend, LinearMatrix shape, uint32_t seed, bool zero) {
  const uint64_t params = uint64_t(shape.outputSize) * shape.inputSize / 64;
  Q4Projection p{backend.allocateBuffer(params * 32), backend.allocateBuffer(params * 2),
                 backend.allocateBuffer(params * 2), shape.outputSize, shape.inputSize};
  auto *q = static_cast<uint8_t *>(p.weights.contents());
  auto *s = static_cast<uint16_t *>(p.scales.contents());
  auto *b = static_cast<uint16_t *>(p.biases.contents());
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
Exact exact(const Q4Projection &p, const uint16_t *input, uint32_t row, uint32_t col) {
  const auto *q = static_cast<const uint8_t *>(p.weights.contents());
  const auto *sc = static_cast<const uint16_t *>(p.scales.contents());
  const auto *bi = static_cast<const uint16_t *>(p.biases.contents());
  double value = 0, magnitude = 0, quantMagnitude = 0;
  const uint32_t groups = p.inputSize / 64;
  for (uint32_t g = 0; g < groups; ++g) {
    const uint64_t at = (uint64_t(col / 256) * groups + g) * 256 + col % 256;
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
void runCase(metal::MetalBackend &backend, uint32_t n, uint32_t k, uint32_t splits,
             LinearEpilogue epilogue, uint32_t fixture, uint32_t rows) {
  const LinearWorkload workload{{n, k}, rows, LinearPhase::Decode, epilogue};
  const auto plan = Q4Linear::plan(workload,
      {LinearTile::Simdgroup, n / (epilogue == LinearEpilogue::GateUp ? 32 : 64), LinearSimdgroups::Four, splits});
  const auto size = plan.scratchSize();
  Guarded input(backend, 2ULL * rows * k), output(backend, 2ULL * rows * n), residual(backend, 2ULL * rows * n);
  Guarded table(backend, size.input), sums(backend, size.sums), partials(backend, size.partials), counters(backend, size.counters);
  LinearScratch scratch{table.view, sums.view, partials.view, counters.view};
  std::memset(counters.view.contents(), 0, size.counters);
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
  Q4Linear linear(backend.capabilities());
  LinearBuffers b{input.view, output.view, {}, epilogue == LinearEpilogue::Residual ? residual.view : metal::MetalBuffer{}, {}, {}, scratch};
  metal::CommandGraph graph;
  // Reuse one workspace repeatedly in a single command to expose incomplete
  // publication, stale counters and dependencies between consecutive dispatches.
  for (uint32_t repeat = 0; repeat < 3; ++repeat)
    linear.add(graph, b, p, plan, epilogue == LinearEpilogue::GateUp ? &gate : nullptr);
  (void)backend.submitCommand(graph.dispatches());
  const auto *prepared = static_cast<const uint16_t *>(table.view.contents());
  const auto *rowSums = static_cast<const float *>(sums.view.contents());
  for (uint32_t g=0;g<k/64;++g) for (uint32_t row=0;row<rows;++row) {
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
  (void)backend.submitCommand(graph.dispatches());
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
        std::cerr << "M=" << rows << " N=" << n << " K=" << k << " S=" << splits << " epilogue=" << int(epilogue)
                  << " fixture=" << fixture << " row=" << row << " col=" << col
                  << " actual=" << bf16ToFloat(value) << " reference=" << ref.value << " error=" << ref.error << '\n';
        throw std::runtime_error("simdgroup result exceeds independent fp64 error bound");
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
  Q4Projection weights, gate;
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
void splitVisibility(metal::MetalBackend &backend,
                     std::array<std::pair<LinearMatrix, LinearEpilogue>, 2> pair, uint32_t lanes) {
  const uint32_t rows = lanes * 8;
  std::vector<SplitOperand> operands;
  for (uint32_t i = 0; i < 2; ++i) {
    const auto [matrix, epilogue] = pair[i];
    const auto [n, k] = matrix;
    SplitOperand o{{matrix, rows, LinearPhase::Decode, epilogue},
                   weights(backend, matrix, 31 + 100 * i, false), weights(backend, matrix, 177 + 100 * i, false),
                   backend.allocateBuffer(2ULL * rows * k), backend.allocateBuffer(2ULL * rows * n),
                   backend.allocateBuffer(2ULL * rows * n), {}, {}};
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
    device.appleGpuFamily = 9;
    device.gpuCoreCount = cores;
    const Q4Linear policy(device);
    const auto a = policy.plan(operands[0].workload).configuration();
    const auto b = policy.plan(operands[1].workload).configuration();
    if (a.tile == LinearTile::Simdgroup && b.tile == LinearTile::Simdgroup && std::max(a.splits, b.splits) > 1)
      splitPairs.insert({a.splits, b.splits});
  }
  require(!splitPairs.empty(), "the policy splits neither projection");
  const auto plan = [&](uint32_t i, uint32_t splits) {
    const LinearWorkload &w = operands[i].workload;
    return Q4Linear::plan(w, {LinearTile::Simdgroup, w.matrix.outputSize / (w.epilogue == LinearEpilogue::GateUp ? 32 : 64),
                              LinearSimdgroups::Four, splits});
  };
  LinearScratchSize size;
  const auto grow = [&](const LinearPlan &p) {
    const auto s = p.scratchSize();
    size = {std::max(size.input, s.input), std::max(size.sums, s.sums), std::max(size.partials, s.partials),
            std::max(size.counters, s.counters)};
  };
  for (uint32_t i = 0; i < 2; ++i) grow(plan(i, 1));
  for (const auto &splits : splitPairs) for (uint32_t i = 0; i < 2; ++i) grow(plan(i, splits[i]));
  Guarded table(backend, size.input), sums(backend, size.sums), partials(backend, size.partials), counters(backend, size.counters);
  std::memset(counters.view.contents(), 0, size.counters);
  const LinearScratch scratch{table.view, sums.view, partials.view, counters.view};
  const auto poison = backend.allocateBuffer(size.partials);
  const Q4Linear linear(backend.capabilities());
  const auto add = [&](metal::CommandGraph &graph, uint32_t i, uint32_t splits) {
    const SplitOperand &o = operands[i];
    const bool gateUp = o.workload.epilogue == LinearEpilogue::GateUp;
    linear.add(graph, {o.input, o.output, {}, o.workload.epilogue == LinearEpilogue::Residual ? o.residual : metal::MetalBuffer{},
                       {}, {}, scratch}, o.weights, plan(i, splits), gateUp ? &o.gate : nullptr);
  };
  const std::string shape = std::to_string(lanes) + " lanes, " + std::to_string(pair[0].first.outputSize) + "x" +
      std::to_string(pair[0].first.inputSize) + " then " + std::to_string(pair[1].first.outputSize) + "x" +
      std::to_string(pair[1].first.inputSize);
  metal::CommandGraph unsplit;
  for (uint32_t i = 0; i < 2; ++i) add(unsplit, i, 1);
  (void)backend.submitCommand(unsplit.dispatches());
  for (uint32_t i = 0; i < 2; ++i) requireFp64(operands[i], 1, shape + " unsplit");
  for (const auto &splits : splitPairs) {
    const std::string what = shape + " splits " + std::to_string(splits[0]) + "/" + std::to_string(splits[1]);
    metal::CommandGraph graph;
    for (uint32_t i = 0; i < 2; ++i) {
      // gguf_copy, the production byte copy, poisons the partials in order.
      graph.add("gguf_copy", {poison, partials.view}, GgufCopyParams{0, 0, uint32_t(size.partials)},
                {uint32_t((size.partials + 4095) / 4096), 1, 1}, {256, 1, 1});
      add(graph, i, splits[i]);
    }
    std::vector<std::vector<uint8_t>> first;
    for (uint32_t bits : {0x7E800000U, 0x7FC00000U, 0x7E800000U}) {
      std::fill_n(static_cast<uint32_t *>(poison.contents()), size.partials / 4, bits);
      (void)backend.submitCommand(graph.dispatches());
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
// The kernel that prepares a plain bf16 input in `layout`, the reference for
// the producers that write the table themselves.
const char *prepareKernel(LinearInput layout) {
  return layout == LinearInput::Table16 ? "decode_linear_gguf_prepare" : "decode_linear_q4_prepare";
}
// Rows of widely spread values and a norm's weights, bf16 or F32 as a GGUF
// stores them; the F32 weights carry bits a bf16 rounding would drop.
struct NormCase {
  metal::MetalBuffer input;
  NormWeights weight;
  std::vector<double> weights;  // the values the kernels read
};
NormCase normCase(metal::MetalBackend &backend, uint32_t k, uint32_t rows, bool float32) {
  NormCase c{backend.allocateBuffer(k*rows*2), {backend.allocateBuffer(k*(float32?4:2)), float32},
             std::vector<double>(k)};
  auto *x=static_cast<uint16_t *>(c.input.contents());
  for (uint32_t i=0;i<k*rows;++i) x[i]=floatToBf16(float(int(hash(i)%257)-128)*8192);
  for (uint32_t i=0;i<k;++i) {
    const float value=float(int(i%17)-8)/4;
    if (float32) {
      const float exact=value*(1+float(hash(i)%4093)/65536);
      static_cast<float *>(c.weight.buffer.contents())[i]=exact;
      c.weights[i]=exact;
    } else {
      static_cast<uint16_t *>(c.weight.buffer.contents())[i]=floatToBf16(value);
      c.weights[i]=bf16ToFloat(floatToBf16(value));
    }
  }
  return c;
}
// Every output is the fp64 norm rounded once to bf16, up to the fp32
// arithmetic's noise (well below 2^-8 of half an ulp): weights rounded to
// bf16 anywhere on the way would miss the bound on about a quarter of them.
void requireNorm(const NormCase &c, const metal::MetalBuffer &output, uint32_t k, uint32_t rows,
                 const char *what) {
  const auto *x=static_cast<const uint16_t *>(c.input.contents());
  const auto *out=static_cast<const uint16_t *>(output.contents());
  for (uint32_t r=0;r<rows;++r) {
    double squares=0;
    for (uint32_t i=0;i<k;++i) squares+=double(bf16ToFloat(x[r*k+i]))*bf16ToFloat(x[r*k+i]);
    const double inverse=1/std::sqrt(squares/k+1e-6);
    for (uint32_t i=0;i<k;++i) {
      const double exact=bf16ToFloat(x[r*k+i])*inverse*c.weights[i];
      require(std::fabs(bf16ToFloat(out[r*k+i])-exact)<=0.5*ulpBf16(float(exact))*(1+1.0/256),what);
    }
  }
}
void fusedNorm(metal::MetalBackend &backend, uint32_t k, uint32_t rows, LinearInput layout, bool float32) {
  const uint64_t sumsBytes = tableSumsBytes(layout, k, rows);
  const NormCase c=normCase(backend,k,rows,float32);
  auto output=backend.allocateBuffer(k*rows*2), fused=backend.allocateBuffer(k*rows*2);
  auto a=backend.allocateBuffer(tableBytes(k,rows)), b=backend.allocateBuffer(tableBytes(k,rows));
  auto sa=backend.allocateBuffer(sumsBytes), sb=backend.allocateBuffer(sumsBytes);
  metal::CommandGraph graph;
  Normalization::addRms(graph,c.input,c.weight,output,k,rows);
  graph.add(prepareKernel(layout),{output,a,sa},k,{k/32,rows/8,1},{128,1,1});
  Normalization::addRms(graph,c.input,c.weight,fused,k,rows,{b,sb,{},{}},layout);
  (void)backend.submitCommand(graph.dispatches());
  requireNorm(c,output,k,rows,"norm differs from the fp64 reference");
  require(!std::memcmp(output.contents(),fused.contents(),k*rows*2),"fused norm changed bf16 output");
  require(!std::memcmp(a.contents(),b.contents(),tableBytes(k,rows)),"fused operand permutation mismatch");
  require(!std::memcmp(sa.contents(),sb.contents(),sumsBytes),"fused input sums mismatch");
}
// The packed prefill's norm, whose rows need not fill its 32-row sum tiles.
void prefillNorm(metal::MetalBackend &backend, uint32_t k, uint32_t rows, bool float32) {
  const NormCase c=normCase(backend,k,rows,float32);
  auto output=backend.allocateBuffer(k*rows*2), sums=backend.allocateBuffer((rows+31)/32*32*(k/64)*4);
  metal::CommandGraph graph;
  Normalization::addRmsWithQ4Sums(graph,c.input,c.weight,output,sums,k,rows);
  (void)backend.submitCommand(graph.dispatches());
  requireNorm(c,output,k,rows,"prefill norm differs from the fp64 reference");
}
void fusedAttentionGate(metal::MetalBackend &backend, uint32_t heads, uint32_t kvHeads, uint32_t lanes,
                        LinearInput layout) {
  const uint32_t width = heads * 256, packedWidth = 2 * width + 2 * kvHeads * 256, rows = lanes * 8;
  const uint64_t sumsBytes = tableSumsBytes(layout, width, rows);
  auto packed = backend.allocateBuffer(uint64_t{packedWidth} * 16 * lanes);
  auto attention = backend.allocateBuffer(uint64_t{width} * 32 * 2 * lanes);
  for (auto buffer : {packed, attention}) {
    auto *data = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
      data[i] = floatToBf16(float(int(hash(uint32_t(i)) % 257) - 128) / 16);
  }
  Guarded output(backend, width * 16 * lanes), fused(backend, width * 16 * lanes);
  Guarded a(backend, tableBytes(width, rows)), b(backend, tableBytes(width, rows));
  Guarded sa(backend, sumsBytes), sb(backend, sumsBytes);
  metal::CommandGraph graph;
  PagedAttention::addVerifyGate(graph, packed, attention, output.view, 8, 32, 32,
                                heads, {1, kvHeads, 256}, lanes);
  graph.add(prepareKernel(layout), {output.view, a.view, sa.view}, width,
            {width / 32, lanes, 1}, {128, 1, 1});
  PagedAttention::addVerifyGate(graph, packed, attention, fused.view, 8, 32, 32,
                                heads, {1, kvHeads, 256}, lanes, {b.view, sb.view, {}, {}}, layout);
  (void)backend.submitCommand(graph.dispatches());
  require(!std::memcmp(output.view.contents(), fused.view.contents(), width * 16 * lanes),
          "fused attention gate output");
  require(!std::memcmp(a.view.contents(), b.view.contents(), tableBytes(width, rows)),
          "fused attention gate table");
  require(!std::memcmp(sa.view.contents(), sb.view.contents(), sumsBytes),
          "fused attention gate sums");
  for (auto *buffer : {&output, &fused, &a, &b, &sa, &sb}) buffer->check();
}


}
int main(int argc,char **argv) {
  try {
    require(argc==2,"usage: q4-sgmatrix <production.metallib>");
    metal::MetalBackend backend(argv[1]);
    for (LinearInput layout : {LinearInput::Table64, LinearInput::Table16})
      for (uint32_t lanes : {1U,2U,3U,4U}) {
        fusedAttentionGate(backend, 24, 4, lanes, layout);
        fusedAttentionGate(backend, 16, 2, lanes, layout);
      }
    uint32_t cases=0;
    for (auto [n,k] : std::array<std::array<uint32_t,2>,4>{{{256,256},{768,768},{512,5120},{512,17408}}})
      for (uint32_t splits : {1U,2U,4U,8U}) {
        if ((k/64)%splits) continue;
        for (auto e : {LinearEpilogue::None,LinearEpilogue::Residual,LinearEpilogue::GateUp})
          for (uint32_t fixture=0;fixture<4;++fixture)
            for (uint32_t rows : {8U,16U,24U,32U}) { runCase(backend,n,k,splits,e,fixture,rows); ++cases; }
      }
    for (bool float32 : {false, true}) {
      for (LinearInput layout : {LinearInput::Table64, LinearInput::Table16})
        for (uint32_t width : {64U, 320U, 1984U, 2048U, 2112U, 5120U, 17408U})
          for (uint32_t rows : {8U,16U,24U,32U}) fusedNorm(backend, width, rows, layout, float32);
      for (uint32_t width : {64U, 5120U, 17408U})
        for (uint32_t rows : {1U,37U,64U}) prefillNorm(backend, width, rows, float32);
    }
    // 27B out_proj then down, and gdn_in then gate/up: K 6144, 17408 and 5120.
    for (uint32_t lanes : {1U, 4U}) {
      splitVisibility(backend, {{{{5120, 6144}, LinearEpilogue::Residual}, {{5120, 17408}, LinearEpilogue::Residual}}}, lanes);
      splitVisibility(backend, {{{{16640, 5120}, LinearEpilogue::None}, {{17408, 5120}, LinearEpilogue::GateUp}}}, lanes);
    }
    std::cout << "Q4 simdgroup: PASS cases=" << cases
              << " (fp64, range, cancellation, guards, repeated dispatch, fused norm and attention gate in both"
                 " table layouts, norms with bf16 and F32 weights, shared split scratch)\n";
  } catch (const std::exception &e) { std::cerr << "Q4 simdgroup: FAIL: " << e.what() << '\n'; return 1; }
}
