#include "LinearInputReference.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Normalization.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

// The plain RMS norm (ops::Normalization::addRms) of each epsilon its kernels
// compile, SPLASH_RMS_EPSILON and SPLASH_RMS_EPSILON_E5 (Nemotron-H's, F32
// weights alone), on each of its paths: the staged rows of few narrow rows,
// the streamed rows of many, the register chunks of wide ones, and the
// Table16 producer, against an fp64 reference of the weights' epsilon. Rows of
// small values make the epsilon a sizable part of their mean square, so that
// a kernel of the other epsilon misses the reference.

namespace {

using namespace splash;
using test::bf16ToFloat;
using test::floatToBf16;
using test::rejects;
using test::require;

uint16_t *halves(const metal::MetalBuffer &buffer) { return static_cast<uint16_t *>(buffer.contents()); }

// A deterministic value in [-1, 1).
float unit(uint64_t seed) {
  seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
  return float((seed ^ (seed >> 31)) >> 40) / float(1ULL << 23) - 1.0f;
}

void checkNorm(metal::MetalBackend &backend, uint32_t rows, uint32_t width, double epsilon,
               ops::LinearInput layout, std::string_view kernel) {
  const uint64_t elements = uint64_t{rows} * width;
  const metal::MetalBuffer input = test::sharedBuffer(backend, elements * 2);
  const metal::MetalBuffer output = test::sharedBuffer(backend, elements * 2);
  for (uint64_t i = 0; i < elements; ++i) halves(input)[i] = floatToBf16(2e-3f * unit(i + 1));
  ops::NormWeights norm =
      test::makeNormWeights(backend, width, true, [](uint32_t i) { return float(1.0 + 0.5 * unit(i + (1ULL << 40))); });
  norm.epsilon = epsilon;
  const bool table = layout != ops::LinearInput::Plain;
  const ops::LinearScratch scratch =
      table ? ops::LinearScratch{test::sharedBuffer(backend, ops::tableBytes(width, rows)),
                                 test::sharedBuffer(backend, ops::tableSumsBytes(layout, width, rows)), {}, {}}
            : ops::LinearScratch{};
  metal::CommandGraph graph;
  require(ops::Normalization::addRms(graph, input, norm, output, width, rows, scratch, layout).layout == layout &&
              graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == kernel,
          "the norm ran another kernel than " + std::string(kernel));
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  ops::NormWeights other = norm;
  other.epsilon = epsilon == SPLASH_RMS_EPSILON ? SPLASH_RMS_EPSILON_E5 : SPLASH_RMS_EPSILON;
  uint64_t differing = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    const uint16_t *source = halves(input) + uint64_t{row} * width;
    const std::vector<double> expected = test::rmsNorm(source, norm, width);
    const std::vector<double> otherEpsilon = test::rmsNorm(source, other, width);
    for (uint32_t column = 0; column < width; ++column) {
      const uint16_t got = halves(output)[uint64_t{row} * width + column];
      require(test::roundedOnceToBf16(got, expected[column]),
              std::string(kernel) + " is not the fp64 norm of its epsilon rounded once");
      differing += !test::roundedOnceToBf16(got, otherEpsilon[column]);
    }
  }
  require(differing > elements / 2, std::string(kernel) + "'s rows do not show its epsilon");
  if (table) {
    const metal::MetalBuffer expectedTable = test::sharedBuffer(backend, scratch.input.sizeBytes());
    const metal::MetalBuffer expectedSums = test::sharedBuffer(backend, scratch.sums.sizeBytes());
    metal::CommandGraph reference;
    test::addReferencePreparation(reference, layout, output, expectedTable, expectedSums, width, rows / 8);
    static_cast<void>(backend.submitCommandAsync(reference.dispatches()).wait());
    require(!std::memcmp(scratch.input.contents(), expectedTable.contents(), scratch.input.sizeBytes()) &&
                !std::memcmp(scratch.sums.contents(), expectedSums.contents(), scratch.sums.sizeBytes()),
            std::string(kernel) + "'s table differs from the preparation of its rows");
  }
  std::cout << kernel << ": " << rows << " rows of " << width << ", " << differing << " of " << elements
            << " values apart from the other epsilon's PASS\n";
}

// Only the plain norm's F32 kernels compile SPLASH_RMS_EPSILON_E5, and no
// kernel another epsilon.
void checkRefusals(metal::MetalBackend &backend) {
  const metal::MetalBuffer rows = test::sharedBuffer(backend, 8 * 2048 * 2);
  ops::NormWeights bf16{test::sharedBuffer(backend, 2048 * 2), false, SPLASH_RMS_EPSILON_E5};
  ops::NormWeights other{test::sharedBuffer(backend, 2048 * 4), true, 1e-4};
  ops::NormWeights e5{test::sharedBuffer(backend, 2048 * 4), true, SPLASH_RMS_EPSILON_E5};
  metal::CommandGraph graph;
  rejects([&] { (void)ops::Normalization::addRms(graph, rows, bf16, rows, 2048, 8); },
          "no RMS norm kernel for the norm's epsilon", "a bf16 norm of epsilon 1e-5 was accepted");
  rejects([&] { (void)ops::Normalization::addRms(graph, rows, other, rows, 2048, 8); },
          "no RMS norm kernel for the norm's epsilon", "a norm of epsilon 1e-4 was accepted");
  rejects([&] { (void)ops::normKernel("verify_attention_qkv", e5, 2048); }, "the kernel takes the default RMS epsilon",
          "a kernel of the default epsilon took another");
  require(graph.empty(), "a refused norm encoded a dispatch");
  std::cout << "norm epsilons without kernels: refused PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: normalization METALLIB");
    metal::MetalBackend backend(argv[1]);
    for (const double epsilon : {SPLASH_RMS_EPSILON, SPLASH_RMS_EPSILON_E5}) {
      const std::string e5 = epsilon == SPLASH_RMS_EPSILON ? "" : "_e5";
      checkNorm(backend, 8, 2048, epsilon, ops::LinearInput::Plain, "norm_rms_staged_f32" + e5);
      checkNorm(backend, 100, 2048, epsilon, ops::LinearInput::Plain, "norm_rms_f32" + e5);
      checkNorm(backend, 8, 4096, epsilon, ops::LinearInput::Plain, "norm_rms_f32" + e5);
      checkNorm(backend, 8, 1024, epsilon, ops::LinearInput::Table16, "norm_rms_table16_decode_f32" + e5);
      checkNorm(backend, 32, 4096, epsilon, ops::LinearInput::Table16, "norm_rms_table16_decode_f32" + e5);
    }
    checkRefusals(backend);
    std::cout << "normalization: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "normalization: FAIL: " << error.what() << '\n';
    return 1;
  }
}
