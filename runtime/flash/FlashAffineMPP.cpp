#include "FlashAffineMPP.hpp"

#include "metal/abi/FlashAffineMPP.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {

uint64_t product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash MPP affine byte extent overflows");
  return a * b;
}

uint64_t extent(uint32_t rows, uint64_t stride, uint64_t rowBytes) {
  if (!rows || stride < rowBytes)
    throw std::invalid_argument("Flash MPP affine invalid source stride");
  const uint64_t prior = product(rows - 1, stride);
  if (prior > std::numeric_limits<uint64_t>::max() - rowBytes)
    throw std::invalid_argument("Flash MPP affine byte extent overflows");
  return prior + rowBytes;
}

void requireBuffer(const metal::MetalBuffer &b, uint64_t bytes,
                   const char *what) {
  if (!bytes || !b || b.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash MPP affine insufficient ") + what);
}

void requireTensor(const FlashTensor *t, FlashDType dtype, uint64_t bytes,
                   const char *what) {
  if (!t || t->dtype != dtype || t->logicalBytes < bytes)
    throw std::invalid_argument(std::string("Flash MPP affine invalid ") + what);
  requireBuffer(t->buffer, bytes, what);
}

} // namespace

void addAffineMPP(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const FlashAffineProjection &p, metal::MetalBuffer output,
                  metal::MetalBuffer diagnostics, uint32_t rows,
                  FlashAffineMPPMode mode, FlashAffineMPPTile tile) {
  if (!rows || rows > 8192 || p.experts != 1 || !p.outputSize ||
      !p.inputSize || p.inputSize > 32768 ||
      (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) ||
      (p.groupSize != 32 && p.groupSize != 64 && p.groupSize != 128) ||
      p.inputSize % p.groupSize || p.parameterRowStrideBytes % 2 ||
      static_cast<uint32_t>(mode) > 1)
    throw std::invalid_argument("Flash MPP affine invalid dense geometry/mode");
  uint32_t m = 0, n = 0;
  switch (tile) {
  case FlashAffineMPPTile::M8N64: m = 8; n = 64; break;
  case FlashAffineMPPTile::M16N64: m = 16; n = 64; break;
  case FlashAffineMPPTile::M16N128: m = 16; n = 128; break;
  case FlashAffineMPPTile::M32N64: m = 32; n = 64; break;
  case FlashAffineMPPTile::M32N128: m = 32; n = 128; break;
  default: throw std::invalid_argument("Flash MPP affine invalid tile");
  }
  requireTensor(p.weights, FlashDType::U32,
                extent(p.outputSize, p.weightRowStrideBytes,
                       (product(p.inputSize, p.bits) + 7) / 8), "weights");
  const uint64_t coefficientBytes = extent(
      p.outputSize, p.parameterRowStrideBytes, product(p.inputSize / p.groupSize, 2));
  requireTensor(p.scales, FlashDType::BF16, coefficientBytes, "scales");
  requireTensor(p.biases, FlashDType::BF16, coefficientBytes, "biases");
  requireBuffer(input, product(product(rows, p.inputSize), 2), "input");
  requireBuffer(output, product(product(rows, p.outputSize), 2), "output");
  requireBuffer(diagnostics, sizeof(uint32_t), "diagnostics");
  if (input.sameView(output) || p.weights->buffer.sameView(output) ||
      p.scales->buffer.sameView(output) || p.biases->buffer.sameView(output))
    throw std::invalid_argument("Flash MPP affine input/source and output alias");
  const FlashAffineMPPParams params{
      {rows, 1, p.inputSize, p.outputSize, 1, p.bits, p.groupSize, 0,
       p.weightRowStrideBytes, p.weightExpertStrideBytes,
       p.parameterRowStrideBytes, p.parameterExpertStrideBytes},
      static_cast<uint32_t>(mode), m, n, 0};
  const std::string pipeline = "flash_affine_mpp_m" + std::to_string(m) +
      "_n" + std::to_string(n) + "_g" + std::to_string(p.groupSize);
  graph.add(pipeline,
            {input, p.weights->buffer, p.scales->buffer, p.biases->buffer,
             output, diagnostics}, params,
            {(p.outputSize - 1) / n + 1, (rows - 1) / m + 1, 1},
            {128, 1, 1});
}

} // namespace splash::flash
