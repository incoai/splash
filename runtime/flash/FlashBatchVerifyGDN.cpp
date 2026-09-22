#include "FlashBatchVerifyGDN.hpp"

#include "FlashGDNFused.hpp"
#include "FlashGDNLazyRollback.hpp"
#include "metal/abi/FlashForward.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::flash {
namespace {

constexpr uint64_t kHistoryRowBytes =
    uint64_t{kFlashGDNConvolutionWidth} * sizeof(uint16_t);
constexpr uint64_t kConvolutionStride =
    kFlashBatchVerifyGDNConvolutionRowStrideBytes;
constexpr uint64_t kRecurrentStride =
    kFlashBatchVerifyGDNRecurrentRowStrideBytes;
static_assert(kConvolutionStride ==
              ((flashGDNConvolutionLaneBytes() + 16383) & ~uint64_t{16383}));
static_assert(kRecurrentStride == 3145728);
static_assert(kHistoryRowBytes % sizeof(uint32_t) == 0);

struct Region final {
  metal::MetalBuffer buffer;
  bool writable = true;
};

struct Copy final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  uint64_t bytes = 0;
};

metal::MetalBuffer checkedView(metal::MetalBackend &backend,
                               const metal::MetalBuffer &buffer,
                               uint64_t bytes, const char *name,
                               bool requireShared = true) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash batch verify GDN insufficient ") +
                                name);
  if (requireShared &&
      (buffer.storage() != metal::BufferStorage::Shared || !buffer.contents()))
    throw std::invalid_argument(std::string("Flash batch verify GDN requires Shared ") +
                                name);
  // Also establishes backend ownership before caller graph mutation.
  const auto view = backend.view(buffer, 0, bytes);
  if (requireShared &&
      reinterpret_cast<uintptr_t>(view.contents()) % alignof(uint32_t))
    throw std::invalid_argument(std::string("Flash batch verify GDN unaligned ") +
                                name);
  return view;
}

bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (a.sameView(b)) return true;
  const auto *aContents = a.contents();
  const auto *bContents = b.contents();
  if (!aContents || !bContents) return false;
  const uintptr_t aBegin = reinterpret_cast<uintptr_t>(aContents);
  const uintptr_t bBegin = reinterpret_cast<uintptr_t>(bContents);
  // Subtraction avoids overflowing an address when finding the end.
  return aBegin <= bBegin ? bBegin - aBegin < a.sizeBytes()
                          : aBegin - bBegin < b.sizeBytes();
}

void requireDisjoint(std::span<const Region> regions) {
  for (size_t i = 0; i < regions.size(); ++i)
    for (size_t j = i + 1; j < regions.size(); ++j)
      if ((regions[i].writable || regions[j].writable) &&
          overlaps(regions[i].buffer, regions[j].buffer))
        throw std::invalid_argument(
            "Flash batch verify GDN work/state/tape buffers overlap");
}

void requireRequestStride(uint64_t stride, uint64_t tight) {
  if (stride && (stride < tight || stride % sizeof(uint32_t)))
    throw std::invalid_argument("Flash batch verify GDN invalid request lane stride");
}

void addCopy(metal::CommandGraph &graph, const Copy &copy) {
  const uint64_t words = copy.bytes / sizeof(uint32_t);
  graph.add("flash_forward_copy_words", {copy.input, copy.output},
            FlashForwardCopyParams{words}, {(words + 255) / 256, 1, 1});
}

} // namespace

void addBatchVerifyGDN(
    metal::CommandGraph &graph, metal::MetalBackend &backend,
    const FlashGDNWeights &weights, const FlashGDNBuffers &buffers,
    std::span<const FlashGDNState> actualRequestStates,
    const FlashGDNState &packedState,
    const metal::MetalBuffer &convolutionPrefixes,
    const metal::MetalBuffer &recurrentPrefixes, uint32_t rows,
    uint32_t maximumRows, float normEpsilon, FlashGDNLazyRollback *lazy,
    uint64_t *lazyTicket) {
  if (actualRequestStates.empty() || actualRequestStates.size() > 4 ||
      !rows || rows > maximumRows || !maximumRows || maximumRows > 4)
    throw std::invalid_argument(
        "Flash batch verify GDN requires lanes1..4 and rows1..maximumRows<=4");
  if (bool(lazy) != bool(lazyTicket) || (lazyTicket && *lazyTicket) ||
      (lazy && (lazy->pending() || rows > lazy->maximumRows() ||
                actualRequestStates.size() > lazy->maximumLanes())))
    throw std::invalid_argument("Flash batch verify GDN invalid lazy arena or ticket");
  if ((packedState.convolutionLaneStrideBytes &&
       packedState.convolutionLaneStrideBytes != kConvolutionStride) ||
      (packedState.recurrentLaneStrideBytes &&
       packedState.recurrentLaneStrideBytes != kRecurrentStride))
    throw std::invalid_argument("Flash batch verify GDN packed lane stride differs");

  const uint32_t lanes = static_cast<uint32_t>(actualRequestStates.size());
  const uint64_t prefixRows = maximumRows - 1;
  const uint64_t convolutionLaneStride = prefixRows * kConvolutionStride;
  const uint64_t recurrentLaneStride = prefixRows * kRecurrentStride;
  const FlashGDNState packed{
      checkedView(backend, packedState.convolution,
                  uint64_t{lanes} * kConvolutionStride, "packed convolution"),
      checkedView(backend, packedState.recurrent,
                  uint64_t{lanes} * kRecurrentStride, "packed recurrent"),
      kConvolutionStride, kRecurrentStride};

  metal::MetalBuffer convolutionTape, recurrentTape;
  if (rows > 1 && !lazy) {
    convolutionTape = checkedView(backend, convolutionPrefixes,
        uint64_t{lanes} * convolutionLaneStride, "convolution prefix tape");
    recurrentTape = checkedView(backend, recurrentPrefixes,
        uint64_t{lanes} * recurrentLaneStride, "recurrent prefix tape");
  }
  const FlashGDNCapture capture{recurrentTape, kRecurrentStride,
                               recurrentLaneStride, rows - 1, 512};

  // Reuse the complete qualified GDN geometry/weight/extent validation. This
  // graph is discarded and never submitted; invalid arguments leave graph
  // untouched, including failures discovered while creating copy views.
  metal::CommandGraph validated;
  if (rows > 1 && !lazy)
    addGDNFusedCaptured(validated, weights, buffers, packed, capture, rows,
                        lanes, normEpsilon);
  else
    addGDNFused(validated, weights, buffers, packed, rows, lanes,
                FlashGDNFusion::PersistentHead512, normEpsilon);

  const uint64_t rowCount = uint64_t{lanes} * rows;
  const uint64_t qkvBytes = rowCount * kHistoryRowBytes;
  const uint64_t valueBytes =
      rowCount * kFlashGDNOutputWidth * sizeof(uint16_t);
  const uint64_t gateBytes =
      rowCount * kFlashGDNValueHeads * sizeof(uint16_t);
  const std::array work{
      checkedView(backend, buffers.qkv, qkvBytes, "qkv input"),
      checkedView(backend, buffers.z, valueBytes, "z input"),
      checkedView(backend, buffers.a, gateBytes, "a input"),
      checkedView(backend, buffers.b, gateBytes, "b input"),
      checkedView(backend, buffers.mixed, qkvBytes, "mixed scratch"),
      checkedView(backend, buffers.decay, gateBytes * 2, "decay scratch"),
      checkedView(backend, buffers.beta, gateBytes, "beta scratch"),
      checkedView(backend, buffers.recurrentRows, valueBytes,
                  "recurrent row scratch"),
      checkedView(backend, buffers.output, valueBytes, "output"),
      checkedView(backend, buffers.diagnostics, sizeof(uint32_t), "diagnostics")};
  std::vector<Region> regions;
  regions.reserve(work.size() + 4 + uint64_t{lanes} * 2 + 4);
  // Treat input regions as writable here too, preserving the established
  // qualified GDN requirement that every work buffer is separate.
  const std::array originalWork{buffers.qkv, buffers.z, buffers.a, buffers.b,
      buffers.mixed, buffers.decay, buffers.beta, buffers.recurrentRows,
      buffers.output, buffers.diagnostics};
  for (size_t index = 0; index < work.size(); ++index)
    regions.push_back({lazy ? checkedView(backend, originalWork[index],
        originalWork[index].sizeBytes(), "lazy work") : work[index], true});
  regions.push_back({packed.convolution, true});
  regions.push_back({packed.recurrent, true});
  if (rows > 1 && !lazy) {
    regions.push_back({convolutionTape, true});
    regions.push_back({recurrentTape, true});
  }
  if (lazy)
    for (const auto &buffer : lazy->arenaBuffers())
      regions.push_back({checkedView(backend, buffer, buffer.sizeBytes(),
                                    "lazy arena"), true});

  std::vector<Copy> pack, histories, scatter;
  pack.reserve(uint64_t{lanes} * 2);
  histories.reserve(uint64_t{lanes} * (rows - 1) * 2);
  scatter.reserve(uint64_t{lanes} * 2);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto &request = actualRequestStates[lane];
    requireRequestStride(request.convolutionLaneStrideBytes,
                         flashGDNConvolutionLaneBytes());
    requireRequestStride(request.recurrentLaneStrideBytes,
                         flashGDNRecurrentLaneBytes());
    const auto requestConvolution = checkedView(backend, request.convolution,
        flashGDNConvolutionLaneBytes(), "request convolution");
    const auto requestRecurrent = checkedView(backend, request.recurrent,
        flashGDNRecurrentLaneBytes(), "request recurrent");
    regions.push_back({requestConvolution, true});
    regions.push_back({requestRecurrent, true});
    const auto packedConvolution = backend.view(packed.convolution,
        uint64_t{lane} * kConvolutionStride, flashGDNConvolutionLaneBytes());
    const auto packedRecurrent = backend.view(packed.recurrent,
        uint64_t{lane} * kRecurrentStride, flashGDNRecurrentLaneBytes());
    pack.push_back({requestConvolution, packedConvolution,
                   flashGDNConvolutionLaneBytes()});
    pack.push_back({requestRecurrent, packedRecurrent,
                   flashGDNRecurrentLaneBytes()});
    scatter.push_back({packedConvolution, requestConvolution,
                      flashGDNConvolutionLaneBytes()});
    scatter.push_back({packedRecurrent, requestRecurrent,
                      flashGDNRecurrentLaneBytes()});

    for (uint32_t kept = 1; !lazy && kept < rows; ++kept) {
      const auto destination = backend.view(convolutionTape,
          uint64_t{lane} * convolutionLaneStride +
              uint64_t{kept - 1} * kConvolutionStride,
          flashGDNConvolutionLaneBytes());
      if (kept < 3) {
        const uint64_t bytes = uint64_t{3 - kept} * kHistoryRowBytes;
        histories.push_back({backend.view(requestConvolution,
                                 uint64_t{kept} * kHistoryRowBytes, bytes),
                             backend.view(destination, 0, bytes), bytes});
      }
      const uint64_t bytes = uint64_t{kept} * kHistoryRowBytes;
      histories.push_back({backend.view(work[0],
                               uint64_t{lane} * rows * kHistoryRowBytes, bytes),
                           backend.view(destination,
                               uint64_t{3 - kept} * kHistoryRowBytes, bytes),
                           bytes});
    }
  }
  // Weights were validated above. Check backend ownership and protect their
  // read-only bytes from every writable region. The optional lazy helper uses
  // the production Shared weight ABI; preflight that requirement before pack.
  for (const auto &[tensor, bytes] :
       std::array<std::pair<const FlashTensor *, uint64_t>, 4>{{
           {weights.convolution, uint64_t{kFlashGDNConvolutionWidth} * 4 * 2},
           {weights.aLog, uint64_t{kFlashGDNValueHeads} * 2},
           {weights.timeBias, uint64_t{kFlashGDNValueHeads} * 2},
           {weights.norm, uint64_t{kFlashGDNHeadDimension} * 2}}})
    regions.push_back({checkedView(backend, tensor->buffer,
        lazy ? tensor->buffer.sizeBytes() : bytes,
        "immutable weight", bool(lazy)), bool(lazy)});
  requireDisjoint(regions);

  for (const auto &copy : pack) addCopy(graph, copy);
  if (lazy)
    *lazyTicket = lazy->begin(graph, weights, buffers, packed, rows, lanes,
                             normEpsilon);
  else if (rows > 1)
    addGDNFusedCaptured(graph, weights, buffers, packed, capture, rows, lanes,
                        normEpsilon);
  else
    addGDNFused(graph, weights, buffers, packed, rows, lanes,
                FlashGDNFusion::PersistentHead512, normEpsilon);
  for (const auto &copy : histories) addCopy(graph, copy);
  for (const auto &copy : scatter) addCopy(graph, copy);
}

} // namespace splash::flash
