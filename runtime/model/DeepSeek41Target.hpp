#pragma once

// DeepSeek-V4.1-Flash's target (model/DeepSeek41.hpp): its layers over the
// QwenTarget interface, with steps of its own, as its attention reads its
// window and compressed groups and not the pages' K/V of the shared blocks.
// Each sublayer reads the mix of the four fp32 residual streams and adds its
// output back to each (ops::ManifoldHyperConnection); the Engram layers gate
// their table rows into the streams first; the head reads the last mix
// through the final norm.
//
// A sequence's state (QwenHybridLayout::gdnStateLayout's auxiliary bytes)
// holds each layer's ring of its window's last latents and each paired KV
// source's open group; its pages (kvLayout) hold each KV source's compressed
// groups, a group's latent as the key of the place of the position that
// completes it and its index key in the place's value. A step's rows attend
// their window, of the ring before the step and the step's own latents, and
// the groups their indexer selects, through the sequence's page table
// (ops::LatentAttention, ops::LatentIndex). A prefill writes each
// sequence's state after its rows, and each state it captures, as it goes; a
// verify leaves it to its commit, which writes the state after each lane's
// retained rows from the step's latents and projections.
//
// Its step scratch beside the runtime's arenas: per prefill chunk what lives
// across its layers (its streams, mixes, the indexers' selections and the
// candidate layer's blocks, its staged Engram rows and its MoE output), the
// work of a layer shared by the chunks in tiles of rows, and a verify's.

#include "model/DeepSeek41.hpp"
#include "ops/ManifoldHyperConnection.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

class DeepSeek41Target final : public QwenTarget {
public:
  // A target whose prefill commands run up to `prefillChunks` chunks.
  DeepSeek41Target(const DeepSeek41Weights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                   const ops::ExecutionPlans &operators, uint32_t prefillChunks);
  ~DeepSeek41Target() override;

  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  // Each lane's prior tokens, which the GPU hashes its verify rows' Engram
  // rows of.
  void prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) override;
  // The GPU hashes the verify rows' Engram rows, and a host step of the
  // command stages them from the tables' file before the verify gathers
  // them.
  void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput, metal::MetalBuffer proposals,
                      metal::MetalBuffer verifyInput, uint32_t lanes) const override;
  void addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                 SpilledKvReads &spilled, std::span<const kv::ChunkedPrefillParams> chunks,
                 uint32_t lanes) const override;
  // The hidden rows are the final norm's of the last mix, which the head
  // reads as they are.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const override;
  // Each lane's rings and open groups after its retained rows, from the
  // verify's latents and projections.
  void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers, uint32_t lanes) const override;

  // The step scratch a target of `layout` keeps, whose prefill commands run
  // up to `prefillChunks` chunks and whose Engram rows stage rows of
  // `tableRowBytes` bytes.
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const DeepSeek41Layout &layout,
                                                           const ops::ManifoldHyperShape &mhc, uint32_t prefillChunks,
                                                           uint64_t tableRowBytes);

private:
  struct Scratch;
  struct Rows;

  // A chunk's or a verify's rows through a layer's Engram and attention
  // (the FFN's mix, routing and experts are addPrefill's and addVerify's),
  // a verify's projections through the decode plans of `lanes` lanes.
  void addEngram(metal::CommandGraph &graph, const Scratch &s, const ops::LinearScratch &linear, bool decode,
                 uint32_t lanes, uint32_t index, uint32_t rows, uint32_t staged) const;
  void addAttention(metal::CommandGraph &graph, const Scratch &s, const ops::LinearScratch &linear, bool decode,
                    uint32_t lanes, uint32_t index, std::span<const Rows> sequences, uint32_t rows,
                    const metal::MetalBuffer &normalized, const metal::MetalBuffer &previousMixes,
                    std::span<const SplashKvLayer> kvLayers) const;
  void addSourceGroups(metal::CommandGraph &graph, const Scratch &s, uint32_t index, const Rows &sequence,
                       const metal::MetalBuffer &normalized, std::span<const SplashKvLayer> kvLayers) const;
  void addIndexer(metal::CommandGraph &graph, const Scratch &s, uint32_t index, const Rows &sequence, uint32_t begin,
                  uint32_t first, uint32_t count, const metal::MetalBuffer &normalized,
                  std::span<const SplashKvLayer> kvLayers) const;
  // The Engram rows of the next prefill command, read ahead from the tables'
  // file while this one runs (Qwen4Steps::readAhead): each lane's prompt
  // after its last sequence of `chunks`, a command's chunks of rows at most.
  void readAhead(std::span<const QwenTargetPrefillChunk> chunks) const;
  // The rows' projection through ops::Linear: a prefill's chunks of up to
  // `rows` rows, or a verify's decode plan of `lanes` lanes.
  void project(metal::CommandGraph &graph, metal::MetalBuffer input, const ops::Projection &projection,
               metal::MetalBuffer output, uint32_t rows, bool decode, uint32_t lanes,
               const ops::LinearScratch &linear) const;
  // The KV source of layer `index`: the last compressor layer at or before
  // it, by its index among them, and its ratio.
  [[nodiscard]] uint32_t sourceOf(uint32_t index) const;
  [[nodiscard]] uint32_t ratioOf(uint32_t source) const;
  // The mean of the four streams of `rows` rows (`streams`, from the first
  // row of the capture on), into capture slot `slot` of the captured hidden
  // rows from row `destinationRow`: what DSpark reads at the input of each
  // of its target layers.
  void addStreamCapture(metal::CommandGraph &graph, const metal::MetalBuffer &streams, uint32_t slot,
                        const metal::MetalBuffer &captured, uint32_t destinationRow, uint32_t rows) const;
  // A sequence's state: layer `layer`'s ring, and paired source `paired`'s
  // open group.
  [[nodiscard]] metal::MetalBuffer ring(const metal::MetalBuffer &state, uint32_t layer) const;
  [[nodiscard]] metal::MetalBuffer openGroup(const metal::MetalBuffer &state, uint32_t paired) const;
  // Buffer `buffer` from row `row` of `width` values of `bytes` each to its
  // end: what a prefill projection of a tile from that row reads or writes,
  // whole tiles past its rows.
  [[nodiscard]] metal::MetalBuffer fromRow(const metal::MetalBuffer &buffer, uint32_t row, uint32_t width,
                                           uint32_t bytes) const;

  const DeepSeek41Weights &weights_;
  const DeepSeek41Dimensions &d_;
  ops::ManifoldHyperShape mhc_{};
  ops::MoeShape moeShape_{};
  uint32_t prefillChunks_ = 0;
  // The RoPE inverse frequencies of the uncompressed layers and, YaRN's, of
  // the compressed ones; the first mix's weights of a step's rows (stream 0
  // alone); each Engram layer's gate weights (engram_gate, its weights'
  // buffer); zeros a MoE adds its routes to.
  std::array<metal::MetalBuffer, 2> frequencies_;
  metal::MetalBuffer firstMixes_, zeros_;
  std::vector<metal::MetalBuffer> engramGates_;
  // Each KV source's layer (the compressor layers, ascending).
  std::vector<uint32_t> sources_;
  std::vector<uint32_t> pairedOf_;
  // The scratch of each prefill chunk and of the verify, and the lanes'
  // positions a verify was prepared for.
  std::vector<Scratch> prefill_;
  std::unique_ptr<Scratch> verify_;
  std::array<uint64_t, ExecutionLimits::maximumBatchWidth> preparedPositions_{};
  uint64_t allocatedPrefill_ = 0, allocatedDecode_ = 0;
};

} // namespace splash::model
