#pragma once

// The Qwen3 decoder the Omni models share (docs/omni-design.md): 16 query
// heads over 8 KV heads of 128 dimensions, per-head q/k RMSNorm, full
// rotate-half RoPE, SwiGLU, with projections in either the affine Q4 layout
// (ops::Linear) or dense bf16 (ops::omni::DenseLinear). One instance serves
// the TTS talker (hidden 2048, 28 layers), the TTS code predictor (hidden
// 1024, 5 layers) and the ASR text decoder (hidden 2048, 28 layers).
//
// addForward encodes one forward over `rows` rows whose embeddings the
// arena's hidden buffer holds: ragged prefill (several sequences, up to
// 2048 rows) or a decode step (one row per lane, up to 32 lanes), reading
// and extending the lanes' KV pages, then the final norm and the head over
// the selected rows into fp32 logits. A verify arena runs a decode arena's
// projections over ragged sequences of any rows (up to 32 in all) with the
// fused verify attention, or the prefill attention beyond its rows and
// pages: speculative verification, and a draft catching up.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Omni.hpp"
#include "ops/Weights.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace splash::omni {

struct Qwen3Dims final {
  uint32_t layers = 0;
  uint32_t hidden = 0;
  uint32_t intermediate = 0;
  // Rows of the head (padded to whole Q4 tiles when affine) and the logits
  // it means (the real vocabulary).
  uint32_t headRows = 0;
  uint32_t vocabulary = 0;
  float rmsEpsilon = 1e-6F;
  double ropeTheta = 1e6;
};

using Qwen3Projection = std::variant<ops::Projection, ops::omni::DenseWeights>;

struct Qwen3LayerWeights final {
  ops::NormWeights inputNorm, postAttentionNorm;
  Qwen3Projection qkv; // [4096][hidden]: q (2048), k (1024), v (1024)
  metal::MetalBuffer queryNorm, keyNorm; // bf16 [128]
  Qwen3Projection output;                // [hidden][2048]
  Qwen3Projection gate, up, down;
};

struct Qwen3Weights final {
  Qwen3Dims dims;
  std::vector<Qwen3LayerWeights> layers;
  ops::NormWeights finalNorm;
  std::optional<Qwen3Projection> head; // [headRows][hidden]
};

// The pages and page tables a forward reads (omni/KvPages.hpp).
struct KvBinding final {
  metal::MetalBuffer pageTable;
  uint32_t pageTableEntries = 0;
  metal::MetalBuffer pages;
  uint32_t pageCount = 0;
};

// The scratch of forwards of up to maxRows rows over up to maxSequences
// sequences, selecting up to maxLogitRows rows for logits. Prefill arenas
// take up to 2048 rows; decode arenas up to 32, and verify arenas (decode
// projections, ragged attention) up to 32.
class Qwen3Arena final {
public:
  Qwen3Arena(metal::MetalBackend &backend, const ops::Linear &linear, const Qwen3Weights &weights, uint32_t maxRows,
             uint32_t maxSequences, uint32_t maxLogitRows, bool decode, bool verify = false);
  [[nodiscard]] uint32_t maxRows() const noexcept { return maxRows_; }
  [[nodiscard]] uint32_t maxSequences() const noexcept { return maxSequences_; }
  [[nodiscard]] uint32_t maxLogitRows() const noexcept { return maxLogitRows_; }
  // Whether projections run the decode plans (decode and verify arenas).
  [[nodiscard]] bool decode() const noexcept { return decode_; }
  // Whether attention takes ragged sequences through the prefill kernels.
  [[nodiscard]] bool verify() const noexcept { return verify_; }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

  // Inputs the caller fills before addForward. The attention tables live in
  // a ring of slots, so several forwards of one arena can be encoded into
  // one command (the code predictor's sub-steps): uploadTables advances the
  // slot and addForward reads the current one.
  static constexpr uint32_t kTableSlots = 16;
  struct TableSlot final {
    metal::MetalBuffer rowSequence; // u32 [maxRows]
    metal::MetalBuffer sequences;   // OmniSequenceParams [maxSequences]
    metal::MetalBuffer tiles;       // OmniPrefillTile
  };
  metal::MetalBuffer hidden;      // bf16 [rowCapacity][hidden]: the embeddings in, the last layer's output out
  std::array<TableSlot, kTableSlots> slots;
  uint32_t slot = 0;
  [[nodiscard]] const TableSlot &currentSlot() const noexcept { return slots[slot]; }
  // The tables a forward's attention reads when they differ from the ones
  // its keys and values are stored with (ForwardStep::visibility), sized
  // for a sequence per row (uploadVisibility).
  TableSlot visibilitySlot;
  metal::MetalBuffer logitRows;   // u32 [maxLogitRows]: dispatch rows whose logits are taken
  // Outputs.
  metal::MetalBuffer selected;    // bf16 [logit rows padded][hidden]: the selected rows' final hidden (normalized)
  metal::MetalBuffer logits;      // fp32 [maxLogitRows padded][headRows]

  // Scratch.
  metal::MetalBuffer residual, normalized, sums, qkv, attention, attentionSums, intermediate, gateScratch, downSums;
  metal::MetalBuffer queries, partials, statistics, attentionCounters, selectedRaw;
  // The fused decode attention's rotation table (decode arenas).
  metal::MetalBuffer rotations;
  ops::LinearScratch scratch;
  // The dense projections' K-split partials and counters
  // (ops::omni::GemmBuffers).
  metal::MetalBuffer denseSplitPartials, denseSplitCounters;
  uint32_t rowCapacity = 0;
  uint32_t logitCapacity = 0;
  uint32_t tileCapacity = 0;

private:
  uint32_t maxRows_, maxSequences_, maxLogitRows_;
  bool decode_, verify_;
  uint64_t bytes_ = 0;
};

// The outputs of some of a forward's layers, kept for a DFlash draft
// (AsrEngine): after layer layers[j], rows 0..rows-1 of its output (the
// residual stream, before the final norm) are copied into `buffer` as
// [row][j][hidden], through the identity `rowIndex`, so a row's captures
// are one input row of the draft's fc projection.
struct HiddenCapture final {
  // Views of one bf16 [rowCapacity][layers.size()][hidden] buffer, the j-th
  // from slot j of row 0 to the buffer's end.
  std::span<const metal::MetalBuffer> slots;
  metal::MetalBuffer rowIndex; // u32 identity over at least the forward's rows
  std::span<const uint32_t> layers;
  uint32_t rowCapacity = 0;
};

struct ForwardStep final {
  const ops::omni::AttentionTables *tables = nullptr; // the dispatch's sequences; uploaded by the caller
  uint32_t logitRows = 0;                             // rows selected through arena.logitRows (0: no logits)
  // A prefill forward's FFN split with the Neural Engine (aneFfnFor), on
  // which the caller called begin() and through which it commits the
  // command; none runs the FFN on the GPU alone.
  ops::AneFfn *aneFfn = nullptr;
  // A decode step whose arena.qkv already holds layer 0's QKV rows, as
  // addFirstQkv computes them for a step of rows that sameFirstQkv
  // matches: the forward starts at layer 0's attention.
  bool firstQkvReady = false;
  // A decoder without a head leaves the selected rows unnormalized (the
  // hidden rows when the logit rows are the step's own, else
  // arena.selectedRaw): the caller's head folds the final norm.
  bool rawSelected = false;
  // Layer outputs to keep (HiddenCapture), or none.
  const HiddenCapture *capture = nullptr;
  // The tables attention reads instead of `tables`, which then only store
  // the rows' keys and values and rotate the queries at their positions: a
  // DFlash draft's block, each row a one-row sequence that sees the context
  // and the whole block (uploaded with uploadVisibility). Prefill-path
  // arenas only; decode and verify arenas store and attend in one dispatch.
  const ops::omni::AttentionTables *visibility = nullptr;
};

// A split of the prefill FFN with the Neural Engine (ops::AneFfn): the ANE
// takes `units` of the intermediate channels' units of 512 for chunks of
// functionRows.front() up to functionRows.back() rows, each on the smallest
// of the functions that holds it.
struct AneFfnPolicy final {
  uint32_t units = 0;
  std::vector<uint32_t> functionRows;
};

class Qwen3Decoder final {
public:
  Qwen3Decoder(metal::MetalBackend &backend, Qwen3Weights weights);
  [[nodiscard]] const Qwen3Weights &weights() const noexcept { return weights_; }
  [[nodiscard]] const Qwen3Dims &dims() const noexcept { return weights_.dims; }
  [[nodiscard]] const ops::Linear &linear() const noexcept { return linear_; }
  [[nodiscard]] bool affine() const noexcept;
  // Uploads `tables` into the arena (its sequences, tiles and row-sequence
  // buffers); the caller sets logitRows itself.
  static void uploadTables(Qwen3Arena &arena, const ops::omni::AttentionTables &tables);
  // Uploads a forward's visibility tables (ForwardStep::visibility) into the
  // arena's visibility slot.
  static void uploadVisibility(Qwen3Arena &arena, const ops::omni::AttentionTables &tables);
  // A DFlash draft's context rows (AsrEngine): each layer's key and value
  // projections of arena.hidden's `tables.rows` rows as they are (no input
  // norm), the keys normalized and rotated at their positions, stored in the
  // pages `tables` place them at. No queries are kept, no attention runs.
  void addKeysValues(metal::CommandGraph &graph, Qwen3Arena &arena, const KvBinding &kv,
                     const ops::omni::AttentionTables &tables) const;
  // Builds the prefill FFN's Neural Engine split of an affine decoder and
  // checks it against the GPU alone with `prefill`'s buffers
  // (ops::AneFfn::verify), which it overwrites; returns the check's largest
  // error. Throws, keeping no split, if it does not come up or check out.
  double enableAneFfn(metal::MetalBackend &backend, const AneFfnPolicy &policy, Qwen3Arena &prefill);
  // The split, if it takes a prefill chunk of `rows` rows; none when the GPU
  // runs its FFN alone.
  [[nodiscard]] ops::AneFfn *aneFfnFor(uint32_t rows) const noexcept;
  // The split, or none.
  [[nodiscard]] ops::AneFfn *aneFfn() const noexcept { return aneFfn_.get(); }
  void addForward(metal::CommandGraph &graph, Qwen3Arena &arena, const KvBinding &kv, const ForwardStep &step) const;
  // Layer 0's input norm and QKV projection of `rows` rows of `input` into
  // `qkv` (bf16 [rows][4096]), as a decode step of `rows` rows computes
  // them, in a decode arena; `input` holds whole lanes of 8 rows when the
  // layers are affine.
  void addFirstQkv(metal::CommandGraph &graph, Qwen3Arena &arena, metal::MetalBuffer input, metal::MetalBuffer qkv,
                   uint32_t rows) const;
  // Whether decode steps of `rows` and of `otherRows` rows compute each
  // row's layer-0 QKV with the same arithmetic, so one's rows stand in for
  // the other's (ForwardStep::firstQkvReady).
  [[nodiscard]] bool sameFirstQkv(uint32_t rows, uint32_t otherRows) const;
  // Whether a decode step of `rows` rows projects layer 0's QKV row by row
  // on a kernel whose sums differ from the tiles' of more rows only by
  // their association: the Qmv tile of one row or two sharing it
  // (ops::LinearTile::Qmv, each row the one-row kernel's bits), or the
  // dense GEMV against the dense MMA tiles.
  [[nodiscard]] bool firstQkvRowwise(uint32_t rows) const;

private:
  void addLayer(metal::CommandGraph &graph, Qwen3Arena &arena, const KvBinding &kv,
                const ops::omni::AttentionTables &tables, uint32_t layer, uint32_t rows, ops::AneFfn *split,
                bool qkvReady = false, const ops::omni::AttentionTables *visibility = nullptr) const;
  void addAttention(metal::CommandGraph &graph, const Qwen3Arena &arena, const ops::omni::AttentionBuffers &attention,
                    const ops::omni::AttentionTables &tables, uint32_t layer, const KvBinding &kv,
                    const ops::omni::AttentionTables *visibility = nullptr) const;
  [[nodiscard]] ops::omni::AttentionBuffers attentionBuffers(const Qwen3Arena &arena, const KvBinding &kv,
                                                             uint32_t layer) const;
  void addHead(metal::CommandGraph &graph, Qwen3Arena &arena, uint32_t rows, uint32_t logitRows,
               bool rawSelected = false) const;
  // Both projection formats behind one call: output = act(W x) (+residual).
  void addProjection(metal::CommandGraph &graph, Qwen3Arena &arena, const Qwen3Projection &projection,
                     metal::MetalBuffer input, metal::MetalBuffer inputSums, metal::MetalBuffer output,
                     metal::MetalBuffer residual, uint32_t rows, ops::PreparedInput prepared,
                     ops::FloatOutput destination = ops::FloatOutput::BFloat16) const;

  Qwen3Weights weights_;
  ops::Linear linear_;
  metal::MetalBuffer inverseFrequencies_;
  std::unique_ptr<ops::AneFfn> aneFfn_;
};

} // namespace splash::omni
