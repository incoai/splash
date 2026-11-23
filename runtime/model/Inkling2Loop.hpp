#pragma once

// Inkling-2-Mini (GGUF architecture inkling2_loop, or inkling2 in Unsloth's
// own layout; text model type inkling2_loop_text): a decoder of 24 weight
// layers that all attend, its layers 6 to 17 looped three times a token, over
// a sparse MoE FFN.
//  - The loop: a token's forward runs layers 0-5, layers 6-17 once a pass,
//    three passes, then layers 18-23: 48 layer applications, each with its
//    own KV and convolution state. Passes after the first restart from the
//    states before layer 6 and gate each layer's q, k, v and FFN inputs by
//    messages of the pass before.
//  - Attention of 24 query heads over 8 KV heads of 128 without RoPE: a
//    learned scale and an L2 clip of the queries and keys, a value norm
//    without a weight, a relative bias of a rank-16 score per row and head
//    over the 1024 distances that end at a row, a sink logit per row and
//    head, a sigmoid output gate, and a log scaling of the queries and
//    relative scores in the 4 layers that attend every key; the other 20
//    slide over 1024 keys, the row's own included.
//  - Short causal convolutions of 4 taps on the keys, the values and every
//    mixer's and FFN's output; on the 4 full layers the keys' and values'
//    taps add a projection of the layer's input.
//  - A block residual: the inputs of layers 0, 6, 12 and 18 join a stack of
//    block states, which every mixer, FFN and the head mix by learned
//    queries in place of the residual.
//  - A MoE of 128 experts of 1152, its top 8 selected by sigmoid score plus a
//    bias and weighted by their normalized scores times 8, its gate and up
//    softcapped, no shared expert; logits softcapped at 30.
//  - A vocabulary head of 201024 rows, 200058 of them tokens; one stop
//    token, and no mask token of its own.
//  - Its DFlash2 draft (incoai/Inkling-2-Mini-DFlash2) reads the
//    rows applications 5, 23 and 41 read and the final norm's, its token
//    rows through the table's norm, and its head's finish: no padding row,
//    and the softcap on its candidates' scores after its selector's edges.
// The shared runtime runs it as a target of 48 layers that all attend
// (Inkling2LoopLayout's QwenTargetDimensions, a layer an application); what
// is its own (its weight layers, the loop, the mapping of an application to
// its weight layer, its attention's terms) is Inkling2LoopLayout's alone,
// which its loader, its GGUF plan (model/Inkling2LoopGguf.cpp), its config
// check and its target (model/Inkling2LoopTarget.hpp) read. Its weights load
// from a GGUF.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/ShortConvolution.h"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/RelativeAttention.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::model {

struct GgufTargetPlan;
struct SafetensorsTargetPlan;
struct SparseRowStoreOptions;

// Where an application reads its input rows and writes its output rows, bf16
// rows of hiddenSize values: buffers.hidden[index] of the step, plane `index`
// of the block stack, or plane `index` of the pass before's states
// (Inkling2LoopScratch).
enum class Inkling2LoopRowsKind : uint8_t { Hidden, Stack, Previous };
struct Inkling2LoopRows final {
  Inkling2LoopRowsKind kind = Inkling2LoopRowsKind::Hidden;
  uint32_t index = 0;
  bool operator==(const Inkling2LoopRows &) const = default;
};

// The four short convolutions of an application, in the order their state
// columns follow one another (Inkling2LoopSchedule::convolutionColumn).
enum class Inkling2LoopConvolution : uint8_t { Key, Value, Mixer, Ffn };

// The loop's shape, as the family's layout states it
// (Inkling2LoopLayout::loopShape): `weightLayers` weight layers, of which
// loopFirstLayer to loopEndLayer - 1 run loopPasses times a token; a block
// of the block residual every blockResidualPeriod layers; the weight layers
// that slide (a bit each); and the widths its convolutions' state and its
// step scratch take. These are the family's alone: the shared dimensions
// hold the layers a step runs, one per application (applications()), and
// nothing of the loop.
struct Inkling2LoopShape final {
  uint32_t weightLayers = 0;
  uint32_t loopFirstLayer = 0;
  uint32_t loopEndLayer = 0;
  uint32_t loopPasses = 0;
  uint32_t blockResidualPeriod = 0;
  uint64_t slidingWeightLayers = 0;
  uint32_t hiddenSize = 0;
  uint32_t kvWidth = 0;
  uint32_t convolutionTaps = 0;
  // The rank of a row and head's relative score and of a message gate,
  // which the step scratch holds.
  uint32_t relativeRank = 0;
  uint32_t gateRank = 0;
  bool operator==(const Inkling2LoopShape &) const = default;
};

// The order of a token's layer applications and what each one reads, as the
// source's forward runs them: the layers before the loop, the loop's layers
// once a pass, the layers after it. Every rule is a function of the
// application, so that a prefill command may stop between any two
// applications and the next command go on from the rows the schedule names.
//
// The block stack holds a plane per block (blockResidualPeriod layers): the
// input rows of each boundary layer (layer % period == 0). A boundary
// application reads its input from its plane, which the application before it
// writes, so the push is that write. Every pass of the loop reads the same
// planes before the loop's first layer, which nothing writes after the layer
// before the loop, so they are the state the source keeps before the loop and
// each later pass restarts from (pre_loop_hidden_states, pre_loop_block_states):
// a restarting application reads its input from the loop's first plane as
// the first pass did. The pass before's states, which a later pass's messages
// mix (the block states its pass pushed after its first, previous_block_states,
// and its output), are the `previous` planes: its last application writes its
// output there, and the restart copies its pushed planes there before the
// pass writes its own over them. The other applications pass their rows from
// one of the step's two hidden buffers to the other.
struct Inkling2LoopSchedule final {
  Inkling2LoopShape shape;

  constexpr explicit Inkling2LoopSchedule(const Inkling2LoopShape &value) : shape(value) {}

  // The loop and the stack are whole blocks, and an application is a bit of
  // a kv::Layout's windowed layers.
  [[nodiscard]] constexpr bool valid() const noexcept {
    const Inkling2LoopShape &s = shape;
    const uint32_t period = s.blockResidualPeriod;
    return period && s.weightLayers % period == 0 && s.loopPasses && s.loopFirstLayer % period == 0 &&
           s.loopEndLayer % period == 0 && s.loopFirstLayer < s.loopEndLayer && s.loopEndLayer <= s.weightLayers &&
           applications() <= 64 && s.convolutionTaps > 1 && s.hiddenSize && s.kvWidth;
  }

  [[nodiscard]] constexpr uint32_t loopLayers() const noexcept { return shape.loopEndLayer - shape.loopFirstLayer; }
  [[nodiscard]] constexpr bool isLoopLayer(uint32_t layer) const noexcept {
    return layer >= shape.loopFirstLayer && layer < shape.loopEndLayer;
  }
  // The layers a step runs, one per application: the family's shared
  // QwenTargetDimensions::layers (Inkling2LoopLayout).
  [[nodiscard]] constexpr uint32_t applications() const noexcept {
    return shape.weightLayers + (shape.loopPasses - 1) * loopLayers();
  }
  [[nodiscard]] constexpr uint32_t weightLayer(uint32_t application) const noexcept {
    if (application < shape.loopFirstLayer) return application;
    const uint32_t looped = application - shape.loopFirstLayer;
    if (looped < shape.loopPasses * loopLayers()) return shape.loopFirstLayer + looped % loopLayers();
    return application - (shape.loopPasses - 1) * loopLayers();
  }
  // Its pass of the loop from 0; 0 outside the loop.
  [[nodiscard]] constexpr uint32_t pass(uint32_t application) const noexcept {
    if (application < shape.loopFirstLayer) return 0;
    const uint32_t pass = (application - shape.loopFirstLayer) / loopLayers();
    return pass < shape.loopPasses ? pass : 0;
  }
  // Whether the application pushes its input rows onto the block stack
  // (is_block_boundary): its mixer's output is then the residual alone.
  [[nodiscard]] constexpr bool boundary(uint32_t application) const noexcept {
    return weightLayer(application) % shape.blockResidualPeriod == 0;
  }
  // Whether it attends every key rather than a window, which its three mixer
  // queries, input norms, key and value taps and log scaling follow.
  [[nodiscard]] constexpr bool full(uint32_t application) const noexcept {
    return !(shape.slidingWeightLayers >> weightLayer(application) & 1);
  }
  // The applications that slide, a bit each (kv::Layout::windowedLayers).
  [[nodiscard]] constexpr uint64_t slidingApplications() const noexcept {
    uint64_t bits = 0;
    for (uint32_t application = 0; application < applications() && application < 64; ++application)
      if (!full(application)) bits |= uint64_t{1} << application;
    return bits;
  }
  // Whether messages of the pass before gate its inputs: a loop layer's in
  // every pass after the first.
  [[nodiscard]] constexpr bool gated(uint32_t application) const noexcept {
    return isLoopLayer(weightLayer(application)) && pass(application) > 0;
  }
  // Whether a pass restarts at it from the state before the loop, after the
  // restart copies the pass before's states (previousPlanes).
  [[nodiscard]] constexpr bool restarts(uint32_t application) const noexcept {
    return gated(application) && weightLayer(application) == shape.loopFirstLayer;
  }

  // The stack's planes, one per block.
  [[nodiscard]] constexpr uint32_t stackPlanes() const noexcept {
    return shape.weightLayers / shape.blockResidualPeriod;
  }
  // The block states the mix before its mixer reads beside its input: those
  // pushed before its weight layer. Layer 0's mixer reads its input alone.
  [[nodiscard]] constexpr uint32_t mixerStates(uint32_t application) const noexcept {
    const uint32_t period = shape.blockResidualPeriod;
    return (weightLayer(application) + period - 1) / period;
  }
  // The block states the mix before its FFN reads beside its residual: its
  // own push too.
  [[nodiscard]] constexpr uint32_t ffnStates(uint32_t application) const noexcept {
    return weightLayer(application) / shape.blockResidualPeriod + 1;
  }
  // The head's mix reads every plane beside the last application's output.
  [[nodiscard]] constexpr uint32_t headStates() const noexcept { return stackPlanes(); }

  // The pass before's states a later pass's messages mix: the block states a
  // pass pushes after the loop's first (previousPlanes() - 1, from stack
  // plane firstPassPlane()), then the pass's output, the mix's current rows.
  [[nodiscard]] constexpr uint32_t previousPlanes() const noexcept {
    return loopLayers() / shape.blockResidualPeriod;
  }
  [[nodiscard]] constexpr uint32_t firstPassPlane() const noexcept {
    return shape.loopFirstLayer / shape.blockResidualPeriod + 1;
  }

  [[nodiscard]] constexpr Inkling2LoopRows input(uint32_t application) const noexcept {
    const uint32_t layer = weightLayer(application);
    if (layer % shape.blockResidualPeriod == 0)
      return {Inkling2LoopRowsKind::Stack, layer / shape.blockResidualPeriod};
    return {Inkling2LoopRowsKind::Hidden, application & 1};
  }
  // The next application's input, but a pass's output where a pass follows,
  // and the last application's, which the head's mix reads.
  [[nodiscard]] constexpr Inkling2LoopRows output(uint32_t application) const noexcept {
    if (application + 1 == applications()) return {Inkling2LoopRowsKind::Hidden, (application + 1) & 1};
    const uint32_t layer = weightLayer(application);
    if (isLoopLayer(layer) && layer + 1 == shape.loopEndLayer && pass(application) + 1 < shape.loopPasses)
      return {Inkling2LoopRowsKind::Previous, previousPlanes() - 1};
    return input(application + 1);
  }

  // A lane's convolution state (the target's auxiliary state): a bf16 matrix
  // of taps - 1 rows, the inputs before a row oldest first, by stateColumns()
  // columns, each application's four convolutions' channels side by side,
  // the key's and value's kvWidth, the mixer's and FFN's hiddenSize
  // (ops::ShortConvolution). Its values are exact in bf16: every convolution
  // reads a projection's or the experts' bf16 output.
  [[nodiscard]] constexpr uint32_t applicationColumns() const noexcept {
    return 2 * shape.kvWidth + 2 * shape.hiddenSize;
  }
  [[nodiscard]] constexpr uint32_t stateColumns() const noexcept { return applications() * applicationColumns(); }
  [[nodiscard]] constexpr uint32_t stateRows() const noexcept { return shape.convolutionTaps - 1; }
  [[nodiscard]] constexpr uint64_t stateBytes() const noexcept {
    return uint64_t{stateRows()} * stateColumns() * GdnStateLayout::bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint32_t convolutionColumn(uint32_t application,
                                                     Inkling2LoopConvolution convolution) const noexcept {
    const uint32_t offsets[] = {0, shape.kvWidth, 2 * shape.kvWidth, 2 * shape.kvWidth + shape.hiddenSize};
    return application * applicationColumns() + offsets[static_cast<uint32_t>(convolution)];
  }
  [[nodiscard]] constexpr uint32_t convolutionChannels(Inkling2LoopConvolution convolution) const noexcept {
    return convolution == Inkling2LoopConvolution::Key || convolution == Inkling2LoopConvolution::Value
               ? shape.kvWidth
               : shape.hiddenSize;
  }
};

// Inkling-2-Mini's layout. It is a layout of the Qwen hybrid's
// (QwenHybridLayout: a family's name, the dimensions the shared runtime runs
// on and the KV and state layouts they make) whose dimensions describe the
// layers a step runs: one per application, 48, every one attending
// (fullAttentionPeriod 1) without RoPE, those over its sliding weight layers
// windowed, so the base's kvLayout() is its KV layout as it is. Its own
// members hold the rest: its weight layers, the loop and the applications'
// mapping to them, and its attention's, convolutions', block residual's,
// FFN's and head's own values. Its state layout (gdnStateLayout) adds its
// convolutions' rows to the base's, which has no GDN layers.
struct Inkling2LoopLayout final : QwenHybridLayout {
  // Its config's text model type and its GGUFs' architecture as llama.cpp's
  // converter writes them (model/Inkling2LoopGguf.cpp plans Unsloth's layout
  // too).
  static constexpr std::string_view modelType = "inkling2_loop_text";
  static constexpr std::string_view ggufArchitecture = "inkling2_loop";
  // Its layers attend through its own operator (ops::RelativeAttention), not
  // the shared attention.
  static constexpr bool sharedAttention = false;

  // Its weight layers, and the loop: weight layers loopFirstLayer to
  // loopEndLayer - 1 run loopPasses times a token, with their weights shared;
  // each pass after the first gates its layers' inputs through message gates
  // of rank loopGateRank. The block residual: the input of each
  // blockResidualPeriod-th weight layer joins the stack of block states. The
  // weight layers that slide over slidingWindow keys, a bit each (its
  // config's local_layer_ids): layers 0 to 23 but 5, 11, 17 and 23, the last
  // of each block. The loop's schedule reads them (loopShape).
  uint32_t weightLayers = 24;
  uint32_t loopFirstLayer = 6;
  uint32_t loopEndLayer = 18;
  uint32_t loopPasses = 3;
  uint32_t loopGateRank = 64;
  uint32_t blockResidualPeriod = 6;
  uint64_t slidingWeightLayers = 0x7D'F7DF;
  // The relative bias: a score of relativeRank values per row and head, times
  // its learned scale and L2-clipped per head to relativeClip, against the
  // bias row of the row's distance to a key under relativeExtent, zero past.
  uint32_t relativeRank = 16;
  uint32_t relativeExtent = 1024;
  float relativeClip = 8.0F;
  // The queries' and keys' L2 clip per head, the queries' after their learned
  // scale: its config's qk_l2_clip_threshold, 16 sqrt(2), as F32 (22.627417,
  // as its GGUF states it).
  float queryClip = 22.627416997969522F;
  // The full layers' log scaling of the queries and relative scores:
  // 1 + logScalingAlpha * ln(max((position + 1) / logScalingFloor, 1)).
  uint32_t logScalingFloor = 8192;
  float logScalingAlpha = 0.1F;
  // The taps of each short convolution.
  uint32_t convolutionTaps = 4;

  // The model's own window (its config's model_max_length, its GGUF's
  // context_length): 2^20, the runtime's KV ceiling.
  uint32_t modelContextTokens = 1'048'576;

  // Whether its attention's and convolutions' values are the ones their
  // kernels compile (metal/abi/RelativeAttention.h, ShortConvolution.h,
  // ops::RelativeAttention), which compute with their own: a file is held to
  // the layout's (its config and GGUF checks), so the two must be one.
  [[nodiscard]] constexpr bool matchesKernels() const noexcept {
    return attentionQueryHeads == ops::RelativeAttention::kQueryHeads &&
           relativeRank == SPLASH_RELATIVE_ATTENTION_STATES && relativeExtent == SPLASH_RELATIVE_ATTENTION_EXTENT &&
           relativeClip == SPLASH_RELATIVE_ATTENTION_STATE_CLIP && queryClip == SPLASH_RELATIVE_ATTENTION_QK_CLIP &&
           static_cast<float>(logScalingFloor) == SPLASH_RELATIVE_ATTENTION_LOG_SCALE_FLOOR &&
           logScalingAlpha == SPLASH_RELATIVE_ATTENTION_LOG_SCALE_ALPHA && convolutionTaps == SHORT_CONVOLUTION_TAPS;
  }

  // Inkling-2-Mini's, its first family (kFamilies).
  constexpr Inkling2LoopLayout()
      : QwenHybridLayout{{// Served up to Splash's serving ceiling of 1,010,000 tokens
                          // (serve_options.MAX_YARN_CONTEXT_TOKENS, which every family's served
                          // context stays within), below its own window (modelContextTokens) until
                          // that ceiling is raised; it has no RoPE to stretch.
                          .nativeContextTokens = 1'010'000,
                          // Its applications (applications()).
                          .layers = 48,
                          .hiddenSize = 2304,
                          // The head's and the token table's rows, 64 x 3141 (its config's
                          // vocab_size); its tokens are the first vocabularyTokens.
                          .vocabularySize = 201'024,
                          // Its packed projection of a q input: queries | gate | relative
                          // scores | sinks, to whole 64-column tiles (queryProjectionWidth).
                          .packedFullWidth = 6592,
                          .attentionWidth = 3072,
                          .attentionQueryHeads = 24,
                          .attentionKvHeads = 8,
                          .attentionHeadDimension = 128,
                          .float32NormsOnly = true,
                          // It has no attention q and k norms, and its weights load from GGUFs,
                          // whose norms are their multipliers as stored: nothing reads it.
                          .attentionNormMean = 1.0F,
                          .slidingWindow = 1024,
                          .fullAttentionPeriod = 1,
                          // Its vocabulary keeps no mask token: its DFlash2 draft reads its last
                          // unused special token, <|unused_200056|>, as its mask.
                          .maskToken = 200'056,
                          .stopTokens = {200'006},
                          // Its logits softcapped at 30 (its config's final_logit_softcapping),
                          // over its first 200,058 rows, its tokens (unpadded_vocab_size).
                          .logitSoftcap = 30.0F,
                          .vocabularyTokens = 200'058,
                          // Its drafts read the rows passed into an application, its block
                          // residual's partial sum before it (not the mixes its norms read), and
                          // its final norm at id 48, as its producer captured them; their token
                          // rows take the table's norm (token_embd_norm), as its first
                          // application's do.
                          .draftReadsLayerInputs = true,
                          .draftEmbeddingNorm = true,
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 128,
                          .expertsPerToken = 8,
                          .expertIntermediateSize = 1152,
                          .moeRouter = ops::MoeRouter::SigmoidNormalized,
                          // No shared expert; the selection adds each expert's bias
                          // (exp_probs_b) and scales the weights by 8 (route_scale); the
                          // experts' gate and up softcapped at 4 and 25.
                          .sharedExpertSlices = 0,
                          .sharedExpertGate = false,
                          .expertSelectionBias = true,
                          .expertRoutedScale = 8.0F,
                          .expertGateSoftcap = 4,
                          .expertUpSoftcap = 25},
                         "Inkling-2-Mini"} {
    // An application slides as its weight layer does; none rotates.
    slidingLayers = schedule().slidingApplications();
    unrotatedLayers = (uint64_t{1} << layers) - 1;
  }

  // Its loop's shape and schedule (the order of its applications and what
  // each reads), from its values.
  [[nodiscard]] constexpr Inkling2LoopShape loopShape() const noexcept {
    return {.weightLayers = weightLayers,
            .loopFirstLayer = loopFirstLayer,
            .loopEndLayer = loopEndLayer,
            .loopPasses = loopPasses,
            .blockResidualPeriod = blockResidualPeriod,
            .slidingWeightLayers = slidingWeightLayers,
            .hiddenSize = hiddenSize,
            .kvWidth = kvWidth(),
            .convolutionTaps = convolutionTaps,
            .relativeRank = relativeRank,
            .gateRank = loopGateRank};
  }
  [[nodiscard]] constexpr Inkling2LoopSchedule schedule() const noexcept { return Inkling2LoopSchedule(loopShape()); }

  // What a weight layer's image holds: whether it slides, and the block mix
  // queries before its mixer: none before layer 0, whose input is the
  // residual itself; three before a full layer, of its q, k and v inputs;
  // one before a sliding layer, whose inputs are one row.
  [[nodiscard]] constexpr bool slides(uint32_t layer) const noexcept { return slidingWeightLayers >> layer & 1; }
  [[nodiscard]] constexpr uint32_t mixerQueries(uint32_t layer) const noexcept {
    return layer == 0 ? 0 : slides(layer) ? 1 : 3;
  }

  // The widths of its attention's projections: the relative scores', a
  // score of relativeRank values per query head; the packed projection of a
  // q input, queries | gate | relative scores | a sink logit per query head
  // (6552 columns), its rows to whole 64-column tiles, as every projection's
  // are (ops::Linear); the keys' or the values' of every KV head; and a full
  // layer's taps of its key or value convolution, a row per KV head.
  [[nodiscard]] constexpr uint32_t relativeWidth() const noexcept { return attentionQueryHeads * relativeRank; }
  [[nodiscard]] constexpr uint32_t querySegmentsWidth() const noexcept {
    return 2 * attentionWidth + relativeWidth() + attentionQueryHeads;
  }
  [[nodiscard]] constexpr uint32_t queryProjectionWidth() const noexcept {
    return (querySegmentsWidth() + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
  }
  [[nodiscard]] constexpr uint32_t kvWidth() const noexcept { return attentionKvHeads * attentionHeadDimension; }
  [[nodiscard]] constexpr uint32_t convolutionTapsWidth() const noexcept {
    return attentionKvHeads * convolutionTaps;
  }
  // No GDN layers: its state is its convolutions' input rows alone
  // (Inkling2LoopSchedule::stateBytes, bf16 as their producers write them),
  // the target's auxiliary state (GdnStateLayout::auxiliaryBytes).
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    GdnStateLayout layout = QwenHybridLayout::gdnStateLayout();
    layout.auxiliaryBytes = schedule().stateBytes();
    return layout;
  }
  bool operator==(const Inkling2LoopLayout &) const = default;
};

// Inkling-2-Mini's schedule, checked at compile time against the source's
// forward (modeling_inkling2_loop.py Inkling2LoopTextModel.forward): 48
// applications, the shared dimensions' layers, every one attending; a pass
// restarts at 18 and 30 from plane 1; the pass before's layer-12 state and
// output are mixed into messages. Its KV is 48 layers, 40 windowed over the
// 1023 keys before a row's; its state 1.9 MB a lane of convolution rows. Its
// attention's and convolutions' values are their kernels' (matchesKernels).
static_assert([] {
  constexpr Inkling2LoopLayout layout;
  constexpr Inkling2LoopSchedule s = layout.schedule();
  using K = Inkling2LoopRowsKind;
  const kv::Layout kv = layout.kvLayout();
  return layout.matchesKernels() && s.valid() && s.applications() == 48 && layout.layers == s.applications() &&
         layout.attentionLayerCount() == layout.layers && layout.gdnLayerCount() == 0 &&
         layout.attentionWidth == layout.attentionQueryHeads * layout.attentionHeadDimension &&
         layout.packedFullWidth == layout.queryProjectionWidth() && layout.querySegmentsWidth() == 6552 &&
         layout.vocabularyTokens <= layout.vocabularySize && s.weightLayer(17) == 17 && s.weightLayer(18) == 6 &&
         s.weightLayer(41) == 17 && s.weightLayer(42) == 18 && s.weightLayer(47) == 23 && s.pass(30) == 2 &&
         s.pass(42) == 0 && s.restarts(18) && s.restarts(30) && !s.restarts(6) && s.gated(29) && !s.gated(17) &&
         !s.gated(42) && s.full(5) && s.full(29) && !s.full(28) && s.boundary(24) && !s.boundary(25) &&
         s.input(0) == Inkling2LoopRows{K::Stack, 0} && s.output(5) == Inkling2LoopRows{K::Stack, 1} &&
         s.input(18) == Inkling2LoopRows{K::Stack, 1} && s.output(17) == Inkling2LoopRows{K::Previous, 1} &&
         s.output(29) == Inkling2LoopRows{K::Previous, 1} && s.output(41) == Inkling2LoopRows{K::Stack, 3} &&
         s.output(23) == Inkling2LoopRows{K::Stack, 2} && s.output(47) == Inkling2LoopRows{K::Hidden, 0} &&
         s.mixerStates(0) == 0 && s.mixerStates(6) == 1 && s.ffnStates(6) == 2 && s.mixerStates(47) == 4 &&
         s.previousPlanes() == 2 && s.firstPassPlane() == 2 && s.stateColumns() == 48 * 6656 &&
         s.stateBytes() == 1'916'928 && s.slidingApplications() == 0x7DF7DF7DF7DFull &&
         kv.attentionLayers == 48 && kv.windowedLayers == 0x7DF7DF7DF7DFull && kv.windowTokens == 1023 &&
         layout.gdnStateLayout().auxiliaryBytes == 1'916'928;
}(), "Inkling-2-Mini's schedule follows its source's forward");

// A loop layer's message gate of one input (its mixer's q, k or v input,
// or its FFN's), which the passes after the first apply: down
// [loopGateRank, 2 hiddenSize] of the unweighted norms of the input and of
// the message side by side, forget and gain [hiddenSize, loopGateRank] of
// that; F32 or BF16 rows as stored (a quantized gate's F32 values), which the
// float projection runs in fp32 into fp32 (FloatOutput::Float32).
struct Inkling2LoopGateWeights final {
  ops::Projection down;
  ops::Projection forget;
  ops::Projection gain;
};

// A loop layer's messages: the four queries [4][hiddenSize] F32 that mix the
// pass before's two states into the messages of its q, k, v and FFN inputs,
// and those inputs' gates in that order.
struct Inkling2LoopMessageWeights final {
  metal::MetalBuffer queries;
  std::array<Inkling2LoopGateWeights, 4> gates;
};

// A sliding layer's one norm of its mixer's input (attn_norm), which its q,
// k and v inputs each take (one row but where messages gate them apart).
struct Inkling2LoopSlidingInputs final {
  ops::NormWeights norm;
};
// A full layer's norms of its q, k and v inputs, and the projections of its
// k and v inputs that add to its key and value convolutions' taps
// [convolutionTapsWidth, hiddenSize] (a row of taps per KV head): one segment
// of BF16 or F32 rows as stored, or of a quantized tensor with zero rows to a
// whole tile; the projection is a whole tile wide (64 columns), those past
// its 32 rows padding, and writes bf16.
struct Inkling2LoopFullInputs final {
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::NormWeights valueNorm;
  ops::Projection keyTaps;
  ops::Projection valueTaps;
};

// A weight layer's attention. inputProjection is the packed projection of
// its q input, bf16, its segments in this order: queries (3072), gate (3072),
// relative scores (384), the three quantized ones, which the fused decode
// runs in one dispatch, then a sink logit per query head (24, F32 or BF16 as
// a loop layer's gates, a float segment run apart): querySegmentsWidth()
// columns of queryProjectionWidth() (6592), those past them padding. key and
// value project its k and v inputs (kvWidth each). F32 vectors as the GGUF
// holds them: queryScale [heads][headDimension] and relativeScale
// [heads][relativeRank], the learned scales before the clips; relativeBias
// [relativeExtent][relativeRank], a distance's row; the static taps of the
// key and value convolutions [kvWidth][taps] and of the attention output's
// [hiddenSize][taps], a channel's taps adjacent in the order the GGUF holds
// them. outputProjection takes the gated attention rows.
struct Inkling2LoopAttentionWeights final {
  ops::Projection inputProjection;
  ops::Projection key;
  ops::Projection value;
  metal::MetalBuffer queryScale;
  metal::MetalBuffer relativeScale;
  metal::MetalBuffer relativeBias;
  metal::MetalBuffer keyConvolution;
  metal::MetalBuffer valueConvolution;
  ops::Projection outputProjection;
  metal::MetalBuffer outputConvolution;
};

// A weight layer, which each of its applications runs: the block mix queries
// of its mixer's input ([mixerQueries(layer)][hiddenSize] F32, none in layer
// 0), its input norms by kind, its attention, the block mix query of its
// FFN's input [hiddenSize] F32, the norm before its FFN, its MoE block (no
// shared expert; its experts' NVFP4 scales are in their rows, so every pass
// reads the same weights; its router's bias [experts] F32, which the
// selection adds, is its selectionBias), the FFN output's convolution taps
// [hiddenSize][taps] F32, and a loop layer's messages.
struct Inkling2LoopLayerWeights final {
  metal::MetalBuffer mixerQueries;
  std::variant<Inkling2LoopSlidingInputs, Inkling2LoopFullInputs> inputs;
  Inkling2LoopAttentionWeights mixer;
  metal::MetalBuffer ffnQuery;
  ops::NormWeights ffnNorm;
  ops::BlockMoeWeights ffn;
  metal::MetalBuffer ffnConvolution;
  std::optional<Inkling2LoopMessageWeights> messages;
};

// The target's weights: its weightLayers layers, the norm of the token
// table's gathered rows (token_embd_norm), and the head's block mix query
// [hiddenSize] F32 and final norm (output_norm) before logitsProjection,
// whose rows past vocabularyTokens are padding.
struct Inkling2LoopWeights final : QwenTargetWeightsBase {
  Inkling2LoopLayout layout;
  std::vector<Inkling2LoopLayerWeights> layers;
  ops::NormWeights embeddingNorm;
  metal::MetalBuffer outputQuery;
  ops::NormWeights finalNorm;
};

// Throws unless the layout's values agree with each other, with the shared
// dimensions it runs on and with its kernels' constants (matchesKernels): its
// own check, as requireQwenLayout holds a Qwen layout, whose layers are its
// weight layers and whose attention rotates.
void requireInkling2Layout(const Inkling2LoopLayout &layout);

// Its weights from the images of its GGUF or its checkpoint
// (model/Inkling2LoopSections.hpp): a layer image a weight layer, then the
// head's and the embedding's.
[[nodiscard]] Inkling2LoopWeights loadInkling2LoopWeights(metal::MetalBackend &backend, Inkling2LoopLayout layout,
                                                          const QwenTargetFiles &files);

// How its GGUFs plan: its weightLayers layer images, then the head's and the
// embedding's, its metadata checked key by key (model/Inkling2LoopGguf.cpp).
[[nodiscard]] GgufTargetPlan ggufTargetPlan(const Inkling2LoopLayout &layout);
// How its safetensors checkpoints plan, its release's and its MLX
// conversions': the images its GGUFs' plan lays out, of the tensors each
// names for them (model/Inkling2LoopSafetensors.cpp).
[[nodiscard]] SafetensorsTargetPlan safetensorsTargetPlan(const Inkling2LoopLayout &layout);

// The projections its layers dispatch beyond their mixers' fused queries and
// outputs, which qwenTargetGeometry takes as any mixer's (includeMixer):
// keys, values, a full layer's taps and a loop layer's gates.
void includeFamilyProjections(QwenTargetGeometry &geometry, const Inkling2LoopWeights &weights);

// The family's target (model/Inkling2LoopTarget.hpp), whose prefill commands
// run up to `prefillChunks` chunks, and the step scratch a target of its
// layout keeps itself: its block states, the states before the loop, the
// pass before's states and its messages.
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const Inkling2LoopWeights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const Inkling2LoopLayout &layout,
                                                            const QwenTargetGeometry &geometry,
                                                            uint32_t prefillChunks, uint64_t tableRowBytes);
// The workspace its own attention takes, a prompt chunk's sequence's or a
// decode lane's (StepKind::Prefill, Decode), sized by its operator
// (model/Inkling2LoopTarget.cpp) rather than the shared attention's.
[[nodiscard]] ops::AttentionWorkspace qwenTargetAttentionWorkspace(const Inkling2LoopLayout &layout,
                                                                   const ops::ExecutionPlans &operators,
                                                                   const QwenTargetGeometry &geometry,
                                                                   ops::StepKind step);
// It has no table to stream.
[[nodiscard]] SparseRowStoreOptions streamedTableRows(const Inkling2LoopLayout &layout);

} // namespace splash::model
