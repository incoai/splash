#pragma once

#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace splash::flash {

inline constexpr uint32_t kFlashPLEInvalidIndex = 1;
inline constexpr uint32_t kFlashPLEInvalidParameters = 2;
inline constexpr uint32_t kFlashPLEInvalidNumeric = 4;
inline constexpr uint32_t kFlashPLEHeads = 16;
inline constexpr uint32_t kFlashPLEHeadWidth = 160;

struct FlashPLEGeometry final {
  uint32_t lanes = 1;
  uint32_t rows = 1; // Per lane, in temporal order; never padded logical rows.
  uint32_t width = 2560;
  uint32_t streams = 4;
  uint32_t eosToken = 248044;
  uint32_t vocabularySize = 248320;
  float epsilon = 1e-6f;
};

// Copies retain the original aligned source allocations. PLE rows use Q4/G32,
// not Splash's StorageN layout, and no table or coefficient is transformed.
struct FlashPLEShard final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint64_t rows = 0;
  uint64_t weightRowStrideBytes = 0;
  uint64_t parameterRowStrideBytes = 0;
};

struct FlashPLEWeights final {
  FlashTensor multipliers; // I64[3], from the checkpoint, never regenerated.
  FlashTensor headVocabularySizes; // I64[16].
  FlashTensor headOffsets; // I64[16].
  FlashTensor sharedScale; // BF16[1], applied after affine BF16 conversion.
  FlashTensor normKey;
  FlashTensor normQuery;
  FlashTensor normConvolution;
  FlashTensor convolution; // BF16[streams*width,4,1], dilation 3.
  NormConvention normConvention = NormConvention::OnePlusWeight;
  std::vector<FlashPLEShard> shards;
  // Optional disk-only source route. Exactly one of shards/diskTableRows is
  // populated; hash metadata and all post-PLE tensors remain resident.
  uint64_t diskTableRows = 0;

  [[nodiscard]] uint64_t tableRows() const;
  // The inspected checkpoint route requires 128 unchanged Q4/G32 shards.
  [[nodiscard]] static FlashPLEWeights fromWeights(
      const FlashWeights &weights,
      std::string_view prefix = "language_model.model.layers.1.ple");
};

// tokenIDs is I64[lanes,rows], history is I64[lanes,2] in chronological order.
// Initialize a cold history to two eosToken values. Current EOS hashes its
// previous context; the following token starts a new segment. History is
// updated by a separate ordered dispatch after all hash reads have completed.
// ngramIDs is I64[lanes,rows,16], bigram heads then trigram heads.
void addPLENgramIDs(metal::CommandGraph &graph,
                   const FlashPLEWeights &weights,
                   metal::MetalBuffer tokenIDs,
                   metal::MetalBuffer tokenHistory,
                   metal::MetalBuffer ngramIDs,
                   metal::MetalBuffer diagnostics, FlashPLEGeometry geometry);

// Sparse GPU gather of 16 original source rows per token. Eight source shards
// share a dispatch; nonmatching shards never access table payloads. Output is
// BF16[lanes,rows,2560]. Affine reconstruction rounds to BF16 before multiplying
// the checkpoint's shared BF16 weight_scale. The caller clears and checks the
// sticky diagnostics uint32. Invalid IDs produce NaN, never a silent zero.
void addPLEGather(metal::CommandGraph &graph, const FlashPLEWeights &weights,
                  metal::MetalBuffer ngramIDs, metal::MetalBuffer output,
                  metal::MetalBuffer diagnostics, FlashPLEGeometry geometry);

// Post-projection scratch, each BF16[lanes,rows,streams*width]. All four views
// must be disjoint. Key/value affine projections run through FlashAffine.
struct FlashPLEPostScratch final {
  metal::MetalBuffer normalizedKeys;
  metal::MetalBuffer normalizedQueries;
  metal::MetalBuffer gatedValues;
  metal::MetalBuffer normalizedConvolution;
};

// Frozen per-process opt-in. Missing/0 preserves the original route; only the
// literal value1 selects post fusion, and malformed values reject startup.
[[nodiscard]] bool flashPLEPostFusedEnabled();
[[nodiscard]] const char *flashPLEPostRouteSemantics();

// keyProjected/hyperInput are BF16[lanes,rows,streams*width]; valueProjected is
// BF16[lanes,rows,width]. convState is BF16[lanes,9,streams*width], cold-zeroed.
// The state stores masked normed gated values, not convolution output. It is
// updated after convolution in a separate dispatch and remains batch-safe.
// Optional mask is U32[lanes,rows] (zero false); an empty buffer means no mask.
// pleOutput = BF16(gatedValues + BF16(silu(BF16(depthwiseConv(normed))))).
// This operation does not add hyperInput; addPLEInject supplies that boundary.
void addPLEPostProject(metal::CommandGraph &graph,
                       const FlashPLEWeights &weights,
                       metal::MetalBuffer hyperInput,
                       metal::MetalBuffer keyProjected,
                       metal::MetalBuffer valueProjected,
                       const FlashPLEPostScratch &scratch,
                       metal::MetalBuffer convState,
                       metal::MetalBuffer pleOutput,
                       metal::MetalBuffer diagnostics,
                       FlashPLEGeometry geometry,
                       metal::MetalBuffer mask = {});

// BF16 addition before the layer's attention hyper connection. output may be
// the exact hyperInput view, but may not equal pleOutput.
void addPLEInject(metal::CommandGraph &graph, metal::MetalBuffer hyperInput,
                  metal::MetalBuffer pleOutput, metal::MetalBuffer output,
                  FlashPLEGeometry geometry);

// Combined production entry point. The arguments match addPLEPostProject;
// the resulting hyper state overwrites the exact hyperInput view after the
// same BF16 injection boundary. Both routes retain pleOutput and all scratch
// arrays, including normalizedConvolution for speculative-prefix restoration.
void addPLEPostProjectAndInject(
    metal::CommandGraph &graph, const FlashPLEWeights &weights,
    metal::MetalBuffer hyperInput, metal::MetalBuffer keyProjected,
    metal::MetalBuffer valueProjected, const FlashPLEPostScratch &scratch,
    metal::MetalBuffer convolutionState, metal::MetalBuffer pleOutput,
    metal::MetalBuffer diagnostics, FlashPLEGeometry geometry,
    metal::MetalBuffer mask = {});

// Rebuild state from a retained pre-command snapshot after speculative verify.
// keptTokens is U32[lanes], explicitly the number of incoming tokens retained
// per lane (0..geometry.rows); the runtime maps its accepted+1 convention here.
// normalizedInputs is the verify command's BF16 normalizedConvolution scratch.
// Both outputs must be disjoint from the snapshots, which remain unchanged.
void addPLERestorePrefix(metal::CommandGraph &graph,
                         metal::MetalBuffer beforeTokenHistory,
                         metal::MetalBuffer inputTokenIDs,
                         metal::MetalBuffer beforeConvolutionState,
                         metal::MetalBuffer normalizedInputs,
                         metal::MetalBuffer keptTokens,
                         metal::MetalBuffer outputTokenHistory,
                         metal::MetalBuffer outputConvolutionState,
                         metal::MetalBuffer diagnostics,
                         FlashPLEGeometry geometry);

// Host-only reference: exact I64 wraparound multiplication, XOR and positive
// modulo, using checkpoint arrays. This neither maps nor materializes tables.
// Returned IDs match [lanes,rows,16]; histories are updated in place only after
// input validation and all IDs have been formed.
[[nodiscard]] std::vector<int64_t> computePLENgramIDs(
    std::span<const int64_t> tokenIDs, std::span<int64_t> tokenHistory,
    std::span<const int64_t> multipliers,
    std::span<const int64_t> headVocabularySizes,
    std::span<const int64_t> headOffsets, FlashPLEGeometry geometry,
    uint64_t tableRows);

} // namespace splash::flash
