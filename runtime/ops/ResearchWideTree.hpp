#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ResearchTree.h"
#include "metal/abi/PagedAttention.h"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace splash::ops {

// The wide tree verify of a dense target's single-request decode cycle
// (--verify-tree): sixteen rows in one step, the draft's chain plus the
// likeliest alternatives its selector ranks, kept in a ResearchTreeWideState
// the kernels below read. The target's own selections at the tree's nodes (the
// argmax, or a draw from the node's distribution with its own uniform) decide
// which children a cycle follows, and a commit gathers the rows, K/V, and
// recurrent tape of the followed path into the eight rows the chain verify
// leaves.
class ResearchWideTree final {
public:
  // Chooses the tree's nodes from the selector's candidates, edge scores and
  // chain proposals, and writes the proposals of every node.
  static void addSelect(metal::CommandGraph &graph, metal::MetalBuffer candidates, metal::MetalBuffer unary,
                        metal::MetalBuffer partials, metal::MetalBuffer proposals, metal::MetalBuffer tree);
  // Gives each of the sixteen rows the rotary angles of its depth.
  static void addRope(metal::CommandGraph &graph, metal::MetalBuffer cosine, metal::MetalBuffer sine,
                      metal::MetalBuffer tree);
  // The verify input rows: the anchor, then every node's proposal.
  static void addInput(metal::CommandGraph &graph, metal::MetalBuffer anchor, metal::MetalBuffer proposals,
                       metal::MetalBuffer input, metal::MetalBuffer verifyRows, uint32_t vocabulary);
  // Walks the tree through the children that match the target's selections
  // in `output`, and writes the retained tokens over them.
  static void addAccept(metal::CommandGraph &graph, metal::MetalBuffer proposals, metal::MetalBuffer output,
                        metal::MetalBuffer retained, metal::MetalBuffer accepted, metal::MetalBuffer tree,
                        uint32_t remaining, const std::array<uint32_t, 4> &stop);
  // Gathers the followed path's recurrent tape and captured hidden rows into
  // their chain positions.
  static void addPathCommit(metal::CommandGraph &graph, metal::MetalBuffer tree, metal::MetalBuffer tapeScratch,
                            metal::MetalBuffer tape, const ResearchTreeTapeGather &tapeLayout,
                            metal::MetalBuffer capturedHidden, uint32_t capturedWidth);
  // Gathers one attention layer's K/V rows of the followed path and stores
  // them in the request's pages.
  static void addKvCommit(metal::CommandGraph &graph, metal::MetalBuffer tree, metal::MetalBuffer keys,
                          metal::MetalBuffer values, metal::MetalBuffer pages, const ::SplashChunkedPrefillParams &store);

  // Whether every decode start a request of `promptTokens` and a budget of
  // `maximumNewTokens` can reach splits the verify attention's pages the same
  // way, so that a tree cycle and a chain cycle read the same numeric history.
  [[nodiscard]] static bool stablePagePartition(uint32_t promptTokens, uint32_t maximumNewTokens) noexcept {
    if (!maximumNewTokens)
      return false;
    const uint64_t end = uint64_t{promptTokens} + maximumNewTokens - 1;
    if (end + kv::kVerifyRows > std::numeric_limits<uint32_t>::max())
      return false;
    const uint64_t first = (uint64_t{promptTokens} + kv::kVerifyRows + kv::kPageTokens - 1) / kv::kPageTokens;
    const uint64_t last = (end + kv::kVerifyRows + kv::kPageTokens - 1) / kv::kPageTokens;
    const auto width = [](uint64_t pages) {
      const uint64_t splits = std::clamp((pages + kv::kVerifyPagesPerSplit - 1) / kv::kVerifyPagesPerSplit,
                                         uint64_t{kv::kVerifySplits}, uint64_t{kv::kVerifyMaximumSplits});
      return (pages + splits - 1) / splits;
    };
    const uint64_t reference = width(first);
    for (uint64_t pages = first; pages <= last; ++pages)
      if (width(pages) != reference)
        return false;
    return true;
  }
};

} // namespace splash::ops
