#include "ops/ResearchWideTree.hpp"

#include <algorithm>
#include <array>

namespace splash::ops {

using metal::CommandGraph;
using metal::MetalBuffer;

void ResearchWideTree::addSelect(CommandGraph &graph, MetalBuffer candidates, MetalBuffer unary, MetalBuffer partials,
                                 MetalBuffer proposals, MetalBuffer tree) {
  graph.add("research_wide_select", {candidates, unary, partials, proposals, tree}, {1, 1, 1}, {1, 1, 1});
}

void ResearchWideTree::addRope(CommandGraph &graph, MetalBuffer cosine, MetalBuffer sine, MetalBuffer tree) {
  graph.add("research_wide_rope", {cosine, sine, tree}, {1, 1, 1}, {32, 1, 1});
}

void ResearchWideTree::addInput(CommandGraph &graph, MetalBuffer anchor, MetalBuffer proposals, MetalBuffer input,
                                MetalBuffer verifyRows, uint32_t vocabulary) {
  graph.add("research_wide_input", {anchor, proposals, input, verifyRows}, vocabulary, {1, 1, 1}, {32, 1, 1});
}

void ResearchWideTree::addAccept(CommandGraph &graph, MetalBuffer proposals, MetalBuffer output, MetalBuffer retained,
                                 MetalBuffer accepted, MetalBuffer tree, uint32_t remaining,
                                 const std::array<uint32_t, 4> &stop) {
  ResearchTreeAccept params{remaining, {}};
  std::copy(stop.begin(), stop.end(), params.stop);
  graph.add("research_wide_accept", {proposals, output, retained, accepted, tree}, params, {1, 1, 1}, {1, 1, 1});
}

void ResearchWideTree::addPathCommit(CommandGraph &graph, MetalBuffer tree, MetalBuffer tapeScratch, MetalBuffer tape,
                                     const ResearchTreeTapeGather &layout, MetalBuffer capturedHidden,
                                     uint32_t capturedWidth) {
  graph.add("research_wide_tape_gather", {tapeScratch, tape, tree}, layout,
            {(uint64_t(layout.conv + layout.kv + layout.heads) * layout.layers + 255) / 256, 1, 1});
  graph.add("research_wide_rows_gather", {capturedHidden, tree}, capturedWidth, {(capturedWidth + 255) / 256, 1, 1});
}

void ResearchWideTree::addKvCommit(CommandGraph &graph, MetalBuffer tree, MetalBuffer keys, MetalBuffer values,
                                   MetalBuffer pages, const kv::ChunkedPrefillParams &store) {
  graph.add("research_wide_kv_gather", {keys, values, tree}, {4, 1, 1}, {256, 1, 1});
  std::array<kv::ChunkedPrefillParams, 4> stores{};
  stores[0] = store;
  graph.add("verify_attention_q8_store", {keys, values, pages, pages, pages, pages}, stores, {64, 1, 1},
            {256, 1, 1});
}

} // namespace splash::ops
