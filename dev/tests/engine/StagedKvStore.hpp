#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/LaneBindings.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::test {

// Stores K/V a test staged itself, keys [KV head][row][dimension] and values
// [KV head][dimension][row] at each sequence's staging rows (each sequence's
// kvHeads x chunk_stride rows after the one before it), into the pages of
// `layer` through each sequence's page table: the test libraries' staged
// store (attention_store_test.metal) over the production store routine, for
// attention tests whose K/V are no q/k projection's. A row past its
// sequence's rows (its entry of `rows`' layout) is not stored.
inline void addStagedStore(metal::CommandGraph &graph, SplashKvLayer layer, const metal::MetalBuffer &keys,
                           const metal::MetalBuffer &values, std::span<const metal::MetalBuffer> pageTables,
                           std::span<const kv::ChunkedPrefillParams> sequences, const ops::AttentionRows &rows,
                           kv::Layout layout) {
  if (sequences.empty() || sequences.size() != pageTables.size())
    throw std::invalid_argument("a staged store takes a page table per sequence");
  // The kernel's suffix of the layout's KV heads, head dimension and value
  // order, as the path commit's (ops::PagedAttention).
  const std::string suffix = layout.valuesTokenMajor        ? "_kv2_vt"
                             : layout.headDimension == 128 ? (layout.kvHeads == 2 ? "_kv2_d128" : "_kv4_d128")
                             : layout.kvHeads == 2         ? "_kv2_g8"
                                                           : "";
  kv::AttentionParams params{.sequences = static_cast<uint32_t>(sequences.size()),
                             .first_sequence = 0,
                             .rows = rows.rows,
                             .tiles = 0,
                             .trees = 0,
                             .last_rows = 0,
                             .kv = layer,
                             .sequence = {}};
  uint32_t stageRow = 0, mostRows = 0;
  for (size_t s = 0; s < sequences.size(); ++s) {
    const kv::ChunkedPrefillParams &chunk = sequences[s];
    params.sequence[s] = {.committed_tokens = chunk.committed_tokens,
                          .row_bound = chunk.chunk_tokens,
                          .layout_row = 0,
                          .stage_row = stageRow,
                          .chunk_stride = chunk.chunk_stride,
                          .page_table_entries = chunk.page_table_entries,
                          .split_count = 0,
                          .first_tile = 0,
                          .first_slot = 0,
                          .block_stride = 0,
                          .block_list = 0};
    stageRow += layout.kvHeads * chunk.chunk_stride;
    mostRows = std::max(mostRows, chunk.chunk_tokens);
  }
  std::vector<metal::MetalBuffer> bindings{rows.layout, keys, values};
  const auto tables = ops::laneBindings(pageTables);
  bindings.insert(bindings.end(), tables.begin(), tables.end());
  graph.add("test_staged_kv_store_" + std::string(layout.format == kv::Format::BFloat16 ? "bf16" : "q8") + suffix,
            std::move(bindings), params, {layout.kvHeads, mostRows, sequences.size()}, {layout.headDimension, 1, 1});
}

} // namespace splash::test
