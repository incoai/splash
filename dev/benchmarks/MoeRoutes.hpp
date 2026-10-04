#pragma once

#include "metal/MetalBackend.hpp"
#include "metal/abi/MoE.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace splash::benchmark {

// The routes of one MoE layer call, read after its moe_group_routes
// dispatch: rows, routed experts per row and in the router, tile rows, and
// the distinct routed experts and live tiles (routed plus shared) it grouped.
struct MoeRoutes final {
  ::MoeGroupParams params{};
  uint32_t experts = 0;
  uint32_t tiles = 0;
};

inline const metal::MetalBuffer *
boundBuffer(const metal::ComputeDispatch &dispatch, uint32_t index) {
  for (const metal::BufferBinding &binding : dispatch.buffers)
    if (binding.index == index) return &binding.buffer;
  return nullptr;
}

// Reads the shared-storage scratch of a moe_group_routes dispatch (bindings
// of kernels/shared/moe.metal): selected expert ids at 0, the tile count at 2
// and MoeGroupParams at 5. Other dispatches, or unreadable bindings, are
// skipped.
inline void recordRoutes(const metal::ComputeDispatch &dispatch,
                         std::vector<MoeRoutes> &routes) {
  if (dispatch.pipelineName != "moe_group_routes") return;
  const metal::MetalBuffer *selected = boundBuffer(dispatch, 0);
  const metal::MetalBuffer *tileCount = boundBuffer(dispatch, 2);
  const auto params = std::find_if(
      dispatch.bytes.begin(), dispatch.bytes.end(),
      [](const metal::BytesBinding &binding) { return binding.index == 5; });
  if (!selected || !tileCount || !selected->contents() || !tileCount->contents() ||
      params == dispatch.bytes.end() || params->sizeBytes != sizeof(::MoeGroupParams))
    return;
  MoeRoutes entry;
  std::memcpy(&entry.params, params->data, sizeof(entry.params));
  const uint32_t perRow = entry.params.top_k + 1;
  const auto *ids = static_cast<const uint32_t *>(selected->contents());
  std::vector<bool> seen(entry.params.experts, false);
  for (uint64_t route = 0; route < uint64_t{entry.params.rows} * perRow; ++route) {
    if (route % perRow == entry.params.top_k || ids[route] >= seen.size() ||
        seen[ids[route]])
      continue;
    seen[ids[route]] = true;
    ++entry.experts;
  }
  entry.tiles = *static_cast<const uint32_t *>(tileCount->contents());
  routes.push_back(entry);
}

} // namespace splash::benchmark
