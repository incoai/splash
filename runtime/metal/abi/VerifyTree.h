#pragma once

// The rows of a decode lane's verify block as a tree, shared by host dispatch
// code and Metal kernels. One entry per row of a command's packed rows (lane
// l's rows from its first row, ops::LaneRows): row 0 of a block holds the
// lane's anchor, and every other row a proposal that follows its parent row.
// Rows are in topological order (a parent comes before its children), so the
// set bits of a row's ancestors, ascending, are its path from the anchor. A
// chain is the tree whose row r follows row r - 1.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The anchor's parent.
#define SPLASH_TREE_ROOT 0xffffu

// ancestors: bit j for each row j of the block on the row's path from the
// anchor, the row itself included; parent: its parent row in the block;
// depth: its steps from the anchor, so its position is the block's first
// position plus its depth.
struct SplashTreeRow {
  uint32_t ancestors;
  uint16_t parent;
  uint16_t depth;
};

static_assert(sizeof(SplashTreeRow) == 8,
              "Verify tree rows are 8 bytes on both sides");

// A row's K/V and its GDN tape row sit at its own row of the block (the
// block's first position plus the row), so that siblings never collide; its
// RoPE position is the block's first position plus its depth. A block's
// accepted path is a row mask of the same form as `ancestors`: the ancestors
// of the deepest row it accepted.

// A lane's rows of a command's packed rows (ops::LaneRows): `rows` rows from
// row `first`, as a step's proposer writes them on the device for the
// kernels after it.
struct SplashLaneRows {
  uint32_t first;
  uint32_t rows;
};

static_assert(sizeof(SplashLaneRows) == 8,
              "Lane rows are 8 bytes on both sides");
