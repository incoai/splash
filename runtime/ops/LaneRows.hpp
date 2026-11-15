#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/VerifyTree.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace splash::ops {

// A lane's rows among a command's packed rows: `rows` rows from row `first`.
// A decode lane's are its verify block, the anchor's row and its proposals';
// a lane selecting the first token after its prompt has the one row it
// selects from. The batched kernels address a lane's rows by these (their
// parameters' per-lane first_row and rows), so lanes of different row counts
// share a dispatch.
struct LaneRows {
  uint32_t first = 0;
  uint32_t rows = 0;
};

// A row of a decode lane's verify block as a tree (metal/abi/VerifyTree.h),
// one per row of a command's packed rows.
using TreeRow = SplashTreeRow;

// The most rows of a lane at the narrow lane cap (verifyLaneCap), which the
// narrow GDN decode kernels hold: a draft's block, the most a step of
// --verify-rows all gives a lane.
inline constexpr uint32_t kNarrowLaneRows = SPLASH_GDN_NARROW_ROWS;
static_assert(kNarrowLaneRows == SPLASH_DRAFT_QUERY_ROWS && kNarrowLaneRows <= SPLASH_TARGET_VERIFY_ROWS,
              "a draft's block runs on the narrow GDN kernels");

// The most rows of a lane's verify block in a decode step of `rows` rows over
// `lanes` lanes, which the step's proposer chooses on the device within it:
// those of the narrow GDN decode kernels (SPLASH_GDN_NARROW_ROWS) while
// the step's rows give each lane no more, which took 13-19% less time than
// those of sixteen over a layer's 8-row chains (DRAM-cold, an M6); else a
// verify block's most.
[[nodiscard]] constexpr uint32_t verifyLaneCap(uint32_t rows, uint32_t lanes) noexcept {
  return rows <= uint64_t{kNarrowLaneRows} * lanes ? kNarrowLaneRows : SPLASH_TARGET_VERIFY_ROWS;
}

// Lane rows as a proposer writes them on the device for the kernels after it
// (SplashLaneRows).
static_assert(sizeof(LaneRows) == sizeof(SplashLaneRows));

// A decode step's rows, which its proposer lays out on the device for the
// kernels after it (SplashLaneRows, ops::DraftSelector::addVerifyRows):
// `rows` rows over `lanes` lanes, packed in lane order, each lane's 1 to
// `laneCap` of them (verifyLaneCap). A kernel's grid of a lane's rows spans
// the cap.
struct StepRows final {
  uint32_t lanes = 0;
  uint32_t rows = 0;
  uint32_t laneCap = 0;
};

// Throws unless `step` holds 1 to a batch of lanes of 1 to a verify block's
// rows each, its rows as many as its lanes may hold.
inline void requireStepRows(const StepRows &step) {
  if (!step.lanes || step.lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !step.laneCap ||
      step.laneCap > SPLASH_TARGET_VERIFY_ROWS || step.rows < step.lanes ||
      step.rows > uint64_t{step.lanes} * step.laneCap)
    throw std::invalid_argument("invalid decode step rows");
}

// A command's rows as its layout on the device holds them, a SplashLaneRows
// a sequence (its first row and rows among the command's packed rows):
// `sequences` sequences of the command's `rows` rows, each of 1 to `bound`
// of them. A prompt chunk's host writes its sequences' entries, their bound
// its longest sequence's rows; a decode step's selector writes its lanes'
// (StepRows), their bound the cycle's lane cap. A kernel's grid of a
// sequence's rows spans the bound; a row past its sequence's rows, or a
// sequence whose rows run past the command's, writes nothing.
struct LayoutRows final {
  uint32_t sequences = 0;
  uint32_t rows = 0;
  uint32_t bound = 0;
};

// Throws unless `rows` holds 1 to a batch of sequences of 1 to its bound of
// rows each, a bound of at most `limit`, its rows as many as its sequences
// may hold.
inline void requireLayoutRows(const LayoutRows &rows, uint32_t limit = std::numeric_limits<uint32_t>::max()) {
  if (!rows.sequences || rows.sequences > SPLASH_MAXIMUM_BATCH_WIDTH || !rows.bound || rows.bound > limit ||
      rows.rows < rows.sequences || rows.rows > uint64_t{rows.sequences} * rows.bound)
    throw std::invalid_argument("invalid layout rows");
}

} // namespace splash::ops
