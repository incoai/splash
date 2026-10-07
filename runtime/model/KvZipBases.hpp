#pragma once

#include "ops/PagedKv.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace splash::model {

// The ZBF16 bases calibrated for a model family's KV (abi/KvZip.h), as
// kv::PageStorage takes them, or none when the family has no calibration or
// its layout differs. dev/tools/kvzip_calibrate.py writes them.
[[nodiscard]] std::span<const uint8_t> kvZipBases(std::string_view family, const kv::Layout &layout);

} // namespace splash::model
