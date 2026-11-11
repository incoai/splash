#pragma once

#include "model/RopeScaling.hpp"
#include "model/StateLayout.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace splash::model {

// Canonical records of what a model's cached copies hold. The engine adds
// the model identity and the cache file format, including a page slot's
// storage token, before hashing them into a persistent cache namespace.
[[nodiscard]] inline std::string persistentKvLayoutRecord(const kv::Layout &layout) {
  std::ostringstream record;
  record << "kv " << kv::kPageTokens << ' ' << layout.attentionLayers << ' ' << layout.kvHeads << ' '
         << layout.headDimension << ' ' << kv::formatName(layout.format)
         // Dimension-major values add nothing, so dense layouts keep the
         // caches they wrote.
         << (layout.valuesTokenMajor ? " values-token-major" : "");
  // A block index's pooled keys, bf16, one a block of tokens. A change to
  // how a block's key is pooled, normalized or rotated changes this line.
  if (layout.indexDimension)
    record << " index " << layout.indexDimension << " bf16 block " << SPLASH_ATTENTION_BLOCK_TOKENS;
  // A sliding window also orders a page's regions in its slot and labels
  // the slot with its page's form. Without one, keep the prior record.
  if (layout.windowedLayers)
    record << " window " << layout.windowTokens << " layers 0x" << std::hex << layout.windowedLayers << std::dec;
  return record.str();
}

[[nodiscard]] inline std::string persistentStateLayoutRecord(const CompositeStateLayout &states,
                                                            uint32_t draftBlockSize,
                                                            const RopeScaling &rope) {
  const GdnStateLayout &gdn = states.target;
  const DraftStateLayout &draft = states.draft;
  std::ostringstream record;
  record << "gdn " << gdn.layers << ' ' << gdn.convolutionHistory << ' '
         << gdn.convolutionChannels << ' ' << gdn.recurrentGroups << ' ' << gdn.recurrentRows
         << ' ' << gdn.recurrentColumns
         // fp32 states keep their prior record; without GDN layers the
         // format stores nothing.
         << (gdn.layers && gdn.format == ops::GdnStateFormat::BFloat16 ? " bf16" : "");
  // State a family keeps after its GDN layers, such as PLE history and QSA
  // pending keys.
  if (gdn.auxiliaryBytes) record << " auxiliary " << gdn.auxiliaryBytes;
  record << '\n'
         << "draft " << draft.layers << ' ' << draft.kvHeads << ' ' << draft.headDimension << ' '
         << draft.windowTokens << '\n'
         << "context " << draft.contextWidth << " q4 " << SPLASH_DRAFT_CONTEXT_GROUP << '\n';
  if (draftBlockSize != 8) record << "draft-block " << draftBlockSize << '\n';
  // Without YaRN omit the line, keeping earlier caches valid. Preserve all
  // digits that round-trip a factor so distinct scalings cannot share it.
  if (rope.yarn())
    record << "rope yarn " << std::setprecision(std::numeric_limits<double>::max_digits10) << rope.factor
           << ' ' << rope.originalContextTokens << '\n';
  return record.str();
}

} // namespace splash::model
