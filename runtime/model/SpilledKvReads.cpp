#include "model/SpilledKvReads.hpp"

#include "model/KvBlockCache.hpp"

#include <stdexcept>
#include <utility>

namespace splash::model {

SpilledKvReads::SpilledKvReads(KvBlockCache &cache, std::vector<std::shared_ptr<const KvPageSources>> lanes,
                               metal::MetalBuffer locations)
    : cache_(&cache), sources_(std::move(lanes)), locations_(std::move(locations)) {}

SpilledKvReads::SpilledKvReads(KvHistoryStream &stream, uint32_t sequence,
                               std::shared_ptr<const KvPageSources> sources, uint32_t pages,
                               metal::MetalBuffer poolTable, std::span<const SplashKvLayer> poolLayers)
    : stream_(&stream), sources_{std::move(sources)}, sequence_(sequence), pages_(pages),
      poolTable_(std::move(poolTable)), poolLayers_(poolLayers) {
  if (!sources_.front()) throw std::invalid_argument("a streamed prefill has no page sources");
}

// What a command dropped unconsumed, or whose encode failed, started ends;
// what it could not read has no one to read it.
SpilledKvReads::~SpilledKvReads() {
  try {
    end();
  } catch (...) {
  }
}

bool SpilledKvReads::active() const noexcept { return cache_ || stream_; }

ops::AttentionBlocks SpilledKvReads::verify(metal::CommandGraph &graph, uint32_t layer,
                                            const ops::AttentionBlocks &blocks) {
  if (!cache_) throw std::logic_error("only a decode command's reads of spilled K/V resolve blocks");
  if (!running_) {
    stats_ = std::make_shared<KvReadStats>();
    running_ = true;
  }
  return cache_->addResolve(graph, layer, sources_, blocks, locations_, stats_);
}

// The sequence's first chunk of the command begins the stream, and its first
// chunk of each layer stages the layer.
PrefillKv SpilledKvReads::prefill(metal::CommandGraph &graph, uint32_t layer, uint32_t sequence,
                                  uint32_t chunkFirstPage, uint32_t chunkPages, const PrefillKv &pool) {
  if (!stream_ || sequence != sequence_) return pool;
  if (!running_) {
    stream_->begin(*sources_.front(), pages_, poolTable_, poolLayers_);
    running_ = true;
  }
  if (stagedLayer_ != layer) {
    staged_ = stream_->addLayer(graph, layer);
    stagedLayer_ = layer;
  }
  stream_->addChunk(graph, layer, chunkFirstPage, chunkPages);
  return staged_;
}

void SpilledKvReads::layerEnd(metal::CommandGraph &graph, uint32_t layer) {
  if (stagedLayer_ == layer) stream_->addLayerEnd(graph, layer);
}

void SpilledKvReads::end() {
  if (!running_) return;
  running_ = false;
  if (cache_) {
    cache_->mergeStats(*stats_);
    const std::vector<KvReadFailure> failed = cache_->takeFailures();
    failures_.insert(failures_.end(), failed.begin(), failed.end());
    return;
  }
  const KvStreamResult result = stream_->end();
  for (const KvReadFailure &failure : result.failures) failures_.push_back({sequence_, failure.block});
  streamMilliseconds_ += result.streamMilliseconds;
  stagedLayer_.reset();
}

} // namespace splash::model
