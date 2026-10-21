#pragma once

// How a block-sparse target's attention reads K/V whose pages hold only
// their index (kv::PageForm::Index), with two calls in its family code: the
// runtime builds one per command, over the hot block cache (decode) and the
// history stream (prefill) and the command's page sources. A command none of
// whose lanes holds an Index page gets the null one, which changes nothing:
// its graph is the one it encodes without spilling.
//
// The command owns it until it completed, failed or was abandoned: it holds
// the page sources its reads reach (KvPageSources, whose copies stay alive
// with them) and its resolves' counts, which it shares with the host steps
// that read and count, so that a step whose command was given up may still
// run; end() hands what the reads could not read and their statistics back
// on the engine thread.

#include "metal/CommandGraph.hpp"
#include "model/KvHistoryStream.hpp"
#include "model/KvReadStats.hpp"
#include "model/Model.hpp"
#include "ops/PagedAttention.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace splash::model {

class KvBlockCache;

class SpilledKvReads final {
public:
  // The null one.
  SpilledKvReads() = default;
  // A decode command's: `lanes` holds each batch lane's page sources (null
  // for a lane whose pages all hold their K/V), whose Index pages' blocks
  // resolve through `cache`; `locations` takes the lists' located entries.
  SpilledKvReads(KvBlockCache &cache, std::vector<std::shared_ptr<const KvPageSources>> lanes,
                 metal::MetalBuffer locations);
  // A prefill command's: the history of its lane `sequence`, whose pages
  // the command reads up to page `pages` (`sources`, at the pool's
  // `poolLayers` through `poolTable`), streams through `stream`.
  SpilledKvReads(KvHistoryStream &stream, uint32_t sequence, std::shared_ptr<const KvPageSources> sources,
                 uint32_t pages, metal::MetalBuffer poolTable, std::span<const SplashKvLayer> poolLayers);
  // Ends a run end() did not.
  ~SpilledKvReads();
  SpilledKvReads(const SpilledKvReads &) = delete;
  SpilledKvReads &operator=(const SpilledKvReads &) = delete;

  // Some lane of the command holds an Index page.
  [[nodiscard]] bool active() const noexcept;
  // After a layer's block lists, before its attention: the lists with the
  // locations of their entries' blocks (KvBlockCache::addResolve), for the
  // located split. Only an active one, for lists.
  [[nodiscard]] ops::AttentionBlocks verify(metal::CommandGraph &graph, uint32_t layer,
                                            const ops::AttentionBlocks &blocks);
  // After a prefill chunk's store: the K/V its attention reads, staged when
  // the history of `sequence` holds Index pages (KvHistoryStream), else
  // `pool`. The chunk's pages are [chunkFirstPage, chunkFirstPage +
  // chunkPages).
  [[nodiscard]] PrefillKv prefill(metal::CommandGraph &graph, uint32_t layer, uint32_t sequence,
                                  uint32_t chunkFirstPage, uint32_t chunkPages, const PrefillKv &pool);
  // After a layer's attention in every chunk of the command: the hand-off
  // of a stream with one buffer to the next layer's reads.
  void layerEnd(metal::CommandGraph &graph, uint32_t layer);

  // Engine thread, once the command completed, failed or was abandoned, and
  // before it is encoded again (a rerun): ends what its encode started (a
  // stream's batches, which it cancels and drains) and keeps what its reads
  // could not read; merges its resolves' counts into the cache's.
  void end();
  // What end() kept of every run: the blocks the command's reads could not
  // read, each with the command's lane that needed it, and the time its GPU
  // work waited on its history's reads (ModelBatchTicket::streamMilliseconds).
  [[nodiscard]] std::span<const KvReadFailure> failures() const noexcept { return failures_; }
  [[nodiscard]] double streamMilliseconds() const noexcept { return streamMilliseconds_; }

private:
  KvBlockCache *cache_ = nullptr;
  KvHistoryStream *stream_ = nullptr;
  // The page sources the reads reach: a decode's by batch lane (null for a
  // lane whose pages all hold their K/V), a prefill's streamed sequence's
  // alone.
  std::vector<std::shared_ptr<const KvPageSources>> sources_;
  metal::MetalBuffer locations_;
  // The run's resolves' counts.
  std::shared_ptr<KvReadStats> stats_;
  uint32_t sequence_ = 0;
  uint32_t pages_ = 0;
  metal::MetalBuffer poolTable_;
  std::span<const SplashKvLayer> poolLayers_;
  // Whether this run encoded reads that end() has not ended.
  bool running_ = false;
  // The layer the streamed sequence's chunks attend through staging.
  std::optional<uint32_t> stagedLayer_;
  PrefillKv staged_;
  std::vector<KvReadFailure> failures_;
  double streamMilliseconds_ = 0.0;
};

} // namespace splash::model
