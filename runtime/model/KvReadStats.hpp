#pragma once

// What the reads of spilled K/V report (model::KvBlockCache,
// model::KvHistoryStream, /status), apart from the readers, whose headers
// bring Metal and the attention operators with them.

#include "model/LatencyHistogram.hpp"

#include <cstdint>
#include <vector>

namespace splash::model {

// A block whose tier copy could not be read for a lane of a command: the
// command's item `lane` lists it, and its ModelStepResult::unreadableKvBlocks
// names `block` (KvPageSource::block).
struct KvReadFailure final {
  uint32_t lane = 0;
  uint64_t block = 0;

  bool operator==(const KvReadFailure &) const = default;
};

// What one command's resolves did (KvBlockCache::addResolve): counted on its
// host steps, owned by the command, and merged on the engine thread when it
// completes (KvBlockCache::mergeStats). Entries are the lists' entries, paged
// those of Full pages; a hit found its block cached, or read for another
// lane by the same resolve; a miss read it, once for every lane that listed
// it; a ghost hit missed a block the cache recently gave up. The bytes read
// are what the reads brought, a page's records from its first missed block
// to its last, and a failure a block read twice in vain.
struct KvReadStats final {
  uint64_t resolves = 0;
  uint64_t entries = 0;
  uint64_t paged = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t ghostHits = 0;
  uint64_t readBytes = 0;
  uint64_t readFailures = 0;

  KvReadStats &operator+=(const KvReadStats &other) noexcept {
    resolves += other.resolves;
    entries += other.entries;
    paged += other.paged;
    hits += other.hits;
    misses += other.misses;
    ghostHits += other.ghostHits;
    readBytes += other.readBytes;
    readFailures += other.readFailures;
    return *this;
  }
};

// The hot block cache as /status reports it: its memory, target and floor,
// its slots per layer and the lanes that hold a floor, everything its
// commands' resolves did, and how long a resolve, a read and its host steps
// took (metal::HostStepTiming).
struct KvBlockCacheStats final {
  uint64_t bytes = 0;
  uint64_t targetBytes = 0;
  uint64_t floorBytes = 0;
  uint32_t slotsPerLayer = 0;
  uint32_t lanes = 0;
  KvReadStats reads;
  LatencyStats resolve, read, pickup, work, resume;
};

// What a prefill's history stream did when its command ended
// (KvHistoryStream::end): the blocks it could not read, and the time the
// command waited on its reads, which the scheduler sizes a streaming
// prefill's rows by.
struct KvStreamResult final {
  std::vector<KvReadFailure> failures;
  double streamMilliseconds = 0.0;
};

// The history streams as /status reports them: commands that streamed, the
// layers, pages and bytes they read, the staging they hold and its buffers,
// how long a layer's reads took from their start to their last page, which
// with the bytes gives the streams' bandwidth, and how long commands waited
// on them.
struct KvHistoryStreamStats final {
  uint64_t commands = 0;
  uint64_t layers = 0;
  uint64_t pages = 0;
  uint64_t bytes = 0;
  uint64_t stagingBytes = 0;
  uint32_t buffers = 0;
  LatencyStats read, wait;
};

} // namespace splash::model
