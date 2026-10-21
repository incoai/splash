#include "engine/RuntimeResources.hpp"
#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "StderrLine.hpp"
#include "engine/DiskLabels.hpp"
#include "engine/Engine.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"
#include "model/ExpertStore.hpp"
#include "model/ModelSizing.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/WeightStore.hpp"
#include "ops/AneFfnMeasurement.hpp"
#include "ops/KvSlotFormat.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <utility>

namespace splash::engine {
namespace {

static_assert(model::ExecutionLimits::maximumBatchWidth ==
              SPLASH_MAXIMUM_BATCH_WIDTH);
static_assert(model::ExecutionLimits::prefillTokenBudget ==
              SPLASH_PREFILL_TOKEN_BUDGET);
static_assert(model::ExecutionLimits::draftQueryRows ==
              SPLASH_DRAFT_QUERY_ROWS);
static_assert(model::ExecutionLimits::draftProposalTokens ==
              SPLASH_DRAFT_PROPOSAL_TOKENS);
static_assert(model::ExecutionLimits::targetVerifyRows ==
              SPLASH_TARGET_VERIFY_ROWS);
static_assert(model::ExecutionLimits::draftContextTokens ==
              SPLASH_DRAFT_SLIDING_WINDOW);
static_assert(model::ExecutionLimits::speculativeScratchTokens ==
              SPLASH_SPECULATIVE_SCRATCH_TOKENS);

uint8_t hexNibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<uint8_t>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<uint8_t>(value - 'A' + 10);
  }
  throw std::invalid_argument("manifest SHA-256 is not hexadecimal");
}

uint64_t mebibytes(uint64_t bytes) noexcept { return bytes / kMiB; }

// The startup admission rule. Deliberately independent of the model size:
// it is checked here again at every Metal operation, each image's
// allocation included, as loading and warmup build the residency up.
void requireStartupHeadroom(
    const MemoryGovernor::HostMemoryProvider &hostMemory, uint64_t reserveBytes,
    MemoryPressure pressure) {
  const std::optional<HostMemory> host = hostMemory();
  const std::optional<uint64_t> available =
      host ? std::optional<uint64_t>(host->available) : std::nullopt;
  if (!available || *available <= reserveBytes ||
      pressure == MemoryPressure::Critical) {
    std::ostringstream message;
    message << "not enough free memory to start (";
    if (!available)
      message << "reclaimable host memory cannot be measured";
    else
      message << mebibytes(*available) << " MiB reclaimable, "
              << mebibytes(reserveBytes) << " MiB protected for macOS, system "
              << "pressure " << memoryPressureName(pressure);
    message << "): close other applications for the best performance, or pass "
               "--max-memory to limit Splash's memory";
    throw metal::MetalAllocationError(message.str(),
                                      metal::AllocationFailure::HostPressure);
  }
}

// The format of a persistent cache's files and of what its copies hold. A
// change to either, such as a fix to a kernel that writes KV or state or a
// change to the block identities its labels record (blockIdentity), bumps
// it, so that no start takes back copies of the old one.
constexpr uint32_t kPersistentCacheFormat = 2;
// How long a start waits for the cache directory another process holds: one
// that is closing has finished its flush, or been stopped by the server, by
// then (server/runtime.py `_shutdown_grace_seconds`).
constexpr std::chrono::seconds kCacheLockWait{20};
// A cache nobody opened or wrote for this long is removed at the next start,
// whether or not that start keeps a cache itself.
constexpr std::chrono::hours kStaleCache{24 * 14};
// How long a start serves on probation: the server counts a failure within
// that time toward a crash loop (server/backend.py
// `CRASH_LOOP_WINDOW_SECONDS`) and stops restarting after three in a row.
constexpr std::chrono::seconds kProbation{60};

// A cache root the engine made stays out of Time Machine backups, as
// ~/Library/Caches and $TMPDIR do: its files change with every write.
void excludeFromBackups(const std::filesystem::path &root) {
  NSURL *url = [NSURL fileURLWithPath:@(root.c_str()) isDirectory:YES];
  NSError *error = nil;
  if (![url setResourceValue:@YES forKey:NSURLIsExcludedFromBackupKey error:&error])
    logLine("Persistent cache root ", root.string(), " stays in Time Machine backups (",
            error.localizedDescription.UTF8String, ").");
}

// A persistent tier's directory under root and its files, each with its
// slots' size and options, empty when another process holds the directory or
// it cannot be used: the tier then keeps temporary files, as without one. A
// root it makes is excluded from backups.
PersistentCacheFiles openPersistentCache(const std::filesystem::path &root,
                                         const std::string &cacheNamespace,
                                         uint64_t kvSlotBytes, const model::SlotFile::Options &kvOptions,
                                         uint64_t stateSlotBytes, const model::SlotFile::Options &stateOptions,
                                         const std::shared_ptr<model::DiskBudget> &budget,
                                         const std::function<bool()> &cancelled) {
  try {
    if (root.empty())
      throw std::runtime_error("no home directory to keep it in");
    std::error_code ignored;
    const bool made = !std::filesystem::exists(root, ignored);
    PersistentCacheFiles files;
    files.directory = CacheDirectory::open(root, cacheNamespace, kCacheLockWait, cancelled);
    if (made && std::filesystem::exists(root, ignored))
      excludeFromBackups(root);
    if (!files.directory) {
      if (!cancelled || !cancelled())
        logLine("Persistent cache ", (root / cacheNamespace).string(),
                " is in use by another process; this one keeps a temporary cache.");
      return {};
    }
    const std::vector<std::byte> tag(reinterpret_cast<const std::byte *>(cacheNamespace.data()),
                                     reinterpret_cast<const std::byte *>(cacheNamespace.data()) +
                                         cacheNamespace.size());
    files.kv = std::make_shared<model::SlotFile>(
        kvSlotBytes, budget,
        model::SlotFile::Persistence{files.directory->kvSlots(), files.directory->kvRecords(),
                                     tag, sizeof(KvBlockLabel)},
        kvOptions);
    files.states = std::make_shared<model::SlotFile>(
        stateSlotBytes, budget,
        model::SlotFile::Persistence{files.directory->stateSlots(),
                                     files.directory->stateRecords(), tag, sizeof(StateLabel)},
        stateOptions);
    return files;
  } catch (const std::exception &error) {
    logLine("Persistent cache disabled (", error.what(), "); this process keeps a temporary cache.");
    return {};
  }
}

// The disk tier's startup line: where its files are, its limit and why, its
// write budget and what passes it when one is set, what it holds and stages
// through, and how to turn it off.
void reportDiskTier(const DiskSpace &space, const model::DiskBudget &budget, uint64_t writesPerDay,
                    std::string_view files, uint64_t kvSlotBytes, uint64_t stateBytes) {
  std::ostringstream line;
  line << "SSD cache: up to " << gibibytes(budget.limitBytes()) << " of " << files;
  if (space.othersBytes())
    line << " beside " << gibibytes(space.othersBytes()) << " of other caches there";
  const std::optional<DiskVolume> &volume = space.volume();
  line << " (";
  if (!space.followed()) {
    line << "its quota, on a volume whose free space it does not follow";
  } else {
    const uint64_t floor = diskFloorBytes(volume->sizeBytes);
    if (budget.limitBytes() < budget.capacityBytes())
      line << (space.automatic() ? "a quarter of the " : "the ")
           << gibibytes(diskRoomBytes(volume->availableBytes, space.heldBytes(), floor))
           << " its volume has";
    else
      line << (space.automatic() ? "its automatic quota" : "its quota") << ", while its volume has room";
    line << " beyond the " << gibibytes(floor) << " it keeps free";
  }
  line << ")";
  if (writesPerDay)
    line << "; a write budget of " << gibibytes(writesPerDay)
         << " a day, which the states conversations resume from and KV pages pass, starting with "
            "a quarter of a day's";
  line << "; ";
  if (kvSlotBytes)
    line << "KV pages of " << kvSlotBytes / 1024 << " KiB and ";
  line << "states of " << mebibytes(stateBytes) << " MiB, a state's write staging through "
       << mebibytes(stateBytes) << " MiB of the memory plan; --max-cache-disk 0 turns it off.";
  logLine(line.str());
}

// The layers' part of what ane::recall() keeps a calibration under: their
// shapes and formats.
std::string calibrationKey(std::span<const ops::SwiGluProjections> layers) {
  std::string key = "ane-ffn calibration";
  for (const ops::SwiGluProjections &layer : layers)
    for (const ops::Projection *projection : {layer.gate, layer.up, layer.down})
      key += " " + std::to_string(projection->outputSize) + "x" + std::to_string(projection->inputSize) + ":g" +
             std::to_string(projection->blocks().segments.front().formatId);
  return key;
}

// The split's part of what the engine gives back while idle, if it runs one.
std::optional<ReleasableMemory::Split> idleSplit(ops::AneFfn *split) {
  if (!split)
    return std::nullopt;
  return ReleasableMemory::Split{[split] { return split->release(); },
                                 [split] { split->restore(); }};
}

// The SSD cache tier as a start opens its files (openDiskTier): its budget,
// its state and KV files, a persistent tier's directory, and its volume's
// measure; none of them when it is off, and no KV file when that alone
// could not be made.
struct DiskTier final {
  std::shared_ptr<model::DiskBudget> budget;
  std::shared_ptr<model::SlotFile> states;
  std::shared_ptr<model::SlotFile> kv;
  PersistentCacheFiles persistent;
  std::unique_ptr<DiskSpace> space;
};

// Opens the tier's files where the disk takes them, with `quotaBytes` (none
// leaves it off), each with its slots' size and options: a persistent tier
// keeps its files in a cache directory of its own; when another process
// holds it or it cannot be used, the tier keeps temporary files, as without
// one. The automatic tier opens files only on a volume where its limit would
// hold two states and 1 GiB of KV: a persistent tier's root, whose caches
// nobody holds count as the tier's, else the temporary files'. A relaunch
// whose plan spills (`relaunchSpills`) opens temporary files whatever room
// their volume has: the first engine's context it keeps may rest on the
// tier's KV, and the room bounds each request's commitment instead
// (Engine::admitSpill). Every start removes the root's stale caches.
DiskTier openDiskTier(const RuntimeResourcesConfig &config, const std::string &cacheNamespace,
                      uint64_t quotaBytes, uint64_t kvSlotBytes, const model::SlotFile::Options &kvOptions,
                      uint64_t stateBytes, const model::SlotFile::Options &stateOptions, bool relaunchSpills) {
  DiskTier tier;
  // The cache root's caches nobody holds, this start's own included, looked
  // at once: they are room an automatic persistent tier may take, the stale
  // ones go once this start holds its own, and the rest share the root's
  // limit with a persistent tier (DiskSpace).
  const std::vector<CacheDirectory::Idle> rootCaches =
      config.cacheRoot.empty() ? std::vector<CacheDirectory::Idle>{} : CacheDirectory::idle(config.cacheRoot);
  const uint64_t stateSlotBytes = model::SlotFile::slotBytesFor(stateBytes);
  if (quotaBytes) {
    // A write budget, when one is set, refills on the awake clock the
    // engine schedules its wake-ups on (awakeMilliseconds), so a write it
    // holds back is taken up again when the budget admits it. Without one
    // every write passes.
    tier.budget = std::make_shared<model::DiskBudget>(quotaBytes);
    if (config.cacheDiskWritesPerDay)
      tier.budget->setWriteBudget(config.cacheDiskWritesPerDay);
    std::optional<std::string> rootRefusal, temporaryRefusal;
    if (config.cacheDisk.automatic) {
      const uint64_t minimumBytes = 2 * stateSlotBytes + (uint64_t{1} << 30);
      std::error_code ignored;
      temporaryRefusal = automaticDiskRefusal(
          config.diskVolume(std::filesystem::temp_directory_path(ignored)), 0, relaunchSpills ? 0 : minimumBytes);
      if (config.persistentCache && !config.cacheRoot.empty()) {
        uint64_t heldBytes = 0;
        for (const CacheDirectory::Idle &cache : rootCaches)
          heldBytes += cache.bytes;
        rootRefusal = automaticDiskRefusal(config.diskVolume(config.cacheRoot), heldBytes, minimumBytes);
      }
    }
    if (config.persistentCache && !rootRefusal) {
      tier.persistent = openPersistentCache(config.cacheRoot, cacheNamespace, kvSlotBytes, kvOptions,
                                            stateSlotBytes, stateOptions, tier.budget, config.cancelled);
      tier.states = tier.persistent.states;
    }
    std::optional<std::string> off;
    if (!tier.states && temporaryRefusal) {
      if (rootRefusal)
        off = "for its root, " + *rootRefusal + "; for temporary files, " + *temporaryRefusal;
      else
        off = config.persistentCache ? "for temporary files, " + *temporaryRefusal : *temporaryRefusal;
    } else if (!tier.states) {
      if (rootRefusal)
        logLine("Persistent cache off: ", *rootRefusal, "; the SSD cache keeps temporary files.");
      try {
        tier.states = std::make_shared<model::SlotFile>(stateSlotBytes, tier.budget, stateOptions);
      } catch (const std::exception &error) {
        off = error.what();
      }
    }
    if (off) {
      logLine("SSD cache off: ", *off, "; the state staging the plan set aside stays unused.");
      tier.budget.reset();
    }
  }
  // After the open, which holds this start's own cache: a held one stays.
  // The stale ones go, and the others share the root's limit with the
  // tier.
  std::vector<CacheDirectory::Idle> otherCaches = CacheDirectory::removeStale(rootCaches, kStaleCache);
  std::erase_if(otherCaches,
                [&](const CacheDirectory::Idle &cache) { return cache.path.filename() == cacheNamespace; });
  if (!tier.budget)
    return tier;
  tier.kv = tier.persistent.kv;
  try {
    if (!tier.kv)
      tier.kv = std::make_shared<model::SlotFile>(kvSlotBytes, tier.budget, kvOptions);
  } catch (const std::exception &error) {
    // The persistent files are open by now; without the tier nothing would
    // keep or replace their copies.
    if (tier.persistent.directory)
      throw;
    logLine("SSD cache KV storage disabled; state storage remains enabled (", error.what(), ").");
  }
  // The volume sizes the tier before it takes anything back. A persistent
  // tier's root holds the other caches nobody holds too.
  std::vector<std::shared_ptr<model::SlotFile>> files{tier.states};
  if (tier.kv)
    files.push_back(tier.kv);
  tier.space = std::make_unique<DiskSpace>(
      tier.budget, config.cacheDisk.automatic, std::move(files),
      tier.persistent.directory ? std::move(otherCaches) : std::vector<CacheDirectory::Idle>{});
  reportDiskTier(*tier.space, *tier.budget, config.cacheDiskWritesPerDay,
                 tier.persistent.directory
                     ? "persistent files in " + tier.persistent.directory->path().string()
                     : "temporary files in " + std::filesystem::temp_directory_path().string(),
                 tier.kv ? kvSlotBytes : 0, stateBytes);
  return tier;
}

// A setting the planner refuses, or a plan the budget cannot hold, refused
// with what its floors take of those the model has: the stores of parts
// that stream, and the floors of a block-sparse target's KV on the SSD
// cache.
RuntimeResourcesError planRefusal(const StreamedMemoryResult &result, uint64_t hardBudgetBytes,
                                  const model::StreamableWeights &streamable, const model::ModelDescriptor &model) {
  if (!result.requiredBytes)
    return RuntimeResourcesError(RuntimeResourceStage::Configuration, result.error);
  std::string floors = "model weights with the runtime reserves, one lane's state, the KV runway";
  if (streamable.expertBytes || streamable.tableBytes)
    floors += ", the streamed stores at their floors";
  if (model.capabilities.sparseKv)
    floors += ", the floors of KV on the SSD cache";
  return RuntimeResourcesError(RuntimeResourceStage::MemoryPlanning,
                               floors + " and any SSD cache state staging require " +
                                   std::to_string(result.requiredBytes) + " bytes but the Metal memory budget is " +
                                   std::to_string(hardBudgetBytes) + " bytes",
                               {}, RuntimeResourceFailure::EngineCapacity);
}

// An explicit size of `option` that changes nothing: the model has no `what`
// to stream, or the size holds all of it, which then loads.
void reportIgnoredSetting(std::string_view option, const StreamedSetting &setting, uint64_t allBytes,
                          std::string_view modelName, std::string_view what, std::string_view loads) {
  if (setting.mode != StreamedSetting::Mode::Size)
    return;
  if (!allBytes)
    logLine(option, " ignored: ", modelName, " has no ", what, " to stream");
  else if (setting.bytes >= allBytes)
    logLine(option, " ignored: its ", setting.bytes / kMiB, " MiB hold all of the ", what, " (",
            allBytes / kMiB, " MiB), which ", loads);
}

// What `plan` streams that `other` loads whole, the routed experts before
// the PLE table, and that it does; empty for neither.
std::string_view streamedOnlyIn(const StreamedMemoryPlan &plan, const StreamedMemoryPlan &other) {
  if (plan.expertBytes && !other.expertBytes)
    return "routed experts, which load whole without it";
  if (plan.tableBytes && !other.tableBytes)
    return "PLE table, which loads whole without it";
  return {};
}

// What streams of the loaded model, and how an automatic part's store was
// sized.
void reportStreamedStores(const model::LoadedModel &loaded, const model::StreamableWeights &streamable,
                          const StreamedSetting &experts, const StreamedSetting &table) {
  const auto automatic = [](const StreamedSetting &setting, std::string_view option) {
    return setting.mode == StreamedSetting::Mode::Auto
               ? std::string(", all that ") + std::string(option) +
                     " auto leaves it beside the floors and the KV kept resident"
               : std::string();
  };
  if (const model::SparseRowStore *store = loaded.streamedTable()) {
    const model::SparseRowStats stats = store->stats();
    logLine("Table ", stats.table, " streams from its file: ", stats.tableBytes / kMiB,
            " MiB stay on the SSD; row cache of ", stats.cacheCapacityRows, " rows in ",
            store->committedBytes() / kMiB, " MiB", automatic(table, "--ple-memory"), "; ", model::kSsdReaders,
            " readers.");
  } else if (streamable.tableBytes && table.mode == StreamedSetting::Mode::Auto) {
    logLine("The PLE table loads with the weights: memory holds its ", streamable.tableBytes / kMiB,
            " MiB (--ple-memory auto).");
  }
  if (const model::ExpertStore *store = loaded.expertStore()) {
    const model::ExpertStoreStats stats = store->stats();
    logLine("Routed experts stream from their files: ", stats.slots, " slots for ", stats.experts, " experts (",
            stats.slots * 100 / std::max<uint64_t>(stats.experts, 1), "%) in ", stats.arenaBytes / kMiB, " MiB",
            automatic(experts, "--expert-memory"), ".");
  } else if (streamable.expertBytes && experts.mode == StreamedSetting::Mode::Auto) {
    logLine("Routed experts load with their layers: memory holds their ", streamable.expertBytes / kMiB,
            " MiB (--expert-memory auto).");
  }
}

// The rows the target's decode cycles verify (--verify-rows), and why.
void reportVerifyRows(const model::LoadedModel &loaded, model::VerifyRows rows) {
  if (model::budgetsVerifyRows(loaded, rows))
    logLine("Decode cycles verify the rows the verify budget chooses (--verify-rows auto, as ", loaded.name(),
            " is a MoE target): greedy and seeded outputs can differ from run to run.");
  else if (rows == model::VerifyRows::All)
    logLine("Decode cycles verify every row of the draft's block (--verify-rows all).");
  else
    logLine("Decode cycles verify every row of the draft's block (--verify-rows auto, as ", loaded.name(),
            " is dense).");
}

// Why a target's long requests keep their KV in memory, empty when they
// spill.
std::string spillOffReason(const model::ModelDescriptor &model, const kv::Layout &layout,
                           const KvSpillPlan &spill, bool tierWritable) {
  if (spill.enabled)
    return {};
  if (!model.capabilities.sparseKv)
    return "its attention is dense";
  if (model.capabilities.sparseKv->localTokens)
    return "its rows read a local window, which the SSD cache does not keep in memory";
  if (!layout.spillable())
    return "its KV layout cannot keep a page's index alone";
  if (!tierWritable)
    return "the SSD cache is off";
  return "the floors of KV on the SSD cache do not fit beside the model";
}

// A block-sparse target's startup line: where long requests' KV lives, the
// context and what bounds it, and the readers' memory.
void reportSpill(const model::ModelDescriptor &model, const EngineMemoryBreakdown &budget,
                 std::string_view offReason, uint64_t activeLimitBytes) {
  if (!model.capabilities.sparseKv)
    return;
  const ContextBounds &bounds = budget.contextBounds;
  const KvSpillPlan &spill = budget.kvSpill;
  if (!spill.enabled) {
    logLine("Long requests keep their KV in memory, ", offReason, ": ", bounds.maximum(), " tokens of context.");
    return;
  }
  logLine("Long requests keep their KV on the SSD cache: ", bounds.maximum(), " tokens of context (bound: ",
          bounds.bound(), "; window ", bounds.window, ", resident ", bounds.resident, ", spilled ", bounds.spilled,
          ", disk ", bounds.disk, "); a request whose KV fits ", budget.residentKvBytes / kMiB,
          " MiB stays in memory; hot block cache ", spill.hotTargetBytes / kMiB, " MiB (floor ",
          spill.hotFloorBytes / kMiB, " MiB); history staging in ", spill.stagingBuffers,
          " buffers; SSD room for active KV ", gibibytes(activeLimitBytes), ".");
}

std::array<uint8_t, 32> parseSha256(std::string_view value) {
  if (value.size() != 64) {
    throw std::invalid_argument(
        "manifest SHA-256 must contain exactly 64 hex characters");
  }
  std::array<uint8_t, 32> result{};
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<uint8_t>((hexNibble(value[index * 2]) << 4) |
                                         hexNibble(value[index * 2 + 1]));
  }
  return result;
}

} // namespace

void requireLoadedModel(const model::LoadedModel &loaded) {
  if (!loaded.targetActualAllocatedBytes() ||
      !loaded.draft.actualAllocatedBytes ||
      (loaded.descriptor.hasVision() && !loaded.vision.actualAllocatedBytes) ||
      loaded.manifestFingerprintSha256.empty() ||
      loaded.targetManifestFingerprint().empty()) {
    throw std::invalid_argument(
        "loaded model has incomplete allocation accounting");
  }
}

std::string_view runtimeResourceStageName(RuntimeResourceStage stage) {
  switch (stage) {
  case RuntimeResourceStage::Configuration:
    return "configuration";
  case RuntimeResourceStage::BackendCreation:
    return "backend_creation";
  case RuntimeResourceStage::CapabilityValidation:
    return "capability_validation";
  case RuntimeResourceStage::ModelLoading:
    return "model_loading";
  case RuntimeResourceStage::MemoryPlanning:
    return "memory_planning";
  case RuntimeResourceStage::StorageAllocation:
    return "storage_allocation";
  }
  return "unknown";
}

RuntimeCacheIdentity
makeRuntimeCacheIdentity(std::string_view combinedManifestSha256,
                         std::string_view targetManifestSha256,
                         std::string_view buildId,
                         kv::Layout targetKvLayout,
                         model::RopeScaling ropeScaling) {
  if (buildId.empty()) {
    throw std::invalid_argument("runtime build id is required");
  }
  if (!targetKvLayout.valid()) {
    throw std::invalid_argument("runtime target KV layout is invalid");
  }
  // Parsing rejects a malformed manifest digest before the KV pool and the
  // cache are built.
  RuntimeCacheIdentity result;
  result.modelLayoutSha256 = model::digestHex(parseSha256(combinedManifestSha256));
  result.buildId = buildId;
  result.kvLayout = targetKvLayout;
  result.ropeScaling = ropeScaling;
  result.targetModelSha256 = model::digestHex(parseSha256(targetManifestSha256));
  return result;
}

std::string persistentCacheNamespace(const RuntimeCacheIdentity &identity,
                                     const model::CompositeStateLayout &states) {
  const kv::Layout &kv = identity.kvLayout;
  const model::GdnStateLayout &gdn = states.target;
  const model::DraftStateLayout &draft = states.draft;
  std::ostringstream canonical;
  canonical << "splash-persistent-cache-v" << kPersistentCacheFormat << '\n'
            << identity.modelLayoutSha256 << '\n'
            << "kv " << kv::kPageTokens << ' ' << kv.attentionLayers << ' ' << kv.kvHeads << ' '
            << kv.headDimension << ' ' << kv::formatName(kv.format)
            // Dimension-major values add nothing, so dense layouts keep the
            // caches they wrote.
            << (kv.valuesTokenMajor ? " values-token-major" : "");
  // A block index's pooled keys (kv::Layout::indexDimension), bf16, one a
  // block of SPLASH_ATTENTION_BLOCK_TOKENS tokens. A change to how a block's
  // key is pooled, normalized or rotated changes this line.
  if (kv.indexDimension)
    canonical << " index " << kv.indexDimension << " bf16 block " << SPLASH_ATTENTION_BLOCK_TOKENS;
  // How a slot holds a page; nothing for the spans' order, so the caches
  // written before slots had another order stay valid.
  canonical << KvPageTier::slotFormatToken(kv) << '\n'
            << "gdn " << gdn.layers << ' ' << gdn.convolutionHistory << ' '
            << gdn.convolutionChannels << ' ' << gdn.recurrentGroups << ' ' << gdn.recurrentRows
            << ' ' << gdn.recurrentColumns
            // fp32 states keep the line they had before states had a format,
            // so the caches they wrote stay valid.
            << (gdn.format == ops::GdnStateFormat::BFloat16 ? " bf16" : "");
  // A family's state after the GDN layers' (Qwen3.8-Flash-Next's PLE history
  // and QSA pending keys, model::QwenHybridLayout::gdnStateLayout).
  if (gdn.auxiliaryBytes) canonical << " auxiliary " << gdn.auxiliaryBytes;
  canonical << '\n'
            << "draft " << draft.layers << ' ' << draft.kvHeads << ' ' << draft.headDimension << ' '
            << SPLASH_DRAFT_SLIDING_WINDOW << '\n'
            << "context " << draft.contextWidth << " q4 " << SPLASH_DRAFT_CONTEXT_GROUP << '\n';
  // Without YaRN the line is left out, so the caches written before it
  // existed stay valid. The factor takes the digits that read back as the
  // same double, so no two factors share a namespace.
  if (const model::RopeScaling &rope = identity.ropeScaling; rope.yarn())
    canonical << "rope yarn " << std::setprecision(std::numeric_limits<double>::max_digits10) << rope.factor
              << ' ' << rope.originalContextTokens << '\n';
  // 128 bits name it.
  return model::weightDigest(canonical.str()).substr(0, CacheDirectory::kNamespaceDigits);
}

RuntimeResourcesError::RuntimeResourcesError(RuntimeResourceStage stage,
                                             std::string message,
                                             std::string budgetDescription,
                                             RuntimeResourceFailure failure)
    : std::runtime_error(std::move(message)), stage_(stage), failure_(failure),
      budgetDescription_(std::move(budgetDescription)) {}

RuntimeResources::RuntimeResources(
    PersistentCacheFiles persistentCache,
    std::unique_ptr<metal::MetalBackend> backend, std::shared_ptr<model::SsdReaders> readers,
    model::LoadedModel model, ops::ExecutionPlans operators, EngineMemoryPlan memoryPlan,
    RuntimeCacheIdentity cacheIdentity,
    std::unique_ptr<MemoryGovernor> memoryGovernor,
    std::unique_ptr<kv::PageStorage> kvPages,
    std::unique_ptr<model::QwenStateStorage> stateStorage,
    std::unique_ptr<KvPageTier> kvTier, std::unique_ptr<DiskSpace> diskSpace,
    std::unique_ptr<KvPool> kvPool, std::unique_ptr<engine::Cache> cache,
    std::unique_ptr<ops::AneFfn> aneFfn, AneFfnOutcome aneFfnOutcome,
    std::unique_ptr<model::KvBlockCache> blockCache,
    std::unique_ptr<model::KvHistoryStream> historyStream, bool spillAlways,
    model::VerifyRows verifyRows)
    : persistentCache_(std::move(persistentCache)),
      backend_(std::move(backend)), readers_(std::move(readers)), model_(std::move(model)),
      operators_(std::move(operators)),
      memoryPlan_(std::move(memoryPlan)),
      cacheIdentity_(std::move(cacheIdentity)),
      memoryGovernor_(std::move(memoryGovernor)), kvPages_(std::move(kvPages)),
      stateStorage_(std::move(stateStorage)), kvTier_(std::move(kvTier)),
      diskSpace_(std::move(diskSpace)), kvPool_(std::move(kvPool)),
      cache_(std::move(cache)), aneFfn_(std::move(aneFfn)),
      blockCache_(std::move(blockCache)), historyStream_(std::move(historyStream)),
      releasableMemory_({model_.images.get(), model_.expertStore()}, idleSplit(aneFfn_.get())),
      aneFfnOutcome_(std::move(aneFfnOutcome)), spillAlways_(spillAlways), verifyRows_(verifyRows) {}

std::unique_ptr<RuntimeResources>
RuntimeResources::create(const RuntimeResourcesConfig &config,
                         uint32_t requestedContextTokens) {
  if (config.metallibPath.empty() || config.modelRoot.empty() ||
      !kv::validFormat(config.kvFormat) || !ops::validFormat(config.stateFormat) ||
      !config.model.valid() || !config.hostMemory || !config.diskVolume ||
      config.buildId.empty() || !config.maximumImagePatches ||
      config.maximumImagePatches % 4 ||
      config.maximumImagePatches > ops::kMaximumImagePatches) {
    throw RuntimeResourcesError(
        RuntimeResourceStage::Configuration,
        "metallib path, model root, a valid model layout, KV and state format, "
        "host memory and volume probes, build id and a merge-aligned image "
        "patch limit no larger than the protocol's are required");
  }
  std::unique_ptr<metal::MetalBackend> backend;
  try {
    backend = std::make_unique<metal::MetalBackend>(config.metallibPath.string(),
                                                    config.idleReleaseSeconds);
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what(), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what());
  }

  const DeviceCapabilities &device = backend->capabilities();
  if (auto error = device.validationError()) {
    throw RuntimeResourcesError(RuntimeResourceStage::CapabilityValidation,
                                *error);
  }

  const uint64_t hostReserveBytes =
      EngineMemoryPolicy::hostAvailableReserveBytes(device.physicalMemoryBytes);
  // Startup work stops on cancellation and keeps its reserve of host memory.
  const auto throwIfCancelled = [cancelled = config.cancelled] {
    if (cancelled && cancelled())
      throw metal::MetalBackendError("startup cancelled");
  };
  const auto currentPressure = [pressure = config.memoryPressure] {
    return pressure ? pressure() : MemoryPressure::Normal;
  };
  const auto admitMetalOperation = [throwIfCancelled, currentPressure,
                                    hostMemory = config.hostMemory, hostReserveBytes] {
    throwIfCancelled();
    requireStartupHeadroom(hostMemory, hostReserveBytes, currentPressure());
  };
  backend->setOperationGuard(admitMetalOperation);
  // A synchronous command wait gives up when the process shuts down, also
  // after startup.
  backend->setWaitInterrupt(config.cancelled);
  // One disk quota serves KV pages and states. Without room for a state,
  // disk KV cannot preserve a restorable prefix, so the tier stays off, and
  // no state's write needs the staging buffer the plan would set aside. The
  // plan of a target whose KV cannot spill never depends on the disk, so
  // that every start of the model serves the same context: an explicit quota
  // that holds a state sets the staging aside, and the automatic one where
  // the plan holds the model, its parts as they load without it and the
  // requested context beside it and the Neural Engine split does not need
  // its memory (below). The disk decides only whether the tier's files open,
  // once the weights have loaded: where they do not, the tier is off and the
  // staging stays unused. The volume sizes the tier once its files are open
  // (DiskSpace); a target whose KV spills plans its stores and context
  // (ContextBounds::disk) with the room it leaves active KV, before the
  // weights load and again once the files open.
  model::CompositeStateLayout stateLayout = config.model.stateLayout;
  stateLayout.target.format = config.stateFormat;
  const uint64_t stateBytes = stateLayout.cachedBytes();
  const bool automaticDisk = config.cacheDisk.automatic;
  const uint64_t cacheDiskBytes = automaticDisk ? kAutomaticCacheDiskBytes : config.cacheDisk.bytes;
  // A state's write to the disk tier stages through one buffer of a state's
  // size. It is the backend's like every other, so the governor charges it
  // beside the weights and the plan sets it aside before it sizes KV.
  uint64_t stateStagingBytes = 0;
  if (cacheDiskBytes >= model::SlotFile::slotBytesFor(stateBytes))
    stateStagingBytes = stateBytes;
  else if (cacheDiskBytes)
    logLine("SSD cache off: its quota holds no state of ", mebibytes(stateBytes),
            " MiB; no state staging is set aside.");
  // The automatic tier turns off before the engine's plan is made with it.
  const auto diskOff = [&](std::string_view reason) {
    logLine("SSD cache off: ", reason, "; no state staging is set aside.");
    stateStagingBytes = 0;
  };
  // The one pool of SSD readers, which the model's streamed weights and the
  // reads of the KV the tier holds share. It goes after the model and the
  // tier, which read on it.
  auto readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);

  // One plan (planStreamedMemory) sizes the parts of the weights that may
  // stream from the SSD, a MoE target's routed experts and a GGUF's PLE
  // table: a part set to a size or off loads as set, and the plan sizes an
  // automatic one. It rests on the weights' sizes, which their files give
  // before they load; on the arenas, which only a loaded model gives, as they
  // rest on its weights' formats, but which one load gives for whatever
  // streams of it (model::StreamedParts); and on the SSD cache's room for
  // active KV and whether its KV file opened.
  kv::Layout kvLayout = config.model.targetKvLayout;
  kvLayout.format = config.kvFormat;
  // How the KV tier lays a page out in its slot, and so the slots' size.
  const kv::SlotFormat kvFormat(kvLayout);
  const uint64_t kvSlotBytes = model::SlotFile::slotBytesFor(kvFormat.payloadBytes());
  const uint64_t hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
      device.recommendedMaxWorkingSetBytes, config.maximumMemoryBytes);
  model::StreamableWeights streamable;
  // The arenas of the loaded model by what streams of it, and those of a
  // runtime whose K/V spills, which only a block-sparse target with a
  // spillable layout may have; none before it loads.
  const bool kvMaySpill = config.model.capabilities.sparseKv && kvLayout.spillable();
  std::array<std::array<uint64_t, 2>, 2> arenaBytes{};
  std::array<uint64_t, 2> spillArenaBytes{};
  // The room for active KV, which only a target whose KV may spill plans
  // with: until the tier's files open, what the volume they will take
  // leaves now, a persistent tier's root, whose caches nobody holds count as
  // the tier's, else the temporary files'; the room the open measures then
  // bounds it, so that the floors the stores were sized beside only shrink.
  uint64_t activeLimitBytes = cacheDiskBytes;
  if (kvMaySpill) {
    std::error_code ignored;
    const bool root = config.persistentCache && !config.cacheRoot.empty();
    uint64_t heldBytes = 0;
    if (root)
      for (const CacheDirectory::Idle &cache : CacheDirectory::idle(config.cacheRoot))
        heldBytes += cache.bytes;
    activeLimitBytes = volumeActiveLimitBytes(
        cacheDiskBytes,
        config.diskVolume(root ? config.cacheRoot : std::filesystem::temp_directory_path(ignored)), heldBytes);
  }
  bool kvFileOpen = true;
  // The context a relaunch keeps, which only an automatic context does.
  const uint32_t relaunchContext = requestedContextTokens ? 0 : config.relaunchContext;
  // The plan with the parts set as `experts` and `table`, `stagingBytes` set
  // aside for the disk tier's state staging and `aneFfnBytes` for the
  // prefill FFN's Neural Engine split.
  const auto streamedPlan = [&](const StreamedSetting &experts, const StreamedSetting &table,
                                uint64_t stagingBytes, uint64_t aneFfnBytes) {
    return planStreamedMemory(
        {.hardBudgetBytes = hardBudgetBytes,
         .fixedBytes = streamable.loadedBytes + model::kPipelineReserveBytes + model::kRuntimeOverheadReserveBytes +
                       stagingBytes + aneFfnBytes,
         .arenaBytes = arenaBytes,
         .spillArenaBytes = spillArenaBytes,
         .laneStateBytes = stateLayout.laneBytes(),
         .kvLayout = kvLayout,
         .windowTokens = config.model.capabilities.maximumContextTokens,
         .requestedContextTokens = requestedContextTokens,
         .relaunchContextTokens = relaunchContext,
         .experts = {streamable.expertBytes, streamable.expertFloorBytes, experts},
         .table = {streamable.tableBytes, streamable.tableFloorBytes, table},
         .hotCacheBytes = config.sparseKvMemory,
         .sparseKv = config.model.capabilities.sparseKv,
         .tierWritable = stagingBytes && kvFileOpen,
         .kvSlotBytes = kvSlotBytes,
         .stateSlotBytes = model::SlotFile::slotBytesFor(stateBytes),
         .activeLimitBytes = activeLimitBytes,
         .groupPages = model::kPrefillGroupChunks * model::ExecutionLimits::prefillTokenBudget / kv::kPageTokens});
  };
  // The plan of the parts as configured, which sizes their stores.
  const auto sizingPlan = [&](uint64_t stagingBytes) {
    return streamedPlan(config.expertMemory, config.pleMemory, stagingBytes, 0);
  };
  // The sizing plan with the staging the tier keeps. The automatic tier gives
  // it up where the plan cannot hold the model beside it, and, once the
  // model `loaded` gave the arenas, unless the plan spills, which the tier
  // makes possible, where the staging would stream a part that loads whole
  // without it: a tier that only caches costs the weights nothing, as it
  // costs the context nothing (below).
  const auto storesPlan = [&](bool loaded) {
    StreamedMemoryResult planned = sizingPlan(stateStagingBytes);
    if (!stateStagingBytes || !automaticDisk || (planned.plan && planned.plan->spill.enabled))
      return planned;
    StreamedMemoryResult without = sizingPlan(0);
    if (!without.plan)
      return planned;
    std::string reason;
    if (!planned.plan && planned.requiredBytes)
      reason = "the memory budget has no room for its " + std::to_string(mebibytes(stateBytes)) +
               " MiB of state staging beside the model";
    else if (planned.plan && loaded)
      if (const std::string_view part = streamedOnlyIn(*planned.plan, *without.plan); !part.empty())
        reason = "its " + std::to_string(mebibytes(stateBytes)) + " MiB of state staging would stream the " +
                 std::string(part);
    if (reason.empty())
      return planned;
    diskOff(reason);
    return without;
  };
  // The stores the plan gives the parts, except that automatic experts the
  // plan streams load at their floor while `floors`: until the loaded model
  // gives the arenas, which take from all that the experts stream through.
  const auto storesOf = [&](const StreamedMemoryPlan &plan, bool floors) {
    const bool experts = floors && plan.expertBytes && config.expertMemory.mode == StreamedSetting::Mode::Auto;
    return model::ModelLoadOptions{.streamedTableBytes = plan.tableBytes,
                                   .expertMemoryBytes = experts ? streamable.expertFloorBytes : plan.expertBytes,
                                   .readers = readers};
  };
  model::ModelLoadOptions loadOptions;
  try {
    // A table set off is among the images that load: its streaming is never
    // planned.
    streamable = model::streamableWeights(
        config.modelRoot, config.model,
        {.streamExperts = true, .streamTable = config.pleMemory.mode != StreamedSetting::Mode::Off});
    if (!streamable.tableRefusal.empty()) {
      if (config.pleMemory.mode == StreamedSetting::Mode::Size)
        throw RuntimeResourcesError(RuntimeResourceStage::Configuration,
                                    "--ple-memory cannot stream the PLE table: " + streamable.tableRefusal);
      logLine("The PLE table loads with the weights: ", streamable.tableRefusal, " (--ple-memory auto).");
    }
    reportIgnoredSetting("--ple-memory", config.pleMemory, streamable.tableBytes, config.model.name, "PLE table",
                         "loads with the weights");
    reportIgnoredSetting("--expert-memory", config.expertMemory, streamable.expertBytes, config.model.name,
                         "routed experts", "load with their layers");
    // Refuse a model that cannot fit before loading its weights: beside them
    // the plan needs at least the runtime reserves, one lane's state, the
    // KV runway, the streamed stores at their floors, the floors of KV on the
    // SSD cache and any SSD cache state staging; the plan after the load
    // adds the arenas.
    const StreamedMemoryResult planned = storesPlan(false);
    if (!planned.plan)
      throw planRefusal(planned, hardBudgetBytes, streamable, config.model);
    loadOptions = storesOf(*planned.plan, true);
    // Fail before loading the model when the machine has no headroom at
    // all; the guard installed above keeps checking as residency grows.
    admitMetalOperation();
  } catch (const RuntimeResourcesError &) {
    throw;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what());
  }

  // One plan owner is used both before allocation and during encoding. The
  // engine lends it to model execution without inspecting its plans.
  ops::ExecutionPlans operators(device);
  model::LoadedModel loaded;
  // The runtime's lane state and arenas as the model loaded, with its K/V in
  // memory and where it spills.
  model::ModelMemoryPlan modelMemoryPlan, spillModelMemoryPlan;
  // Loads the model as loadOptions says, `again` once it loaded, and plans
  // its arenas: as it loaded, and for whatever streams of it.
  const auto load = [&](bool again) {
    try {
      const auto started = AwakeClock::now();
      // The images of the last load go before the next allocates its own,
      // and the backend's peaks start again: the memory audit holds the load
      // that serves to its plan, not one the plan with the arenas replaced,
      // which may have held a PLE table whole that now streams through a
      // row cache the audit's budget for Metal leaves out.
      loaded = {};
      backend->restartPeaks();
      loaded = model::loadModel(*backend, config.modelRoot, config.model, loadOptions);
      requireLoadedModel(loaded);
      const std::chrono::duration<double> loading = AwakeClock::now() - started;
      logLine("Weights loaded ", again ? "again " : "", "in ", std::fixed, std::setprecision(2), loading.count(),
              " s.");
    } catch (const metal::MetalAllocationError &error) {
      throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                  error.what(), {},
                                  resourceAllocationFailure(error.failure()));
    } catch (const std::exception &error) {
      throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                  error.what());
    }
    try {
      modelMemoryPlan = model::plannedRuntimeMemory(loaded, operators, config.kvFormat, config.stateFormat,
                                                    model::StreamedParts::loaded(loaded));
      if (kvMaySpill)
        spillModelMemoryPlan = model::plannedRuntimeMemory(loaded, operators, config.kvFormat, config.stateFormat,
                                                           model::StreamedParts::loaded(loaded, true));
      const auto arenas = [&](model::StreamedParts parts) {
        const model::ModelMemoryPlan streamed =
            model::plannedRuntimeMemory(loaded, operators, config.kvFormat, config.stateFormat, parts);
        return streamed.sharedPrefillPlannedAllocatedBytes + streamed.sharedDecodePlannedAllocatedBytes;
      };
      for (const bool table : {false, true}) {
        const uint64_t tableRowBytes = table ? streamable.tableRowBytes : 0;
        for (const bool experts : {false, true})
          arenaBytes[experts][table] =
              arenas({.experts = experts && streamable.expertBytes, .tableRowBytes = tableRowBytes});
        if (kvMaySpill)
          spillArenaBytes[table] = arenas({.tableRowBytes = tableRowBytes, .kv = true});
      }
    } catch (const std::exception &error) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning,
          std::string("model allocated-size plan is invalid: ") + error.what());
    }
  };
  load(false);

  // The engine's memory plan, with `stagingBytes` set aside for the disk
  // tier's state staging and `aneFfnBytes` for the prefill FFN's Neural Engine
  // split, each part as it loaded, whole or through its store, and the spill
  // plan beside them, with the arenas of a runtime whose K/V spills where it
  // spills. Floors that the spill plan refuses refuse it with the engine
  // plan's breakdown.
  const auto planWith = [&](uint64_t stagingBytes, uint64_t aneFfnBytes) -> EngineMemoryPlanResult {
    const auto asLoaded = [](uint64_t storeBytes) {
      return storeBytes ? StreamedSetting{StreamedSetting::Mode::Size, storeBytes}
                        : StreamedSetting{StreamedSetting::Mode::Off, 0};
    };
    const StreamedMemoryResult streamed = streamedPlan(
        asLoaded(loadOptions.expertMemoryBytes), asLoaded(loadOptions.streamedTableBytes), stagingBytes, aneFfnBytes);
    ModelMemoryFootprint footprint{
        loaded.targetActualAllocatedBytes(),
        loaded.draft.actualAllocatedBytes,
        loaded.vision.actualAllocatedBytes,
        streamed.plan && streamed.plan->spill.enabled ? spillModelMemoryPlan : modelMemoryPlan,
        stagingBytes,
        aneFfnBytes,
        loaded.streamedTable() ? loaded.streamedTable()->committedBytes() : 0,
    };
    ModelMemoryProfile modelProfile{
        loaded.name(), loaded.maximumContextTokens(), loaded.targetKvLayout(config.kvFormat), footprint,
        config.stateFormat, streamed.plan ? streamed.plan->spill : KvSpillPlan{}};
    EngineMemoryPlanResult result = evaluateEngineMemoryPlan(device, modelProfile, config.maximumMemoryBytes);
    if (streamed.plan || !result.plan)
      return result;
    BudgetValidationStatus &status = result.status;
    status.valid = false;
    status.code = BudgetErrorCode::KvPoolDoesNotFit;
    status.message = streamed.error;
    if (streamed.requiredBytes) {
      status.breakdown.minimumRequiredBytes = streamed.requiredBytes;
      status.breakdown.deficitBytes = streamed.requiredBytes - status.breakdown.hardBudgetBytes;
    }
    return {std::nullopt, std::move(status)};
  };
  // The plan with the staging the tier keeps.
  const auto planMemory = [&](uint64_t aneFfnBytes) { return planWith(stateStagingBytes, aneFfnBytes); };

  // The plan with the loaded model's arenas sizes the stores, and the start
  // loads again where it gives a part another store: it loads once where the
  // plan without the arenas already loads each part so, as for a model with
  // no part to stream, and twice otherwise, as where the routed experts
  // stream. Floors that the arenas take beyond the budget stop the start
  // with the engine plan's breakdown (requirePlan, below).
  const StreamedMemoryResult planned = storesPlan(true);
  if (!planned.plan && !planned.requiredBytes)
    throw planRefusal(planned, hardBudgetBytes, streamable, config.model);
  if (planned.plan) {
    const model::ModelLoadOptions next = storesOf(*planned.plan, false);
    if (next.expertMemoryBytes != loadOptions.expertMemoryBytes ||
        next.streamedTableBytes != loadOptions.streamedTableBytes) {
      loadOptions = next;
      load(true);
    }
  }
  reportStreamedStores(loaded, streamable, config.expertMemory, config.pleMemory);
  reportVerifyRows(loaded, config.verifyRows);
  // The model's side of the split (startAneFfn).
  const AneFfnModel splitModel =
      aneFfnModel(*backend, loaded, operators, config.kvFormat, config.buildId, config.cancelled);

  // The automatic tier keeps its staging only where the plan holds the model
  // and the requested context beside it, the parts as they load. The split
  // wins over it: where the plan refuses the split beside the staging but not
  // without it, the tier turns off before the split is decided. A relaunch
  // decides it as its first engine, an automatic start, did, so that it keeps
  // the staging that engine kept; only its split leaves that engine's context
  // (below).
  if (stateStagingBytes && automaticDisk) {
    const auto withoutStaging = [&](uint64_t aneFfnBytes) { return planWith(0, aneFfnBytes); };
    const EngineMemoryPlanResult planResult = planMemory(0);
    if (!planResult.plan ||
        planResult.plan->maximumContextTokens() < std::max<uint32_t>(1, requestedContextTokens))
      diskOff(requestedContextTokens
                  ? "the memory plan cannot hold a " + std::to_string(requestedContextTokens) +
                        "-token context beside its state staging"
                  : std::string("the memory plan cannot hold the model beside its state staging"));
    else if (aneFfnRefused(splitModel, config.aneFfn, requestedContextTokens, planMemory) &&
             !aneFfnRefused(splitModel, config.aneFfn, requestedContextTokens, withoutStaging))
      diskOff(requestedContextTokens
                  ? "the memory plan cannot hold the Neural Engine FFN split and a " +
                        std::to_string(requestedContextTokens) +
                        "-token context beside its state staging"
                  : std::string("the memory plan cannot hold the Neural Engine FFN split and any "
                                "context beside its state staging"));
  }

  // The engine's plan with the staging the tier keeps; one that cannot hold
  // the model stops the start with its breakdown, before the tier's files
  // open.
  const auto requirePlan = [&] {
    EngineMemoryPlanResult planResult = planMemory(0);
    if (!planResult.plan) {
      throw RuntimeResourcesError(RuntimeResourceStage::MemoryPlanning,
                                  planResult.status.message,
                                  planResult.status.describe());
    }
    return std::move(*planResult.plan);
  };
  EngineMemoryPlan memoryPlan = requirePlan();

  RuntimeCacheIdentity cacheIdentity;
  try {
    cacheIdentity = makeRuntimeCacheIdentity(
        loaded.manifestFingerprintSha256,
        loaded.targetManifestFingerprint(), config.buildId,
        loaded.targetKvLayout(config.kvFormat), loaded.descriptor.ropeScaling);
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(),
                                memoryPlan.breakdown().describe());
  }

  try {
    // The tier's files open now, where the disk takes them (openDiskTier):
    // the plan set the staging aside without asking it, and where they do
    // not open the staging stays unused. A plan that spills takes the room
    // the tier's volume leaves active KV, and spills only where its KV file
    // opened: its floors can only shrink. A relaunch whose plan spills opens
    // them whatever room the volume has, as the first engine's context it
    // keeps may rest on them.
    const std::string cacheNamespace =
        persistentCacheNamespace(cacheIdentity, loaded.stateLayout(config.stateFormat));
    // The KV file's slots hold a page as the tier lays it out, checksummed by
    // segment when persistent, and the writes of both files hold back while
    // a step waits for reads.
    const model::SlotFile::Options kvFileOptions{.segments = kvFormat.segments(),
                                                 .deferWrites = readers->criticalBusyProbe()};
    const model::SlotFile::Options stateFileOptions{.deferWrites = readers->criticalBusyProbe()};
    DiskTier disk = openDiskTier(config, cacheNamespace, stateStagingBytes ? cacheDiskBytes : 0, kvSlotBytes,
                                 kvFileOptions, stateBytes, stateFileOptions,
                                 relaunchContext && memoryPlan.breakdown().kvSpill.enabled);
    if (memoryPlan.breakdown().kvSpill.enabled) {
      kvFileOpen = disk.kv != nullptr;
      if (disk.budget)
        activeLimitBytes = std::min(activeLimitBytes, disk.budget->activeLimitBytes());
      memoryPlan = requirePlan();
    }

    // The governor holds the complete Metal footprint to the hard budget. The
    // plan budgets pipelines and driver allocations inside the pipeline and
    // allocator reserves, so memory outside the backend's buffers is charged
    // only beyond them, and elastic state and KV never grow into them. The
    // split only moves bytes from KV to fixed runtime memory, so the governor
    // is the same with it or without.
    auto memoryGovernor = std::make_unique<MemoryGovernor>(*backend, memoryPlan.breakdown(), config.hostMemory);
    if (config.memoryPressure)
      memoryGovernor->setPressure(config.memoryPressure());
    // What the start allocates, the KV runway, the arenas and the warmup, is
    // memory a request in service needs: like loading, it needs only the
    // reserve macOS keeps short of critical pressure, which every Metal
    // operation checks. The host's margins and macOS's warning hold back
    // only what serving adds (beginServing).
    memoryGovernor->setServing(true);
    logLine("Kernel policy for GPU family ", device.appleGpuFamily,
            " with ", device.gpuCoreCount, " cores.");
    if (const model::RopeScaling &rope = config.model.ropeScaling; rope.yarn())
      logLine("YaRN is on: the target's RoPE is scaled by ", rope.factor, " for ",
              config.model.capabilities.maximumContextTokens, " tokens of context, past the native window of ",
              rope.originalContextTokens, ". Static YaRN changes answers at every length, so pass a --max-context "
              "above ", rope.originalContextTokens, " only when a request needs it.");
    // The split leaves the context asked for, else a relaunch's.
    const RequestedContext splitContext =
        requestedContextTokens ? RequestedContext{.tokens = requestedContextTokens}
                               : RequestedContext{.tokens = config.relaunchContext,
                                                  .relaunch = config.relaunchContext != 0};
    AneFfnStart aneFfn = startAneFfn(splitModel, config.aneFfn, splitContext, planMemory, config.cancelled);
    if (aneFfn.plan)
      memoryPlan = std::move(*aneFfn.plan);
    const EngineMemoryBreakdown &budget = memoryPlan.breakdown();
    const KvSpillPlan &spill = budget.kvSpill;
    const std::string spillOff = spillOffReason(config.model, kvLayout, spill, stateStagingBytes && kvFileOpen);
    reportSpill(config.model, budget, spillOff, activeLimitBytes);
    if (!spill.enabled) {
      if (config.kvSpillAlways)
        logLine("--kv-spill always ignored: ", config.model.name, " keeps its KV in memory, ", spillOff, ".");
      if (config.sparseKvMemory)
        logLine("--sparse-kv-memory ignored: ", config.model.name, " keeps its KV in memory, ", spillOff, ".");
    }

    // Page ids for every extent the hard budget could hold, of Index pages
    // where spilled pages hold only their index: the governor, never the id
    // range, limits the pool.
    const uint64_t idExtentBytes =
        spill.enabled ? uint64_t{budget.kvExtentPages} * budget.kvIndexPageBytes : budget.kvExtentBytes;
    const uint64_t poolExtents = std::min<uint64_t>(
        (budget.hardBudgetBytes + idExtentBytes - 1) / idExtentBytes,
        std::numeric_limits<uint32_t>::max() / budget.kvExtentPages);
    auto kvPages = std::make_unique<kv::PageStorage>(
        *backend, memoryGovernor->allocationAdmission(), kvLayout,
        static_cast<uint32_t>(poolExtents * budget.kvExtentPages), budget.kvExtentPages);
    auto kvPool = std::make_unique<KvPool>(*kvPages, model::ExecutionLimits::warmupKvPages);
    auto stateStorage = std::make_unique<model::QwenStateStorage>(
        *backend, memoryGovernor->allocationAdmission(), loaded.stateLayout(config.stateFormat),
        disk.states);
    std::unique_ptr<KvPageTier> kvTier;
    if (disk.kv)
      kvTier = std::make_unique<KvPageTier>(*kvPages, kvFormat, disk.kv, readers);
    std::optional<CacheSpillPolicy> spillPolicy;
    if (spill.enabled)
      spillPolicy = CacheSpillPolicy{.densePages = spill.densePages, .windowPages = spill.windowPages};
    auto cache = std::make_unique<engine::Cache>(*kvPool, kvTier.get(), disk.budget, spillPolicy);
    if (disk.space)
      cache->setDiskMeasure([space = disk.space.get()] { space->measure(); });
    // A target whose KV spills reads it back through one hot block cache and
    // one history stream, which take KV memory from the governor.
    std::unique_ptr<model::KvBlockCache> blockCache;
    std::unique_ptr<model::KvHistoryStream> historyStream;
    if (spill.enabled) {
      blockCache = std::make_unique<model::KvBlockCache>(
          *backend, memoryGovernor->allocationAdmission(), kvLayout, budget.kvExtentPages,
          model::KvBlockCacheOptions{.laneSlots = static_cast<uint32_t>(spill.laneSlots),
                                     .targetBytes = spill.hotTargetBytes,
                                     .readers = readers,
                                     .format = kvFormat});
      historyStream = std::make_unique<model::KvHistoryStream>(
          *backend, memoryGovernor->allocationAdmission(), kvLayout, budget.kvExtentPages, readers, kvFormat,
          spill.stagingBuffers);
    }

    if (stateStorage->actualAllocatedBytes() != 0) {
      throw std::runtime_error("lane state was allocated eagerly");
    }
    metal::MetalMemoryStats memory = backend->memoryStats();
    if (!backend->healthy()) {
      throw std::runtime_error(
          "Metal backend became unhealthy during resource allocation: " +
          backend->unhealthyReason());
    }
    if (memory.allocatedBytes > budget.hardBudgetBytes ||
        memory.deviceCurrentAllocatedBytes > budget.hardBudgetBytes) {
      throw std::runtime_error(
          "base Metal allocation exceeds immutable hard budget");
    }

    auto result = std::unique_ptr<RuntimeResources>(new RuntimeResources(
        std::move(disk.persistent), std::move(backend), std::move(readers), std::move(loaded),
        std::move(operators), std::move(memoryPlan), std::move(cacheIdentity),
        std::move(memoryGovernor), std::move(kvPages), std::move(stateStorage),
        std::move(kvTier), std::move(disk.space), std::move(kvPool), std::move(cache),
        std::move(aneFfn.split), std::move(aneFfn.outcome), std::move(blockCache),
        std::move(historyStream), config.kvSpillAlways, config.verifyRows));
    result->adoptPersistentCache();
    return result;
  } catch (const RuntimeResourcesError &) {
    throw;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(),
                                memoryPlan.breakdown().describe(),
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(),
                                memoryPlan.breakdown().describe());
  }
}

RuntimeResources::~RuntimeResources() {
  stopProbation();
  if (persistentCache_.directory) {
    persistentCache_.kv->seal();
    persistentCache_.states->seal();
  }
}

void RuntimeResources::adoptPersistentCache() {
  if (!persistentCache_.directory)
    return;
  CacheDirectory &directory = *persistentCache_.directory;
  if (!directory.coldReason().empty())
    logLine("Persistent cache emptied: ", directory.coldReason(), ".");
  std::vector<PersistedKv> blocks;
  for (const model::SlotRecord &record : persistentCache_.kv->records())
    blocks.push_back({record.label, [this, record] { return kvTier_->adopt(record); }});
  std::vector<PersistedState> states;
  for (const model::SlotRecord &record : persistentCache_.states->records())
    states.push_back({record.label, [this, record](uint32_t tokens) {
                        return stateStorage_->adopt(record, tokens);
                      }});
  const CacheAdoption adoption = cache_->adopt(std::move(blocks), std::move(states));
  persistentCache_.kv->finishAdoption();
  persistentCache_.states->finishAdoption();
  logLine("Persistent cache ", directory.path().string(), ": took back ", adoption.states,
          " restore points over ", adoption.blocks, " KV blocks (", adoption.bytes / kMiB,
          " MiB); left ", adoption.dropped, " copies behind.",
          directory.uncleanExit() ? " The last process did not stop cleanly." : "");
}

void RuntimeResources::beginServing() {
  memoryGovernor_->setServing(false);
  CacheDirectory *directory = persistentCache_.directory.get();
  if (!directory)
    return;
  try {
    directory->beginServing();
  } catch (const std::exception &error) {
    logLine("Persistent cache cannot mark this process serving (", error.what(),
            "); the next start will not know how it ended.");
    return;
  }
  probation_ = std::thread([this, directory] {
    std::unique_lock lock(probationMutex_);
    if (probationWake_.wait_until(lock, AwakeClock::now() + kProbation,
                                  [this] { return probationStopped_; }))
      return;
    try {
      directory->endProbation();
    } catch (const std::exception &error) {
      logLine("Persistent cache probation did not end (", error.what(), ").");
    }
  });
}

void RuntimeResources::engineFailed() noexcept {
  CacheDirectory *directory = persistentCache_.directory.get();
  if (!directory)
    return;
  // A probation that has not ended yet ends failed.
  stopProbation();
  try {
    directory->engineFailed();
  } catch (const std::exception &error) {
    logLine("Persistent cache cannot record the engine's failure (", error.what(), ").");
  }
}

void RuntimeResources::stopProbation() noexcept {
  {
    std::lock_guard lock(probationMutex_);
    probationStopped_ = true;
  }
  probationWake_.notify_all();
  if (probation_.joinable())
    probation_.join();
}

bool RuntimeResources::closePersistentCache() {
  if (!persistentCache_.directory)
    return true;
  stopProbation();
  cache_->relabelStates();
  // Each file runs its operations in order: the writes and labels submitted
  // before land first.
  const auto kv = persistentCache_.kv->synchronize({});
  const auto states = persistentCache_.states->synchronize({});
  const bool kvSynchronized = kv->wait();
  if (!states->wait() || !kvSynchronized)
    return false;
  try {
    persistentCache_.directory->close();
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

model::RuntimeContext RuntimeResources::modelContext() noexcept {
  return {
      *backend_,
      model_,
      *kvPages_,
      *stateStorage_,
      operators_,
      aneFfn_.get(),
      blockCache_.get(),
      historyStream_.get(),
      verifyRows_,
  };
}

EngineSpillConfig RuntimeResources::spillConfig() const noexcept {
  const KvSpillPlan &spill = memoryPlan_.breakdown().kvSpill;
  return {.capable = spill.enabled,
          .always = spillAlways_,
          .denseTokens = spill.densePages * kv::kPageTokens,
          .geometry = memoryPlan_.spillGeometry()};
}

std::optional<KvSpillStatus> RuntimeResources::kvSpillStatus() const {
  if (!memoryPlan_.breakdown().kvSpill.enabled)
    return std::nullopt;
  KvSpillStatus result{.cache = cache_->spillSnapshot()};
  if (blockCache_)
    result.blockCache = blockCache_->stats();
  if (historyStream_)
    result.historyStream = historyStream_->stats();
  return result;
}

std::optional<model::SparseRowStats> RuntimeResources::streamedTableStats() const {
  if (const model::SparseRowStore *table = model_.streamedTable()) return table->stats();
  return std::nullopt;
}

std::optional<model::ExpertStoreStats> RuntimeResources::streamedExpertStats() const {
  if (const model::ExpertStore *experts = model_.expertStore()) return experts->stats();
  return std::nullopt;
}

AneFfnSnapshot RuntimeResources::aneFfnSnapshot() const {
  if (!aneFfn_)
    return {.reason = aneFfnOutcome_.reason};
  const bool stopped = aneFfn_->retired();
  const ops::AneFfn::Served &served = aneFfn_->served();
  return {stopped ? AneFfnSnapshot::State::Stopped : AneFfnSnapshot::State::Split,
          stopped ? aneFfn_->reason() : aneFfnOutcome_.reason,
          aneFfn_->share(),
          aneFfn_->minimumRows(),
          served.commands,
          served.evaluations,
          served.milliseconds};
}

ActualMemoryReport RuntimeResources::actualMemoryReport(
    const model::ModelMemoryActual &modelMemory) const {
  ActualMemoryReport report;
  report.targetWeightsBytes = model_.targetActualAllocatedBytes();
  report.draftWeightsBytes = model_.draft.actualAllocatedBytes;
  report.visionWeightsBytes = model_.vision.actualAllocatedBytes;
  report.stateAllocatedBytes = modelMemory.stateActualAllocatedBytes;
  report.sharedPrefillBytes = modelMemory.sharedPrefillActualAllocatedBytes;
  report.sharedDecodeBytes = modelMemory.sharedDecodeActualAllocatedBytes;
  if (kvPool_->allocatedBytes() != kvPages_->actualAllocatedBytes()) {
    throw std::logic_error("the KV pool and its storage disagree on allocated extents");
  }
  report.kvAllocatedBytes = kvPool_->allocatedBytes();
  if (blockCache_)
    report.kvBlockCacheBytes = blockCache_->stats().bytes;
  if (historyStream_)
    report.kvHistoryStagingBytes = historyStream_->stats().stagingBytes;
  report.stateStagingBytes = modelMemory.stateStagingBytes;
  report.aneFfnBytes = aneFfn_ ? aneFfn_->allocatedBytes() : 0;
  if (const auto table = streamedTableStats()) report.streamedTableBytes = table->cacheBytes;
  // Optional warmup may end with a rolled-back allocation and no subsequent
  // command. Refresh the current counts after that rollback; peaks stay intact.
  metal::MetalMemoryStats memory = backend_->refreshMemoryStats();
  report.backendAllocatedBytes = memory.allocatedBytes;
  report.deviceCurrentAllocatedBytes = memory.deviceCurrentAllocatedBytes;
  report.devicePeakAllocatedBytes = memory.devicePeakAllocatedBytes;
  // A capacity-limited warmup can roll back a partial allocation before it
  // returns a result; the backend's own high-water mark keeps it.
  report.backendPeakAllocatedBytes = memory.peakAllocatedBytes;
  return report;
}

AneFfnModel aneFfnModel(metal::MetalBackend &backend, const model::LoadedModel &loaded,
                        const ops::ExecutionPlans &operators, kv::Format format, std::string_view buildId,
                        std::function<bool()> cancelled) {
  auto layers = std::make_shared<const std::vector<ops::SwiGluProjections>>(model::aneFfnLayers(loaded));
  AneFfnModel model;
  model.dense = !layers->empty();
  if (!model.dense) return model;
  if (const char *reason = ops::AneFfn::unsupported(*layers)) {
    model.unsupported = reason;
    return model;
  }
  model.unavailable = [] { return ane::unavailable(); };
  model.units = ops::AneFfn::units(*layers);
  model.plannedBytes = [layers](uint32_t aneUnits) { return ops::AneFfn::plannedBytes(*layers, aneUnits); };
  const auto onArena = [&backend, &loaded, &operators, format](const auto &use) {
    model::withPrefillArena(backend, loaded, operators, format, use);
  };
  model.time = [&backend, layers, cancelled, onArena] {
    ops::ane_ffn::Timings timings;
    onArena([&](const ops::PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden) {
      timings = ops::ane_ffn::Measurement(backend, *layers, ffn, hidden, cancelled).time();
    });
    return timings;
  };
  model.prepare = [&backend, layers, cancelled, onArena](uint32_t aneUnits, bool timeChunks) {
    AneFfnPrepared prepared;
    onArena([&](const ops::PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden) {
      prepared.split = std::make_unique<ops::AneFfn>(backend, *layers, aneUnits, cancelled);
      prepared.error = prepared.split->verify(*layers, ffn, hidden);
      if (timeChunks)
        prepared.chunks =
            ops::ane_ffn::Measurement(backend, *layers, ffn, hidden, cancelled).chunks(*prepared.split);
    });
    return prepared;
  };
  const std::string key = calibrationKey(*layers) + " device " + backend.capabilities().deviceName + " macos " +
                          [[NSProcessInfo processInfo] operatingSystemVersionString].UTF8String + " build " +
                          std::string(buildId) + " most ";
  model.recall = [key](uint32_t maxAneUnits) -> std::optional<ops::ane_ffn::Calibration> {
    const std::optional<std::string> line = ane::recall(key + std::to_string(maxAneUnits));
    return line ? ops::ane_ffn::Calibration::parse(*line) : std::nullopt;
  };
  model.remember = [key](uint32_t maxAneUnits, const ops::ane_ffn::Calibration &calibration) {
    ane::remember(key + std::to_string(maxAneUnits), calibration.text());
  };
  model.forget = [key](uint32_t maxAneUnits) { ane::forget(key + std::to_string(maxAneUnits)); };
  model.healthy = [&backend] { return backend.healthy(); };
  return model;
}

void connectToGovernor(EngineConfig &config, MemoryGovernor &governor) {
  config.growthPaused = [&governor] {
    return !governor.snapshot().hostGrowthAllowed;
  };
  config.serving = [&governor](bool serving) { return governor.setServing(serving); };
}

} // namespace splash::engine
