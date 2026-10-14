#pragma once

// Enabled-history reservation tests reuse the engine suite's request/tick
// helpers and reservation-aware TestExecutor.
struct HistoryReservationFixture {
  test::TestKvStorage storage{4096, 1, 4};
  KvPool pool{storage, 0};
  Cache cache{pool, nullptr, nullptr, {8192, 2, 128}};
  Executor executor{1};
  Events events;
  Engine engine;
  uint64_t ceiling = 8192, freshStateBytes = 0, peakBytes = 0;

  explicit HistoryReservationFixture(EngineConfig config = test::engineConfig())
      : engine(std::move(config), cache, executor, events) {
    executor.kv = &storage;
    guardReleases(storage, engine);
    executor.reservationRoom = [&] { return reserveBytes(); };
    storage.growthAllowed = [&](uint32_t) {
      const uint64_t bytes = storage.extentPages() * storage.bytesPerPage();
      const bool fits = bytes <= ceiling - std::min(ceiling, used());
      if (fits)
        peakBytes = std::max(peakBytes, used() + bytes);
      return fits;
    };
  }
  // Cached states and outstanding reservations return their 64 bytes to the
  // executor's pool. Those bytes remain charged until reclaimIdleState frees
  // them; a later reserve consumes the pooled bytes without another admission.
  uint64_t used() const {
    require(freshStateBytes >= executor.reclaimedIdleStateBytes,
            "shared-byte state account underflow");
    return storage.allocatedPages() * storage.bytesPerPage() + freshStateBytes -
           executor.reclaimedIdleStateBytes;
  }
  bool reserveBytes() {
    if (*executor.evictedStateBytes >= 64) {
      *executor.evictedStateBytes -= 64;
      return true;
    }
    if (64 > ceiling - std::min(ceiling, used()))
      return false;
    freshStateBytes += 64;
    peakBytes = std::max(peakBytes, used());
    return true;
  }
  StateCheckpoint seed(uint64_t id, const std::vector<uint32_t> &prompt, bool retained = true) {
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 8192).granted(), "seed KV admission failed");
    const auto table = cache.pageTable(id);
    for (uint32_t page = 0; page < 8192 / KvCache::pageTokens; ++page)
      storage.content.at(table.pages[page]) =
          Executor::rows(page, prompt[page * KvCache::pageTokens]);
    const uint64_t block = test::publishBlocks(cache, id, prompt, 8192);
    require(reserveBytes(), "seed state admission failed");
    cache.publishCompositeState(block, std::make_shared<State>(executor.evictedStateBytes), true);
    cache.endRequest(id);
    const auto point = cache.checkpointState(block);
    require(!retained || cache.retainCheckpointState(point), "seed history not retained");
    return point;
  }
  void drain() {
    require(idle(engine), "test left an active command");
    require(used() <= ceiling, "shared byte limit exceeded");
    const auto history = cache.snapshot().stateCache;
    require(history.historyEntries <= 2 && history.historyBytes <= 128, "history quota exceeded");
    static_cast<void>(cache.evictAll());
    static_cast<void>(executor.reclaimIdleState(false, model::IdleMemory::Buffers));
    require(used() == 0 && cache.snapshot().stateCache.entries == 0,
            "shared byte account did not drain");
  }
};

// Retention is decided only after a successful command. In particular, a
// refused reservation must not turn the lane's recovery point into a victim.
void testHistoryRetainsOnlyAfterTheReplacementCommand() {
  enum class End { Complete, Cancelled, Expired, Failed };
  for (const bool refused : {false, true}) {
    for (const End end : {End::Complete, End::Cancelled, End::Expired, End::Failed}) {
      HistoryReservationFixture f;
      const std::vector<uint32_t> prompt(25001, 701);
      auto first = request(1, prompt);
      first.deadlineMilliseconds = 1000;
      f.engine.submit(std::move(first));
      double now = 1;
      tickUntil(f.engine, now, [&] {
        return !f.engine.commandInFlight() && f.executor.requests.contains(1) &&
               f.executor.requests.at(1).position == 10240;
      }, "retention fixture did not reach the reservation seam");
      const auto owner = f.cache.checkpointState(f.cache.blockAt(1, 8192));
      require(owner && f.cache.snapshot().stateCache.historyEntries == 0,
              "checkpoint marked before its replacement");
      if (refused)
        f.executor.deniedReservations = 1;
      if (end == End::Failed)
        f.executor.nonFinitePrefills.insert(1);
      f.executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
      require(f.engine.tick(now++) && f.engine.commandInFlight(),
              "replacement command did not start");
      require(f.cache.snapshot().stateCache.historyEntries == 0 &&
                  f.cache.checkpointState(owner.kvBlock).publication == owner.publication &&
                  f.engine.snapshot().checkpointPublications == 2 &&
                  f.engine.snapshot().recycledStatePublications == 0,
              "reservation marked or retired the recovery point before its command");
      if (end == End::Cancelled)
        f.engine.cancel(1);
      if (end == End::Expired)
        static_cast<void>(f.engine.tick(now = 1000));
      *f.executor.holdPrefillUntil = true;
      tickUntil(f.engine, now, [&] { return !f.engine.commandInFlight(); },
                "replacement command did not finish");
      require(f.cache.checkpointState(owner.kvBlock).publication == owner.publication &&
                  f.cache.snapshot().stateCache.historyEntries == (end == End::Complete ? 1U : 0U) &&
                  f.engine.snapshot().checkpointPublications == (end == End::Complete ? 3U : 2U),
              "replacement completion or interruption changed the wrong checkpoint lifetime");
      f.executor.holdPrefillUntil.reset();
      if (end == End::Complete)
        f.engine.cancel(1);
      runUntilIdle(f.engine);
      f.drain();
    }
  }
}

// The optional victim is leased when reserve is refused. It can yield only
// after the command completes and the lease releases. Without it, the native
// handoff still takes the lane's own checkpoint buffers after completion.
void testHistoryPostCommandSnapshotYieldAndNativeHandoff() {
  for (const bool spareEligible : {true, false}) {
    HistoryReservationFixture f;
    const auto sparePrompt = std::vector<uint32_t>(8193, 714);
    const auto spare = f.seed(900, sparePrompt);
    auto lease = f.cache.lookup(sparePrompt, {});
    const std::vector<uint32_t> prompt(25001, 715);
    f.engine.submit(request(1, prompt));
    double now = 1;
    tickUntil(f.engine, now, [&] {
      return !f.engine.commandInFlight() && f.executor.requests.contains(1) &&
             f.executor.requests.at(1).position == 10240;
    }, "snapshot fixture did not reach its checkpoint seam");
    const auto owner = f.cache.checkpointState(f.cache.blockAt(1, 8192));
    static_cast<void>(f.executor.reclaimIdleState(false, model::IdleMemory::Buffers));
    f.executor.deniedReservations = 1;
    f.executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
    require(f.engine.tick(now++) && f.engine.commandInFlight() &&
                f.cache.checkpointState(owner.kvBlock).publication == owner.publication &&
                f.cache.stateResident(spare.kvBlock) && f.executor.snapshots == 0,
            "refused reserve changed protected recovery before the command");
    f.ceiling = f.used();
    if (spareEligible)
      lease = {};
    *f.executor.holdPrefillUntil = true;
    tickUntil(f.engine, now, [&] { return !f.engine.commandInFlight(); },
              "post-command snapshot did not finish");
    require(f.executor.snapshots == 1 && f.engine.snapshot().checkpointPublications == 3 &&
                f.engine.snapshot().checkpointPublicationFailures == 0 &&
                f.cache.stateResident(owner.kvBlock) == spareEligible &&
                f.cache.stateResident(spare.kvBlock) == !spareEligible &&
                f.engine.snapshot().recycledStatePublications == (spareEligible ? 1U : 0U) &&
                f.cache.snapshot().stateCache.checkpointEvictions == (spareEligible ? 1U : 0U) &&
                f.cache.probe(prompt, {}).cachedTokens() == 12288 && f.used() <= f.ceiling,
            "post-command fallback lost recovery instead of eligible history or blocked native handoff");
    lease = {};
    f.executor.holdPrefillUntil.reset();
    f.engine.cancel(1);
    runUntilIdle(f.engine);
    f.drain();
  }
}

// A cause-independent refusal can persist after optional history yields and
// the native handoff. Do not require history loss; require a bounded attempt
// count, no false publication and continued foreground progress.
void testHistoryPostCommandRefusalRemainsBounded() {
  HistoryReservationFixture f;
  const std::vector<uint32_t> prompt(25001, 718);
  const auto owner = f.seed(900, prompt, false);
  const std::vector<uint32_t> sparePrompt(8193, 719);
  f.seed(901, sparePrompt);
  auto lease = f.cache.lookup(sparePrompt, {});
  f.engine.submit(request(1, prompt));
  double now = 1;
  tickUntil(f.engine, now, [&] {
    return !f.engine.commandInFlight() && f.executor.requests.contains(1) &&
           f.executor.requests.at(1).position == 10240;
  }, "refusal fixture did not reach its replacement command");
  const uint32_t attempts = f.executor.reservationAttempts;
  f.executor.denyReservationAt = 12288;
  f.executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  require(f.engine.tick(now++) && f.engine.commandInFlight() &&
              f.cache.checkpointState(owner.kvBlock).publication == owner.publication,
          "refused replacement lost its recovery point before completion");
  lease = {};
  *f.executor.holdPrefillUntil = true;
  tickUntil(f.engine, now, [&] { return !f.engine.commandInFlight(); },
            "refused post-command snapshot did not return");
  const uint32_t tried = f.executor.reservationAttempts - attempts;
  const auto state = f.cache.snapshot().stateCache;
  require(tried >= 2 && tried <= 4 && f.executor.snapshots == 0 &&
              f.engine.snapshot().checkpointPublicationFailures == 1 &&
              f.engine.snapshot().checkpointPublications == 0 &&
              f.engine.snapshot().recycledStatePublications == 0 &&
              state.historyEntries <= 2 && state.historyBytes <= 128 &&
              state.checkpointEvictions <= 2 && f.events.failedCount == 0 &&
              f.executor.requests.at(1).position == 12288,
          "post-command refusal retried beyond history and native handoff or stopped foreground");
  f.executor.holdPrefillUntil.reset();
  f.engine.cancel(1);
  runUntilIdle(f.engine);
  f.drain();
}

// Native host-pressure reclaim keeps its class ordering and newest replay
// protection. History accounting drops only when its checkpoint is reclaimed.
void testHistoryRemainsANativePressureVictim() {
  HistoryReservationFixture f;
  const auto optional = f.seed(900, std::vector<uint32_t>(8193, 716));
  const std::vector<uint32_t> recent(65, 717);
  f.engine.submit(request(1, recent));
  runUntilIdle(f.engine);
  const auto before = f.cache.snapshot().stateCache;
  const auto result = f.engine.reclaimMemory({.targetBytes = 64, .keepResumePoint = true});
  require(result.outcome == ReclaimOutcome::Met && result.releasedBytes >= 64 &&
              !f.cache.stateResident(optional.kvBlock) &&
              f.cache.snapshot().stateCache.historyEntries == 0 &&
              f.cache.snapshot().stateCache.historyBytes == 0 &&
              f.cache.snapshot().stateCache.checkpointEvictions == before.checkpointEvictions + 1 &&
              f.cache.probe(recent, {}).cachedTokens() == 64,
          "history bypassed native pressure accounting, class ordering, or newest replay protection");
  f.drain();
}

void testHistoryBytePressureYieldsBeforePublicationAndCancellationCountsNothing() {
  for (bool cancel : {false, true}) {
    HistoryReservationFixture f;
    const auto spare = f.seed(900, std::vector<uint32_t>(8193, 702));
    // One 4-byte KV extent fits, but another 64-byte state allocation does not.
    f.ceiling = f.used() + 4;
    f.executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
    f.engine.submit(request(1, std::vector<uint32_t>(65, 703)));
    require(f.engine.tick(1) && f.engine.commandInFlight(), "pressure capture did not submit");
    require(!f.cache.stateResident(spare.kvBlock) && f.executor.reservationAttempts == 2 &&
                f.engine.snapshot().recycledStatePublications == 0 &&
                f.cache.snapshot().stateCache.checkpointEvictions == 1 && f.used() <= f.ceiling,
            "optional yield did not release real pooled bytes before the command");
    if (cancel)
      f.engine.cancel(1);
    *f.executor.holdPrefillUntil = true;
    runUntilIdle(f.engine);
    require(f.engine.snapshot().recycledStatePublications == (cancel ? 0U : 1U) &&
                f.engine.snapshot().replayStatePublications == (cancel ? 0U : 1U) &&
                f.events.failedCount == 0,
            "recycled publication accounting happened before a successful publication");
    f.drain();
  }
}

void testHistoryPinnedPredecessorDoesNotYieldAnotherState() {
  HistoryReservationFixture f;
  const std::vector<uint32_t> prompt(25001, 704);
  const auto owner = f.seed(900, prompt, false);
  const auto spare = f.seed(901, std::vector<uint32_t>(8193, 705));
  auto pinned = f.cache.lookup(prompt, {});
  require(pinned.resumeBoundary() == 8192, "fixture failed to pin the predecessor");
  // A boundary-specific refusal, not a memory-pressure model. No optional
  // victim is useful: the pinned predecessor forbids checkpoint publication.
  f.executor.denyReservationAt = 12288;
  f.engine.submit(request(1, prompt));
  double now = 1;
  tickUntil(
      f.engine, now, [&] { return f.engine.snapshot().checkpointPublicationFailures == 1; },
      "pinned predecessor did not reject the replacement");
  require(
      f.cache.checkpointState(owner.kvBlock).publication == owner.publication &&
          f.cache.stateResident(spare.kvBlock) &&
          f.cache.snapshot().stateCache.historyEntries == 1 &&
          f.cache.snapshot().stateCache.evictions == 0 &&
          f.engine.snapshot().recycledStatePublications == 0,
      "pinned predecessor caused unnecessary optional yield or recycled-publication accounting");
  f.engine.cancel(1);
  pinned = {};
  runUntilIdle(f.engine);
  f.drain();
}

void testEnabledHistoryDoesNotCaptureASupersededPlannedBoundary() {
  HistoryReservationFixture f(test::engineConfig({.prefillCheckpointTokens = 3072}));
  const auto spare = f.seed(900, std::vector<uint32_t>(8193, 706));
  std::vector<uint32_t> prompt(6001);
  std::iota(prompt.begin(), prompt.end(), 1);
  std::vector<uint32_t> other(prompt.begin(), prompt.begin() + 3584);
  other.resize(3685, 0);
  f.engine.submit(request(1, other));
  runUntilIdle(f.engine);
  f.engine.submit(request(2, prompt));
  runUntilIdle(f.engine);
  require(
      capturedStates(f.executor, 2) ==
              std::vector<std::pair<uint32_t, uint64_t>>{{3584, 4096}, {5984, 6001}} &&
          f.engine.snapshot().checkpointPublications == 0 && f.engine.snapshot().captureCuts == 0 &&
          f.cache.stateResident(spare.kvBlock) && f.cache.snapshot().stateCache.historyEntries == 1,
      "enabled history manufactured a capture at a planned checkpoint superseded inside a command");
  f.drain();
}

void testHistoryYieldBeforeReplayReservationRetarget() {
  // The preemption fixture puts checkpoint 6144 and replay point 6656 in
  // the same resumed command. An unrelated retained 8192 state is optional.
  bool paused = false;
  auto config = test::engineConfig({.prefillCheckpointTokens = 3072});
  config.growthPaused = [&] { return paused; };
  HistoryReservationFixture f(config);
  f.executor.decodeFinishes = false;
  f.executor.decodeTokens = 8;
  std::vector<uint32_t> prompt(6657);
  std::iota(prompt.begin(), prompt.end(), 1);
  double now = 1;
  suspendWithoutTheReplayPoint(f.engine, f.cache, f.executor, f.events, now, prompt, 1600);
  tickUntil(
      f.engine, now, [&] { return f.events.usage.contains(2); },
      "foreground preemptor did not finish");
  const auto spare = f.seed(900, std::vector<uint32_t>(8193, 707));
  // Reserve denies the replay exactly once, independently of the byte budget.
  // This isolates the direct second-reservation seam before native retargeting.
  bool deniedReplay = false;
  f.executor.reservationRoom = [&] {
    if (!deniedReplay && f.executor.lastReservationTokens == 6656) {
      deniedReplay = true;
      return false;
    }
    return f.reserveBytes();
  };
  tickUntil(
      f.engine, now,
      [&] { return f.executor.prefillChunks.at(1) == 6 && !f.engine.commandInFlight(); },
      "resumed command did not complete");
  const auto captures = capturedStates(f.executor, 1);
  require(
      deniedReplay && captures.size() >= 2 &&
          captures[captures.size() - 2] == std::pair<uint32_t, uint64_t>{6144, 7168} &&
          captures.back() == std::pair<uint32_t, uint64_t>{6656, 7168} &&
          !f.cache.stateResident(spare.kvBlock) &&
          f.engine.snapshot().recycledStatePublications == 1 &&
          f.cache.probe(prompt, {}).cachedTokens() == 6656,
      "optional history did not yield before the replay would retarget its command's checkpoint");
  f.executor.reservationRoom = [&] { return f.reserveBytes(); };
  f.executor.decodeFinishes = true;
  runUntilIdle(f.engine);
  f.drain();
}

void testHistoryReservationYieldSkipsAnUnleasedActivePeer() {
  HistoryReservationFixture f;
  f.executor.maximumLanes = 2;
  const std::vector<uint32_t> backgroundPrompt(25001, 708);
  const auto owner = f.seed(900, backgroundPrompt);
  const auto spare = f.seed(901, std::vector<uint32_t>(8193, 709));
  auto background = request(1, backgroundPrompt);
  background.priority = RequestPriority::Background;
  f.engine.submit(std::move(background));
  require(f.engine.tick(1) && f.engine.commandInFlight() && f.events.starts.back() == 8192 &&
              f.cache.snapshot().stateCache.pinned == 0,
          "peer fixture did not own an unleased checkpoint");
  require(f.cache.reuseStoredState(spare.kvBlock, true), "peer fixture lost spare history");
  f.ceiling = f.used() + 4;
  auto foreground = request(2, std::vector<uint32_t>(65, 710));
  foreground.priority = RequestPriority::Foreground;
  f.engine.submit(std::move(foreground));
  double now = 2;
  tickUntil(
      f.engine, now, [&] { return f.events.usage.contains(2); },
      "history yield blocked foreground");
  require(f.cache.checkpointState(owner.kvBlock).publication == owner.publication &&
              !f.cache.stateResident(spare.kvBlock) &&
              f.cache.snapshot().stateCache.historyEntries == 1 &&
              f.cache.snapshot().stateCache.checkpointEvictions == 1 &&
              f.engine.snapshot().recycledStatePublications == 1 && f.events.failedCount == 0 &&
              !f.events.usage.contains(1) && f.used() <= f.ceiling,
          "reservation history yield took an unleased active owner or blocked foreground");
  f.engine.cancel(1);
  runUntilIdle(f.engine);
  f.drain();
}

void testReservationDenialBoundsHistoryRetriesAndPreservesFallback() {
  auto config = test::engineConfig();
  config.growthPaused = [] { return true; };
  HistoryReservationFixture f(config);
  f.seed(900, std::vector<uint32_t>(8193, 711));
  f.seed(901, std::vector<uint32_t>(8193, 712));
  f.executor.denyReservationAt = 64;
  f.engine.submit(request(1, std::vector<uint32_t>(65, 713)));
  runUntilIdle(f.engine);
  const uint32_t refusedAttempts = f.executor.reservationAttempts;
  const auto refusedHistory = f.cache.snapshot().stateCache;
  require(refusedAttempts >= 1 && refusedAttempts <= 3 && f.executor.reservations == 0 &&
              refusedHistory.historyEntries <= 2 && refusedHistory.historyBytes <= 128 &&
              refusedHistory.checkpointEvictions <= 2 &&
              f.engine.snapshot().replayStatePublicationFailures == 1 &&
              f.engine.snapshot().recycledStatePublications == 0 && f.events.failedCount == 0,
          "non-memory refusal retried beyond optional history or counted an unpublished capture");
  f.executor.denyReservationAt.reset();
  f.engine.submit(request(2, std::vector<uint32_t>(65, 713)));
  runUntilIdle(f.engine);
  require(f.executor.reservationAttempts == refusedAttempts + 1 &&
              f.cache.probe(std::vector<uint32_t>(65, 713), {}).cachedTokens() == 64,
          "bounded refusal prevented later publication");
  f.drain();
}

void testCheckpointHistoryReservationSeams() {
  testHistoryRetainsOnlyAfterTheReplacementCommand();
  testHistoryPostCommandSnapshotYieldAndNativeHandoff();
  testHistoryPostCommandRefusalRemainsBounded();
  testHistoryRemainsANativePressureVictim();
  testHistoryBytePressureYieldsBeforePublicationAndCancellationCountsNothing();
  testHistoryPinnedPredecessorDoesNotYieldAnotherState();
  testEnabledHistoryDoesNotCaptureASupersededPlannedBoundary();
  testHistoryYieldBeforeReplayReservationRetarget();
  testHistoryReservationYieldSkipsAnUnleasedActivePeer();
  testReservationDenialBoundsHistoryRetriesAndPreservesFallback();
}
