#include "TestChecks.hpp"
#include "engine/KvCache.hpp"
#include "TestKvPool.hpp"

#include <array>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <random>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;

std::array<uint32_t, KvCache::pageTokens> page(uint32_t token) {
  std::array<uint32_t, KvCache::pageTokens> result{};
  result.fill(token);
  return result;
}

void testImageIdentityKeysBlocks() {
  test::TestKvStorage storage(8, 100, 1);
  KvPool pool(storage, 8);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(2);
  require(acquired.granted() && acquired.pages.size() == 2,
          "test pages were not acquired");

  // Two images with the same grid and placeholder tokens differ only by
  // content digest; text-only lookups carry the zero identity.
  const ImageSpan red{8, 16, 8, 8, 0x1111, 0x2222};
  const ImageSpan blue{8, 16, 8, 8, 0x3333, 0x4444};
  const std::array<ImageSpan, 1> redSpans{red};
  const std::array<ImageSpan, 1> blueSpans{blue};
  require(blockImageIdentity(0, 32, {}) == BlockIdentity{},
          "text-only block identity must be zero");
  require(blockImageIdentity(32, 32, redSpans) == BlockIdentity{},
          "a block after the image must not carry image identity");
  const BlockIdentity redIdentity = blockImageIdentity(0, 32, redSpans);
  const BlockIdentity blueIdentity = blockImageIdentity(0, 32, blueSpans);
  require(redIdentity != BlockIdentity{} && redIdentity != blueIdentity,
          "image identity does not depend on the content digest");
  const ImageSpan redLater{40, 16, 8, 8, 0x1111, 0x2222};
  const std::array<ImageSpan, 1> laterSpans{redLater};
  require(blockImageIdentity(32, 32, laterSpans) != redIdentity,
          "image identity ignores the block's alignment inside the image");
  // Of sorted spans, only those that overlap the block make its identity:
  // not those that end before it, nor those that start after it.
  const ImageSpan first{0, 8, 4, 8, 0x5555, 0x6666};
  const std::array<ImageSpan, 3> threeSpans{first, blue, redLater};
  const std::array<ImageSpan, 1> firstSpan{first};
  require(blockImageIdentity(32, 32, threeSpans) == blockImageIdentity(32, 32, laterSpans) &&
              blockImageIdentity(0, 8, threeSpans) == blockImageIdentity(0, 8, firstSpan),
          "a span outside the block changed its identity");

  const auto tokens = page(248056);
  auto redBlock = cache.insert(0, tokens, acquired.pages[0], redIdentity);
  require(!cache.find(0, tokens, {}), "text-only lookup matched an image block");
  require(!cache.find(0, tokens, blueIdentity),
          "a different image matched a token-identical block");
  auto found = cache.find(0, tokens, redIdentity);
  require(found && found->id == redBlock.id,
          "identical image content did not match its block");
  auto blueBlock = cache.insert(0, tokens, acquired.pages[1], blueIdentity);
  require(blueBlock.id != redBlock.id,
          "token-identical blocks with different images were merged");
  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  std::cout << "KV image identity ok\n";
}

// A persistent tier's recorded block identities must match a later process's lookups; a change to
// blockIdentity bumps kPersistentCacheFormat and pins the new values here.
void testBlockIdentitiesArePinned() {
  const CacheSalt salt{0x0123456789abcdefULL, 0xfedcba9876543210ULL};
  // Two images in the first block, the second through the third.
  const std::array<ImageSpan, 2> images{
      ImageSpan{8, 16, 8, 8, 0x1111222233334444ULL, 0x5555666677778888ULL},
      ImageSpan{24, 72, 12, 24, 0x99990000aaaabbbbULL, 0xccccddddeeeeffffULL}};
  require(blockIdentity(0, 32, {}, salt) ==
                  BlockIdentity{0x7ebd72de8d5302c4ULL, 0xfc6aad21c68d9802ULL} &&
              blockIdentity(0, 32, images, {}) ==
                  BlockIdentity{0x9979f264297f2e0dULL, 0x2a53365bb9c20351ULL} &&
              blockIdentity(32, 32, images, {}) ==
                  BlockIdentity{0xbb5b6de52eeab8baULL, 0xfa8765be1d4ad307ULL} &&
              blockIdentity(64, 32, images, {}) ==
                  BlockIdentity{0xbb5b6de52eeab89aULL, 0xfa8765be1d4ad327ULL} &&
              blockIdentity(0, 32, images, salt) ==
                  BlockIdentity{0xa23cc0b0f9bab89dULL, 0x386a63737598e476ULL} &&
              blockIdentity(32, 32, images, salt) == blockIdentity(32, 32, images, {}),
          "block identities changed: persisted salted and image chains would match "
          "nothing; bump kPersistentCacheFormat and pin the new values");
  std::cout << "KV block identities pinned ok\n";
}

void testExactChainedBlocksAndPhysicalOwnership() {
  test::TestKvStorage storage(8, 100, 1);
  KvPool pool(storage, 8);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(4);
  require(acquired.granted() && acquired.pages.size() == 4,
          "test pages were not acquired");

  const auto rootTokens = page(11);
  const auto leftTokens = page(12);
  const auto rightTokens = page(13);
  const auto otherTokens = page(14);
  auto root = cache.insert(0, rootTokens, acquired.pages[0], {});
  auto left = cache.insert(root.id, leftTokens, acquired.pages[1], {});
  auto right = cache.insert(root.id, rightTokens, acquired.pages[2], {});
  auto other = cache.insert(0, otherTokens, acquired.pages[3], {});
  require(pool.snapshot().pagesPrefix == 4,
          "page cache did not retain four physical blocks");

  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  require(pool.snapshot().pagesPrefix == 4 &&
              pool.snapshot().pagesActive == 0,
          "page cache ownership was not isolated from active requests");

  auto foundLeft = cache.find(root.id, leftTokens, {});
  auto wrongTokens = leftTokens;
  wrongTokens.back() ^= 1;
  require(foundLeft && foundLeft->id == left.id &&
              foundLeft->physicalPage == acquired.pages[1] &&
              !cache.find(root.id, wrongTokens, {}),
          "page lookup trusted a hash collision");
  const KvCache::Chain chain = cache.chain(left.id);
  require(chain.blocks == std::vector<uint64_t>{root.id, left.id} &&
              chain.pages == std::vector<uint32_t>{acquired.pages[0],
                                                   acquired.pages[1]} &&
              cache.chainLength(left.id) == 2,
          "page cache did not reconstruct the exact parent chain");

  auto duplicate = cache.insert(root.id, leftTokens, acquired.pages[1], {});
  require(duplicate.id == left.id &&
              duplicate.physicalPage == acquired.pages[1] &&
              pool.snapshot().pagesPrefix == 4,
          "exact duplicate created a second KV block");

  rejects([&] { cache.erase(root.id); }, "cannot evict a referenced KV cache block",
          "parent block was evicted before children");
  cache.retainActive(left.id);
  require(cache.evictionCandidate(0).value().id == right.id &&
              cache.evictionCandidate(right.id).value().id == other.id &&
              !cache.evictionCandidate(other.id),
          "active page block remained evictable");
  cache.releaseActive(left.id);
  require(cache.evictionCandidate(0).value().id == right.id &&
              cache.evictionCandidate(right.id).value().id == other.id &&
              cache.evictionCandidate(other.id).value().id == left.id &&
              !cache.evictionCandidate(left.id),
          "leaf eviction did not follow deterministic LRU order");
  cache.touch(right.id);
  require(cache.evictionCandidate(0).value().id == other.id &&
              cache.evictionCandidate(other.id).value().id == left.id &&
              cache.evictionCandidate(left.id).value().id == right.id &&
              !cache.evictionCandidate(right.id),
          "touch did not refresh the matched leaf without walking its chain");

  cache.erase(right.id);
  cache.erase(left.id);
  cache.erase(root.id);
  cache.erase(other.id);
  require(pool.snapshot().pagesPrefix == 0 && pool.freePageCount() == pool.pageCount(),
          "page cache erase leaked physical references");
}

void testErasedLeafParentInheritsRecency() {
  test::TestKvStorage storage(8, 100, 1);
  KvPool pool(storage, 8);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(4);
  require(acquired.granted() && acquired.pages.size() == 4,
          "test pages were not acquired");

  // Chain A is inserted first and never touched again; chain B is newer.
  const auto a0Tokens = page(21);
  const auto a1Tokens = page(22);
  const auto b0Tokens = page(23);
  const auto b1Tokens = page(24);
  auto a0 = cache.insert(0, a0Tokens, acquired.pages[0], {});
  auto a1 = cache.insert(a0.id, a1Tokens, acquired.pages[1], {});
  auto b0 = cache.insert(0, b0Tokens, acquired.pages[2], {});
  auto b1 = cache.insert(b0.id, b1Tokens, acquired.pages[3], {});
  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  require(cache.evictionCandidate(0).value().id == a1.id &&
              cache.evictionCandidate(a1.id).value().id == b1.id,
          "leaf order did not start with the cold chain");

  // Erasing A's leaf exposes A's root, which is still colder than B's leaf.
  cache.erase(a1.id);
  require(cache.evictionCandidate(0).value().id == a0.id &&
              cache.evictionCandidate(a0.id).value().id == b1.id &&
              !cache.evictionCandidate(b1.id),
          "exposed parent jumped ahead of a warmer chain");
  cache.erase(a0.id);
  cache.erase(b1.id);
  require(cache.evictionCandidate(0).value().id == b0.id &&
              !cache.evictionCandidate(b0.id),
          "last exposed parent was not the sole candidate");
  cache.erase(b0.id);
  require(pool.snapshot().pagesPrefix == 0 &&
              pool.freePageCount() == pool.pageCount(),
          "recency test leaked physical references");
}

void testInputValidation() {
  test::TestKvStorage storage(2, 100, 1);
  KvPool pool(storage, 2);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(1);
  require(acquired.granted(), "validation page was not acquired");
  const std::array<uint32_t, 1> shortTokens{1};
  rejects(
      [&] {
        static_cast<void>(
            cache.insert(0, shortTokens, acquired.pages[0], {}));
      },
      "KV cache block must contain one full page",
      "partial token page was accepted");
  rejects(
      [&] {
        static_cast<void>(
            cache.insert(999, page(2), acquired.pages[0], {}));
      },
      "KV cache parent block is unknown",
      "nonresident parent was accepted");
  pool.releasePage(acquired.pages[0], false);
  // The pool tells one cache of its pages' activity: a second cache on it
  // would leave the first's counts behind.
  CacheRecency otherRecency;
  rejects([&] { KvCache second(pool, otherRecency); }, "already has its observer",
          "a second KV cache on one pool was accepted");
}

void testCandidateOrderThroughChurn() {
  constexpr uint32_t count = 256;
  test::TestKvStorage storage(count, 100, 1);
  KvPool pool(storage, count);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(count);
  require(acquired.granted(), "churn test pages were not acquired");
  struct Reference {
    uint64_t parent = 0;
    uint64_t lastUsed = 0;
    uint32_t children = 0;
    uint32_t activeUsers = 0;
    bool resident = true;
  };
  std::array<Reference, count + 1> entries{};
  uint64_t clock = 0;
  std::mt19937 random(571);
  for (uint32_t index = 0; index < count; ++index) {
    const uint64_t parent = index % 3 ? 1 + random() % index : 0;
    const auto inserted = cache.insert(parent, page(index + 1),
                                       acquired.pages[index], {});
    require(inserted.id == index + 1, "unexpected test block identity");
    entries[inserted.id] = {parent, ++clock};
    if (parent) ++entries[parent].children;
    pool.releasePage(acquired.pages[index], false);
  }
  auto checkOrder = [&] {
    std::vector<std::pair<uint64_t, uint64_t>> expected;
    for (uint64_t id = 1; id <= count; ++id) {
      const auto &entry = entries[id];
      if (entry.resident && !entry.activeUsers && !entry.children)
        expected.emplace_back(entry.lastUsed, id);
    }
    std::sort(expected.begin(), expected.end());
    auto candidate = cache.evictionCandidate(0);
    for (const auto &[lastUsed, id] : expected) {
      require(candidate && candidate->id == id &&
                  candidate->lastUsed == lastUsed,
              "candidate index disagrees with exact reference recency");
      candidate = cache.evictionCandidate(id);
    }
    require(!candidate, "candidate index retained a pinned/non-leaf block");
  };
  for (uint32_t step = 0; step < 8192; ++step) {
    const uint64_t id = 1 + random() % count;
    auto &entry = entries[id];
    if (!entry.resident) continue;
    switch (random() % 3) {
    case 0:
      cache.touch(id);
      entry.lastUsed = ++clock;
      break;
    case 1:
      if (!entry.activeUsers) {
        cache.retainActive(id);
        ++entry.activeUsers;
        entry.lastUsed = ++clock;
      } else {
        cache.releaseActive(id);
        --entry.activeUsers;
        if (!entry.children) entry.lastUsed = ++clock;
      }
      break;
    case 2:
      if (entry.activeUsers || entry.children) break;
      cache.erase(id);
      entry.resident = false;
      if (entry.parent) {
        auto &parent = entries[entry.parent];
        --parent.children;
        if (!parent.children && !parent.activeUsers)
          parent.lastUsed = std::max(parent.lastUsed, entry.lastUsed);
      }
      break;
    }
    checkOrder();
  }
  for (uint64_t id = 1; id <= count; ++id) {
    if (entries[id].resident && entries[id].activeUsers)
      cache.releaseActive(id);
  }
  while (auto candidate = cache.evictionCandidate(0)) cache.erase(candidate->id);
  require(pool.snapshot().pagesPrefix == 0 && pool.freePageCount() == count,
          "candidate churn leaked a block or physical reference");
}

// On random trees grown by inserts and shrunk by leaf erasures, subtree() returns every block below
// a block, each before its parent, or nothing while one of them is in use.
void testSubtreeThroughChurn() {
  constexpr uint32_t steps = 1000;
  test::TestKvStorage storage(steps, 100, 1);
  KvPool pool(storage, steps);
  CacheRecency recency;
  KvCache cache(pool, recency);
  struct Reference {
    uint64_t parent = 0;
    bool live = false;
    bool used = false;
  };
  std::vector<Reference> blocks(steps + 1);
  uint64_t inserted = 0;
  std::mt19937 random(907);
  for (uint32_t step = 0; step < steps; ++step) {
    const uint64_t pick = inserted ? 1 + random() % inserted : 0;
    const bool live = pick && blocks[pick].live;
    switch (random() % 5) {
    case 0:
    case 1: {
      // A resident block owns its page, so each block takes the next one;
      // only the shape of the tree matters here.
      const uint64_t parent = live ? pick : 0;
      const auto result =
          cache.insert(parent, page(step + 1), static_cast<uint32_t>(inserted), {});
      require(result.id == ++inserted, "unexpected test block identity");
      blocks[result.id] = {parent, true};
      break;
    }
    case 2:
    case 3:
      if (live && !blocks[pick].used &&
          std::none_of(blocks.begin(), blocks.end(), [&](const Reference &other) {
            return other.live && other.parent == pick;
          })) {
        cache.erase(pick);
        blocks[pick].live = false;
      }
      break;
    case 4:
      if (!live) break;
      if (blocks[pick].used)
        cache.releaseActive(pick);
      else
        cache.retainActive(pick);
      blocks[pick].used = !blocks[pick].used;
      break;
    }
    std::vector<std::vector<uint64_t>> below(inserted + 1);
    std::vector<bool> busy(inserted + 1);
    for (uint64_t id = 1; id <= inserted; ++id) {
      if (!blocks[id].live) continue;
      for (uint64_t above = blocks[id].parent; above; above = blocks[above].parent) {
        below[above].push_back(id);
        busy[above] = busy[above] || blocks[id].used;
      }
    }
    for (uint64_t id = 1; id <= inserted; ++id) {
      if (!blocks[id].live) continue;
      const std::vector<uint64_t> order = cache.subtree(id);
      for (auto at = order.begin(); at != order.end(); ++at)
        require(blocks[*at].parent == id ||
                    std::find(at + 1, order.end(), blocks[*at].parent) != order.end(),
                "subtree listed a block before one of its children");
      std::vector<uint64_t> sorted = order;
      std::sort(sorted.begin(), sorted.end());
      require(busy[id] ? order.empty() : sorted == below[id],
              "subtree disagrees with the reference tree");
    }
  }
  for (uint64_t id = 1; id <= inserted; ++id)
    if (blocks[id].live && blocks[id].used) cache.releaseActive(id);
  while (auto candidate = cache.evictionCandidate(0)) cache.erase(candidate->id);
  require(pool.snapshot().pagesPrefix == 0 && pool.freePageCount() == pool.pageCount(),
          "subtree churn leaked a block or a page reference");
}

struct FakeSlot final : engine::KvDiskSlot {};

// On random trees, as blocks, states, holders, transfers, waiting chains and the resume point change, the counts the
// cache keeps give what its rule names (pageReach): for a walk's reasons `mask`, a resident block frees now unless
// one of the reasons sits at it or below it (it or a resident block below is used, has a page a request holds, a
// restore fills or a write drains, or lacks a copy below a state that needs its KV; a state, a waiting chain's end
// or the resume point below it has a reason `mask` names), or it is on the chain of a `kept` state or above a write
// no reclaim waits for (`unwaited`); the written state's chain frees once that state has left RAM and needs its KV;
// a block is reachable at all unless it or one below is used, held or filled, it is above a write no reclaim waits
// for, or, idle, it has a chain reason `mask` names at it or below or is on a kept chain; and the RAM copies a
// reclaim may take count at the blocks that free now. Every step checks the counts against a recount
// (checkInvariants).
void testPageReachFollowsItsRule() {
  using Keep = KvCache::Keep;
  constexpr uint32_t count = 48;
  test::TestKvStorage storage(count, 100, 1);
  KvPool pool(storage, count);
  CacheRecency recency;
  KvCache cache(pool, recency);
  struct Reference {
    uint64_t parent = 0;
    uint32_t page = KvCache::noPage;
    bool live = false;
    bool used = false;    // retainActive
    bool held = false;    // a request holds its page
    bool moving = false;  // in transfer: a write drains it (copied) or a restore fills it
    bool copied = false;  // has a disk copy
    bool state = false;   // a state sits at it
    KvCache::Keeps keeps = 0;
    bool needed = false;
    bool candidate = false;  // a RAM copy a reclaim may take
    bool ram = false;        // the state has a RAM copy
    uint32_t wanted = 0;
  };
  std::vector<Reference> blocks(1);
  uint64_t resumePoint = 0;
  std::mt19937 random(733);
  const auto pick = [&](uint64_t last) { return random() % (last + 1); };
  const auto resident = [&](uint64_t id) { return blocks[id].live && blocks[id].page != KvCache::noPage; };
  const auto hasChild = [&](uint64_t id, bool residentOnly) {
    for (uint64_t other = 1; other < blocks.size(); ++other)
      if (blocks[other].live && blocks[other].parent == id && (!residentOnly || resident(other)))
        return true;
    return false;
  };
  const auto atOrBelow = [&](uint64_t id, uint64_t above) {
    for (; id; id = blocks[id].parent)
      if (id == above)
        return true;
    return false;
  };
  const auto bit = [](Keep keep) { return KvCache::keepBit(keep); };
  uint32_t checks = 0, freeing = 0, written = 0, reachable = 0;
  for (uint32_t step = 0; step < 6000; ++step) {
    const uint64_t id = blocks.size() > 1 ? 1 + pick(blocks.size() - 2) : 0;
    Reference *entry = id && blocks[id].live ? &blocks[id] : nullptr;
    // A use, a held page or a transfer ends whenever drawn and starts a
    // third of the time, so that most blocks have none.
    const bool flip = !pick(2);
    switch (pick(9)) {
    case 0:
    case 1: {
      // Each resident block owns a page.
      if (std::ranges::count_if(blocks, [](const Reference &at) { return at.live && at.page != KvCache::noPage; }) ==
          count)
        break;
      const uint64_t parent = entry && resident(id) && pick(3) ? id : 0;
      auto acquired = pool.acquirePages(1);
      const auto inserted = cache.insert(parent, page(static_cast<uint32_t>(blocks.size())), acquired.pages[0], {});
      pool.releasePage(acquired.pages[0], false);
      require(inserted.id == blocks.size(), "unexpected test block identity");
      blocks.push_back({.parent = parent, .page = acquired.pages[0], .live = true});
      break;
    }
    case 2:
      if (entry && !entry->used && !entry->held && !entry->moving && !entry->state && !entry->wanted &&
          id != resumePoint && !hasChild(id, false)) {
        cache.erase(id);
        entry->live = false;
      }
      break;
    case 3:
      if (entry && (entry->used || flip)) {
        entry->used ? cache.releaseActive(id) : cache.retainActive(id);
        entry->used = !entry->used;
      }
      break;
    case 4:
      if (entry && resident(id) && (entry->held || flip)) {
        entry->held ? pool.releasePage(entry->page, false) : pool.retainPage(entry->page, false);
        entry->held = !entry->held;
      }
      break;
    case 5:
      // A write reads a page with its copy; a restore fills one without.
      if (entry && resident(id) && (entry->moving || flip)) {
        cache.setTransfer(id, entry->moving  ? KvCache::Transfer::None
                              : entry->copied ? KvCache::Transfer::Draining
                                              : KvCache::Transfer::Filling);
        entry->moving = !entry->moving;
      }
      break;
    case 6:
      if (!entry || !resident(id) || entry->moving)
        break;
      // A RAM copy and the resume point sit on a page.
      if (entry->copied && !entry->used && !entry->held && !hasChild(id, true) &&
          !(entry->keeps & (bit(Keep::Writer) | bit(Keep::UsedWriter))) && id != resumePoint && pick(1)) {
        cache.dropPage(id);
        entry->page = KvCache::noPage;
      } else {
        cache.setSlot(id, entry->copied ? nullptr : std::make_shared<FakeSlot>());
        entry->copied = !entry->copied;
      }
      break;
    case 7:
      // A state comes or goes, or changes what it asks of its KV: a RAM
      // copy that leaves only through a write (sitting on a page), in use or
      // not, or one that needs its KV, in use or not.
      if (entry && (!entry->state || pick(1))) {
        entry->state = !entry->state || pick(1);
        const bool use = pick(1);
        const bool writer = entry->state && resident(id) && pick(1);
        entry->keeps = !entry->state ? 0
                       : writer      ? static_cast<KvCache::Keeps>(bit(use ? Keep::UsedWriter : Keep::Writer) |
                                                              (use ? bit(Keep::Used) : 0))
                                     : static_cast<KvCache::Keeps>(use ? bit(Keep::Used) : 0);
        entry->needed = entry->state && !writer;
        entry->candidate = entry->state && resident(id) && (writer || pick(1));
        entry->ram = entry->candidate || (entry->state && pick(1));
        cache.setStateMarks(id, {.keeps = entry->keeps, .needed = entry->needed, .candidate = entry->candidate},
                            entry->ram);
      }
      break;
    case 8:
      if (entry && (entry->wanted || flip)) {
        const bool added = !entry->wanted || pick(1);
        cache.countWanted(id, added);
        added ? ++entry->wanted : --entry->wanted;
      }
      break;
    case 9:
      if (entry || flip) {
        const uint64_t to = entry && entry->state && resident(id) ? id : 0;
        cache.moveResumePoint(resumePoint, to);
        resumePoint = to;
      }
      break;
    }
    cache.checkInvariants();
    // A walk: its reasons, some states it keeps, and the state it writes.
    KvCache::Keeps mask = bit(Keep::Busy);
    for (const Keep keep : {Keep::Moving, Keep::Uncopied, Keep::Writer, Keep::UsedWriter, Keep::Used, Keep::Wanted,
                            Keep::ResumePoint})
      if (pick(1))
        mask |= bit(keep);
    std::vector<uint64_t> keptStates;
    std::vector<uint64_t> writers;
    std::vector<uint64_t> unwaited;
    for (uint64_t block = 1; block < blocks.size(); ++block) {
      if (!blocks[block].live)
        continue;
      if (!pick(15))
        keptStates.push_back(block);
      if (resident(block) && blocks[block].moving && blocks[block].copied && pick(1))
        unwaited.push_back(block);
      if (blocks[block].keeps & mask & (bit(Keep::Writer) | bit(Keep::UsedWriter)))
        writers.push_back(block);
    }
    const uint64_t stateWritten = !writers.empty() && pick(1) ? writers[pick(writers.size() - 1)] : 0;
    // The rule, block by block, with the written state as it is or once it
    // has left RAM.
    const auto filled = [&](const Reference &at) { return at.moving && !at.copied; };
    const auto drained = [&](const Reference &at) { return at.moving && at.copied; };
    const auto staysHere = [&](uint64_t block, bool afterWrite) {
      const Reference &at = blocks[block];
      if (at.used || at.held || filled(at) || (drained(at) && (mask & bit(Keep::Moving))))
        return true;
      bool needed = false;
      for (uint64_t below = 1; below < blocks.size(); ++below) {
        const Reference &state = blocks[below];
        if (!state.live || !atOrBelow(below, block))
          continue;
        // Once written, a state no longer waits for its write; one in use
        // stays in use.
        const bool gone = afterWrite && below == stateWritten;
        const KvCache::Keeps keeps = static_cast<KvCache::Keeps>(
            (gone ? state.keeps & ~(bit(Keep::Writer) | bit(Keep::UsedWriter)) : state.keeps) |
            (state.wanted ? bit(Keep::Wanted) : 0) | (below == resumePoint ? bit(Keep::ResumePoint) : 0));
        if (keeps & mask)
          return true;
        needed = needed || (state.state && (state.needed || gone));
      }
      return (mask & bit(Keep::Uncopied)) && !at.copied && needed;
    };
    const auto cut = [&](uint64_t block) {
      for (const uint64_t state : keptStates)
        if (atOrBelow(state, block))
          return true;
      for (const uint64_t write : unwaited)
        if (atOrBelow(write, block))
          return true;
      return false;
    };
    const auto frees = [&](uint64_t block, bool afterWrite) {
      if (!resident(block) || cut(block))
        return false;
      for (uint64_t below = 1; below < blocks.size(); ++below)
        if (resident(below) && atOrBelow(below, block) && staysHere(below, afterWrite))
          return false;
      return true;
    };
    // Reachable: nothing at or below it used, held or filled; and for an
    // idle page, which no write drains, not on a kept chain and no chain
    // reason `mask` names there. Above a write no reclaim waits for, none.
    const KvCache::Keeps chains = mask & KvCache::kChainKeeps;
    const auto reaches = [&](uint64_t block) {
      if (!resident(block))
        return false;
      for (const uint64_t write : unwaited)
        if (atOrBelow(write, block))
          return false;
      const bool idle = !blocks[block].held && !blocks[block].moving;
      if (idle)
        for (const uint64_t state : keptStates)
          if (atOrBelow(state, block))
            return false;
      for (uint64_t below = 1; below < blocks.size(); ++below) {
        const Reference &at = blocks[below];
        if (!at.live || !atOrBelow(below, block))
          continue;
        if (resident(below) && (at.used || at.held || filled(at)))
          return false;
        const KvCache::Keeps keeps = static_cast<KvCache::Keeps>(
            at.keeps | (at.wanted ? bit(Keep::Wanted) : 0) | (below == resumePoint ? bit(Keep::ResumePoint) : 0));
        if (idle && (keeps & chains))
          return false;
      }
      return true;
    };
    uint32_t expectedNow = 0, expectedWritten = 0, expectedReachable = 0, expectedCandidates = 0;
    for (uint64_t block = 1; block < blocks.size(); ++block) {
      const bool now = frees(block, false);
      expectedNow += now;
      expectedWritten += !now && stateWritten && frees(block, true);
      expectedReachable += reaches(block);
      expectedCandidates += now && blocks[block].candidate;
    }
    const KvCache::PageReach reach = cache.pageReach(mask, keptStates, unwaited, stateWritten);
    require(reach.now == std::array<uint32_t, 3>{expectedNow, 0, 0} &&
                reach.written == std::array<uint32_t, 3>{expectedWritten, 0, 0} &&
                reach.reachable == std::array<uint32_t, 3>{expectedReachable, 0, 0} &&
                reach.candidates == expectedCandidates,
            "pageReach counted " + std::to_string(reach.now[0]) + " now, " + std::to_string(reach.written[0]) +
                " written and " + std::to_string(reach.reachable[0]) + " reachable pages and " +
                std::to_string(reach.candidates) + " RAM copies, the rule " + std::to_string(expectedNow) + ", " +
                std::to_string(expectedWritten) + ", " + std::to_string(expectedReachable) + " and " +
                std::to_string(expectedCandidates) + " at step " + std::to_string(step));
    ++checks;
    freeing += expectedNow;
    written += expectedWritten;
    reachable += expectedReachable;
  }
  require(freeing > checks && written && reachable > freeing, "the random trees rarely had a page each count names");
  for (uint64_t block = 1; block < blocks.size(); ++block) {
    Reference &entry = blocks[block];
    if (!entry.live)
      continue;
    if (entry.used)
      cache.releaseActive(block);
    if (entry.held)
      pool.releasePage(entry.page, false);
    if (entry.moving)
      cache.setTransfer(block, KvCache::Transfer::None);
  }
  std::cout << "pageReach follows its rule: " << checks << " counts\n";
}

// A resident block gains a disk copy, gives up its page, and takes a page
// back; the orders and the parent follow each step.
void testDiskTierTransitions() {
  test::TestKvStorage storage(8, 100, 1);
  KvPool pool(storage, 8);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(4);
  require(acquired.granted(), "test pages were not acquired");
  const auto rootTokens = page(21);
  const auto leafTokens = page(22);
  auto root = cache.insert(0, rootTokens, acquired.pages[0], {});
  auto leaf = cache.insert(root.id, leafTokens, acquired.pages[1], {});
  pool.releasePage(acquired.pages[0], false);
  pool.releasePage(acquired.pages[1], false);
  require(cache.evictionCandidate(0).value().id == leaf.id &&
              !cache.diskCandidate(true) && !cache.diskCandidate(false),
          "fresh blocks hold disk copies");
  rejects([&] { cache.dropPage(leaf.id); }, "KV cache block has no page to drop or no disk copy",
          "a block without a disk copy dropped its page");

  // A copy of a resident block is redundant; the block stays a RAM leaf.
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
  require(cache.diskCandidate(true).value().id == leaf.id &&
              !cache.diskCandidate(false) &&
              cache.evictionCandidate(0).value().id == leaf.id &&
              pool.snapshot().pagesPrefix == 2 && cache.diskBlocks() == 1,
          "a resident block with a disk copy is not a duplicate");
  cache.setSlot(leaf.id, nullptr);
  require(!cache.diskCandidate(true) && cache.diskBlocks() == 0,
          "a resident block could not drop its copy");
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());

  // Dropping the page leaves a disk-only leaf. The parent is the RAM leaf
  // now, no newer than the child was.
  cache.touch(leaf.id);
  const uint64_t leafUsed = cache.evictionCandidate(0).value().lastUsed;
  const uint32_t freePages = pool.freePageCount();
  cache.dropPage(leaf.id);
  require(pool.freePageCount() == freePages + 1 && pool.snapshot().pagesPrefix == 1 &&
              cache.diskBlocks() == 1 && cache.page(leaf.id) == KvCache::noPage,
          "dropped page did not return to the pool");
  require(cache.evictionCandidate(0).value().id == root.id &&
              cache.evictionCandidate(0).value().lastUsed == leafUsed &&
              !cache.evictionCandidate(root.id) && cache.hasDiskChildren(root.id),
          "parent did not become the RAM leaf with the child's recency");
  require(cache.diskCandidate(false).value().id == leaf.id && !cache.diskCandidate(true),
          "disk-only leaf is not replaceable");
  auto found = cache.find(root.id, leafTokens, {});
  require(found && found->id == leaf.id && found->physicalPage == KvCache::noPage &&
              cache.chain(leaf.id).pages ==
                  std::vector<uint32_t>{acquired.pages[0], KvCache::noPage},
          "disk-only block did not match through the chain");
  rejects([&] { cache.setSlot(leaf.id, nullptr); },
          "a disk-only KV cache block is erased, not stripped",
          "a disk-only block was stripped instead of erased");
  rejects([&] { cache.erase(root.id); }, "cannot evict a referenced KV cache block",
          "a parent with a disk child was erased");

  // A user protects a disk copy; a transfer hides the block from every order.
  cache.retainActive(leaf.id);
  require(!cache.diskCandidate(false), "a used disk block stayed replaceable");
  cache.releaseActive(leaf.id);
  require(cache.diskCandidate(false).value().id == leaf.id, "released disk block left the order");
  cache.setTransfer(leaf.id, KvCache::Transfer::Filling);
  require(!cache.diskCandidate(false), "a block in transfer stayed replaceable");
  rejects([&] { cache.setTransfer(leaf.id, KvCache::Transfer::Draining); },
          "KV cache block transfer state did not change",
          "a block in transfer started another");
  rejects([&] { cache.erase(leaf.id); }, "cannot evict a referenced KV cache block",
          "a block in transfer was erased");

  // Adopting a page makes the block resident again while its content is on
  // the way; a writer that recomputes it meanwhile keeps its own page.
  cache.adoptPage(leaf.id, acquired.pages[2]);
  const uint32_t adoptedFreePages = pool.freePageCount();
  pool.releasePage(acquired.pages[2], false);
  require(cache.page(leaf.id) == acquired.pages[2] && !cache.evictionCandidate(0) &&
              pool.snapshot().pagesPrefix == 2 && pool.freePageCount() == adoptedFreePages,
          "adopted page was not retained by the block");
  auto writer = cache.insert(root.id, leafTokens, acquired.pages[3], {});
  require(writer.id == leaf.id && writer.physicalPage == acquired.pages[3],
          "a writer was switched to a page still being filled");
  pool.releasePage(acquired.pages[3], false);
  cache.setTransfer(leaf.id, KvCache::Transfer::None);
  require(cache.evictionCandidate(0).value().id == leaf.id &&
              cache.diskCandidate(true).value().id == leaf.id,
          "restored block did not rejoin the orders");
  rejects([&] { cache.adoptPage(leaf.id, acquired.pages[3]); }, "KV cache block already has a page",
          "a resident block adopted a second page");

  cache.erase(leaf.id);
  cache.erase(root.id);
  require(pool.snapshot().pagesPrefix == 0 && cache.diskBlocks() == 0 &&
              pool.freePageCount() == pool.pageCount(),
          "erasing both tiers leaked a page or a copy");
}

// A disk-only block whose content is published takes the writer's page; a poisoned block matches
// nothing and leaves with its last user, taking a poisoned parent once its subtree is gone.
void testDiskOnlyAdoptionAndPoison() {
  test::TestKvStorage storage(8, 100, 1);
  KvPool pool(storage, 8);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(4);
  require(acquired.granted(), "test pages were not acquired");
  const auto rootTokens = page(31);
  const auto leafTokens = page(32);
  auto root = cache.insert(0, rootTokens, acquired.pages[0], {});
  auto leaf = cache.insert(root.id, leafTokens, acquired.pages[1], {});
  pool.releasePage(acquired.pages[0], false);
  pool.releasePage(acquired.pages[1], false);
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
  cache.dropPage(leaf.id);

  auto adopted = cache.insert(root.id, leafTokens, acquired.pages[2], {});
  require(adopted.id == leaf.id &&
              adopted.physicalPage == acquired.pages[2] &&
              cache.page(leaf.id) == acquired.pages[2] &&
              cache.diskCandidate(true).value().id == leaf.id,
          "a disk-only block did not adopt the writer's page");
  const uint32_t freePages = pool.freePageCount();
  pool.releasePage(acquired.pages[2], false);
  require(pool.freePageCount() == freePages, "adopted page was not retained");

  // A restore that fails: the block is unmatchable at once, keeps nothing
  // on disk, and a fresh publication of the same content stands beside it.
  cache.dropPage(leaf.id);
  cache.adoptPage(leaf.id, acquired.pages[2]);
  cache.retainActive(leaf.id);
  cache.setTransfer(leaf.id, KvCache::Transfer::Filling);
  cache.setTransfer(leaf.id, KvCache::Transfer::None);
  cache.poison(leaf.id);
  require(!cache.find(root.id, leafTokens, {}) && cache.contains(leaf.id) &&
              cache.diskBlocks() == 0 && !cache.evictionCandidate(0) &&
              !cache.diskCandidate(true) && !cache.diskCandidate(false),
          "poisoned block still matched or waited in an order");
  auto fresh = cache.insert(root.id, leafTokens, acquired.pages[3], {});
  pool.releasePage(acquired.pages[3], false);
  require(fresh.id != leaf.id &&
              cache.find(root.id, leafTokens, {}).value().id == fresh.id,
          "poisoned block blocked republication of its content");
  cache.releaseActive(leaf.id);
  require(!cache.contains(leaf.id) && pool.freePageCount() == freePages + 1 &&
              pool.snapshot().pagesPrefix == 2,
          "poisoned block outlived its last user");

  cache.poison(root.id);
  require(cache.contains(root.id) && !cache.find(0, rootTokens, {}),
          "a poisoned parent left before its child");
  cache.erase(fresh.id);
  require(pool.snapshot().pagesPrefix == 0 && pool.freePageCount() == pool.pageCount(),
          "a poisoned parent outlived its subtree");
}

// A block is matchable while find() can return it; a poisoned block its user still holds stays in
// the graph but is not matchable.
void testMatchableExcludesPoisonedBlocks() {
  test::TestKvStorage storage(8, 100, 1);
  KvPool pool(storage, 8);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(2);
  require(acquired.granted(), "test pages were not acquired");
  auto root = cache.insert(0, page(41), acquired.pages[0], {});
  auto leaf = cache.insert(root.id, page(42), acquired.pages[1], {});
  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  require(cache.matchable(root.id) && cache.matchable(leaf.id) &&
              !cache.matchable(leaf.id + 1),
          "a block in the graph was not matchable, or an unknown one was");
  cache.retainActive(leaf.id);
  cache.poison(leaf.id);
  require(cache.contains(leaf.id) && !cache.matchable(leaf.id) && cache.matchable(root.id),
          "a poisoned block stayed matchable");
  cache.releaseActive(leaf.id);
  cache.erase(root.id);
  require(!cache.matchable(leaf.id) && !cache.matchable(root.id),
          "an erased block stayed matchable");
}


// With spilling, a block a request's page reference holds is in no order, so neither RAM reclaim
// nor quota takes it or its copy; without a spill policy nothing is held.
void testRequestsHoldTheirBlocks() {
  for (const bool spill : {true, false}) {
    test::TestKvStorage storage(8, 100, 1, 10);
    KvPool pool(storage, 8);
    CacheRecency recency;
    KvCache cache(pool, recency, spill);
    auto acquired = pool.acquirePages(2);
    require(acquired.granted(), "test pages were not acquired");
    auto root = cache.insert(0, page(51), acquired.pages[0], {});
    auto leaf = cache.insert(root.id, page(52), acquired.pages[1], {});
    cache.setSlot(root.id, std::make_shared<FakeSlot>());
    cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
    require(cache.held(root.id) == spill && cache.held(leaf.id) == spill &&
                cache.heldCopies() == (spill ? 2u : 0u) && cache.diskCandidate(true).has_value() == !spill &&
                cache.evictionCandidate(0).has_value() == !spill,
            "a block a request holds was not held, or held without a spill policy");
    // A second reference keeps the mark when the first goes.
    cache.holdPage(acquired.pages[1]);
    cache.unholdPage(acquired.pages[1]);
    require(cache.held(leaf.id) == spill, "a block lost its mark while a request still held its page");
    cache.unholdPage(acquired.pages[0]);
    cache.unholdPage(acquired.pages[1]);
    require(!cache.held(root.id) && !cache.held(leaf.id) && cache.heldCopies() == 0 &&
                cache.diskCandidate(true).has_value() && cache.evictionCandidate(0).value().id == leaf.id,
            "a block nothing holds did not rejoin the orders");
    cache.holdPage(acquired.pages[1]);
    require(cache.held(leaf.id) == spill && cache.heldCopies() == (spill ? 1u : 0u),
            "a request's new reference did not hold the block");
    if (spill)
      rejects([&] { cache.erase(leaf.id); }, "cannot evict a referenced KV cache block",
              "a held block was erased");
    cache.unholdPage(acquired.pages[1]);
    cache.checkInvariants();
  }
}

// A spilled block keeps its index in RAM and its K/V only in its copy, which quota takes only with
// the block; back on a Full page its copy is redundant again.
void testSpilledBlocksTakeTheirPagesForm() {
  test::TestKvStorage storage(16, 100, 4, 10);
  KvPool pool(storage, 4);
  CacheRecency recency;
  KvCache cache(pool, recency, true);
  auto acquired = pool.acquirePages(2);
  require(acquired.granted(), "test pages were not acquired");
  auto root = cache.insert(0, page(61), acquired.pages[0], {});
  auto leaf = cache.insert(root.id, page(62), acquired.pages[1], {});
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
  const KvPageMoveResult spilled = pool.movePages(std::array{acquired.pages[1]}, kv::PageForm::Index);
  require(spilled.granted(), "the leaf's page did not move to the index form");
  cache.followPages(spilled.moves);
  const uint32_t indexPage = cache.page(leaf.id);
  require(pool.form(indexPage) == kv::PageForm::Index && cache.indexBlocks() == 1 && cache.held(leaf.id) &&
              cache.blockOn(indexPage) == leaf.id && cache.blockOn(acquired.pages[1]) == 0,
          "the spilled block did not follow its page to the index form");
  rejects([&] { cache.setSlot(leaf.id, nullptr); }, "copy holds its only K/V",
          "a spilled block gave up its only K/V");
  cache.unholdPage(acquired.pages[0]);
  cache.unholdPage(indexPage);
  require(!cache.diskCandidate(true) && cache.diskCandidate(false).value().id == leaf.id &&
              cache.evictionCandidate(0).value().id == leaf.id,
          "an idle spilled leaf was a redundant copy, or not the only copy or a RAM leaf");
  cache.holdPage(indexPage);
  const KvPageMoveResult restored = pool.movePages(std::array{indexPage}, kv::PageForm::Full);
  require(restored.granted(), "the page did not move back to the Full form");
  cache.followPages(restored.moves);
  cache.unholdPage(cache.page(leaf.id));
  require(cache.indexBlocks() == 0 && cache.diskCandidate(true).value().id == leaf.id &&
              !cache.diskCandidate(false),
          "a block back on a Full page did not hold a redundant copy");
  cache.checkInvariants();
}

// A block on a Slim page keeps its K/V there, and its copy holds what its page held when written;
// a writer's Full page drops a copy of the Slim page, which holds less.
void testSlimBlocksKeepWhatTheirCopiesHold() {
  test::TestKvStorage storage(16, 100, 4, 0, 30);
  KvPool pool(storage, 4);
  CacheRecency recency;
  KvCache cache(pool, recency);
  auto acquired = pool.acquirePages(3);
  require(acquired.granted(), "test pages were not acquired");
  auto root = cache.insert(0, page(91), acquired.pages[0], {});
  auto middle = cache.insert(root.id, page(92), acquired.pages[1], {});
  auto leaf = cache.insert(middle.id, page(93), acquired.pages[2], {});
  cache.setSlot(middle.id, std::make_shared<FakeSlot>());
  const KvPageMoveResult trimmed =
      pool.movePages(std::array{acquired.pages[1], acquired.pages[2]}, kv::PageForm::Slim);
  require(trimmed.granted(), "the pages did not move to the Slim form");
  cache.followPages(trimmed.moves);
  require(pool.form(cache.page(middle.id)) == kv::PageForm::Slim &&
              pool.form(cache.page(leaf.id)) == kv::PageForm::Slim && cache.indexBlocks() == 0 &&
              cache.blockOn(cache.page(leaf.id)) == leaf.id && cache.kvCopied(middle.id) &&
              cache.copyForm(middle.id) == kv::PageForm::Full,
          "a block did not follow its page to the Slim form, or its Full copy stopped holding its K/V");
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
  require(cache.kvCopied(leaf.id) && cache.copyForm(leaf.id) == kv::PageForm::Slim,
          "a copy of a Slim page did not hold what the page holds");
  cache.unholdPage(acquired.pages[0]);
  cache.unholdPage(cache.page(middle.id));
  cache.unholdPage(cache.page(leaf.id));
  require(cache.diskCandidate(true).value().id == middle.id && !cache.diskCandidate(false) &&
              cache.evictionCandidate(0).value().id == leaf.id,
          "the copy of a Slim page was not redundant, or a Slim leaf not a RAM leaf");
  cache.dropPage(leaf.id);
  require(cache.page(leaf.id) == KvCache::noPage && cache.copyForm(leaf.id) == kv::PageForm::Slim &&
              cache.kvCopied(leaf.id),
          "a disk-only block lost the form its copy holds");
  cache.checkInvariants();
  const auto writer = pool.acquirePages(1);
  require(writer.granted() && pool.form(writer.pages[0]) == kv::PageForm::Full, "a writer's page was not acquired");
  require(cache.insert(middle.id, page(93), writer.pages[0], {}).physicalPage == writer.pages[0] &&
              cache.page(leaf.id) == writer.pages[0] && !cache.slot(leaf.id) &&
              cache.copyForm(leaf.id) == kv::PageForm::Full && cache.diskBlocks() == 1,
          "a block a writer computed again kept a copy of its Slim page beside the writer's Full one");
  cache.unholdPage(writer.pages[0]);
  cache.checkInvariants();

  // Taken back after a restart, into a pool of its own.
  test::TestKvStorage nextStorage(16, 100, 4, 0, 30);
  KvPool nextPool(nextStorage, 4);
  CacheRecency later;
  KvCache next(nextPool, later);
  require(next.adoptDiskBlock(40, 0, page(94), {}, std::make_shared<FakeSlot>(), 1, kv::PageForm::Slim) &&
              next.copyForm(40) == kv::PageForm::Slim,
          "a block taken back lost the form its copy holds");
  rejects([&] {
    static_cast<void>(next.adoptDiskBlock(41, 0, page(95), {}, std::make_shared<FakeSlot>(), 1,
                                          kv::PageForm::Index));
  }, "invalid adopted KV cache block", "a copy of an index alone was taken back");
  next.checkInvariants();
}

// With a window of three pages a state pins only the three blocks ending at it; a Slim block takes
// a writer's Full page, making its tail Full again, unless a write is reading it.
void testStateTailsPinBoundedWindows() {
  test::TestKvStorage storage(32, 100, 4, 0, 30);
  KvPool pool(storage, 4);
  CacheRecency recency;
  KvCache cache(pool, recency, false, false, 3);
  auto acquired = pool.acquirePages(13);
  require(acquired.granted(), "test pages were not acquired");
  std::vector<uint64_t> chain;
  for (uint32_t index = 0; index < 10; ++index)
    chain.push_back(cache.insert(index ? chain.back() : 0, page(100 + index), acquired.pages[index], {}).id);
  const auto candidates = [&] {
    std::vector<uint64_t> blocks;
    for (auto candidate = cache.trimCandidate(0); candidate; candidate = cache.trimCandidate(candidate->id))
      blocks.push_back(candidate->id);
    std::ranges::sort(blocks);
    return blocks;
  };
  cache.countState(chain[9], true, false);
  cache.countState(chain[7], true, false);
  require(cache.pinnedBlocks() == 5 &&
              candidates() == std::vector<uint64_t>(chain.begin(), chain.begin() + 5),
          "two states two blocks apart did not pin five blocks, or the rest were no trim candidates");
  cache.countState(chain[9], false, false);
  require(cache.pinnedBlocks() == 3 &&
              candidates() == std::vector<uint64_t>{chain[0], chain[1], chain[2], chain[3],
                                                    chain[4], chain[8], chain[9]} &&
              cache.tailFull(chain[7]),
          "a state's removal did not unpin what its tail alone held");
  cache.checkInvariants();

  // Blocks 2 and 3 give up their windowed layers, 2 with a copy of its Slim
  // page and 3 with one of its Full page; block 4 too, being written.
  cache.setSlot(chain[3], std::make_shared<FakeSlot>());
  const KvPageMoveResult trimmed = pool.movePages(
      std::array{acquired.pages[2], acquired.pages[3], acquired.pages[4]}, kv::PageForm::Slim);
  require(trimmed.granted(), "the pages did not move to the Slim form");
  cache.followPages(trimmed.moves);
  cache.setSlot(chain[2], std::make_shared<FakeSlot>());
  cache.setSlot(chain[4], std::make_shared<FakeSlot>());
  cache.setTransfer(chain[4], KvCache::Transfer::Draining);
  require(!cache.tailFull(chain[3]) && candidates().size() == 4, "a Slim tail was taken for Full");
  const uint32_t slim = cache.page(chain[2]);
  require(cache.insert(chain[1], page(102), acquired.pages[10], {}).physicalPage == acquired.pages[10] &&
              cache.page(chain[2]) == acquired.pages[10] && !cache.slot(chain[2]) && cache.blockOn(slim) == 0 &&
              cache.insert(chain[2], page(103), acquired.pages[11], {}).physicalPage == acquired.pages[11] &&
              cache.slot(chain[3]) && !cache.kvCopied(chain[3]) && cache.upgrades() == 2 && cache.tailFull(chain[3]),
          "a Slim block did not take the writer's Full page, or kept a copy of what it held");
  require(cache.insert(chain[3], page(104), acquired.pages[12], {}).physicalPage == acquired.pages[12] &&
              pool.form(cache.page(chain[4])) == kv::PageForm::Slim && cache.upgrades() == 2,
          "a Slim block being written took the writer's page");
  cache.setTransfer(chain[4], KvCache::Transfer::None);
  cache.checkInvariants();
}

// A writer moves onto its block's page while a write reads it and keeps its own while a restore
// fills it; an adopted disk-only block's copy of another command's K/V is no active KV.
void testWritersAndTheContentOfABlock() {
  for (const bool spill : {true, false}) {
    test::TestKvStorage storage(8, 100, 1, 10);
    KvPool pool(storage, 8);
    CacheRecency recency;
    KvCache cache(pool, recency, spill);
    auto acquired = pool.acquirePages(4);
    require(acquired.granted(), "test pages were not acquired");
    auto root = cache.insert(0, page(71), acquired.pages[0], {});
    cache.setSlot(root.id, std::make_shared<FakeSlot>());
    cache.setTransfer(root.id, KvCache::Transfer::Draining);
    require(cache.insert(0, page(71), acquired.pages[1], {}).physicalPage == acquired.pages[0],
            "a writer kept its own page beside a page being written");
    cache.setTransfer(root.id, KvCache::Transfer::None);
    cache.setTransfer(root.id, KvCache::Transfer::Filling);
    require(cache.insert(0, page(71), acquired.pages[1], {}).physicalPage == acquired.pages[1],
            "a writer moved onto a page being filled");
    cache.setTransfer(root.id, KvCache::Transfer::None);

    auto leaf = cache.insert(root.id, page(72), acquired.pages[2], {});
    cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
    cache.unholdPage(acquired.pages[2]);
    cache.dropPage(leaf.id);
    const auto adopted = cache.insert(root.id, page(72), acquired.pages[3], {});
    require(adopted.id == leaf.id && adopted.physicalPage == acquired.pages[3] && cache.slot(leaf.id) &&
                !cache.kvCopied(leaf.id) && cache.diskBlocks() == 2,
            "a disk-only block a writer filled gave its copy up, or took it for a copy of the page");
    const auto redundant = cache.diskCandidate(true);
    require(cache.held(leaf.id) == spill && cache.heldCopies() == (spill ? 1u : 0u) && redundant &&
                redundant->id == (spill ? leaf.id : root.id),
            "a copy of another command's K/V beside a held page was active KV, or quota could not take it");
    cache.checkInvariants();
    cache.unholdPage(acquired.pages[3]);
    cache.dropPage(leaf.id);
    require(cache.kvCopied(leaf.id), "the copy of a block without its page did not hold its K/V");
    const auto writer = pool.acquirePages(1);
    require(writer.granted() && cache.insert(root.id, page(72), writer.pages[0], {}).id == leaf.id &&
                !cache.kvCopied(leaf.id),
            "a disk-only block a writer filled again took its copy for one of the page");
    cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
    require(cache.kvCopied(leaf.id) && cache.rewrites() == 1 && cache.heldCopies() == (spill ? 2u : 0u),
            "a copy of the page did not replace the copy of another command's K/V as active KV");
    for (const uint32_t held : {acquired.pages[0], acquired.pages[1], writer.pages[0]})
      cache.unholdPage(held);
    cache.checkInvariants();
  }
}

// A copy leaves with its block once the last request holding it lets go; a reader that outlives
// them, as a command's cut does, keeps the slot until it lets go.
void testCopiesLeaveWithTheirBlocks() {
  test::TestKvStorage storage(8, 100, 1, 10);
  KvPool pool(storage, 8);
  CacheRecency recency;
  KvCache cache(pool, recency, true);
  auto acquired = pool.acquirePages(2);
  require(acquired.granted(), "test pages were not acquired");
  auto root = cache.insert(0, page(81), acquired.pages[0], {});
  auto leaf = cache.insert(root.id, page(82), acquired.pages[1], {});
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
  cache.poison(leaf.id);
  require(cache.contains(leaf.id) && cache.slot(leaf.id) && cache.diskBlocks() == 1,
          "a held poisoned block lost its copy");
  cache.unholdPage(acquired.pages[1]);
  require(!cache.contains(leaf.id) && cache.diskBlocks() == 0,
          "a poisoned block outlived the last request that held it");

  cache.setSlot(root.id, std::make_shared<FakeSlot>());
  cache.setSlot(root.id, nullptr);
  cache.setSlot(root.id, std::make_shared<FakeSlot>());
  require(cache.rewrites() == 1, "a block written again after its copy went was not counted");
  std::shared_ptr<engine::KvDiskSlot> reader = cache.slot(root.id);
  const std::weak_ptr<engine::KvDiskSlot> slot = reader;
  cache.checkInvariants();
  cache.unholdPage(acquired.pages[0]);
  rejects([&] { cache.checkInvariants(); }, "has no reader",
          "a copy whose block nothing holds kept a reader unnoticed");
  cache.erase(root.id);
  require(!cache.contains(root.id) && cache.diskBlocks() == 0 && !slot.expired(),
          "a block whose copy a reader holds did not leave, or took the copy from the reader");
  reader.reset();
  require(slot.expired(), "the copy outlived its last reader");
  cache.checkInvariants();
}
} // namespace

int main() {
  try {
    testExactChainedBlocksAndPhysicalOwnership();
    testImageIdentityKeysBlocks();
    testBlockIdentitiesArePinned();
    testErasedLeafParentInheritsRecency();
    testInputValidation();
    testCandidateOrderThroughChurn();
    testSubtreeThroughChurn();
    testDiskTierTransitions();
    testPageReachFollowsItsRule();
    testDiskOnlyAdoptionAndPoison();
    testMatchableExcludesPoisonedBlocks();
    testRequestsHoldTheirBlocks();
    testSpilledBlocksTakeTheirPagesForm();
    testSlimBlocksKeepWhatTheirCopiesHold();
    testStateTailsPinBoundedWindows();
    testWritersAndTheContentOfABlock();
    testCopiesLeaveWithTheirBlocks();
    std::cout << "KV page cache tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "KV page cache test failed: " << error.what() << '\n';
    return 1;
  }
}
