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

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void requireThrows(Function &&function, const char *message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

CacheNamespace cacheNamespace(uint8_t salt = 0x5a) {
  CacheNamespace result;
  result.digest.fill(salt);
  return result;
}

std::array<uint32_t, KvCache::pageTokens> page(uint32_t token) {
  std::array<uint32_t, KvCache::pageTokens> result{};
  result.fill(token);
  return result;
}

void testImageIdentityKeysBlocks() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(2, false);
  require(acquired.granted() && acquired.pages.size() == 2,
          "test pages were not acquired");

  // Two images with the same grid and placeholder tokens differ only by
  // content digest; text-only lookups carry the zero identity.
  const ImageSpan red{8, 16, 8, 8, 0x1111, 0x2222};
  const ImageSpan blue{8, 16, 8, 8, 0x3333, 0x4444};
  const std::array<ImageSpan, 1> redSpans{red};
  const std::array<ImageSpan, 1> blueSpans{blue};
  require(blockImageIdentity(0, 32, {}) == ImageIdentity{},
          "text-only block identity must be zero");
  require(blockImageIdentity(32, 32, redSpans) == ImageIdentity{},
          "a block after the image must not carry image identity");
  const ImageIdentity redIdentity = blockImageIdentity(0, 32, redSpans);
  const ImageIdentity blueIdentity = blockImageIdentity(0, 32, blueSpans);
  require(redIdentity != ImageIdentity{} && redIdentity != blueIdentity,
          "image identity does not depend on the content digest");
  const ImageSpan redLater{40, 16, 8, 8, 0x1111, 0x2222};
  const std::array<ImageSpan, 1> laterSpans{redLater};
  require(blockImageIdentity(32, 32, laterSpans) != redIdentity,
          "image identity ignores the block's alignment inside the image");

  const auto tokens = page(248056);
  auto redBlock = cache.insert(0, tokens, acquired.pages[0], redIdentity);
  require(redBlock.inserted, "image block was not inserted");
  require(!cache.find(0, tokens), "text-only lookup matched an image block");
  require(!cache.find(0, tokens, blueIdentity),
          "a different image matched a token-identical block");
  auto found = cache.find(0, tokens, redIdentity);
  require(found && found->id == redBlock.id,
          "identical image content did not match its block");
  auto blueBlock = cache.insert(0, tokens, acquired.pages[1], blueIdentity);
  require(blueBlock.inserted && blueBlock.id != redBlock.id,
          "token-identical blocks with different images were merged");
  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  std::cout << "KV image identity ok\n";
}

void testExactChainedBlocksAndPhysicalOwnership() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(4, false);
  require(acquired.granted() && acquired.pages.size() == 4,
          "test pages were not acquired");

  const auto rootTokens = page(11);
  const auto leftTokens = page(12);
  const auto rightTokens = page(13);
  const auto otherTokens = page(14);
  auto root = cache.insert(0, rootTokens, acquired.pages[0]);
  auto left = cache.insert(root.id, leftTokens, acquired.pages[1]);
  auto right = cache.insert(root.id, rightTokens, acquired.pages[2]);
  auto other = cache.insert(0, otherTokens, acquired.pages[3]);
  require(root.inserted && left.inserted && right.inserted && other.inserted &&
              cache.snapshot().blocks == 4 &&
              cache.snapshot().bytes == 400,
          "page cache did not retain four physical blocks");

  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  require(pool.snapshot().pagesPrefix == 4 &&
              pool.snapshot().pagesActive == 0,
          "page cache ownership was not isolated from active requests");

  auto foundLeft = cache.find(root.id, leftTokens);
  auto wrongTokens = leftTokens;
  wrongTokens.back() ^= 1;
  require(foundLeft && foundLeft->id == left.id &&
              foundLeft->physicalPage == acquired.pages[1] &&
              !cache.find(root.id, wrongTokens),
          "page lookup trusted a hash collision");
  const KvCache::Chain chain = cache.chain(left.id);
  require(chain.blocks == std::vector<uint64_t>{root.id, left.id} &&
              chain.pages == std::vector<uint32_t>{acquired.pages[0],
                                                   acquired.pages[1]} &&
              cache.chainLength(left.id) == 2,
          "page cache did not reconstruct the exact parent chain");

  auto duplicate = cache.insert(root.id, leftTokens, acquired.pages[1]);
  require(!duplicate.inserted && duplicate.id == left.id &&
              duplicate.physicalPage == acquired.pages[1] &&
              cache.snapshot().blocks == 4,
          "exact duplicate created a second KV block");

  requireThrows<std::logic_error>([&] { cache.erase(root.id); },
                                  "parent block was evicted before children");
  cache.retainActive(left.id);
  require(cache.evictionCandidate().value().id == right.id &&
              cache.evictionCandidate(right.id).value().id == other.id &&
              !cache.evictionCandidate(other.id),
          "active page block remained evictable");
  cache.releaseActive(left.id);
  require(cache.evictionCandidate().value().id == right.id &&
              cache.evictionCandidate(right.id).value().id == other.id &&
              cache.evictionCandidate(other.id).value().id == left.id &&
              !cache.evictionCandidate(left.id),
          "leaf eviction did not follow deterministic LRU order");
  cache.touch(right.id);
  require(cache.evictionCandidate().value().id == other.id &&
              cache.evictionCandidate(other.id).value().id == left.id &&
              cache.evictionCandidate(left.id).value().id == right.id &&
              !cache.evictionCandidate(right.id),
          "touch did not refresh the matched leaf without walking its chain");

  cache.erase(right.id);
  cache.erase(left.id);
  cache.erase(root.id);
  cache.erase(other.id);
  require(cache.snapshot().blocks == 0 &&
              pool.snapshot().pagesPrefix == 0 &&
              pool.freePageCount() == pool.pageCount(),
          "page cache erase leaked physical references");
}

void testErasedLeafParentInheritsRecency() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(4, false);
  require(acquired.granted() && acquired.pages.size() == 4,
          "test pages were not acquired");

  // Chain A is inserted first and never touched again; chain B is newer.
  const auto a0Tokens = page(21);
  const auto a1Tokens = page(22);
  const auto b0Tokens = page(23);
  const auto b1Tokens = page(24);
  auto a0 = cache.insert(0, a0Tokens, acquired.pages[0]);
  auto a1 = cache.insert(a0.id, a1Tokens, acquired.pages[1]);
  auto b0 = cache.insert(0, b0Tokens, acquired.pages[2]);
  auto b1 = cache.insert(b0.id, b1Tokens, acquired.pages[3]);
  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  require(cache.evictionCandidate().value().id == a1.id &&
              cache.evictionCandidate(a1.id).value().id == b1.id,
          "leaf order did not start with the cold chain");

  // Erasing A's leaf exposes A's root, which is still colder than B's leaf.
  cache.erase(a1.id);
  require(cache.evictionCandidate().value().id == a0.id &&
              cache.evictionCandidate(a0.id).value().id == b1.id &&
              !cache.evictionCandidate(b1.id),
          "exposed parent jumped ahead of a warmer chain");
  cache.erase(a0.id);
  cache.erase(b1.id);
  require(cache.evictionCandidate().value().id == b0.id &&
              !cache.evictionCandidate(b0.id),
          "last exposed parent was not the sole candidate");
  cache.erase(b0.id);
  require(cache.snapshot().blocks == 0 &&
              pool.freePageCount() == pool.pageCount(),
          "recency test leaked physical references");
}

void testInputValidation() {
  test::TestKvBacking backing(2, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(1, false);
  require(acquired.granted(), "validation page was not acquired");
  const std::array<uint32_t, 1> shortTokens{1};
  requireThrows<std::invalid_argument>(
      [&] {
        static_cast<void>(
            cache.insert(0, shortTokens, acquired.pages[0]));
      },
      "partial token page was accepted");
  requireThrows<std::invalid_argument>(
      [&] {
        static_cast<void>(
            cache.insert(999, page(2), acquired.pages[0]));
      },
      "nonresident parent was accepted");
  pool.releasePage(acquired.pages[0], false);
}

void testCandidateOrderThroughChurn() {
  constexpr uint32_t count = 256;
  test::TestKvBacking backing(count, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(count, false);
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
                                       acquired.pages[index]);
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
    auto candidate = cache.evictionCandidate();
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
  while (auto candidate = cache.evictionCandidate()) cache.erase(candidate->id);
  require(cache.snapshot().blocks == 0 && pool.freePageCount() == count,
          "candidate churn leaked a block or physical reference");
}

void testHashCollisionStillRequiresExactTokens() {
  const auto left = page(7);
  auto right = left;
  right.back() = 8;
  constexpr uint64_t forcedCollision = 0x12345678;
  const KvBlockKeyView stored{11, forcedCollision, left, {}};
  const KvBlockKeyView colliding{11, forcedCollision, right, {}};
  const KvBlockKeyView exact{11, forcedCollision, left, {}};
  require(!exactKvBlockKeyMatch(stored, colliding) &&
              exactKvBlockKeyMatch(stored, exact),
          "KV block matching trusted a colliding index hash");
}

struct FakeSlot final : model::KvDiskSlot {};

// A resident block gains a disk copy, gives up its page, and takes a page
// back; the orders and the parent follow each step.
void testDiskTierTransitions() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(4, false);
  require(acquired.granted(), "test pages were not acquired");
  const auto rootTokens = page(21);
  const auto leafTokens = page(22);
  auto root = cache.insert(0, rootTokens, acquired.pages[0]);
  auto leaf = cache.insert(root.id, leafTokens, acquired.pages[1]);
  pool.releasePage(acquired.pages[0], false);
  pool.releasePage(acquired.pages[1], false);
  require(cache.evictionCandidate().value().id == leaf.id &&
              !cache.diskCandidate(true) && !cache.diskCandidate(false),
          "fresh blocks hold disk copies");
  requireThrows<std::logic_error>([&] { cache.dropPage(leaf.id); },
                                  "a block without a disk copy dropped its page");

  // A copy of a resident block is redundant; the block stays a RAM leaf.
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
  require(cache.diskCandidate(true).value().id == leaf.id &&
              !cache.diskCandidate(false) &&
              cache.evictionCandidate().value().id == leaf.id &&
              cache.snapshot().blocks == 2 && cache.snapshot().diskBlocks == 1,
          "a resident block with a disk copy is not a duplicate");
  cache.setSlot(leaf.id, nullptr);
  require(!cache.diskCandidate(true) && cache.snapshot().diskBlocks == 0,
          "a resident block could not drop its copy");
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());

  // Dropping the page leaves a disk-only leaf. The parent is the RAM leaf
  // now, no newer than the child was.
  cache.touch(leaf.id);
  const uint64_t leafUsed = cache.evictionCandidate().value().lastUsed;
  cache.dropPage(leaf.id);
  require(pool.pageFree(acquired.pages[1]) && cache.snapshot().blocks == 1 &&
              cache.snapshot().diskBlocks == 1 && cache.page(leaf.id) == KvCache::noPage,
          "dropped page did not return to the pool");
  require(cache.evictionCandidate().value().id == root.id &&
              cache.evictionCandidate().value().lastUsed == leafUsed &&
              !cache.evictionCandidate(root.id) && cache.hasDiskChildren(root.id),
          "parent did not become the RAM leaf with the child's recency");
  require(cache.diskCandidate(false).value().id == leaf.id && !cache.diskCandidate(true),
          "disk-only leaf is not replaceable");
  auto found = cache.find(root.id, leafTokens);
  require(found && found->id == leaf.id && found->physicalPage == KvCache::noPage &&
              cache.chain(leaf.id).pages ==
                  std::vector<uint32_t>{acquired.pages[0], KvCache::noPage},
          "disk-only block did not match through the chain");
  requireThrows<std::logic_error>([&] { cache.setSlot(leaf.id, nullptr); },
                                  "a disk-only block was stripped instead of erased");
  requireThrows<std::logic_error>([&] { cache.erase(root.id); },
                                  "a parent with a disk child was erased");

  // A user protects a disk copy; a transfer hides the block from every order.
  cache.retainActive(leaf.id);
  require(!cache.diskCandidate(false), "a used disk block stayed replaceable");
  cache.releaseActive(leaf.id);
  require(cache.diskCandidate(false).value().id == leaf.id, "released disk block left the order");
  cache.setTransferring(leaf.id, true);
  require(!cache.diskCandidate(false) && cache.transferring(leaf.id),
          "a block in transfer stayed replaceable");
  requireThrows<std::logic_error>([&] { cache.erase(leaf.id); },
                                  "a block in transfer was erased");

  // Adopting a page makes the block resident again while its content is on
  // the way; a writer that recomputes it meanwhile keeps its own page.
  cache.adoptPage(leaf.id, acquired.pages[2]);
  pool.releasePage(acquired.pages[2], false);
  require(cache.page(leaf.id) == acquired.pages[2] && !cache.evictionCandidate() &&
              cache.snapshot().blocks == 2 && !pool.pageFree(acquired.pages[2]),
          "adopted page was not retained by the block");
  auto writer = cache.insert(root.id, leafTokens, acquired.pages[3]);
  require(!writer.inserted && writer.id == leaf.id && writer.physicalPage == acquired.pages[3],
          "a writer was switched to a page still being filled");
  pool.releasePage(acquired.pages[3], false);
  cache.setTransferring(leaf.id, false);
  require(cache.evictionCandidate().value().id == leaf.id &&
              cache.diskCandidate(true).value().id == leaf.id,
          "restored block did not rejoin the orders");
  requireThrows<std::logic_error>([&] { cache.adoptPage(leaf.id, acquired.pages[3]); },
                                  "a resident block adopted a second page");

  cache.erase(leaf.id);
  cache.erase(root.id);
  require(cache.snapshot().blocks == 0 && cache.snapshot().diskBlocks == 0 &&
              pool.freePageCount() == pool.pageCount(),
          "erasing both tiers leaked a page or a copy");
}

// Publishing the content of a disk-only block gives it the writer's page. A
// poisoned block matches nothing, leaves with its last user, and takes a
// poisoned parent with it once the subtree is gone.
void testDiskOnlyAdoptionAndPoison() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(4, false);
  require(acquired.granted(), "test pages were not acquired");
  const auto rootTokens = page(31);
  const auto leafTokens = page(32);
  auto root = cache.insert(0, rootTokens, acquired.pages[0]);
  auto leaf = cache.insert(root.id, leafTokens, acquired.pages[1]);
  pool.releasePage(acquired.pages[0], false);
  pool.releasePage(acquired.pages[1], false);
  cache.setSlot(leaf.id, std::make_shared<FakeSlot>());
  cache.dropPage(leaf.id);

  auto adopted = cache.insert(root.id, leafTokens, acquired.pages[2]);
  require(!adopted.inserted && adopted.id == leaf.id &&
              adopted.physicalPage == acquired.pages[2] &&
              cache.page(leaf.id) == acquired.pages[2] &&
              cache.diskCandidate(true).value().id == leaf.id,
          "a disk-only block did not adopt the writer's page");
  pool.releasePage(acquired.pages[2], false);
  require(!pool.pageFree(acquired.pages[2]), "adopted page was not retained");

  // A restore that fails: the block is unmatchable at once, keeps nothing
  // on disk, and a fresh publication of the same content stands beside it.
  cache.dropPage(leaf.id);
  cache.adoptPage(leaf.id, acquired.pages[2]);
  cache.retainActive(leaf.id);
  cache.setTransferring(leaf.id, true);
  cache.setTransferring(leaf.id, false);
  cache.poison(leaf.id);
  require(!cache.find(root.id, leafTokens) && cache.contains(leaf.id) &&
              cache.snapshot().diskBlocks == 0 && !cache.evictionCandidate() &&
              !cache.diskCandidate(true) && !cache.diskCandidate(false),
          "poisoned block still matched or waited in an order");
  auto fresh = cache.insert(root.id, leafTokens, acquired.pages[3]);
  pool.releasePage(acquired.pages[3], false);
  require(fresh.inserted && fresh.id != leaf.id &&
              cache.find(root.id, leafTokens).value().id == fresh.id,
          "poisoned block blocked republication of its content");
  cache.releaseActive(leaf.id);
  require(!cache.contains(leaf.id) && pool.pageFree(acquired.pages[2]) &&
              cache.snapshot().blocks == 2,
          "poisoned block outlived its last user");

  cache.poison(root.id);
  require(cache.contains(root.id) && !cache.find(0, rootTokens),
          "a poisoned parent left before its child");
  cache.erase(fresh.id);
  require(cache.snapshot().blocks == 0 && pool.freePageCount() == pool.pageCount(),
          "a poisoned parent outlived its subtree");
}

} // namespace

int main() {
  try {
    testExactChainedBlocksAndPhysicalOwnership();
    testImageIdentityKeysBlocks();
    testErasedLeafParentInheritsRecency();
    testInputValidation();
    testCandidateOrderThroughChurn();
    testHashCollisionStillRequiresExactTokens();
    testDiskTierTransitions();
    testDiskOnlyAdoptionAndPoison();
    std::cout << "KV page cache tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "KV page cache test failed: " << error.what() << '\n';
    return 1;
  }
}
