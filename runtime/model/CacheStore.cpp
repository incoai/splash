#include "model/CacheStore.hpp"

#include <arm_acle.h>
#include <array>
#include <fcntl.h>
#include <sqlite3.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <csignal>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

namespace splash::model {
namespace {
void check(int result, sqlite3 *db) {
  if (result != SQLITE_OK && result != SQLITE_DONE && result != SQLITE_ROW)
    throw std::runtime_error(std::string("prefix cache: ") +
                             sqlite3_errmsg(db));
}
void sql(sqlite3 *db, const char *query) {
  check(sqlite3_exec(db, query, nullptr, nullptr, nullptr), db);
}
class Statement {
public:
  Statement(sqlite3 *db, const char *query) : db_(db) {
    check(sqlite3_prepare_v2(db, query, -1, &stmt_, nullptr), db);
  }
  ~Statement() { sqlite3_finalize(stmt_); }
  Statement(const Statement &) = delete;
  Statement &operator=(const Statement &) = delete;
  void integer(int index, uint64_t value) {
    check(sqlite3_bind_int64(stmt_, index, static_cast<sqlite3_int64>(value)),
          db_);
  }
  void blob(int index, const void *data, size_t size) {
    if (size > INT_MAX)
      throw std::runtime_error("prefix cache metadata too large");
    check(sqlite3_bind_blob(stmt_, index, data, static_cast<int>(size),
                            SQLITE_TRANSIENT),
          db_);
  }
  bool step() {
    const int code = sqlite3_step(stmt_);
    check(code, db_);
    return code == SQLITE_ROW;
  }
  uint64_t integer(int column) const {
    return sqlite3_column_int64(stmt_, column);
  }
  std::vector<uint64_t> words(int column) const {
    const int bytes = sqlite3_column_bytes(stmt_, column);
    if (bytes < 0 || bytes > 8 * 1024 * 1024 || bytes % sizeof(uint64_t))
      throw std::runtime_error("bad cache metadata");
    std::vector<uint64_t> result(bytes / sizeof(uint64_t));
    if (bytes)
      std::memcpy(result.data(), sqlite3_column_blob(stmt_, column), bytes);
    return result;
  }

private:
  sqlite3 *db_;
  sqlite3_stmt *stmt_ = nullptr;
};
class Transaction {
public:
  explicit Transaction(sqlite3 *db) : db_(db) { sql(db_, "BEGIN IMMEDIATE"); }
  ~Transaction() {
    if (!done_)
      sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
  }
  Transaction(const Transaction &) = delete;
  Transaction &operator=(const Transaction &) = delete;
  void commit() {
    sql(db_, "COMMIT");
    done_ = true;
  }

private:
  sqlite3 *db_;
  bool done_ = false;
};
// CRC32C detects accidental cache corruption with ARM's CRC instructions.
class Checksum {
public:
  void update(const std::byte *bytes, size_t size) {
    while (size >= sizeof(uint64_t)) {
      uint64_t word;
      std::memcpy(&word, bytes, sizeof(word));
      value_ = __crc32cd(value_, word);
      bytes += sizeof(word);
      size -= sizeof(word);
    }
    while (size--)
      value_ = __crc32cb(value_, static_cast<uint8_t>(*bytes++));
  }
  uint64_t finish() const { return value_ ^ UINT32_MAX; }

private:
  uint32_t value_ = UINT32_MAX;
};
constexpr uint64_t alignment = 16384;
uint64_t aligned(uint64_t bytes) {
  if (!bytes || bytes > uint64_t{INT64_MAX} - alignment)
    throw std::invalid_argument("invalid cache payload size");
  return (bytes + alignment - 1) / alignment * alignment;
}
void fileCheck(bool ok, const char *operation) {
  if (!ok)
    throw std::system_error(errno, std::generic_category(), operation);
}
bool transfer(int fd, uint64_t offset, std::byte *data, size_t bytes,
              bool writing, const std::atomic<bool> &cancelled) {
  while (bytes) {
    if (cancelled.load())
      return false;
    const auto count = std::min<size_t>(bytes, 1 << 20);
    const auto done = writing ? ::pwrite(fd, data, count, offset)
                              : ::pread(fd, data, count, offset);
    if (done < 0 && errno == EINTR)
      continue;
    if (done <= 0)
      return false;
    data += done;
    offset += done;
    bytes -= done;
  }
  return true;
}
} // namespace

struct CacheStore::Impl {
  sqlite3 *db = nullptr;
  bool owned = false;
  int payload = -1;
  uint64_t highWater = alignment;
  struct Extent {
    uint64_t offset, bytes, checksum;
  };
  std::unordered_map<uint64_t, Extent> records;
  // Free extents, coalesced by offset. Accessed under mutex; in-flight writes
  // reserve their extent before dropping the metadata lock.
  std::map<uint64_t, uint64_t> free;
  std::mutex mutex;
  std::mutex retiredMutex;
  std::vector<uint64_t> retired;
  std::atomic<uint64_t> next{1};
  uint64_t clock = 0;
  ~Impl() {
    if (db) {
      // All SlotFile backings are gone. Only committed prefix ownership
      // survives a graceful close; an abrupt exit does this at next open.
      if (owned)
        sqlite3_exec(db,
                     "DELETE FROM slots WHERE NOT EXISTS(SELECT 1 FROM refs "
                     "WHERE slot=slots.id)",
                     nullptr, nullptr, nullptr);
      sqlite3_close(db);
    }
    if (payload >= 0)
      ::close(payload);
  }
  void release(uint64_t offset, uint64_t bytes) {
    bytes = aligned(bytes);
    auto next = free.lower_bound(offset);
    if (next != free.begin()) {
      auto prior = std::prev(next);
      if (prior->first + prior->second == offset) {
        offset = prior->first;
        bytes += prior->second;
        free.erase(prior);
      }
    }
    if (next != free.end() && offset + bytes == next->first) {
      bytes += next->second;
      free.erase(next);
    }
    free.emplace(offset, bytes);
  }
  uint64_t reserve(uint64_t bytes) {
    const uint64_t size = aligned(bytes);
    auto best = free.end();
    for (auto it = free.begin(); it != free.end(); ++it)
      if (it->second >= size &&
          (best == free.end() || it->second < best->second))
        best = it;
    if (best == free.end()) {
      if (highWater > uint64_t{INT64_MAX} - size)
        throw std::runtime_error("cache payload offset overflow");
      const auto offset = highWater;
      highWater += size;
      return offset;
    }
    const auto [offset, available] = *best;
    free.erase(best);
    if (available > size)
      free.emplace(offset + size, available - size);
    return offset;
  }
  // Startup only: reconstruct holes from the committed index and truncate
  // only the unused tail. Live payloads are never copied or relocated.
  void rebuild() {
    struct stat info{};
    fileCheck(::fstat(payload, &info) == 0, "stat cache payload");
    free.clear();
    records.clear();
    highWater = alignment;
    Statement rows(
        db, "SELECT id,offset,bytes,checksum FROM slots ORDER BY offset");
    while (rows.step()) {
      const auto offset = rows.integer(1), bytes = rows.integer(2);
      const auto size = aligned(bytes);
      if (offset < highWater || offset % alignment ||
          offset > uint64_t{INT64_MAX} - size ||
          offset + bytes > static_cast<uint64_t>(info.st_size))
        throw std::runtime_error("invalid or truncated cache payload extent");
      if (offset > highWater)
        free.emplace(highWater, offset - highWater);
      highWater = offset + size;
      records.emplace(rows.integer(0), Extent{offset, bytes, rows.integer(3)});
    }
    if (static_cast<uint64_t>(info.st_size) != highWater)
      fileCheck(::ftruncate(payload, highWater) == 0,
                "trim cache payload tail");
    // APFS can return interior holes without moving live data. Other
    // filesystems may keep these reusable extents allocated; logical quotas are
    // unchanged.
    for (const auto &[offset, bytes] : free) {
      fpunchhole_t hole{};
      hole.fp_offset = offset;
      hole.fp_length = bytes;
      static_cast<void>(::fcntl(payload, F_PUNCHHOLE, &hole));
    }
  }
  void collect() {
    std::vector<uint64_t> ids;
    {
      std::lock_guard lock(retiredMutex);
      ids.swap(retired);
    }
    if (ids.empty())
      return;
    std::vector<uint64_t> released;
    Transaction transaction(db);
    for (auto id : ids) {
      Statement referenced(db, "SELECT 1 FROM refs WHERE slot=? LIMIT 1");
      referenced.integer(1, id);
      if (referenced.step()) {
        std::lock_guard lock(retiredMutex);
        retired.push_back(id);
      } else {
        if (records.contains(id))
          released.push_back(id);
        Statement remove(db, "DELETE FROM slots WHERE id=?");
        remove.integer(1, id);
        remove.step();
      }
    }
    transaction.commit();
    for (auto id : released) {
      const auto extent = records.at(id);
      release(extent.offset, extent.bytes);
      records.erase(id);
    }
  }
};

CacheStore::CacheStore(const std::filesystem::path &path,
                       const std::string &identity) try
    : impl_(std::make_unique<Impl>()) {
  if (path.empty())
    throw std::invalid_argument("persistent cache path is empty");
  // Like the temporary backend, report file-size limits as failed writes.
  // SQLite may write schema pages before any SlotFile has been constructed.
  struct sigaction fileSize{};
  if (::sigaction(SIGXFSZ, nullptr, &fileSize) == 0 &&
      fileSize.sa_handler == SIG_DFL)
    std::signal(SIGXFSZ, SIG_IGN);
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  const int descriptor =
      ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor < 0)
    throw std::system_error(errno, std::generic_category(),
                            "open persistent cache");
  ::close(descriptor);
  check(
      sqlite3_open_v2(path.c_str(), &impl_->db, SQLITE_OPEN_READWRITE, nullptr),
      impl_->db);
  auto *db = impl_->db;
  {
    Statement application(db, "PRAGMA application_id");
    application.step();
    Statement tables(db,
                     "SELECT COUNT(*) FROM sqlite_master WHERE type='table'");
    tables.step();
    Statement version(db, "PRAGMA user_version");
    version.step();
    if (tables.integer(0) &&
        (application.integer(0) != 0x53504c48 || version.integer(0) != 2))
      throw std::runtime_error("file is not a supported Splash prefix cache");
  }
  sql(db, "PRAGMA locking_mode=EXCLUSIVE;");
  sql(db, "PRAGMA application_id=0x53504c48; PRAGMA user_version=2;");
  sql(db,
      "PRAGMA journal_mode=PERSIST; PRAGMA synchronous=FULL; PRAGMA "
      "cache_size=-2048;"
      "PRAGMA foreign_keys=ON; PRAGMA secure_delete=OFF; PRAGMA "
      "journal_size_limit=1048576;"
      "CREATE TABLE IF NOT EXISTS identity(value BLOB NOT NULL);"
      "CREATE TABLE IF NOT EXISTS storage(value BLOB NOT NULL);"
      "CREATE TABLE IF NOT EXISTS slots(id INTEGER PRIMARY KEY, offset INTEGER "
      "NOT "
      "NULL, bytes INTEGER NOT NULL, checksum INTEGER NOT NULL);"
      "CREATE TABLE IF NOT EXISTS prefixes(id INTEGER PRIMARY KEY, metadata "
      "BLOB NOT NULL, used INTEGER NOT NULL, checksum INTEGER NOT NULL);"
      "CREATE TABLE IF NOT EXISTS refs(prefix INTEGER REFERENCES prefixes(id) "
      "ON DELETE CASCADE,"
      "slot INTEGER REFERENCES slots(id), PRIMARY KEY(prefix,slot));"
      "CREATE INDEX IF NOT EXISTS refs_slot ON refs(slot);");
  // The payload file is paired by a random identity, not just its name.
  const auto payloadPath = std::filesystem::path(path.string() + ".data");
  const bool exists = std::filesystem::exists(payloadPath);
  Statement count(db, "SELECT COUNT(*) FROM slots");
  count.step();
  if (!exists && count.integer(0))
    throw std::runtime_error("persistent payload file is missing");
  impl_->payload = ::open(
      payloadPath.c_str(),
      O_RDWR | O_CLOEXEC | O_NOFOLLOW | (exists ? 0 : O_CREAT | O_EXCL), 0600);
  fileCheck(impl_->payload >= 0, "open cache payload");
  fileCheck(::flock(impl_->payload, LOCK_EX | LOCK_NB) == 0,
            "lock cache payload");
  fileCheck(::fcntl(impl_->payload, F_NOCACHE, 1) == 0,
            "uncached cache payload");
  std::array<uint64_t, 4> header{0x53504c4850415932ULL, 0, 0, 0};
  const std::atomic<bool> running{false};
  if (exists) {
    fileCheck(transfer(impl_->payload, 0,
                       reinterpret_cast<std::byte *>(header.data()),
                       sizeof(header), false, running),
              "read cache payload header");
    if (header[0] != 0x53504c4850415932ULL)
      throw std::runtime_error("unsupported cache payload file");
  } else {
    arc4random_buf(header.data() + 1, 3 * sizeof(uint64_t));
    fileCheck(transfer(impl_->payload, 0,
                       reinterpret_cast<std::byte *>(header.data()),
                       sizeof(header), true, running),
              "write cache payload header");
    fileCheck(::fsync(impl_->payload) == 0, "sync cache payload header");
    const auto parent = path.parent_path().empty() ? std::filesystem::path(".")
                                                   : path.parent_path();
    const int directory = ::open(parent.c_str(), O_RDONLY | O_CLOEXEC);
    fileCheck(directory >= 0, "open cache directory");
    const int synced = ::fsync(directory);
    const int saved = errno;
    ::close(directory);
    errno = saved;
    fileCheck(synced == 0, "sync cache directory");
  }
  {
    Statement pairing(db, "SELECT value=? FROM storage");
    pairing.blob(1, header.data(), sizeof(header));
    if (pairing.step()) {
      if (!pairing.integer(0))
        throw std::runtime_error("cache payload identity mismatch");
    } else {
      Statement insert(db, "INSERT INTO storage VALUES(?)");
      insert.blob(1, header.data(), sizeof(header));
      insert.step();
    }
  }
  {
    Statement existing(db, "SELECT value=? FROM identity");
    const std::string versioned = "splash-prefix-cache-v3-groups-crc32c:" + identity;
    existing.blob(1, versioned.data(), versioned.size());
    if (existing.step()) {
      if (!existing.integer(0)) {
        // Explicit files are reusable after an engine/model change, but none
        // of the old bytes may enter the new model's prefix graph.
        Transaction transaction(db);
        sql(db, "DELETE FROM prefixes; DELETE FROM slots;");
        Statement reset(db, "UPDATE identity SET value=?");
        reset.blob(1, versioned.data(), versioned.size());
        reset.step();
        transaction.commit();
      }
    } else {
      Statement insert(db, "INSERT INTO identity VALUES(?)");
      insert.blob(1, versioned.data(), versioned.size());
      insert.step();
    }
  }
  impl_->owned = true;
  sql(db, "DELETE FROM slots WHERE NOT EXISTS(SELECT 1 FROM refs WHERE "
          "slot=slots.id)");
  Statement maximum(db, "SELECT COALESCE(MAX(id),0) FROM slots");
  maximum.step();
  impl_->next.store(maximum.integer(0) + 1);
  Statement clock(db, "SELECT COALESCE(MAX(used),0) FROM prefixes");
  clock.step();
  impl_->clock = clock.integer(0);
  impl_->rebuild();
} catch (const std::bad_alloc &) {
  throw;
} catch (const std::exception &error) {
  throw std::runtime_error("persistent cache " + path.string() + ": " +
                           error.what());
}
CacheStore::~CacheStore() = default;
uint64_t CacheStore::allocate() noexcept { return impl_->next.fetch_add(1); }
void CacheStore::retire(uint64_t id) noexcept {
  try {
    std::lock_guard lock(impl_->retiredMutex);
    impl_->retired.push_back(id);
  } catch (...) {
  } // startup also collects unreferenced records
}

bool CacheStore::write(uint64_t id,
                       const std::vector<std::span<const std::byte>> &source,
                       const std::atomic<bool> &cancelled) {
  uint64_t bytes = 0;
  Checksum checksum;
  for (auto span : source) {
    if (cancelled.load() || span.size() > uint64_t{INT64_MAX} - bytes)
      return false;
    bytes += span.size();
    checksum.update(span.data(), span.size());
  }
  if (!bytes)
    return false;
  uint64_t offset;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->collect();
    offset = impl_->reserve(bytes);
  }
  bool written = true;
  uint64_t cursor = offset;
  try {
    for (auto span : source) {
      if (!transfer(impl_->payload, cursor,
                    const_cast<std::byte *>(span.data()), span.size(), true,
                    cancelled)) {
        written = false;
        break;
      }
      cursor += span.size();
    }
    if (written && !cancelled.load()) {
      std::lock_guard lock(impl_->mutex);
      if (!impl_->records
               .emplace(id, Impl::Extent{offset, bytes, checksum.finish()})
               .second)
        throw std::logic_error("cache payload is immutable");
      return true;
    }
  } catch (...) {
    std::lock_guard lock(impl_->mutex);
    impl_->release(offset, bytes);
    throw;
  }
  std::lock_guard lock(impl_->mutex);
  impl_->release(offset, bytes);
  return false;
}

bool CacheStore::read(uint64_t id,
                      const std::vector<std::span<std::byte>> &destination,
                      const std::atomic<bool> &cancelled) {
  uint64_t offset, bytes, expected;
  {
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->records.find(id);
    if (found == impl_->records.end())
      return false;
    offset = found->second.offset;
    bytes = found->second.bytes;
    expected = found->second.checksum;
  }
  uint64_t requested = 0;
  for (auto span : destination) {
    if (span.size() > uint64_t{INT64_MAX} - requested)
      return false;
    requested += span.size();
  }
  if (bytes != requested)
    return false;
  Checksum checksum;
  // The caller's live slot owns this immutable extent until IO completes.
  for (auto span : destination) {
    if (!transfer(impl_->payload, offset, span.data(), span.size(), false,
                  cancelled))
      return false;
    checksum.update(span.data(), span.size());
    offset += span.size();
  }
  return checksum.finish() == expected;
}

std::vector<CacheStore::Prefix> CacheStore::load() {
  std::lock_guard lock(impl_->mutex);
  std::vector<Prefix> prefixes;
  Statement rows(impl_->db,
                 "SELECT id,metadata,checksum FROM prefixes ORDER BY used,id");
  while (rows.step()) {
    Prefix prefix{rows.integer(0), {}, {}};
    try {
      prefix.metadata = rows.words(1);
      Checksum checksum;
      checksum.update(
          reinterpret_cast<const std::byte *>(prefix.metadata.data()),
          prefix.metadata.size() * sizeof(uint64_t));
      if (checksum.finish() != rows.integer(2))
        prefix.metadata.clear();
    } catch (const std::runtime_error &) {
      prefix.metadata.clear();
    }
    Statement refs(impl_->db, "SELECT slot,bytes FROM refs JOIN slots "
                              "ON slot=slots.id WHERE prefix=?");
    refs.integer(1, prefix.id);
    while (refs.step())
      prefix.records.push_back({refs.integer(0), refs.integer(1)});
    prefixes.push_back(std::move(prefix));
  }
  return prefixes;
}
void CacheStore::collectUnreferenced() {
  std::lock_guard lock(impl_->mutex);
  sql(impl_->db, "DELETE FROM slots WHERE NOT EXISTS(SELECT 1 FROM refs WHERE "
                 "slot=slots.id)");
  impl_->rebuild();
}

void CacheStore::save(const Prefix &prefix) {
  // Only durable publication synchronizes payloads. Temporary eviction pays
  // for a raw write; a manifest cannot commit before all its payloads are safe.
  fileCheck(::fsync(impl_->payload) == 0, "sync persistent cache payloads");
  std::lock_guard lock(impl_->mutex);
  auto *db = impl_->db;
  {
    Transaction transaction(db);
    Statement put(db, "INSERT OR REPLACE INTO prefixes VALUES(?,?,?,?)");
    put.integer(1, prefix.id);
    put.blob(2, prefix.metadata.data(),
             prefix.metadata.size() * sizeof(uint64_t));
    put.integer(3, ++impl_->clock);
    Checksum checksum;
    checksum.update(reinterpret_cast<const std::byte *>(prefix.metadata.data()),
                    prefix.metadata.size() * sizeof(uint64_t));
    put.integer(4, checksum.finish());
    put.step();
    for (auto record : prefix.records) {
      const auto &extent = impl_->records.at(record.id);
      if (extent.bytes != record.bytes)
        throw std::invalid_argument("persistent record size mismatch");
      Statement slot(db, "INSERT OR IGNORE INTO slots VALUES(?,?,?,?)");
      slot.integer(1, record.id);
      slot.integer(2, extent.offset);
      slot.integer(3, extent.bytes);
      slot.integer(4, extent.checksum);
      slot.step();
      Statement ref(db, "INSERT OR IGNORE INTO refs VALUES(?,?)");
      ref.integer(1, prefix.id);
      ref.integer(2, record.id);
      ref.step();
    }
    transaction.commit();
  }
}
void CacheStore::erase(uint64_t id) {
  std::lock_guard lock(impl_->mutex);
  Transaction transaction(impl_->db);
  Statement remove(impl_->db, "DELETE FROM prefixes WHERE id=?");
  remove.integer(1, id);
  remove.step();
  transaction.commit();
  impl_->collect();
}
void CacheStore::touch(std::span<const uint64_t> ids) {
  if (ids.empty())
    return;
  std::lock_guard lock(impl_->mutex);
  Transaction transaction(impl_->db);
  uint64_t newest = 0;
  {
    Statement latest(impl_->db,
                     "SELECT id FROM prefixes ORDER BY used DESC LIMIT 1");
    if (latest.step())
      newest = latest.integer(0);
  }
  for (uint64_t id : ids) {
    // Accessing the newest prefix cannot change the durable LRU order.
    if (id == newest)
      continue;
    Statement update(impl_->db, "UPDATE prefixes SET used=? WHERE id=?");
    update.integer(1, ++impl_->clock);
    update.integer(2, id);
    update.step();
    if (sqlite3_changes(impl_->db))
      newest = id;
  }
  transaction.commit();
}
} // namespace splash::model
