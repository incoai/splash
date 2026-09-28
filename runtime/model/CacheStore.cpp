#include "model/CacheStore.hpp"

#include <arm_acle.h>
#include <array>
#include <fcntl.h>
#include <sqlite3.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <csignal>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <system_error>

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
  void blob(int index, const void *data, size_t size, bool borrowed = false) {
    if (size > INT_MAX)
      throw std::runtime_error("prefix cache metadata too large");
    check(sqlite3_bind_blob(stmt_, index, data, static_cast<int>(size),
                            borrowed ? SQLITE_STATIC : SQLITE_TRANSIENT),
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
} // namespace

struct CacheStore::Impl {
  sqlite3 *db = nullptr;
  bool owned = false;
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
  }
  void collect() {
    std::vector<uint64_t> ids;
    {
      std::lock_guard lock(retiredMutex);
      ids.swap(retired);
    }
    for (auto id : ids) {
      Statement referenced(db, "SELECT 1 FROM refs WHERE slot=? LIMIT 1");
      referenced.integer(1, id);
      if (referenced.step()) {
        std::lock_guard lock(retiredMutex);
        retired.push_back(id);
      } else {
        Statement remove(db, "DELETE FROM slots WHERE id=?");
        remove.integer(1, id);
        remove.step();
      }
    }
  }
};

CacheStore::CacheStore(const std::filesystem::path &path,
                       const std::string &identity)
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
        (application.integer(0) != 0x53504c48 || version.integer(0) != 1))
      throw std::runtime_error("file is not a supported Splash prefix cache");
  }
  sql(db, "PRAGMA locking_mode=EXCLUSIVE;");
  // New stores keep SQLite's relocation map so reducing the quota can return
  // unused pages to the filesystem at startup, without rebuilding live blobs.
  // Slots are multiples of 16 KiB. A matching database page reduces pager
  // overhead for large state blobs without the slack of 64 KiB pages.
  sql(db, "PRAGMA page_size=16384; PRAGMA auto_vacuum=INCREMENTAL;");
  sql(db, "PRAGMA application_id=0x53504c48; PRAGMA user_version=1;");
  sql(db,
      "PRAGMA journal_mode=PERSIST; PRAGMA synchronous=FULL; PRAGMA "
      "cache_size=-2048;"
      "PRAGMA foreign_keys=ON; PRAGMA secure_delete=OFF; PRAGMA "
      "journal_size_limit=1048576;"
      "CREATE TABLE IF NOT EXISTS identity(value BLOB NOT NULL);"
      "CREATE TABLE IF NOT EXISTS slots(id INTEGER PRIMARY KEY, data BLOB NOT "
      "NULL, checksum INTEGER NOT NULL);"
      "CREATE TABLE IF NOT EXISTS prefixes(id INTEGER PRIMARY KEY, metadata "
      "BLOB NOT NULL, used INTEGER NOT NULL, checksum INTEGER NOT NULL);"
      "CREATE TABLE IF NOT EXISTS refs(prefix INTEGER REFERENCES prefixes(id) "
      "ON DELETE CASCADE,"
      "slot INTEGER REFERENCES slots(id), PRIMARY KEY(prefix,slot));"
      "CREATE INDEX IF NOT EXISTS refs_slot ON refs(slot);");
  {
    Statement existing(db, "SELECT value=? FROM identity");
    const std::string versioned = "splash-prefix-cache-v2-crc32c:" + identity;
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
    if (cancelled.load())
      return false;
    bytes += span.size();
    checksum.update(span.data(), span.size());
  }
  const uint64_t hash = checksum.finish();
  if (bytes > INT_MAX || cancelled.load())
    return false;
  std::lock_guard lock(impl_->mutex);
  auto *db = impl_->db;
  // Retire before allocating: rollback journals need not copy free payload
  // pages. Never delete and reuse their large blobs in the same transaction.
  {
    Transaction reclaim(db);
    impl_->collect();
    reclaim.commit();
  }
  Transaction transaction(db);
  // Production staging is contiguous. Bind it for the duration of step()
  // without a second state-sized allocation or a zeroblob write pass.
  if (source.size() == 1) {
    Statement insert(db, "INSERT INTO slots VALUES(?,?,?)");
    insert.integer(1, id);
    insert.blob(2, source.front().data(), bytes, true);
    insert.integer(3, hash);
    insert.step();
    if (cancelled.load())
      return false;
    transaction.commit();
    return true;
  }
  Statement insert(db, "INSERT INTO slots VALUES(?,zeroblob(?),?)");
  insert.integer(1, id);
  insert.integer(2, bytes);
  insert.integer(3, hash);
  insert.step();
  sqlite3_blob *raw = nullptr;
  check(sqlite3_blob_open(db, "main", "slots", "data", id, 1, &raw), db);
  std::unique_ptr<sqlite3_blob, decltype(&sqlite3_blob_close)> blob(
      raw, sqlite3_blob_close);
  int offset = 0;
  for (auto span : source) {
    while (!span.empty()) {
      if (cancelled.load())
        return false;
      const int count =
          static_cast<int>(std::min<size_t>(span.size(), 1 << 20));
      check(sqlite3_blob_write(raw, span.data(), count, offset), db);
      offset += count;
      span = span.subspan(count);
    }
  }
  blob.reset();
  transaction.commit();
  return true;
}

bool CacheStore::read(uint64_t id,
                      const std::vector<std::span<std::byte>> &destination,
                      const std::atomic<bool> &cancelled) {
  std::lock_guard lock(impl_->mutex);
  auto *db = impl_->db;
  Statement stored(db, "SELECT length(data),checksum FROM slots WHERE id=?");
  stored.integer(1, id);
  if (!stored.step())
    return false;
  uint64_t bytes = 0;
  for (auto span : destination)
    bytes += span.size();
  if (bytes != stored.integer(0) || bytes > INT_MAX)
    return false;
  const uint64_t expected = stored.integer(1);
  sqlite3_blob *raw = nullptr;
  check(sqlite3_blob_open(db, "main", "slots", "data", id, 0, &raw), db);
  std::unique_ptr<sqlite3_blob, decltype(&sqlite3_blob_close)> blob(
      raw, sqlite3_blob_close);
  int offset = 0;
  Checksum checksum;
  for (auto span : destination) {
    while (!span.empty()) {
      if (cancelled.load())
        return false;
      const int count =
          static_cast<int>(std::min<size_t>(span.size(), 1 << 20));
      check(sqlite3_blob_read(raw, span.data(), count, offset), db);
      checksum.update(span.data(), count);
      offset += count;
      span = span.subspan(count);
    }
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
    Statement refs(impl_->db, "SELECT slot,length(data) FROM refs JOIN slots "
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
  sql(impl_->db, "PRAGMA incremental_vacuum;");
}

void CacheStore::save(const Prefix &prefix) {
  std::lock_guard lock(impl_->mutex);
  auto *db = impl_->db;
  // Payload and publication transactions both use FULL synchronization.
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
  impl_->collect();
  transaction.commit();
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
