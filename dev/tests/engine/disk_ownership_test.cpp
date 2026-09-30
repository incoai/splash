#include "model/SlotFile.hpp"
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace splash::model;
namespace {
constexpr uint64_t size = SlotFile::kAlignmentBytes;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Fixture {
  std::filesystem::path directory;
  std::shared_ptr<DiskBudget> temporary =
      std::make_shared<DiskBudget>(2 * size);
  std::shared_ptr<DiskBudget> durable = std::make_shared<DiskBudget>(3 * size);
  std::shared_ptr<CacheStore> store;
  std::shared_ptr<SlotFile> file;
  Fixture() {
    std::string path =
        (std::filesystem::temp_directory_path() / "splash-ownership-XXXXXX")
            .string();
    require(mkdtemp(path.data()), "temporary test directory");
    directory = path;
    store = std::make_shared<CacheStore>(directory / "prefix.sqlite",
                                         "ownership-test");
    file = std::make_shared<SlotFile>(size, temporary, directory, store);
  }
  ~Fixture() {
    file.reset();
    store.reset();
    std::filesystem::remove_all(directory);
  }
};
void reservationLifecycle() {
  Fixture f;
  auto first = f.file->acquire(), second = f.file->acquire();
  require(first && second && !f.file->acquire(), "temporary quota exceeded");
  auto reservation = DiskReservation::acquire(f.durable, 3 * size);
  require(reservation && f.durable->usedBytes() == 3 * size,
          "manifest not fully charged");
  require(!DiskReservation::acquire(f.durable, 1),
          "reserved quota was borrowed");
  first->transfer(*reservation);
  require(first->durable() && f.temporary->usedBytes() == size &&
              f.durable->usedBytes() == 3 * size &&
              reservation->remainingBytes() == 2 * size,
          "transfer double charged a slot");
  auto third = f.file->acquire(*reservation);
  reservation.reset();
  require(f.durable->usedBytes() == 2 * size, "unused reservation leaked");
  auto fill = f.file->acquire();
  require(fill && !first->returnTemporary(),
          "rollback borrowed the temporary quota");
  second.reset();
  require(first->returnTemporary() && !first->durable(),
          "rollback did not transfer charge");
  require(f.temporary->usedBytes() == 2 * size &&
              f.durable->usedBytes() == size,
          "rollback double charged slot");
  auto held = third;
  third.reset();
  require(f.durable->usedBytes() == size, "live read pin lost its charge");
  held->retire();
  require(!held->reusable() && !held->returnTemporary(),
          "retired slot was reused");
  held.reset();
  first.reset();
  fill.reset();
  require(!f.temporary->usedBytes() && !f.durable->usedBytes(),
          "slot quota leaked");
}
void transferDuringQueuedIO() {
  Fixture f;
  auto slot = f.file->acquire();
  std::vector<std::byte> source(size, std::byte{7}), output(size);
  std::promise<void> gate;
  auto released = gate.get_future();
  auto blocker = f.file->metadata([&] { released.wait(); });
  auto write = f.file->write(slot, {source}, {});
  auto reservation = DiskReservation::acquire(f.durable, size);
  slot->transfer(*reservation);
  require(!f.temporary->usedBytes() && f.durable->usedBytes() == size,
          "IO pin duplicated charge");
  gate.set_value();
  require(blocker->wait() && write->wait(), "queued write failed");
  require(f.temporary->writtenBytes() == size && !f.durable->writtenBytes(),
          "ownership transfer raced IO accounting");
  require(f.file->read(slot, {output}, {})->wait() && source == output,
          "transfer rewrote payload");
  require(f.durable->readBytes() == size, "read charged wrong IO owner");
}
void durableOnlyAndReopen() {
  Fixture f;
  auto zero = std::make_shared<DiskBudget>(0);
  auto file = std::make_shared<SlotFile>(size, zero, f.directory, f.store);
  require(!file->acquire(), "durable-only tier admitted temporary offload");
  auto reservation = DiskReservation::acquire(f.durable, 2 * size);
  auto committed = file->acquire(*reservation),
       pending = file->acquire(*reservation);
  std::vector<std::byte> data(size, std::byte{3});
  require(file->write(committed, {data}, {})->wait(), "durable write failed");
  require(file->write(pending, {data}, {})->wait(), "pending write failed");
  const auto id = committed->recordId();
  f.store->save({id, {17}, {{id, size}}});
  pending.reset();
  committed.reset();
  file.reset();
  f.file.reset();
  f.store.reset();
  require(!f.durable->usedBytes(), "shutdown retained process charges");
  f.store = std::make_shared<CacheStore>(f.directory / "prefix.sqlite",
                                         "ownership-test");
  auto prefixes = f.store->load();
  require(prefixes.size() == 1 && prefixes[0].records.size() == 1 &&
              prefixes[0].records[0].id == id,
          "restart retained uncommitted payload");
  f.store->collectUnreferenced();
  file = std::make_shared<SlotFile>(size, zero, f.directory, f.store);
  reservation = DiskReservation::acquire(f.durable, size);
  auto first = file->reopen(id, *reservation),
       second = file->reopen(id, *reservation);
  require(first == second && f.durable->usedBytes() == size &&
              !reservation->remainingBytes(),
          "shared reopen charged twice");
  std::vector<std::byte> output(size);
  require(file->read(first, {output}, {})->wait() && output == data,
          "reopen payload differs");
}
} // namespace
int main() {
  try {
    reservationLifecycle();
    transferDuringQueuedIO();
    durableOnlyAndReopen();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "disk ownership: PASS\n";
}
