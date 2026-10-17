#include "model/WeightReaders.hpp"

#include <pthread/qos.h>

#include <algorithm>
#include <stdexcept>

namespace splash::model {

bool WeightReaders::Batch::claim(size_t &index) noexcept {
  index = next_.fetch_add(1, std::memory_order_acq_rel);
  return index < count_;
}

void WeightReaders::Batch::end(size_t index, std::exception_ptr failure) {
  if (failure) {
    const std::lock_guard lock(mutex_);
    failures_.emplace_back(index, std::move(failure));
  }
  ends_[index].store(true, std::memory_order_release);
  const size_t done = done_.fetch_add(1, std::memory_order_acq_rel) + 1;
  // stop() may lower the limit to the tasks taken after this check; it then
  // finds them all done itself.
  if (done != limit_.load(std::memory_order_acquire)) return;
  if (done == count_ && ended_) ended_(*this);
  {
    const std::lock_guard lock(mutex_);
    endedReturned_ = done == count_;
  }
  changed_.notify_all();
}

void WeightReaders::Batch::wait() {
  std::unique_lock lock(mutex_);
  // Every task taken ended, and a batch whose every task ran also its end:
  // a waiter reads what `ended` wrote.
  changed_.wait(lock, [this] {
    const size_t limit = limit_.load(std::memory_order_acquire);
    return done_.load(std::memory_order_acquire) == limit && (limit < count_ || !ended_ || endedReturned_);
  });
}

bool WeightReaders::Batch::failed(size_t index) const {
  const std::lock_guard lock(mutex_);
  return std::ranges::any_of(failures_, [index](const auto &failure) { return failure.first == index; });
}

WeightReaders::Tally WeightReaders::Batch::tally() const {
  const std::lock_guard lock(mutex_);
  return {retries_.load(std::memory_order_relaxed), failures_.size()};
}

std::exception_ptr WeightReaders::Batch::failure() const {
  const std::lock_guard lock(mutex_);
  if (failures_.empty()) return nullptr;
  return std::ranges::min(failures_, {}, [](const auto &failure) { return failure.first; }).second;
}

WeightReaders::WeightReaders(unsigned readers) {
  threads_.reserve(readers);
  try {
    for (unsigned reader = 0; reader < readers; ++reader) threads_.emplace_back([this] { readerLoop(); });
  } catch (...) {
    {
      const std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    for (std::thread &thread : threads_) thread.join();
    throw;
  }
}

WeightReaders::~WeightReaders() {
  {
    const std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  for (std::thread &thread : threads_) thread.join();
}

void WeightReaders::run(size_t count, const Task &task, Tally *tally) {
  if (tally) *tally = {};
  if (!count) return;
  const std::shared_ptr<Batch> batch(new Batch(count, task, readers(), {}));
  runNow(batch);
  if (tally) *tally = batch->tally();
  if (const std::exception_ptr failure = batch->failure()) std::rethrow_exception(failure);
}

std::shared_ptr<WeightReaders::Batch> WeightReaders::start(size_t count, Task task, unsigned readers,
                                                           std::function<void(const Batch &)> ended) {
  if (!readers) throw std::invalid_argument("a background read without readers");
  const std::shared_ptr<Batch> batch(new Batch(count, std::move(task), readers, std::move(ended)));
  if (!count) {
    if (batch->ended_) batch->ended_(*batch);
    batch->endedReturned_ = true;
    return batch;
  }
  {
    const std::lock_guard lock(mutex_);
    background_.push_back(batch);
  }
  wakeReaders(std::min<size_t>(count, readers));
  return batch;
}

void WeightReaders::halt(Batch &batch) {
  // A halted batch keeps the tasks taken before.
  const size_t taken = batch.next_.exchange(batch.count_, std::memory_order_acq_rel);
  batch.limit_.store(std::min({taken, batch.count_, batch.limit_.load(std::memory_order_acquire)}),
                     std::memory_order_release);
}

void WeightReaders::stop(Batch &batch) {
  halt(batch);
  batch.wait();
  const std::lock_guard lock(mutex_);
  std::erase_if(background_, [&](const std::shared_ptr<Batch> &queued) { return queued.get() == &batch; });
}

void WeightReaders::finish(Batch &batch) {
  std::shared_ptr<Batch> owned;
  {
    const std::lock_guard lock(mutex_);
    const auto queued =
        std::ranges::find_if(background_, [&](const std::shared_ptr<Batch> &each) { return each.get() == &batch; });
    if (queued != background_.end()) {
      owned = *queued;
      background_.erase(queued);
    }
  }
  // A batch no longer queued has no task left to take.
  if (owned) runNow(owned);
  else batch.wait();
}

void WeightReaders::runNow(const std::shared_ptr<Batch> &batch) {
  {
    const std::lock_guard lock(mutex_);
    now_.push_back(batch);
    running_.fetch_add(1, std::memory_order_acq_rel);
  }
  // This thread takes a task too.
  wakeReaders(batch->count_ - 1);
  work(batch, false);
  batch->wait();
  bool background;
  {
    // Under the lock, so that a reader that passed the background over while
    // this batch ran is woken for it.
    const std::lock_guard lock(mutex_);
    std::erase(now_, batch);
    running_.fetch_sub(1, std::memory_order_acq_rel);
    background = !background_.empty();
  }
  if (background) wake_.notify_all();
}

void WeightReaders::wakeReaders(size_t count) {
  if (count >= threads_.size()) wake_.notify_all();
  else
    for (size_t reader = 0; reader < count; ++reader) wake_.notify_one();
}

void WeightReaders::attempt(Batch &batch, size_t index) {
  std::exception_ptr failure;
  try {
    batch.task_(index);
  } catch (...) {
    batch.retries_.fetch_add(1, std::memory_order_relaxed);
    try {
      batch.task_(index);
    } catch (...) {
      failure = std::current_exception();
    }
  }
  batch.end(index, std::move(failure));
}

void WeightReaders::work(const std::shared_ptr<Batch> &batch, bool background) {
  for (size_t index; !(background && running_.load(std::memory_order_acquire)) && batch->claim(index);)
    attempt(*batch, index);
}

void WeightReaders::readerLoop() {
  // A step waits on their reads.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  std::unique_lock lock(mutex_);
  for (;;) {
    std::shared_ptr<Batch> batch;
    bool background = false;
    wake_.wait(lock, [&] {
      if (stopping_) return true;
      std::erase_if(now_, [](const std::shared_ptr<Batch> &queued) { return queued->exhausted(); });
      if (!now_.empty()) {
        batch = now_.front();
        return true;
      }
      if (running_.load(std::memory_order_acquire)) return false;
      std::erase_if(background_, [](const std::shared_ptr<Batch> &queued) { return queued->exhausted(); });
      const auto open = std::ranges::find_if(
          background_, [](const std::shared_ptr<Batch> &queued) { return queued->working_ < queued->readers_; });
      if (open == background_.end()) return false;
      batch = *open;
      background = true;
      return true;
    });
    if (stopping_) return;
    ++batch->working_;
    lock.unlock();
    work(batch, background);
    lock.lock();
    --batch->working_;
  }
}

} // namespace splash::model
