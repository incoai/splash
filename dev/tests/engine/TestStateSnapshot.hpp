#pragma once
#include "engine/Cache.hpp"

namespace splash::test {
class CompletedStateWrite final : public StateWrite<RestoreState> {
public:
  explicit CompletedStateWrite(std::shared_ptr<const RestoreState> state)
      : state_(std::move(state)) {}
  bool ready() const noexcept override { return true; }
  bool finish() override { return true; }
  const std::shared_ptr<const RestoreState> &state() const noexcept override {
    return state_;
  }

private:
  std::shared_ptr<const RestoreState> state_;
};
inline void publishDiskState(engine::StateCache &cache, uint64_t block,
                             std::shared_ptr<const RestoreState> state) {
  if (!cache.publishToDisk(
          block,
          [state](std::function<void()>, model::DiskReservation *) {
            return std::make_unique<CompletedStateWrite>(state);
          },
          {}, {}) ||
      !cache.pollOffload())
    throw std::logic_error("test disk publication failed");
}
inline std::shared_ptr<const RestoreState>
checkpoint(uint32_t boundary, std::shared_ptr<const StatePayload> payload) {
  return std::make_shared<RestoreState>(
      RestoreState{boundary, {{0, boundary, boundary, std::move(payload)}}});
}
inline void publishCheckpoint(engine::Cache &cache, uint64_t block,
                              std::shared_ptr<const StatePayload> payload,
                              bool disposable = false) {
  cache.publishRestoreState(
      block, checkpoint(cache.prefixLength(block), std::move(payload)),
      disposable);
}
class SnapshotWrite final : public StateWrite<RestoreState> {
public:
  SnapshotWrite(uint32_t boundary,
                std::unique_ptr<StateWrite<StatePayload>> write)
      : write_(std::move(write)),
        state_(checkpoint(boundary, write_->state())) {}
  bool ready() const noexcept override { return write_->ready(); }
  bool finish() override { return write_->finish(); }
  const std::shared_ptr<const RestoreState> &state() const noexcept override {
    return state_;
  }

private:
  std::unique_ptr<StateWrite<StatePayload>> write_;
  std::shared_ptr<const RestoreState> state_;
};
inline std::unique_ptr<StateWrite<RestoreState>>
snapshotWrite(uint32_t boundary,
              std::unique_ptr<StateWrite<StatePayload>> write) {
  return write ? std::make_unique<SnapshotWrite>(boundary, std::move(write))
               : nullptr;
}
} // namespace splash::test
