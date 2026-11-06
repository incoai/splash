#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestRegistry.hpp"
#include "engine/ReleasableMemory.hpp"
#include "engine/WeightResidency.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;
using splash::test::Weights;
using Kind = WeightResidency::Step::Kind;

// Admits each allocation, recording its bytes, or refuses it for want of host
// memory while `refuse` is set.
struct Admission final {
  bool refuse = false;
  std::vector<uint64_t> granted;
  [[nodiscard]] metal::AllocationAdmission admit() {
    return [this](uint64_t bytes, const std::function<void()> &allocate) -> metal::AllocationResult {
      if (refuse)
        return metal::AllocationFailure::HostPressure;
      allocate();
      granted.push_back(bytes);
      return {};
    };
  }
};

// The residency refuses a config without the weights, their admission or a
// positive idle release.
void testResidencyNeedsItsWeightsAndAdmission() {
  Weights weights;
  rejects([] { WeightResidency residency({.admit = test::admitAll}); },
          "lacks the weights or their admission", "a residency without its weights was built");
  rejects([&] { WeightResidency residency({.weights = &weights}); },
          "lacks the weights or their admission", "a residency without their admission was built");
  for (const double milliseconds : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()})
    rejects([&] {
      WeightResidency residency(
          {.weights = &weights, .admit = test::admitAll, .idleReleaseMilliseconds = milliseconds});
    }, "the idle release must be positive", "an idle release that is not positive was accepted");
}

// The weights go only while held and the engine is idle: a release request
// at once, the idle release once the idle time has passed since the idle
// period began; never during a restore.
void testReleaseOnlyHeldWeightsOfAnIdleEngine() {
  Weights weights;
  WeightResidency residency(
      {.weights = &weights, .admit = test::admitAll, .idleReleaseMilliseconds = 1000.0});
  residency.idleFrom(100.0);
  require(!residency.release(5000.0, false, true) && !residency.release(5000.0, false, false) &&
              !weights.released(),
          "the weights of an engine that holds a request were released");
  require(!residency.release(1099.0, true, true) && !weights.released(),
          "the idle release came before the idle time");
  require(residency.release(1100.0, true, true) && weights.released() && !residency.held(),
          "the idle release did not release the weights after the idle time");
  require(!residency.release(1100.0, true, false), "released weights were released again");
  require(residency.restore(1200.0).kind == Kind::Progress && !residency.release(5000.0, true, false) &&
              weights.held() == 1,
          "a release request released the weights during a restore");
  while (residency.restore(1300.0).kind == Kind::Progress) {
  }
  require(residency.held() && !residency.release(2299.0, true, true),
          "the idle period did not begin when the weights were restored");
  require(residency.release(1300.0, true, false) && weights.released(),
          "a release request did not release held weights at once");
}

// A restore takes back an image a call: Progress, Progress, then Restored,
// which counts it and begins an idle period; the snapshot follows it.
void testRestoreTakesBackAPartACall() {
  Weights weights;
  WeightResidency residency(
      {.weights = &weights, .admit = test::admitAll, .idleReleaseMilliseconds = 2000.0});
  require(residency.release(0.0, true, false), "the weights were not released");
  WeightsSnapshot snapshot = residency.snapshot();
  require(snapshot.idleReleaseSeconds == 2.0 && snapshot.released && !snapshot.restoring &&
              !snapshot.restores && !snapshot.restoreFailures,
          "the snapshot of released weights is wrong");
  std::vector<Kind> steps;
  for (double now = 10.0; !residency.held(); now += 10.0) {
    steps.push_back(residency.restore(now).kind);
    snapshot = residency.snapshot();
    require(snapshot.restoring == !residency.held() && residency.restoring() == snapshot.restoring,
            "the snapshot did not report the restore under way");
  }
  require(steps == std::vector<Kind>{Kind::Progress, Kind::Progress, Kind::Restored} &&
              weights.restores == Weights::kImages,
          "the restore did not take back an image a call");
  snapshot = residency.snapshot();
  require(!snapshot.released && !snapshot.restoring && snapshot.restores == 1 && !snapshot.restoreFailures,
          "the snapshot did not count the restore");
  require(!residency.release(2029.0, true, true) && residency.release(2030.0, true, true),
          "the idle period did not begin at the restore");
  Weights kept;
  WeightResidency off({.weights = &kept, .admit = test::admitAll});
  require(std::isinf(off.snapshot().idleReleaseSeconds) && !off.release(1e18, true, true) &&
              off.release(1e18, true, false) && kept.released(),
          "with the idle release off the weights went idle, or a release request kept them");
}

// A refused part stays released and gives the failure; what came back stays,
// and the next call tries the part again. A restore abandoned once its last
// waiter is gone gives back what came back, and counts a failure only when a
// request waiting for it failed at its resource wait limit.
void testARefusedRestoreGivesTheFailure() {
  for (const bool waiterFailed : {false, true}) {
    Weights weights;
    Admission admission;
    WeightResidency residency(
        {.weights = &weights, .admit = admission.admit(), .idleReleaseMilliseconds = 1000.0});
    require(residency.release(0.0, true, false), "the weights were not released");
    require(residency.restore(1.0).kind == Kind::Progress, "the first image did not come back");
    admission.refuse = true;
    const WeightResidency::Step refused = residency.restore(2.0);
    require(refused.kind == Kind::Refused && refused.failure == metal::AllocationFailure::HostPressure &&
                weights.held() == 1 && residency.restoring(),
            "a refused image did not give the failure, or gave back what came back");
    admission.refuse = false;
    require(residency.restore(3.0).kind == Kind::Progress && weights.held() == 2,
            "the image refused was not taken back once admitted");
    admission.refuse = true;
    require(residency.restore(4.0).kind == Kind::Refused, "the third image was not refused");
    if (waiterFailed)
      residency.waiterFailed();
    residency.abandon();
    const WeightsSnapshot snapshot = residency.snapshot();
    require(weights.held() == 0 && weights.released() && !residency.restoring() && !snapshot.restoring &&
                snapshot.restoreFailures == (waiterFailed ? 1U : 0U) && !snapshot.restores,
            "an abandoned restore kept what came back, or counted a failure no waiter's");
    admission.refuse = false;
    while (residency.restore(5.0).kind == Kind::Progress) {
    }
    require(residency.held() && weights.restores == 2 + Weights::kImages &&
                residency.snapshot().restoreFailures == (waiterFailed ? 1U : 0U),
            "the next restore did not start over from the first image");
  }
}

// Through ReleasableMemory, a store part after the images comes back in one
// admission of its own, after the last image: the floor and its staging
// alone (model::ExpertStore::restore), as the part takes them.
void testTheStorePartComesBackInOneAdmissionAfterTheImages() {
  Weights weights;
  constexpr uint64_t kFloorBytes = 1ULL << 28;
  struct Store final : model::WeightMemory {
    bool held = true;
    [[nodiscard]] bool released() const noexcept override { return !held; }
    void release() override { held = false; }
    bool restore(const metal::AllocationAdmission &admit) override {
      if (const metal::AllocationResult admitted = admit(kFloorBytes, [] {}); !admitted)
        throw metal::MetalAllocationError("the floor was refused", admitted.failure);
      held = true;
      return true;
    }
  } store;
  ReleasableMemory memory({&weights, &store}, std::nullopt);
  Admission admission;
  WeightResidency residency({.weights = &memory, .admit = admission.admit()});
  require(residency.release(0.0, true, false) && weights.released() && store.released(),
          "the release kept the images or the store");
  std::vector<Kind> steps;
  while (!residency.held())
    steps.push_back(residency.restore(1.0).kind);
  require(steps == std::vector<Kind>{Kind::Progress, Kind::Progress, Kind::Progress, Kind::Restored} &&
              admission.granted == std::vector<uint64_t>{Weights::kImageBytes, Weights::kImageBytes,
                                                         Weights::kImageBytes, kFloorBytes},
          "the store did not come back in one admission after the images");
}

} // namespace

int main(int argc, char **argv) {
  static constexpr test::TestCase kTests[] = {
      {"testResidencyNeedsItsWeightsAndAdmission", testResidencyNeedsItsWeightsAndAdmission},
      {"testReleaseOnlyHeldWeightsOfAnIdleEngine", testReleaseOnlyHeldWeightsOfAnIdleEngine},
      {"testRestoreTakesBackAPartACall", testRestoreTakesBackAPartACall},
      {"testARefusedRestoreGivesTheFailure", testARefusedRestoreGivesTheFailure},
      {"testTheStorePartComesBackInOneAdmissionAfterTheImages",
       testTheStorePartComesBackInOneAdmissionAfterTheImages},
  };
  const test::TestGroup groups[] = {{"residency", kTests}};
  return test::runTests("weight-residency", groups, argc, argv);
}
