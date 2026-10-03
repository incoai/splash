#include "TestChecks.hpp"
#include "engine/Scheduler.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <new>

using namespace splash;
using namespace splash::engine;
using splash::test::require;

namespace {
bool countAllocations = false;
size_t allocations = 0;
int failAfter = -1;
}

void *operator new(size_t size) {
  if (failAfter == 0) {
    failAfter = -1;
    throw std::bad_alloc();
  }
  if (failAfter > 0)
    --failAfter;
  if (countAllocations)
    ++allocations;
  if (void *memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }

namespace {

// Count only the public planning call, not setup or assertion messages.
template <class F> auto measured(F call, size_t limit) {
  allocations = 0;
  countAllocations = true;
  auto result = call();
  countAllocations = false;
  require(allocations <= limit, "warmed planning allocated candidate storage");
  return result;
}

RequestSpec request(uint64_t id) {
  return {id, RequestPriority::Normal, false, 65, 10'000.0};
}

void testIdleDoesNotAllocate() {
  Scheduler scheduler(0.0);
  require(measured([&] { return scheduler.admissionOrder(); }, 0).empty(),
          "idle admission returned work");
  require(measured([&] { return scheduler.prefillAdmissionOrder({}); }, 0).empty(),
          "idle prefill admission returned work");
  require(!measured([&] { return scheduler.next({}); }, 0),
          "idle dispatch returned work");
}

void testWarmedPlanningAllocations() {
  constexpr size_t count = 256;
  Scheduler queued(0.0), prefill(0.0), decode(0.0);
  std::array<PrefillAdmission, count> candidates{};
  for (uint64_t id = 1; id <= count; ++id) {
    queued.submit(request(id));
    prefill.submit(request(id));
    prefill.resourcesReady(id, 0);
    decode.submit(request(id));
    decode.resourcesReady(id, 65);
    candidates[id - 1] = {id, 0};
  }
  (void)queued.admissionOrder();
  (void)queued.prefillAdmissionOrder(candidates);
  (void)prefill.next({});
  (void)decode.next({});
  for (size_t repeat = 0; repeat < 4; ++repeat) {
    // Returned ID and BatchItem vectors still own their storage. At width 4
    // each plan grows at most three times; admission returns one ID vector.
    const auto order = measured([&] { return queued.admissionOrder(); }, 1);
    require(order.size() == count && order.front() == 1 && order.back() == count,
            "queued admission order changed");
    const auto selected = measured(
        [&] { return queued.prefillAdmissionOrder(candidates); }, 6);
    require(selected == std::vector<uint64_t>({1, 2, 3, 4}),
            "prefill admission order changed");
    const auto prefillPlan = measured([&] { return prefill.next({}); }, 3);
    const auto decodePlan = measured([&] { return decode.next({}); }, 3);
    require(prefillPlan && prefillPlan->kind == WorkKind::Prefill &&
                prefillPlan->width() == 4 && prefillPlan->items.front().requestId == 1,
            "prefill dispatch changed");
    require(decodePlan && decodePlan->kind == WorkKind::Decode &&
                decodePlan->width() == 4 && decodePlan->items.front().requestId == 1,
            "decode dispatch changed");
  }
}

void testViewsRebuiltAfterFailureAndRemoval() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1));
  scheduler.submit(request(2));
  const std::array candidates{PrefillAdmission{1, 0}, PrefillAdmission{2, 0}};
  (void)scheduler.prefillAdmissionOrder(candidates);
  // Fail after rebuilding the warmed candidate views, while constructing
  // the returned plan. Removing those requests must leave no usable views.
  failAfter = 0;
  bool failed = false;
  try {
    (void)scheduler.prefillAdmissionOrder(candidates);
  } catch (const std::bad_alloc &) {
    failed = true;
  }
  failAfter = -1;
  require(failed, "planning fault was not exercised");
  const std::array invalid{PrefillAdmission{1, 0}, PrefillAdmission{2, 65}};
  splash::test::rejects([&] { (void)scheduler.prefillAdmissionOrder(invalid); },
                       "invalid pending", "invalid admission was accepted");
  for (uint64_t id : {1U, 2U}) {
    scheduler.cancel(id);
    scheduler.remove(id);
  }
  require(!scheduler.next({}) && scheduler.admissionOrder().empty(),
          "removed requests survived in planning");
  scheduler.submit(request(3));
  scheduler.resourcesReady(3, 0);
  const auto retainedPlan = *scheduler.next({});
  scheduler.submit(request(4));
  const std::array fresh{PrefillAdmission{4, 64}};
  require(scheduler.prefillAdmissionOrder(fresh) == std::vector<uint64_t>({4}),
          "fresh candidate reused an old view");
  scheduler.resourcesReady(4, 64);
  const std::array excluded{uint64_t{4}};
  const auto plan = *scheduler.next(excluded);
  require(plan.width() == 1 && plan.items.front().requestId == 3 &&
              retainedPlan.items.front().requestId == 3 &&
              retainedPlan.items.front().tokenCount == 65,
          "new planning corrupted a retained plan or included an excluded lane");
  scheduler.cancel(3);
  scheduler.remove(3);
  require(scheduler.next({})->items.front().requestId == 4,
          "planning did not rebuild after request removal");
}

} // namespace

int main() {
  try {
    testIdleDoesNotAllocate();
    testWarmedPlanningAllocations();
    testViewsRebuiltAfterFailureAndRemoval();
    std::cout << "scheduler storage tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    countAllocations = false;
    failAfter = -1;
    std::cerr << "scheduler storage tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
