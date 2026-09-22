#include "flash/FlashIdleResidencyPolicy.hpp"
#include "flash/FlashIdleResidencyScheduler.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace idle = splash::flash::idle_maintenance;
namespace {
uint64_t checks = 0;
void require(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
using Time = idle::Scheduler::Time;
using Milliseconds = std::chrono::milliseconds;
Time at(int64_t milliseconds) { return Time{Milliseconds{milliseconds}}; }
}

int main() {
  try {
    for (uint32_t interval = 100; interval <= 1000; ++interval) {
      idle::Scheduler scheduler(interval);
      require(!scheduler.armed(), "startup armed maintenance");
      require(!scheduler.hasUserGPUCommand(), "startup claimed user GPU completion");
      require(!scheduler.due(at(100000)), "startup timer due without successful user");
      scheduler.userFinished(at(100), true, false);
      require(!scheduler.armed(), "finish without user GPU work armed maintenance");

      scheduler.userGPUCompleted(at(200));
      require(scheduler.hasUserGPUCommand(), "user GPU completion was not recorded");
      require(scheduler.lastUserGPUCommand() == at(200), "user GPU timestamp changed");
      require(!scheduler.armed() && !scheduler.due(at(100000)),
              "user GPU command alone armed maintenance");
      scheduler.userFinished(at(210), false, false);
      require(!scheduler.armed(), "zero-output completion armed maintenance");
      scheduler.userFinished(at(220), true, true);
      require(!scheduler.armed(), "cancelled completion armed maintenance");

      scheduler.userFinished(at(250), true, false);
      require(scheduler.armed(), "successful user completion did not arm");
      require(!scheduler.due(at(250 + interval - 1)), "completion timer fired early");
      require(scheduler.due(at(250 + interval)), "completion timer missed exact boundary");
      require(scheduler.due(at(250 + interval + 1)), "completion timer not due after boundary");
      require(!scheduler.due(at(249)), "time before successful finish was due");

      scheduler.userGPUCompleted(at(2000));
      require(scheduler.armed(), "new user command incorrectly disarmed armed maintenance");
      require(!scheduler.due(at(2000 + interval - 1)), "user GPU reset timer fired early");
      require(scheduler.due(at(2000 + interval)), "user GPU reset timer missed boundary");
      scheduler.pressureSuspended(at(3500));
      require(scheduler.armed(), "pressure suspension unexpectedly disarmed");
      require(!scheduler.due(at(3500 + interval - 1)), "pressure retry timer fired early");
      require(scheduler.due(at(3500 + interval)), "pressure retry timer missed boundary");

      const double threshold = double(interval) / 1000;
      scheduler.maintenanceCompleted(at(5000), std::nextafter(threshold, 0.0));
      require(scheduler.armed(), "maintenance below cold threshold disarmed");
      require(!scheduler.due(at(5000 + interval - 1)), "maintenance retry fired early");
      require(scheduler.due(at(5000 + interval)), "maintenance retry missed boundary");
      require(scheduler.lastUserGPUCommand() == at(2000),
              "maintenance overwrote real user timestamp");
      scheduler.maintenanceCompleted(at(7000), threshold);
      require(!scheduler.armed(), "exact cold threshold did not disarm");
      require(!scheduler.due(at(100000)), "cold suspension remained due");
      scheduler.maintenanceCompleted(at(7100), 0.000001);
      require(!scheduler.armed(), "maintenance command rearmed cold suspension");
      scheduler.userGPUCompleted(at(8000));
      require(!scheduler.armed(), "user GPU command alone rearmed cold suspension");
      scheduler.userFinished(at(8100), true, true);
      require(!scheduler.armed(), "cancelled user rearmed cold suspension");
      scheduler.userFinished(at(8200), false, false);
      require(!scheduler.armed(), "zero-output user rearmed cold suspension");
      scheduler.userFinished(at(8300), true, false);
      require(scheduler.armed(), "successful user did not rearm cold suspension");
      require(!scheduler.due(at(8300 + interval - 1)), "rearm timer fired early");
      require(scheduler.due(at(8300 + interval)), "rearm timer missed boundary");

      scheduler.maintenanceCompleted(at(10000), std::nextafter(threshold,
          std::numeric_limits<double>::infinity()));
      require(!scheduler.armed(), "maintenance above cold threshold did not disarm");
      scheduler.userGPUCompleted(at(11000));
      scheduler.userFinished(at(11100), true, false);
      scheduler.failed();
      require(!scheduler.armed() && !scheduler.due(at(100000)),
              "maintenance failure left scheduler armed");
      require(scheduler.lastUserGPUCommand() == at(11000),
              "failure corrupted recorded user timestamp");
      scheduler.pressureSuspended(at(12000));
      require(!scheduler.armed(), "pressure helper rearmed failed scheduler");

      // Controls block a due timer independently of elapsed time. These are
      // the policy helpers called by the Worker, not an alternate scheduler.
      require(!idle::eligible(true, false, true, false, false, false, false, true, true),
              "active grammar-mask request allowed idle maintenance");
      require(!idle::eligible(true, true, false, false, false, false, false, true, true),
              "queued request allowed idle maintenance");
      require(!idle::eligible(true, true, true, true, false, false, true, true, true),
              "shutdown allowed idle maintenance");
      require(!idle::eligible(true, true, true, true, false, true, false, true, true),
              "outstanding backend command allowed idle maintenance");
      require(!idle::eligible(true, true, true, true, false, false, false, true, false),
              "incoming reader control allowed idle maintenance");
    }
    std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":" << checks
              << ",\"actual_scheduler_intervals\":901,"
                 "\"startup_arm_cancel_and_completion\":true,"
                 "\"exact_timer_boundaries\":true,\"pressure_retry\":true,"
                 "\"cold_miss_disarm_and_user_rearm\":true,\"shutdown_guards\":true}"
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "scheduler review failed: " << error.what() << '\n';
    return 1;
  }
}
