#include "engine/ThermalState.hpp"

#import <Foundation/Foundation.h>

namespace splash::engine {

ThermalState queryThermalState() noexcept {
  // The control pass asks twice a second for as long as the engine runs, on
  // a thread whose outer pool never drains; this one keeps whatever the
  // query autoreleases from piling up there.
  @autoreleasepool {
    switch (NSProcessInfo.processInfo.thermalState) {
    case NSProcessInfoThermalStateNominal:
      return ThermalState::Nominal;
    case NSProcessInfoThermalStateFair:
      return ThermalState::Fair;
    case NSProcessInfoThermalStateSerious:
      return ThermalState::Serious;
    case NSProcessInfoThermalStateCritical:
      return ThermalState::Critical;
    }
    return ThermalState::Critical;
  }
}

} // namespace splash::engine
