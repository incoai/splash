#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace splash::engine {

// The Mac's thermal state as macOS reports it (NSProcessInfo.thermalState).
// As the state rises, the CPU and GPU clocks drop, on some Macs already at
// fair, so prefill and decode slow down with nothing in the engine having
// changed. The engine reports it and goes on serving: heat is not a resource
// it can free.
enum class ThermalState : uint8_t {
  Nominal,
  Fair,
  Serious,
  Critical,
};

// One spelling of the states for status JSON and the log.
[[nodiscard]] inline const char *thermalStateName(ThermalState state) noexcept {
  switch (state) {
  case ThermalState::Nominal:
    return "nominal";
  case ThermalState::Fair:
    return "fair";
  case ThermalState::Serious:
    return "serious";
  case ThermalState::Critical:
    return "critical";
  }
  return "critical";
}

// The state macOS reports now; a state this build does not know reads as
// critical.
[[nodiscard]] ThermalState queryThermalState() noexcept;

// Emits only transitions, as MemoryStatusReporter does: the first state when
// it is not nominal, then each change. macOS changes the state over seconds
// to minutes of load, so the lines stay few.
class ThermalStateReporter final {
public:
  [[nodiscard]] std::string update(ThermalState state) {
    const std::optional<ThermalState> previous = state_;
    state_ = state;
    if (previous == state || (!previous && state == ThermalState::Nominal))
      return {};
    std::string line = "Thermal state: ";
    if (previous)
      line = line + thermalStateName(*previous) + " → ";
    return line + thermalStateName(state);
  }
  // The last state given, nominal before the first.
  [[nodiscard]] ThermalState state() const noexcept {
    return state_.value_or(ThermalState::Nominal);
  }

private:
  std::optional<ThermalState> state_;
};

} // namespace splash::engine
