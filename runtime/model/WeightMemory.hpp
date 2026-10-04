#pragma once

namespace splash::model {

// The memory of a model's weights, which can be given back and written
// again from the weights' sources: WeightImages, or a test's stand-in.
class WeightMemory {
public:
  virtual ~WeightMemory() = default;
  // From release() until restore() has written the last image back.
  [[nodiscard]] virtual bool released() const noexcept = 0;
  // Frees the memory while no command is in flight.
  virtual void release() = 0;
  // Allocates and writes the next released image again; true once none is
  // left. On failure the weights are unusable: the engine stops.
  [[nodiscard]] virtual bool restore() = 0;
};

} // namespace splash::model
