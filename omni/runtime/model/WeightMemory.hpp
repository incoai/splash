#pragma once

namespace splash::model {

// Memory the engine gives back while idle and takes again before it runs the
// next request: a model's weights, which can be written again from their
// sources (WeightImages), with whatever else the engine gives back with them,
// or a test's stand-in.
class WeightMemory {
public:
  virtual ~WeightMemory() = default;
  // From release() until restore() has written the last image back.
  [[nodiscard]] virtual bool released() const noexcept = 0;
  // Gives the memory back while no command is in flight.
  virtual void release() = 0;
  // Takes back the next part released, such as an image it allocates and
  // writes again; true once none is left. On failure the weights are
  // unusable: the engine stops.
  [[nodiscard]] virtual bool restore() = 0;
};

} // namespace splash::model
