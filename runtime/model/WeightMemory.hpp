#pragma once

#include "metal/Allocation.hpp"

#include <functional>

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
  // Gives the memory back while no command is in flight. During a restore it
  // gives back the parts restore() took back, which the next restore() starts
  // over from.
  virtual void release() = 0;
  // Takes back the next part released, such as an image whose memory `admit`
  // admits and which it writes again, in one admission; true once none is
  // left. A part whose memory admission or the driver refuses stays released
  // and throws metal::MetalAllocationError with the cause. Memory a part can
  // do without, such as an expert store's chunks above its floor
  // (model::ExpertStore), does not come back: the store grows it again as
  // memory allows. On any other failure the weights are unusable: the engine
  // stops.
  // A part may go on being written on a thread of its own once the call
  // returns (WeightImages), which calls the completion notifier as it ends:
  // writing() is true until then, and the next call, which may come only
  // after it, throws that write's failure as the call that wrote the part
  // would have.
  [[nodiscard]] virtual bool restore(const metal::AllocationAdmission &admit) = 0;
  // A part restore() took back is still being written on its thread, which
  // may allocate and run commands on the backend meanwhile.
  [[nodiscard]] virtual bool writing() const noexcept { return false; }
  // Wakes the engine's loop as the write of a part on its thread ends, for
  // the writes restore() starts from then on; a part written before restore()
  // returns never calls it.
  virtual void setCompletionNotifier(std::function<void()>) {}
};

} // namespace splash::model
