#pragma once

#include "metal/MetalBackend.hpp"

#include <map>
#include <span>
#include <string>

namespace splash::benchmark {

// GPU seconds per pipeline of a command whose every dispatch is submitted as
// a command of its own.
inline std::map<std::string, double>
replayDispatches(metal::MetalBackend &backend,
                 std::span<const metal::ComputeDispatch> dispatches) {
  std::map<std::string, double> seconds;
  for (const metal::ComputeDispatch &dispatch : dispatches)
    seconds[dispatch.pipelineName] += backend.submit(dispatch).gpuSeconds;
  return seconds;
}

} // namespace splash::benchmark
