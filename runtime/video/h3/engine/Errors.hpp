#pragma once

// The engine fails by exception, as Splash does. The C ABI (CApi.mm) returns each failure as its h3_status:
//   std::invalid_argument              H3_ERROR_ARGUMENT  request or inputs inconsistent
//   StateError                         H3_ERROR_STATE     phase called out of order
//   GpuError                           H3_ERROR_GPU       a command failed
//   splash::metal::MetalBackendError   H3_ERROR_METAL     device, kernel library, pipeline or allocation
//   any other std::exception           H3_ERROR_IO        file missing, unreadable or malformed
#include "metal/MetalBackend.hpp"

#include <stdexcept>

namespace h3 {

class StateError final : public std::logic_error {
public:
  using std::logic_error::logic_error;
};

class GpuError final : public splash::metal::MetalBackendError {
public:
  using splash::metal::MetalBackendError::MetalBackendError;
};

} // namespace h3
