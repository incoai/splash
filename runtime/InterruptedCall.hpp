#pragma once

#include <cerrno>

namespace splash {

// Retry only the interrupted syscall, for at most eight attempts. The caller
// must choose an operation safe to repeat after EINTR; never use this for close.
template <class F> int retryInterrupted(F call) {
  for (unsigned attempt = 0; attempt < 8; ++attempt) {
    const int result = call();
    if (result >= 0 || errno != EINTR)
      return result;
  }
  return -1;
}

} // namespace splash
