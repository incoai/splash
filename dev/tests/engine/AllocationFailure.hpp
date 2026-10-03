#pragma once

#include <cstdlib>
#include <new>

// Fault only the caller thread; the file worker must still be able to drain.
inline thread_local int allocationFailureAfter = -1;
void *operator new(size_t size) {
  if (allocationFailureAfter == 0) {
    allocationFailureAfter = -1;
    throw std::bad_alloc();
  }
  if (allocationFailureAfter > 0) --allocationFailureAfter;
  if (void *memory = std::malloc(size ? size : 1)) return memory;
  throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }
