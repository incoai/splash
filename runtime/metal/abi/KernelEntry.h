#pragma once

// The same row-sensitive source compiles in its ordinary eight-row namespace
// and in the sixteen-row namespace. Explicit host names keep pipeline keys
// distinct without changing the default eight-row names.
#ifndef SPLASH_KERNEL_SUFFIX
#define SPLASH_KERNEL_SUFFIX ""
#endif
#define SPLASH_KERNEL_STRING_IMPL(name) #name
#define SPLASH_KERNEL_STRING(name) SPLASH_KERNEL_STRING_IMPL(name)
#define SPLASH_KERNEL(name) \
  [[host_name(SPLASH_KERNEL_STRING(name) SPLASH_KERNEL_SUFFIX)]] kernel void name
