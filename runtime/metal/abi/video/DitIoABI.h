#pragma once

// DiT input embedding and final layer (outside the 50-block loop).
#include "metal/abi/video/Layout.h"

struct DitIoParams {
  int32_t L, Na, Kv, Ka;  // text rows, audio rows, video / audio latent channels
  int32_t D;
  float eps;
};
