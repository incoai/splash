#pragma once
#include "metal/abi/ExecutionGeometry.h"

// Keep the argument list and buffer indices identical at the default width.
#if SPLASH_MAXIMUM_BATCH_WIDTH == 8u
#define SPLASH_LANE_BUFFERS(Type, Name, Base) \
  Type Name##0 [[buffer((Base) + 0)]], \
  Type Name##1 [[buffer((Base) + 1)]], \
  Type Name##2 [[buffer((Base) + 2)]], \
  Type Name##3 [[buffer((Base) + 3)]], \
  Type Name##4 [[buffer((Base) + 4)]], \
  Type Name##5 [[buffer((Base) + 5)]], \
  Type Name##6 [[buffer((Base) + 6)]], \
  Type Name##7 [[buffer((Base) + 7)]]
#define SPLASH_LANE_PARAMETERS(Type, Name) \
  Type Name##0, \
  Type Name##1, \
  Type Name##2, \
  Type Name##3, \
  Type Name##4, \
  Type Name##5, \
  Type Name##6, \
  Type Name##7
#define SPLASH_LANE_ARGUMENTS(Name) \
  Name##0, \
  Name##1, \
  Name##2, \
  Name##3, \
  Name##4, \
  Name##5, \
  Name##6, \
  Name##7
#define SPLASH_LANE_BINDING(lane, lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7) ((lane) == 0 ? (lane0) : ((lane) == 1 ? (lane1) : ((lane) == 2 ? (lane2) : ((lane) == 3 ? (lane3) : ((lane) == 4 ? (lane4) : ((lane) == 5 ? (lane5) : ((lane) == 6 ? (lane6) : (lane7))))))))
#else
#define SPLASH_LANE_BUFFERS(Type, Name, Base) \
  Type Name##0 [[buffer((Base) + 0)]], \
  Type Name##1 [[buffer((Base) + 1)]], \
  Type Name##2 [[buffer((Base) + 2)]], \
  Type Name##3 [[buffer((Base) + 3)]]
#define SPLASH_LANE_PARAMETERS(Type, Name) \
  Type Name##0, \
  Type Name##1, \
  Type Name##2, \
  Type Name##3
#define SPLASH_LANE_ARGUMENTS(Name) \
  Name##0, \
  Name##1, \
  Name##2, \
  Name##3
#define SPLASH_LANE_BINDING(lane, lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7) ((lane) == 0 ? (lane0) : ((lane) == 1 ? (lane1) : ((lane) == 2 ? (lane2) : (lane3))))
#endif
