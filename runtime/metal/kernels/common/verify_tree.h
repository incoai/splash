#pragma once

#include "metal/abi/VerifyTree.h"

// Row `depth` of a row mask, lowest first: of a verify block's path
// (metal/abi/VerifyTree.h), the row at that depth.
inline uint splash_path_row(uint path, uint depth) {
  for (uint i = 0; i < depth; ++i)
    path &= path - 1;
  return ctz(path);
}
