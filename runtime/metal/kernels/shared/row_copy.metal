#include "metal/abi/KernelABI.h"

// One thread per copied value (ops::RowCopy).
kernel void copy_rows_bf16(device const bfloat *source [[buffer(0)]],
                           device bfloat *destination [[buffer(1)]],
                           constant RowCopyParams &p [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.width)
    return;
  const uint row = index / p.width, column = index % p.width;
  destination[ulong(p.destination_row + row) * p.destination_stride +
              p.destination_column + column] =
      source[ulong(p.source_row + row) * p.source_stride + p.source_column +
             column];
}
