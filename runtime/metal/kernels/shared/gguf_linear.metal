// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal. Keep activations BF16.
// Weight layout: plane0 [tile(256 cols)][group32][256 cols][P0 bytes], plane1 likewise with P1 bytes (0 = none).
// Metadata: [tile][unit][256 cols][MetaBytes], unit = super-block (256 K) or group32 (IQ4_NL, Q8_0).
// Activations fp16 or bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output.
#include "metal/abi/Gguf.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;
enum Epilogue : ushort { EpNone = GGUF_EPILOGUE_NONE, EpResidual = GGUF_EPILOGUE_RESIDUAL, EpUpWithGate = GGUF_EPILOGUE_UP_WITH_GATE };
inline float silu_gate(float g) { return g / (1.0f + fast::exp2(-1.44269504089f * g)); }
constant constexpr ushort kStorageN = 256;
constant half kIQ4NL[16] = {-127.0h, -104.0h, -83.0h, -65.0h, -49.0h, -35.0h, -22.0h, -10.0h, 1.0h, 13.0h, 25.0h, 38.0h, 53.0h, 69.0h, 89.0h, 113.0h};
constant half2 kIQ4NL2[256] = {half2(-127.0h, -127.0h), half2(-104.0h, -127.0h), half2(-83.0h, -127.0h), half2(-65.0h, -127.0h), half2(-49.0h, -127.0h), half2(-35.0h, -127.0h), half2(-22.0h, -127.0h), half2(-10.0h, -127.0h), half2(1.0h, -127.0h), half2(13.0h, -127.0h), half2(25.0h, -127.0h), half2(38.0h, -127.0h), half2(53.0h, -127.0h), half2(69.0h, -127.0h), half2(89.0h, -127.0h), half2(113.0h, -127.0h), half2(-127.0h, -104.0h), half2(-104.0h, -104.0h), half2(-83.0h, -104.0h), half2(-65.0h, -104.0h), half2(-49.0h, -104.0h), half2(-35.0h, -104.0h), half2(-22.0h, -104.0h), half2(-10.0h, -104.0h), half2(1.0h, -104.0h), half2(13.0h, -104.0h), half2(25.0h, -104.0h), half2(38.0h, -104.0h), half2(53.0h, -104.0h), half2(69.0h, -104.0h), half2(89.0h, -104.0h), half2(113.0h, -104.0h), half2(-127.0h, -83.0h), half2(-104.0h, -83.0h), half2(-83.0h, -83.0h), half2(-65.0h, -83.0h), half2(-49.0h, -83.0h), half2(-35.0h, -83.0h), half2(-22.0h, -83.0h), half2(-10.0h, -83.0h), half2(1.0h, -83.0h), half2(13.0h, -83.0h), half2(25.0h, -83.0h), half2(38.0h, -83.0h), half2(53.0h, -83.0h), half2(69.0h, -83.0h), half2(89.0h, -83.0h), half2(113.0h, -83.0h), half2(-127.0h, -65.0h), half2(-104.0h, -65.0h), half2(-83.0h, -65.0h), half2(-65.0h, -65.0h), half2(-49.0h, -65.0h), half2(-35.0h, -65.0h), half2(-22.0h, -65.0h), half2(-10.0h, -65.0h), half2(1.0h, -65.0h), half2(13.0h, -65.0h), half2(25.0h, -65.0h), half2(38.0h, -65.0h), half2(53.0h, -65.0h), half2(69.0h, -65.0h), half2(89.0h, -65.0h), half2(113.0h, -65.0h), half2(-127.0h, -49.0h), half2(-104.0h, -49.0h), half2(-83.0h, -49.0h), half2(-65.0h, -49.0h), half2(-49.0h, -49.0h), half2(-35.0h, -49.0h), half2(-22.0h, -49.0h), half2(-10.0h, -49.0h), half2(1.0h, -49.0h), half2(13.0h, -49.0h), half2(25.0h, -49.0h), half2(38.0h, -49.0h), half2(53.0h, -49.0h), half2(69.0h, -49.0h), half2(89.0h, -49.0h), half2(113.0h, -49.0h), half2(-127.0h, -35.0h), half2(-104.0h, -35.0h), half2(-83.0h, -35.0h), half2(-65.0h, -35.0h), half2(-49.0h, -35.0h), half2(-35.0h, -35.0h), half2(-22.0h, -35.0h), half2(-10.0h, -35.0h), half2(1.0h, -35.0h), half2(13.0h, -35.0h), half2(25.0h, -35.0h), half2(38.0h, -35.0h), half2(53.0h, -35.0h), half2(69.0h, -35.0h), half2(89.0h, -35.0h), half2(113.0h, -35.0h), half2(-127.0h, -22.0h), half2(-104.0h, -22.0h), half2(-83.0h, -22.0h), half2(-65.0h, -22.0h), half2(-49.0h, -22.0h), half2(-35.0h, -22.0h), half2(-22.0h, -22.0h), half2(-10.0h, -22.0h), half2(1.0h, -22.0h), half2(13.0h, -22.0h), half2(25.0h, -22.0h), half2(38.0h, -22.0h), half2(53.0h, -22.0h), half2(69.0h, -22.0h), half2(89.0h, -22.0h), half2(113.0h, -22.0h), half2(-127.0h, -10.0h), half2(-104.0h, -10.0h), half2(-83.0h, -10.0h), half2(-65.0h, -10.0h), half2(-49.0h, -10.0h), half2(-35.0h, -10.0h), half2(-22.0h, -10.0h), half2(-10.0h, -10.0h), half2(1.0h, -10.0h), half2(13.0h, -10.0h), half2(25.0h, -10.0h), half2(38.0h, -10.0h), half2(53.0h, -10.0h), half2(69.0h, -10.0h), half2(89.0h, -10.0h), half2(113.0h, -10.0h), half2(-127.0h, 1.0h), half2(-104.0h, 1.0h), half2(-83.0h, 1.0h), half2(-65.0h, 1.0h), half2(-49.0h, 1.0h), half2(-35.0h, 1.0h), half2(-22.0h, 1.0h), half2(-10.0h, 1.0h), half2(1.0h, 1.0h), half2(13.0h, 1.0h), half2(25.0h, 1.0h), half2(38.0h, 1.0h), half2(53.0h, 1.0h), half2(69.0h, 1.0h), half2(89.0h, 1.0h), half2(113.0h, 1.0h), half2(-127.0h, 13.0h), half2(-104.0h, 13.0h), half2(-83.0h, 13.0h), half2(-65.0h, 13.0h), half2(-49.0h, 13.0h), half2(-35.0h, 13.0h), half2(-22.0h, 13.0h), half2(-10.0h, 13.0h), half2(1.0h, 13.0h), half2(13.0h, 13.0h), half2(25.0h, 13.0h), half2(38.0h, 13.0h), half2(53.0h, 13.0h), half2(69.0h, 13.0h), half2(89.0h, 13.0h), half2(113.0h, 13.0h), half2(-127.0h, 25.0h), half2(-104.0h, 25.0h), half2(-83.0h, 25.0h), half2(-65.0h, 25.0h), half2(-49.0h, 25.0h), half2(-35.0h, 25.0h), half2(-22.0h, 25.0h), half2(-10.0h, 25.0h), half2(1.0h, 25.0h), half2(13.0h, 25.0h), half2(25.0h, 25.0h), half2(38.0h, 25.0h), half2(53.0h, 25.0h), half2(69.0h, 25.0h), half2(89.0h, 25.0h), half2(113.0h, 25.0h), half2(-127.0h, 38.0h), half2(-104.0h, 38.0h), half2(-83.0h, 38.0h), half2(-65.0h, 38.0h), half2(-49.0h, 38.0h), half2(-35.0h, 38.0h), half2(-22.0h, 38.0h), half2(-10.0h, 38.0h), half2(1.0h, 38.0h), half2(13.0h, 38.0h), half2(25.0h, 38.0h), half2(38.0h, 38.0h), half2(53.0h, 38.0h), half2(69.0h, 38.0h), half2(89.0h, 38.0h), half2(113.0h, 38.0h), half2(-127.0h, 53.0h), half2(-104.0h, 53.0h), half2(-83.0h, 53.0h), half2(-65.0h, 53.0h), half2(-49.0h, 53.0h), half2(-35.0h, 53.0h), half2(-22.0h, 53.0h), half2(-10.0h, 53.0h), half2(1.0h, 53.0h), half2(13.0h, 53.0h), half2(25.0h, 53.0h), half2(38.0h, 53.0h), half2(53.0h, 53.0h), half2(69.0h, 53.0h), half2(89.0h, 53.0h), half2(113.0h, 53.0h), half2(-127.0h, 69.0h), half2(-104.0h, 69.0h), half2(-83.0h, 69.0h), half2(-65.0h, 69.0h), half2(-49.0h, 69.0h), half2(-35.0h, 69.0h), half2(-22.0h, 69.0h), half2(-10.0h, 69.0h), half2(1.0h, 69.0h), half2(13.0h, 69.0h), half2(25.0h, 69.0h), half2(38.0h, 69.0h), half2(53.0h, 69.0h), half2(69.0h, 69.0h), half2(89.0h, 69.0h), half2(113.0h, 69.0h), half2(-127.0h, 89.0h), half2(-104.0h, 89.0h), half2(-83.0h, 89.0h), half2(-65.0h, 89.0h), half2(-49.0h, 89.0h), half2(-35.0h, 89.0h), half2(-22.0h, 89.0h), half2(-10.0h, 89.0h), half2(1.0h, 89.0h), half2(13.0h, 89.0h), half2(25.0h, 89.0h), half2(38.0h, 89.0h), half2(53.0h, 89.0h), half2(69.0h, 89.0h), half2(89.0h, 89.0h), half2(113.0h, 89.0h), half2(-127.0h, 113.0h), half2(-104.0h, 113.0h), half2(-83.0h, 113.0h), half2(-65.0h, 113.0h), half2(-49.0h, 113.0h), half2(-35.0h, 113.0h), half2(-22.0h, 113.0h), half2(-10.0h, 113.0h), half2(1.0h, 113.0h), half2(13.0h, 113.0h), half2(25.0h, 113.0h), half2(38.0h, 113.0h), half2(53.0h, 113.0h), half2(69.0h, 113.0h), half2(89.0h, 113.0h), half2(113.0h, 113.0h)};
constant uint kIQ3S_GRID[512] = {0x01010101,0x01010103,0x01010105,0x0101010b,0x0101010f,0x01010301,0x01010303,0x01010305,0x01010309,0x0101030d,0x01010501,0x01010503,0x0101050b,0x01010707,0x01010901,0x01010905,0x0101090b,0x0101090f,0x01010b03,0x01010b07,0x01010d01,0x01010d05,0x01010f03,0x01010f09,0x01010f0f,0x01030101,0x01030103,0x01030105,0x01030109,0x01030301,0x01030303,0x0103030b,0x01030501,0x01030507,0x0103050f,0x01030703,0x0103070b,0x01030909,0x01030d03,0x01030d0b,0x01030f05,0x01050101,0x01050103,0x0105010b,0x0105010f,0x01050301,0x01050307,0x0105030d,0x01050503,0x0105050b,0x01050701,0x01050709,0x01050905,0x0105090b,0x0105090f,0x01050b03,0x01050b07,0x01050f01,0x01050f07,0x01070107,0x01070303,0x0107030b,0x01070501,0x01070505,0x01070703,0x01070707,0x0107070d,0x01070909,0x01070b01,0x01070b05,0x01070d0f,0x01070f03,0x01070f0b,0x01090101,0x01090307,0x0109030f,0x01090503,0x01090509,0x01090705,0x01090901,0x01090907,0x01090b03,0x01090f01,0x010b0105,0x010b0109,0x010b0501,0x010b0505,0x010b050d,0x010b0707,0x010b0903,0x010b090b,0x010b090f,0x010b0d0d,0x010b0f07,0x010d010d,0x010d0303,0x010d0307,0x010d0703,0x010d0b05,0x010d0f03,0x010f0101,0x010f0105,0x010f0109,0x010f0501,0x010f0505,0x010f050d,0x010f0707,0x010f0b01,0x010f0b09,0x03010101,0x03010103,0x03010105,0x03010109,0x03010301,0x03010303,0x03010307,0x0301030b,0x0301030f,0x03010501,0x03010505,0x03010703,0x03010709,0x0301070d,0x03010b09,0x03010b0d,0x03010d03,0x03010f05,0x03030101,0x03030103,0x03030107,0x0303010d,0x03030301,0x03030309,0x03030503,0x03030701,0x03030707,0x03030903,0x03030b01,0x03030b05,0x03030f01,0x03030f0d,0x03050101,0x03050305,0x0305030b,0x0305030f,0x03050501,0x03050509,0x03050705,0x03050901,0x03050907,0x03050b0b,0x03050d01,0x03050f05,0x03070103,0x03070109,0x0307010f,0x03070301,0x03070307,0x03070503,0x0307050f,0x03070701,0x03070709,0x03070903,0x03070d05,0x03070f01,0x03090107,0x0309010b,0x03090305,0x03090309,0x03090703,0x03090707,0x03090905,0x0309090d,0x03090b01,0x03090b09,0x030b0103,0x030b0301,0x030b0307,0x030b0503,0x030b0701,0x030b0705,0x030b0b03,0x030d0501,0x030d0509,0x030d050f,0x030d0909,0x030d090d,0x030f0103,0x030f0107,0x030f0301,0x030f0305,0x030f0503,0x030f070b,0x030f0903,0x030f0d05,0x030f0f01,0x05010101,0x05010103,0x05010107,0x0501010b,0x0501010f,0x05010301,0x05010305,0x05010309,0x0501030d,0x05010503,0x05010507,0x0501050f,0x05010701,0x05010705,0x05010903,0x05010907,0x0501090b,0x05010b01,0x05010b05,0x05010d0f,0x05010f01,0x05010f07,0x05010f0b,0x05030101,0x05030105,0x05030301,0x05030307,0x0503030f,0x05030505,0x0503050b,0x05030703,0x05030709,0x05030905,0x05030b03,0x05050103,0x05050109,0x0505010f,0x05050503,0x05050507,0x05050701,0x0505070f,0x05050903,0x05050b07,0x05050b0f,0x05050f03,0x05050f09,0x05070101,0x05070105,0x0507010b,0x05070303,0x05070505,0x05070509,0x05070703,0x05070707,0x05070905,0x05070b01,0x05070d0d,0x05090103,0x0509010f,0x05090501,0x05090507,0x05090705,0x0509070b,0x05090903,0x05090f05,0x05090f0b,0x050b0109,0x050b0303,0x050b0505,0x050b070f,0x050b0901,0x050b0b07,0x050b0f01,0x050d0101,0x050d0105,0x050d010f,0x050d0503,0x050d0b0b,0x050d0d03,0x050f010b,0x050f0303,0x050f050d,0x050f0701,0x050f0907,0x050f0b01,0x07010105,0x07010303,0x07010307,0x0701030b,0x0701030f,0x07010505,0x07010703,0x07010707,0x0701070b,0x07010905,0x07010909,0x0701090f,0x07010b03,0x07010d07,0x07010f03,0x07030103,0x07030107,0x0703010b,0x07030309,0x07030503,0x07030507,0x07030901,0x07030d01,0x07030f05,0x07030f0d,0x07050101,0x07050305,0x07050501,0x07050705,0x07050709,0x07050b01,0x07070103,0x07070301,0x07070309,0x07070503,0x07070507,0x0707050f,0x07070701,0x07070903,0x07070907,0x0707090f,0x07070b0b,0x07070f07,0x07090107,0x07090303,0x0709030d,0x07090505,0x07090703,0x07090b05,0x07090d01,0x07090d09,0x070b0103,0x070b0301,0x070b0305,0x070b050b,0x070b0705,0x070b0909,0x070b0b0d,0x070b0f07,0x070d030d,0x070d0903,0x070f0103,0x070f0107,0x070f0501,0x070f0505,0x070f070b,0x09010101,0x09010109,0x09010305,0x09010501,0x09010509,0x0901050f,0x09010705,0x09010903,0x09010b01,0x09010f01,0x09030105,0x0903010f,0x09030303,0x09030307,0x09030505,0x09030701,0x0903070b,0x09030907,0x09030b03,0x09030b0b,0x09050103,0x09050107,0x09050301,0x0905030b,0x09050503,0x09050707,0x09050901,0x09050b0f,0x09050d05,0x09050f01,0x09070109,0x09070303,0x09070307,0x09070501,0x09070505,0x09070703,0x0907070b,0x09090101,0x09090105,0x09090509,0x0909070f,0x09090901,0x09090f03,0x090b010b,0x090b010f,0x090b0503,0x090b0d05,0x090d0307,0x090d0709,0x090d0d01,0x090f0301,0x090f030b,0x090f0701,0x090f0907,0x090f0b03,0x0b010105,0x0b010301,0x0b010309,0x0b010505,0x0b010901,0x0b010909,0x0b01090f,0x0b010b05,0x0b010d0d,0x0b010f09,0x0b030103,0x0b030107,0x0b03010b,0x0b030305,0x0b030503,0x0b030705,0x0b030f05,0x0b050101,0x0b050303,0x0b050507,0x0b050701,0x0b05070d,0x0b050b07,0x0b070105,0x0b07010f,0x0b070301,0x0b07050f,0x0b070909,0x0b070b03,0x0b070d0b,0x0b070f07,0x0b090103,0x0b090109,0x0b090501,0x0b090705,0x0b09090d,0x0b0b0305,0x0b0b050d,0x0b0b0b03,0x0b0b0b07,0x0b0d0905,0x0b0f0105,0x0b0f0109,0x0b0f0505,0x0d010303,0x0d010307,0x0d01030b,0x0d010703,0x0d010707,0x0d010d01,0x0d030101,0x0d030501,0x0d03050f,0x0d030d09,0x0d050305,0x0d050709,0x0d050905,0x0d050b0b,0x0d050d05,0x0d050f01,0x0d070101,0x0d070309,0x0d070503,0x0d070901,0x0d09050b,0x0d090907,0x0d090d05,0x0d0b0101,0x0d0b0107,0x0d0b0709,0x0d0b0d01,0x0d0d010b,0x0d0d0901,0x0d0f0303,0x0d0f0307,0x0f010101,0x0f010109,0x0f01010f,0x0f010501,0x0f010505,0x0f01070d,0x0f010901,0x0f010b09,0x0f010d05,0x0f030105,0x0f030303,0x0f030509,0x0f030907,0x0f03090b,0x0f050103,0x0f050109,0x0f050301,0x0f05030d,0x0f050503,0x0f050701,0x0f050b03,0x0f070105,0x0f070705,0x0f07070b,0x0f070b07,0x0f090103,0x0f09010b,0x0f090307,0x0f090501,0x0f090b01,0x0f0b0505,0x0f0b0905,0x0f0d0105,0x0f0d0703,0x0f0f0101};

// ---- shared helpers: 8 weights per word. Interleaved order: weights (2p, 2p+1) at bits 4p and 16+4p.
inline void store_affine8(uint v, float2 s2, float2 m2, threadgroup half *dst) {
  const half2 k = half2(1024.0h);
  half4 o0, o1;
  o0.xy = half2(fma(float2(as_type<half2>(((v >> 0) & 0x000F000Fu) | 0x64006400u) - k), s2, m2));
  o0.zw = half2(fma(float2(as_type<half2>(((v >> 4) & 0x000F000Fu) | 0x64006400u) - k), s2, m2));
  o1.xy = half2(fma(float2(as_type<half2>(((v >> 8) & 0x000F000Fu) | 0x64006400u) - k), s2, m2));
  o1.zw = half2(fma(float2(as_type<half2>(((v >> 12) & 0x000F000Fu) | 0x64006400u) - k), s2, m2));
  *((threadgroup half4 *)dst) = o0; *((threadgroup half4 *)(dst + 4)) = o1;
}
// IQ4 codebook, interleaved order, 16-entry constant table
template <typename Scale>
inline void store_lut8(uint v, Scale s2, threadgroup half *dst) {
  half4 o0, o1;
  o0.xy = half2(Scale(half2(kIQ4NL[(v >> 0) & 15], kIQ4NL[(v >> 16) & 15])) * s2);
  o0.zw = half2(Scale(half2(kIQ4NL[(v >> 4) & 15], kIQ4NL[(v >> 20) & 15])) * s2);
  o1.xy = half2(Scale(half2(kIQ4NL[(v >> 8) & 15], kIQ4NL[(v >> 24) & 15])) * s2);
  o1.zw = half2(Scale(half2(kIQ4NL[(v >> 12) & 15], kIQ4NL[(v >> 28) & 15])) * s2);
  *((threadgroup half4 *)dst) = o0; *((threadgroup half4 *)(dst + 4)) = o1;
}
// IQ4 codebook, natural byte order (weights 2p, 2p+1 in byte p), 256-entry half2 table (constant or threadgroup)
template <typename Scale>
inline void store_lutb8(uint v, Scale s2, threadgroup half *dst) {
  half4 o0, o1;
  o0.xy = half2(Scale(kIQ4NL2[v & 255]) * s2); o0.zw = half2(Scale(kIQ4NL2[(v >> 8) & 255]) * s2);
  o1.xy = half2(Scale(kIQ4NL2[(v >> 16) & 255]) * s2); o1.zw = half2(Scale(kIQ4NL2[v >> 24]) * s2);
  *((threadgroup half4 *)dst) = o0; *((threadgroup half4 *)(dst + 4)) = o1;
}
template <typename Scale>
inline void store_lutt8(uint v, Scale s2, threadgroup half2 *tl, threadgroup half *dst) {
  half4 o0, o1;
  o0.xy = half2(Scale(tl[v & 255]) * s2); o0.zw = half2(Scale(tl[(v >> 8) & 255]) * s2);
  o1.xy = half2(Scale(tl[(v >> 16) & 255]) * s2); o1.zw = half2(Scale(tl[v >> 24]) * s2);
  *((threadgroup half4 *)dst) = o0; *((threadgroup half4 *)(dst + 4)) = o1;
}
inline void k4_scale_min(uint4 hdr, ushort j, thread float2 &s2, thread float2 &m2) {   // block_q4_K / block_q5_K header
  // scales[12] = bytes of hdr.y (0-3), hdr.z (4-7), hdr.w (8-11). Extracted with shifts: a thread-local byte array indexed
  // by the group costs ~8% of the M=8 kernel time on Apple10 (the dequantizer is ALU-bound at eight rows).
  uint sc, m;
  if (j < 4) {
    const uint sh = 8u * j;
    sc = (hdr.y >> sh) & 63u; m = (hdr.z >> sh) & 63u;
  } else {
    const uint sh = 8u * (j - 4), w = hdr.w >> sh;
    sc = (w & 0xFu) | (((hdr.y >> sh) >> 6) & 3u) << 4; m = ((w >> 4) & 0xFu) | (((hdr.z >> sh) >> 6) & 3u) << 4;
  }
  const half d = as_type<half>(ushort(hdr.x & 0xFFFF)), dmin = as_type<half>(ushort(hdr.x >> 16));
  s2 = float2(float(d) * float(sc)); m2 = float2(-float(dmin) * float(m));
}

struct P16 { uint4 a; };
struct P20 { uint4 a; uint b; };
struct P24 { uint4 a; uint2 b; };
struct P12 { uint2 a; uint b; };
struct P32 { uint4 a; uint4 b; };

struct FmtQ4K {
  enum : uint { P0 = 16, P1 = 0, MetaBytes = 16 }; enum : ushort { MetaGroups = 8, TgLut = 0 };
  typedef P16 Payload; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static void dequant32(Payload w, Meta hdr, ushort j, threadgroup half2 *, threadgroup half *dst) {
    float2 s2, m2; k4_scale_min(hdr, j, s2, m2);
#pragma unroll
    for (ushort k = 0; k < 4; ++k) store_affine8(w.a[k], s2, m2, dst + 8 * k);
  }
};
// Lut 0: interleaved nibbles + 16-entry constant table; 2: byte order + 256-entry constant half2 table; 3: byte order + threadgroup table
template <int Lut> struct FmtIQ4XS {
  enum : uint { P0 = 16, P1 = 0, MetaBytes = 8 }; enum : ushort { MetaGroups = 8, TgLut = Lut == 3 ? 256 : 0 };
  typedef P16 Payload; typedef uint2 Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint2 *)m); }
  static void dequant32(Payload w, Meta mt, ushort j, threadgroup half2 *tl, threadgroup half *dst) {
    const half d = as_type<half>(ushort(mt.x & 0xFFFF)); const uint sh = mt.x >> 16;
    const int ls = int((mt.y >> (4 * j)) & 0xF) | int(((sh >> (2 * j)) & 3) << 4);
    const float2 s2 = float2(float(d) * float(ls - 32));
#pragma unroll
    for (ushort k = 0; k < 4; ++k) {
      if constexpr (Lut == 0) store_lut8(w.a[k], s2, dst + 8 * k);
      else if constexpr (Lut == 2) store_lutb8(w.a[k], s2, dst + 8 * k);
      else store_lutt8(w.a[k], s2, tl, dst + 8 * k);
    }
  }
};
template <int Lut> struct FmtIQ4NL {
  enum : uint { P0 = 16, P1 = 0, MetaBytes = 2 }; enum : ushort { MetaGroups = 1, TgLut = Lut == 3 ? 256 : 0 };
  typedef P16 Payload; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static void dequant32(Payload w, Meta mt, ushort, threadgroup half2 *tl, threadgroup half *dst) {
    const half2 s2 = half2(as_type<half>(mt));
#pragma unroll
    for (ushort k = 0; k < 4; ++k) {
      if constexpr (Lut == 0) store_lut8(w.a[k], s2, dst + 8 * k);
      else if constexpr (Lut == 2) store_lutb8(w.a[k], s2, dst + 8 * k);
      else store_lutt8(w.a[k], s2, tl, dst + 8 * k);
    }
  }
};
// Q5_K: plane0 interleaved low nibbles; plane1 one uint of 5th bits: bit (4k+p) = weight 8k+2p, bit (16+4k+p) = weight 8k+2p+1
struct FmtQ5K {
  enum : uint { P0 = 16, P1 = 4, MetaBytes = 16 }; enum : ushort { MetaGroups = 8, TgLut = 0 };
  typedef P20 Payload; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static void dequant32(Payload w, Meta hdr, ushort j, threadgroup half2 *, threadgroup half *dst) {
    float2 s2, m2; k4_scale_min(hdr, j, s2, m2);
    const half2 k = half2(1024.0h);
#pragma unroll
    for (ushort kk = 0; kk < 4; ++kk) {
      const uint v = w.a[kk]; half4 o0, o1;
      o0.xy = half2(fma(float2(as_type<half2>(((v >> 0) & 0x000F000Fu) | (((w.b >> (4 * kk + 0)) & 0x00010001u) << 4) | 0x64006400u) - k), s2, m2));
      o0.zw = half2(fma(float2(as_type<half2>(((v >> 4) & 0x000F000Fu) | (((w.b >> (4 * kk + 1)) & 0x00010001u) << 4) | 0x64006400u) - k), s2, m2));
      o1.xy = half2(fma(float2(as_type<half2>(((v >> 8) & 0x000F000Fu) | (((w.b >> (4 * kk + 2)) & 0x00010001u) << 4) | 0x64006400u) - k), s2, m2));
      o1.zw = half2(fma(float2(as_type<half2>(((v >> 12) & 0x000F000Fu) | (((w.b >> (4 * kk + 3)) & 0x00010001u) << 4) | 0x64006400u) - k), s2, m2));
      *((threadgroup half4 *)(dst + 8 * kk)) = o0; *((threadgroup half4 *)(dst + 8 * kk + 4)) = o1;
    }
  }
};
// Q6_K: plane0 interleaved low nibbles; plane1 two uints of 2-bit highs (H_h for weights 16h..16h+15): bits 2(4k'+p) / 16+2(4k'+p), k' = k&1.
// Meta 20 B: packed_uint4 of 16 int8 scales, then half d. value = d*sc[2j+h]*(q6-32).
struct FmtQ6K {
  enum : uint { P0 = 16, P1 = 8, MetaBytes = 20 }; enum : ushort { MetaGroups = 8, TgLut = 0 };
  typedef P24 Payload; struct Meta { packed_uint4 sc; uint d; };
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint2 *)p1)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.sc = *((device packed_uint4 *)m); r.d = *((device uint *)(m + 16)); return r; }
  static void dequant32(Payload w, Meta mt, ushort j, threadgroup half2 *, threadgroup half *dst) {
    const float d = float(as_type<half>(ushort(mt.d & 0xFFFF)));
    // scales 4(j>>1) .. +3 in word j>>1; we need int8 scales 2j, 2j+1 = bytes 2(j&1), 2(j&1)+1 (shifts, no dynamic vector index)
    const uint scw = j < 2 ? mt.sc.x : j < 4 ? mt.sc.y : j < 6 ? mt.sc.z : mt.sc.w;
    const uint pair = scw >> (16u * (j & 1));
    const float2 sA = float2(d * float(int(as_type<char>(uchar(pair & 0xFFu))))), sB = float2(d * float(int(as_type<char>(uchar((pair >> 8) & 0xFFu)))));
    const half2 k = half2(1056.0h);   // 1024 + 32
#pragma unroll
    for (ushort kk = 0; kk < 4; ++kk) {
      const uint v = w.a[kk], H = w.b[kk >> 1]; const ushort kp = (kk & 1) * 8; const float2 s2 = kk < 2 ? sA : sB;
      half4 o0, o1;
      o0.xy = half2(float2(as_type<half2>(((v >> 0) & 0x000F000Fu) | (((H >> (kp + 0)) & 0x00030003u) << 4) | 0x64006400u) - k) * s2);
      o0.zw = half2(float2(as_type<half2>(((v >> 4) & 0x000F000Fu) | (((H >> (kp + 2)) & 0x00030003u) << 4) | 0x64006400u) - k) * s2);
      o1.xy = half2(float2(as_type<half2>(((v >> 8) & 0x000F000Fu) | (((H >> (kp + 4)) & 0x00030003u) << 4) | 0x64006400u) - k) * s2);
      o1.zw = half2(float2(as_type<half2>(((v >> 12) & 0x000F000Fu) | (((H >> (kp + 6)) & 0x00030003u) << 4) | 0x64006400u) - k) * s2);
      *((threadgroup half4 *)(dst + 8 * kk)) = o0; *((threadgroup half4 *)(dst + 8 * kk + 4)) = o1;
    }
  }
};
// Q3_K: plane0 two uints of 2-bit codes (word h = pairs 8h..8h+7: lo weight at bits 2i', hi weight at 16+2i'), plane1 one uint of high bits
// (pair i: lo at bit i, hi at 16+i). Meta 16 B: half d, 2 pad, 12 scale bytes. value = d*(sc6-32)*((q2 | h<<2) - 4).
struct FmtQ3K {
  enum : uint { P0 = 8, P1 = 4, MetaBytes = 16 }; enum : ushort { MetaGroups = 8, TgLut = 0 };
  typedef P12 Payload; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static void dequant32(Payload w, Meta mt, ushort j, threadgroup half2 *, threadgroup half *dst) {
    const float d = float(as_type<half>(ushort(mt.x & 0xFFFF)));
    const uint t0 = mt.y, t1 = mt.z, t2 = mt.w;   // scales[0..3], [4..7], [8..11]
    uint aux;
    switch (j >> 1) {
      case 0: aux = (t0 & 0x0f0f0f0fu) | (((t2 >> 0) & 0x03030303u) << 4); break;
      case 1: aux = (t1 & 0x0f0f0f0fu) | (((t2 >> 2) & 0x03030303u) << 4); break;
      case 2: aux = ((t0 >> 4) & 0x0f0f0f0fu) | (((t2 >> 4) & 0x03030303u) << 4); break;
      default: aux = ((t1 >> 4) & 0x0f0f0f0fu) | (((t2 >> 6) & 0x03030303u) << 4); break;
    }
    const uint ap = aux >> (16u * (j & 1));   // 6-bit scales 2j, 2j+1 (shifts, no dynamic vector index)
    const float2 sA = float2(d * float(int(ap & 0xFFu) - 32)), sB = float2(d * float(int((ap >> 8) & 0xFFu) - 32));
    const half2 k = half2(1028.0h);   // 1024 + 4
#pragma unroll
    for (ushort h = 0; h < 2; ++h) {
      const uint v = w.a[h]; const float2 s2 = h ? sB : sA; half4 o[4];
#pragma unroll
      for (ushort q = 0; q < 4; ++q) {
        o[q].xy = half2(float2(as_type<half2>(((v >> (4 * q)) & 0x00030003u) | (((w.b >> (8 * h + 2 * q)) & 0x00010001u) << 2) | 0x64006400u) - k) * s2);
        o[q].zw = half2(float2(as_type<half2>(((v >> (4 * q + 2)) & 0x00030003u) | (((w.b >> (8 * h + 2 * q + 1)) & 0x00010001u) << 2) | 0x64006400u) - k) * s2);
        *((threadgroup half4 *)(dst + 16 * h + 4 * q)) = o[q];
      }
    }
  }
};
// Q8_0: plane0 32 int8 (natural order); meta half d per 32
struct FmtQ80 {
  enum : uint { P0 = 32, P1 = 0, MetaBytes = 2 }; enum : ushort { MetaGroups = 1, TgLut = 0 };
  typedef P32 Payload; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0), *((device uint4 *)(p0 + 16))}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static void dequant32(Payload w, Meta mt, ushort, threadgroup half2 *, threadgroup half *dst) {
    const half4 s4 = half4(as_type<half>(mt));
#pragma unroll
    for (ushort k = 0; k < 4; ++k) {
      *((threadgroup half4 *)(dst + 4 * k)) = half4(as_type<char4>(w.a[k])) * s4;
      *((threadgroup half4 *)(dst + 16 + 4 * k)) = half4(as_type<char4>(w.b[k])) * s4;
    }
  }
};
// IQ3_S: plane0 16 B = qs[8] | signs[4] | qh (byte 12) | scale nibble (byte 13) | pad; meta half d per 256. value = d*(1+2*scale)*grid*sign
struct FmtIQ3S {
  enum : uint { P0 = 16, P1 = 0, MetaBytes = 2 }; enum : ushort { MetaGroups = 8, TgLut = 0 };
  typedef P16 Payload; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static void dequant32(Payload w, Meta mt, ushort, threadgroup half2 *, threadgroup half *dst) {
    const uint qh = w.a.w & 0xFF, scale = (w.a.w >> 8) & 0xF;
    const float db = float(as_type<half>(mt)) * float(1 + 2 * scale);
    const uchar4 sg = as_type<uchar4>(w.a.z);
#pragma unroll
    for (ushort l = 0; l < 4; ++l) {
      const uint qsw = l < 2 ? w.a.x : w.a.y; const ushort sh = (l & 1) * 16;
      const uint q1 = (qsw >> sh) & 0xFF, q2 = (qsw >> (sh + 8)) & 0xFF;
      const uint g1 = kIQ3S_GRID[q1 | ((qh << (8 - 2 * l)) & 256)], g2 = kIQ3S_GRID[q2 | ((qh << (7 - 2 * l)) & 256)];
      const uchar s = sg[l];
      half4 v1 = half4(float4(as_type<uchar4>(g1)) * db), v2 = half4(float4(as_type<uchar4>(g2)) * db);
      v1 = select(v1, -v1, bool4(s & 1, s & 2, s & 4, s & 8));
      v2 = select(v2, -v2, bool4(s & 16, s & 32, s & 64, s & 128));
      *((threadgroup half4 *)(dst + 8 * l)) = v1; *((threadgroup half4 *)(dst + 8 * l + 4)) = v2;
    }
  }
};

// ---------------- sg: each simdgroup stages its own Cols x KS sub-tile privately and runs matmul2d alone
template <class F, typename TA, typename TO, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, ushort Ep = EpNone>
inline void sg_tile(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, device TO *output,
                    uint output_size, uint input_size, uint output_origin, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint step_begin, uint step_end, uint out_stride = 0, uint out_offset = 0, device bfloat *aux = nullptr) {
  if (out_stride == 0) out_stride = output_size;
  constexpr ushort GPS = KS / 32, Items = Cols * GPS, IPT = (Items + 31) / 32;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint tile = output_origin / kStorageN, tile_offset = output_origin % kStorageN;
  device uchar *tw0 = w0 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * kStorageN + tile_offset) * F::MetaBytes;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + (Buffers > 1 ? KS * Cols : 0), dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
  const uint unit0 = (step_begin * GPS) / F::MetaGroups;
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = simd_lane + it * 32; const bool live = item < Items;
    const uint col = live ? item % Cols : 0, gi = live ? item / Cols : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(step_begin + pf) * GPS + gi;
      if (live && step_begin + pf < step_end) packed[pf][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + (ulong(unit0) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit0;
  }
  for (uint step = step_begin; step < step_end; ++step) {
    threadgroup half *buf = stage + (Buffers > 1 ? (step & 1) * (KS * Cols) : 0);
    if constexpr (Buffers == 1) simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = simd_lane + it * 32; if (item >= Items) break;
      const uint col = item % Cols, gi = item / Cols, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
      if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit; }
      F::dequant32(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort pf = 0; pf + 1 < Prefetch; ++pf)
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) packed[pf][it] = packed[pf + 1][it];
    if (step + Prefetch < step_end) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = simd_lane + it * 32; if (item >= Items) break;
        const uint col = item % Cols, gi = item / Cols; const ulong g = ulong(step + Prefetch) * GPS + gi;
        packed[Prefetch - 1][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
      }
    }
    auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
    if (Buffers > 1 && (step & 1)) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = ulong(index[1]) * out_stride + out_offset + output_origin + index[0];
    float v = acc[i];
    if constexpr (Ep == EpResidual) v += float(aux[o]);
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * silu_gate(float(aux[o]));
    output[o] = TO(v);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}


// ---------------- pf: prefill with a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG rows
template <class F, typename TA, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, ushort Prefetch, ushort Ep = EpNone>
inline void pf_tile(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint output_size, uint input_size, uint output_origin, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint simd_group, uint out_stride = 0, uint out_offset = 0, device bfloat *aux = nullptr) {
  if (out_stride == 0) out_stride = output_size;
  constexpr ushort Threads = Simdgroups * 32, GPS = KS / 32, Items = TileN * GPS, IPT = (Items + Threads - 1) / Threads;
  auto a = tensor(input + ulong(simd_group) * RowsPerSG * input_size, dextents<int, 2>{int(input_size), RowsPerSG}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, TileN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, steps = groups / GPS, units = groups / F::MetaGroups;
  const uint tile = output_origin / kStorageN, tile_offset = output_origin % kStorageN;
  device uchar *tw0 = w0 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * kStorageN + tile_offset) * F::MetaBytes;
  auto a0 = a.template slice<KS, RowsPerSG>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * TileN, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, TileN>(0, 0), b1 = bt1.slice<KS, TileN>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  const uint thread_index = simd_group * 32 + simd_lane;
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = thread_index + it * Threads; const bool live = item < Items;
    const uint col = live ? item % TileN : 0, gi = live ? item / TileN : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(pf) * GPS + gi;
      if (live && pf < steps) packed[pf][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + col * F::MetaBytes); hdr_unit[it] = 0;
  }
  for (uint step = 0; step < steps; ++step) {
    threadgroup half *buf = stage + (step & 1) * (KS * TileN);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = thread_index + it * Threads; if (item >= Items) break;
      const uint col = item % TileN, gi = item / TileN, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
      if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit; }
      F::dequant32(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort pf = 0; pf + 1 < Prefetch; ++pf)
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) packed[pf][it] = packed[pf + 1][it];
    if (step + Prefetch < steps) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = thread_index + it * Threads; if (item >= Items) break;
        const uint col = item % TileN, gi = item / TileN; const ulong g = ulong(step + Prefetch) * GPS + gi;
        packed[Prefetch - 1][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
      }
    }
    auto a_slice = a.template slice<KS, RowsPerSG>(step * KS, 0);
    if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    float v = acc[i];
    if constexpr (Ep == EpResidual) v += float(aux[o]);
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * silu_gate(float(aux[o]));
    output[o] = bfloat(v);
  }
}

#define TGLUT_INIT(F)                                                                                     \
  threadgroup half2 tl[F::TgLut ? F::TgLut : 1];                                                          \
  if constexpr (F::TgLut) { for (uint i = simd_group * 32 + simd_lane; i < F::TgLut; i += Threads) tl[i] = kIQ4NL2[i]; threadgroup_barrier(mem_flags::mem_threadgroup); }
#define ABUF(TA) device TA *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], constant GgufParams &p [[buffer(5)]]
#define ABUFE device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], device bfloat *aux [[buffer(5)]], constant GgufParams &p [[buffer(6)]]
#define IDS uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
// decode: one simdgroup per C columns, S simdgroups per threadgroup, grid = tiles (persistent_groups >= tiles)
#define SG_K(F, f, TA, ta, R, C, S, KS, B, P)                                                             \
  kernel void sg##ta##_##f##_m##R##_c##C##_sg##S##_k##KS##_b##B##_p##P(ABUF(TA), uint group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[S * B * KS * C]; const uint tiles = p.output_size / (S * C), steps = p.input_size / KS; \
    for (uint tile = group; tile < tiles; tile += p.persistent_groups)                                    \
      sg_tile<F, TA, bfloat, R, C, KS, B, P>(input, w0, w1, meta, output, p.output_size, p.input_size, tile * (S * C) + simd_group * C, \
                                             stage + simd_group * (B * KS * C), tl, simd_lane, 0, steps, p.out_stride, p.out_offset); }
// split-K decode: group.y = split, p.persistent_groups = splits, fp32 partials [split][R][N]
#define SGK_K(F, f, TA, ta, R, C, S, KS, B, P)                                                            \
  kernel void sgk##ta##_##f##_m##R##_c##C##_sg##S##_k##KS##_b##B##_p##P(device TA *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], \
             device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]], device float *partials [[buffer(4)]], constant GgufParams &p [[buffer(5)]], \
             uint2 group [[threadgroup_position_in_grid]], IDS) {                                          \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[S * B * KS * C]; const uint steps = p.input_size / KS, per = steps / p.persistent_groups; \
    sg_tile<F, TA, float, R, C, KS, B, P>(input, w0, w1, meta, partials + ulong(group.y) * R * p.output_size, p.output_size, p.input_size, \
        group.x * (S * C) + simd_group * C, stage + simd_group * (B * KS * C), tl, simd_lane, group.y * per, (group.y + 1) * per); }
kernel void gguf_splitk_reduce(device float *partials [[buffer(0)]], device bfloat *output [[buffer(1)]], device bfloat *aux [[buffer(2)]],
                             constant GgufReduceParams &rp [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  const uint count = rp.rows * rp.cols; if (tid >= count) return; float s = 0.0f;
  for (uint k = 0; k < rp.splits; ++k) s += partials[ulong(k) * count + tid];
  const uint row = tid / rp.cols, col = tid % rp.cols; const ulong o = ulong(row) * (rp.out_stride ? rp.out_stride : rp.cols) + rp.out_offset + col;
  if (rp.epilogue == EpResidual) s += float(aux[o]);
  if (rp.epilogue == EpUpWithGate) s = float(bfloat(s)) * silu_gate(float(aux[o]));
  output[o] = bfloat(s);
}
// epilogue entry points (bf16 activations): residual add or silu(gate)*acc, aux in buffer(6)
#define SGE_K(F, f, EP, ep, R, C, S, KS, B, P)                                                            \
  kernel void sg##ep##_##f##_m##R##_c##C##_sg##S##_k##KS##_b##B##_p##P(ABUFE, uint group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[S * B * KS * C]; const uint tiles = p.output_size / (S * C), steps = p.input_size / KS; \
    for (uint tile = group; tile < tiles; tile += p.persistent_groups)                                    \
      sg_tile<F, bfloat, bfloat, R, C, KS, B, P, EP>(input, w0, w1, meta, output, p.output_size, p.input_size, tile * (S * C) + simd_group * C, \
                                             stage + simd_group * (B * KS * C), tl, simd_lane, 0, steps, p.out_stride, p.out_offset, aux); }
#define PFE_K(F, f, EP, ep, R, S, N, KS, P)                                                               \
  kernel void pf##ep##_##f##_r##R##_sg##S##_n##N##_k##KS##_p##P(ABUFE, uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N]; const uint rs = p.out_stride ? p.out_stride : p.output_size;      \
    pf_tile<F, bfloat, R, S, N, KS, P, EP>(input + ulong(group.x) * (R * S) * p.input_size, w0, w1, meta, output + ulong(group.x) * (R * S) * rs, \
                                   p.output_size, p.input_size, group.y * N, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset, aux + ulong(group.x) * (R * S) * rs); }
#define PF_K(F, f, TA, ta, R, S, N, KS, P)                                                                \
  kernel void pf##ta##_##f##_r##R##_sg##S##_n##N##_k##KS##_p##P(ABUF(TA), uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N];                                                                   \
    pf_tile<F, TA, R, S, N, KS, P>(input + ulong(group.x) * (R * S) * p.input_size, w0, w1, meta, output + ulong(group.x) * (R * S) * (p.out_stride ? p.out_stride : p.output_size), \
                                   p.output_size, p.input_size, group.y * N, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset); }
// ---------------- sg_accum: the decode tile loop without the store; acc is created by the caller (gguf_make_acc) so
// several formats can accumulate into the same cooperative tensor type (fused segments, gate+up).
// Initialize in the caller after construction: returning an initialized cooperative tensor
// loses its initial values on Apple9 in runtime-format kernels (also with shader validation).
template <typename TA, ushort Rows, ushort Cols, ushort KS>
inline auto gguf_make_acc(device TA *input, uint input_size, threadgroup half *stage) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0);
  return operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
}
template <class F, typename TA, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, class Acc>
inline void sg_accum(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint output_origin,
                     threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint step_begin, uint step_end, thread Acc &acc) {
  constexpr ushort GPS = KS / 32, Items = Cols * GPS, IPT = (Items + 31) / 32;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint tile = output_origin / kStorageN, tile_offset = output_origin % kStorageN;
  device uchar *tw0 = w0 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * kStorageN + tile_offset) * F::MetaBytes;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + (Buffers > 1 ? KS * Cols : 0), dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
  const uint unit0 = (step_begin * GPS) / F::MetaGroups;
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = simd_lane + it * 32; const bool live = item < Items;
    const uint col = live ? item % Cols : 0, gi = live ? item / Cols : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(step_begin + pf) * GPS + gi;
      if (live && step_begin + pf < step_end) packed[pf][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + (ulong(unit0) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit0;
  }
  for (uint step = step_begin; step < step_end; ++step) {
    threadgroup half *buf = stage + (Buffers > 1 ? (step & 1) * (KS * Cols) : 0);
    if constexpr (Buffers == 1) simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = simd_lane + it * 32; if (item >= Items) break;
      const uint col = item % Cols, gi = item / Cols, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
      if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit; }
      F::dequant32(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort pf = 0; pf + 1 < Prefetch; ++pf)
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) packed[pf][it] = packed[pf + 1][it];
    if (step + Prefetch < step_end) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = simd_lane + it * 32; if (item >= Items) break;
        const uint col = item % Cols, gi = item / Cols; const ulong g = ulong(step + Prefetch) * GPS + gi;
        packed[Prefetch - 1][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
      }
    }
    auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
    if (Buffers > 1 && (step & 1)) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);   // the stage may be reused by a following accumulate
}
// runtime dequantizer selection (uniform per threadgroup)
template <typename TA, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, class Acc>
inline void gguf_accum_any(uint fmt, device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint origin,
                         threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint sb, uint se, thread Acc &acc) {
  switch (fmt) {
  case GGUF_FMT_Q4K: sg_accum<FmtQ4K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_IQ4XS: sg_accum<FmtIQ4XS<3>, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_IQ4NL: sg_accum<FmtIQ4NL<0>, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q5K: sg_accum<FmtQ5K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q6K: sg_accum<FmtQ6K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q3K: sg_accum<FmtQ3K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q80: sg_accum<FmtQ80, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  default: sg_accum<FmtIQ3S, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  }
}
inline void gguf_init_lut(threadgroup half2 *tl, uint thread_index, uint threads) {
  for (uint i = thread_index; i < 256; i += threads) tl[i] = kIQ4NL2[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------- fused segments: one dispatch over up to three column segments of different formats (decode rows)
#define SEGBUF(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_FUSED_K(R)                                                                                            \
  kernel void gguf_fused_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, w0a, w1a, ma), SEGBUF(4, w0b, w1b, mb), SEGBUF(7, w0c, w1c, mc), \
                       device bfloat *output [[buffer(10)]], constant GgufFusedParams &p [[buffer(11)]],       \
                       uint group [[threadgroup_position_in_grid]], IDS) {                                   \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256];                                     \
    gguf_init_lut(tl, simd_group * 32 + simd_lane, 64);                                                      \
    const uint t0 = p.cols[0] / 64, t1 = t0 + p.cols[1] / 64;                                              \
    device uchar *w0 = w0a; device uchar *w1 = w1a; device uchar *meta = ma; uint fmt = p.fmt[0], off = p.offset[0], local = group; \
    if (group >= t1) { w0 = w0c; w1 = w1c; meta = mc; fmt = p.fmt[2]; off = p.offset[2]; local = group - t1; }                    \
    else if (group >= t0) { w0 = w0b; w1 = w1b; meta = mb; fmt = p.fmt[1]; off = p.offset[1]; local = group - t0; }                \
    const uint origin = local * 64 + simd_group * 32, steps = p.input_size / 32;                            \
    threadgroup half *my = stage + simd_group * (2 * 32 * 32);                                              \
    auto acc = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                     \
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;                                    \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(fmt, input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, 0, steps, acc); \
    for (ushort i = 0; i < acc.get_capacity(); ++i) {                                                       \
      if (!acc.is_valid_element(i)) continue;                                                              \
      auto index = acc.get_multidimensional_index(i);                                                      \
      output[ulong(index[1]) * p.out_stride + off + origin + index[0]] = bfloat(acc[i]);                   \
    }                                                                                                       \
  }
GGUF_FUSED_K(8) GGUF_FUSED_K(16) GGUF_FUSED_K(24) GGUF_FUSED_K(32)

// ---------------- gate + up in one dispatch: output = silu(gate) * up (both rounded to bf16 first, as splash does)
#define GGUF_GATEUP_K(R)                                                                                           \
  kernel void gguf_gateup_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, gw0, gw1, gm), SEGBUF(4, uw0, uw1, um),  \
                        device bfloat *output [[buffer(7)]], constant GgufGateUpParams &p [[buffer(8)]],        \
                        uint group [[threadgroup_position_in_grid]], IDS) {                                  \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256];                                     \
    gguf_init_lut(tl, simd_group * 32 + simd_lane, 64);                                                      \
    const uint origin = group * 64 + simd_group * 32, steps = p.input_size / 32;                           \
    threadgroup half *my = stage + simd_group * (2 * 32 * 32);   /* one 8 KB stage for both passes: occupancy */ \
    auto gate = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                    \
    for (ushort i = 0; i < gate.get_capacity(); ++i) gate[i] = 0.0f;                                   \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(p.gate_fmt, input, gw0, gw1, gm, p.input_size, origin, my, tl, simd_lane, 0, steps, gate); \
    auto up = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                      \
    for (ushort i = 0; i < up.get_capacity(); ++i) up[i] = 0.0f;                                     \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(p.up_fmt, input, uw0, uw1, um, p.input_size, origin, my, tl, simd_lane, 0, steps, up); \
    for (ushort i = 0; i < gate.get_capacity(); ++i) {                                                      \
      if (!gate.is_valid_element(i)) continue;                                                             \
      auto index = gate.get_multidimensional_index(i);                                                     \
      const float g = float(bfloat(gate[i])), u = float(bfloat(up[i]));                                    \
      output[ulong(index[1]) * p.out_stride + origin + index[0]] = bfloat(silu_gate(g) * u);               \
    }                                                                                                       \
  }
GGUF_GATEUP_K(8) GGUF_GATEUP_K(16) GGUF_GATEUP_K(24) GGUF_GATEUP_K(32)

// ---------------- split-K with last-arriver reduction (per format), epilogue applied by the reducing threadgroup
template <class F, ushort Rows>
inline void gguf_splitk_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device float *partials,
                           device atomic_uint *counters, device bfloat *output, device bfloat *aux, constant GgufSplitParams &p,
                           uint2 group, uint simd_lane, uint simd_group, threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint steps = p.input_size / 32, per = steps / p.splits, thread_index = simd_group * 32 + simd_lane;
  const uint origin = group.x * 64 + simd_group * 32;
  threadgroup half *my = stage + simd_group * (2 * 32 * 32);
  auto acc = gguf_make_acc<bfloat, Rows, 32, 32>(input, p.input_size, my);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  sg_accum<F, bfloat, Rows, 32, 32, 2, 1>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, group.y * per, (group.y + 1) * per, acc);
  const ulong N = p.output_size;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    partials[(ulong(group.y) * Rows + index[1]) * N + origin + index[0]] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counters + group.x, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (*arrival != p.splits - 1) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    float total = 0.0f;
    for (uint s = 0; s < p.splits; ++s)
      total += s == group.y ? acc[i] : partials[(ulong(s) * Rows + index[1]) * N + origin + index[0]];
    const ulong o = ulong(index[1]) * p.out_stride + p.out_offset + origin + index[0];
    if (p.epilogue == GGUF_EPILOGUE_RESIDUAL) total += float(aux[o]);
    if (p.epilogue == GGUF_EPILOGUE_UP_WITH_GATE) total = float(bfloat(total)) * silu_gate(float(aux[o]));
    output[o] = bfloat(total);
  }
  if (thread_index == 0) atomic_store_explicit(counters + group.x, 0u, memory_order_relaxed);
}
#define GGUF_SPLITK_K(F, f, R)                                                                                      \
  kernel void gguf_splitk_##f##_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, w0, w1, meta), device float *partials [[buffer(4)]], \
                             device atomic_uint *counters [[buffer(5)]], device bfloat *output [[buffer(6)]], device bfloat *aux [[buffer(7)]], \
                             constant GgufSplitParams &p [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]], IDS) { \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[F::TgLut ? F::TgLut : 1]; threadgroup uint arrival; \
    if constexpr (F::TgLut) gguf_init_lut(tl, simd_group * 32 + simd_lane, 64);                                \
    gguf_splitk_tile<F, R>(input, w0, w1, meta, partials, counters, output, aux, p, group, simd_lane, simd_group, stage, tl, &arrival); }
#define GGUF_SPLITK_SET(F, f) GGUF_SPLITK_K(F, f, 8) GGUF_SPLITK_K(F, f, 16) GGUF_SPLITK_K(F, f, 24) GGUF_SPLITK_K(F, f, 32)
GGUF_SPLITK_SET(FmtQ4K, q4k) GGUF_SPLITK_SET(FmtIQ4XS<3>, iq4xs) GGUF_SPLITK_SET(FmtIQ4NL<0>, iq4nl) GGUF_SPLITK_SET(FmtQ5K, q5k)
GGUF_SPLITK_SET(FmtQ6K, q6k) GGUF_SPLITK_SET(FmtQ3K, q3k) GGUF_SPLITK_SET(FmtQ80, q80) GGUF_SPLITK_SET(FmtIQ3S, iq3s)

#define PROD_SET(F, f)                                                                                    \
  SG_K(F, f, bfloat, a, 8, 32, 2, 32, 2, 1) SG_K(F, f, bfloat, a, 16, 32, 2, 32, 2, 1) SG_K(F, f, bfloat, a, 24, 32, 2, 32, 2, 1) SG_K(F, f, bfloat, a, 32, 32, 2, 32, 2, 1) \
  SGE_K(F, f, EpResidual, r, 8, 32, 2, 32, 2, 1) SGE_K(F, f, EpResidual, r, 16, 32, 2, 32, 2, 1) SGE_K(F, f, EpResidual, r, 24, 32, 2, 32, 2, 1) SGE_K(F, f, EpResidual, r, 32, 32, 2, 32, 2, 1) \
  SGE_K(F, f, EpUpWithGate, g, 8, 32, 2, 32, 2, 1) SGE_K(F, f, EpUpWithGate, g, 16, 32, 2, 32, 2, 1) SGE_K(F, f, EpUpWithGate, g, 24, 32, 2, 32, 2, 1) SGE_K(F, f, EpUpWithGate, g, 32, 32, 2, 32, 2, 1) \
  PF_K(F, f, bfloat, a, 32, 4, 64, 64, 1) PFE_K(F, f, EpResidual, r, 32, 4, 64, 64, 1) PFE_K(F, f, EpUpWithGate, g, 32, 4, 64, 64, 1)
PROD_SET(FmtQ4K, q4k)
PROD_SET(FmtIQ4XS<3>, iq4xs)
PROD_SET(FmtIQ4NL<0>, iq4nl)
PROD_SET(FmtQ5K, q5k)
PROD_SET(FmtQ6K, q6k)
PROD_SET(FmtQ3K, q3k)
PROD_SET(FmtQ80, q80)
PROD_SET(FmtIQ3S, iq3s)

// ---------------- token embedding gather from native block_q4_K rows (row = K/256 blocks of 144 B)
kernel void gguf_embed_q4k(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], device bfloat *output [[buffer(2)]],
                      constant GgufEmbedParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *blk = table + (ulong(token) * (p.hidden / 256) + dim / 256) * 144;
  const uint j = (dim % 256) / 32, l = dim % 32;
  const half d = as_type<half>(ushort(blk[0] | (blk[1] << 8))), dmin = as_type<half>(ushort(blk[2] | (blk[3] << 8)));
  device const uchar *sc = blk + 4; uchar s, m;
  if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
  const uchar q = (blk[16 + (j / 2) * 32 + l] >> ((j % 2) * 4)) & 15;
  output[index] = bfloat(float(d) * float(s) * float(q) - float(dmin) * float(m));
}
// native block_q6_K rows (210 B per 256 weights): ql[128] | qh[64] | int8 scales[16] | half d
kernel void gguf_embed_q6k(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], device bfloat *output [[buffer(2)]],
                      constant GgufEmbedParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *blk = table + (ulong(token) * (p.hidden / 256) + dim / 256) * 210;
  const uint l = dim % 256, n = l / 128, r = l % 128, quarter = r / 32, pos = r % 32;
  const uchar lo = (blk[n * 64 + (quarter & 1) * 32 + pos] >> ((quarter >> 1) * 4)) & 15;
  const uchar hi = (blk[128 + n * 32 + pos] >> (2 * quarter)) & 3;
  const char sc = as_type<char>(blk[192 + n * 8 + 2 * quarter + pos / 16]);
  const half d = as_type<half>(ushort(blk[208] | (blk[209] << 8)));
  output[index] = bfloat(float(d) * float(sc) * float(int(lo | (hi << 4)) - 32));
}
// native block_q8_0 rows (34 B per 32 weights): half d | int8 qs[32]
kernel void gguf_embed_q80(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], device bfloat *output [[buffer(2)]],
                      constant GgufEmbedParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *blk = table + (ulong(token) * (p.hidden / 32) + dim / 32) * 34;
  const half d = as_type<half>(ushort(blk[0] | (blk[1] << 8)));
  output[index] = bfloat(float(d) * float(as_type<char>(blk[2 + dim % 32])));
}
// permute the K columns (in 128-wide head blocks) of a bf16 activation: out[row][h*128+e] = in[row][perm[h]*128+e]
kernel void gguf_permute_heads(device const bfloat *input [[buffer(0)]], device bfloat *output [[buffer(1)]], device const uint *perm [[buffer(2)]],
                          constant GgufPermuteParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.width) return;
  const uint row = index / p.width, col = index % p.width, h = col / p.block, e = col % p.block;
  output[index] = input[ulong(row) * p.width + perm[h] * p.block + e];
}

// ---- load-time repack: native GGUF rows -> MDGG0001 planes (layout documented in model/GgufImage.hpp) ----
// One thread per (destination row n, 32-wide K group g). Rows >= permute_from_row are read from
// llama.cpp's tiled value-head order so the image holds splash's grouped order.
static inline uint gguf_repack_source_row(uint n, constant GgufRepackParams &p) {
  if (n < p.permute_from_row) return n;
  const uint head = (n - p.permute_from_row) / p.permute_head_rows, e = (n - p.permute_from_row) % p.permute_head_rows;
  const uint source = (head % p.permute_groups) * p.permute_group_heads + head / p.permute_groups;
  return p.permute_from_row + source * p.permute_head_rows + e;
}
// 32 codes (< 16) -> 4 words; interleaved: code j of each 8 sits at nibble (j&1)*4 + (j>>1) of a 16-bit half.
static inline void gguf_pack_words(thread const uchar *codes, bool interleave, device uint *dst) {
  for (uint k = 0; k < 4; ++k) {
    uint w = 0;
    for (uint j = 0; j < 8; ++j) w |= uint(codes[8 * k + j]) << (interleave ? (j & 1) * 16 + 4 * (j >> 1) : 4 * j);
    dst[k] = w;
  }
}
// 16 small values -> one word: even k at bit step*(k/2), odd k at 16 + step*(k/2).
static inline uint gguf_pair_word(thread const uchar *v, uint step) {
  uint w = 0;
  for (uint k = 0; k < 16; ++k) w |= uint(v[k]) << (((k & 1) ? 16 : 0) + step * (k >> 1));
  return w;
}
kernel void gguf_repack(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                      constant GgufRepackParams &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  const uint G = p.input_size / 32;
  if (t >= p.rows * G) return;
  const uint n = t / G, g = t % G, r = gguf_repack_source_row(n, p);
  const bool k256 = !(p.fmt == GGUF_FMT_IQ4NL || p.fmt == GGUF_FMT_Q80);
  const uint b = k256 ? g / 8 : g, j = k256 ? g % 8 : 0, mg = k256 ? 8 : 1;
  uint blockBytes, p0, p1, mb;
  switch (p.fmt) {
    case GGUF_FMT_Q4K: blockBytes = 144; p0 = 16; p1 = 0; mb = 16; break;
    case GGUF_FMT_IQ4XS: blockBytes = 136; p0 = 16; p1 = 0; mb = 8; break;
    case GGUF_FMT_IQ4NL: blockBytes = 18; p0 = 16; p1 = 0; mb = 2; break;
    case GGUF_FMT_Q5K: blockBytes = 176; p0 = 16; p1 = 4; mb = 16; break;
    case GGUF_FMT_Q6K: blockBytes = 210; p0 = 16; p1 = 8; mb = 20; break;
    case GGUF_FMT_Q3K: blockBytes = 110; p0 = 8; p1 = 4; mb = 16; break;
    case GGUF_FMT_Q80: blockBytes = 34; p0 = 32; p1 = 0; mb = 2; break;
    default: blockBytes = 110; p0 = 16; p1 = 0; mb = 2; break;  // IQ3_S
  }
  device const uchar *blk = src + p.src_offset + ulong(r) * p.src_row_bytes + ulong(b) * blockBytes;
  const uint tile = ((n / 256) * G + g) * 256 + (n % 256);
  device uchar *out0 = dst + p.dst_plane0 + ulong(tile) * p0;
  device uchar *out1 = dst + p.dst_plane1 + ulong(tile) * p1;
  device uchar *meta = dst + p.dst_meta + (ulong((n / 256) * (G / mg) + b) * 256 + (n % 256)) * mb;
  uchar codes[32], bits[32];
  switch (p.fmt) {
    case GGUF_FMT_Q4K: {
      for (uint l = 0; l < 32; ++l) codes[l] = (blk[16 + (j / 2) * 32 + l] >> (4 * (j % 2))) & 15;
      gguf_pack_words(codes, true, (device uint *)out0);
      if (j == 0) for (uint i = 0; i < 16; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_Q5K: {
      for (uint l = 0; l < 32; ++l) { codes[l] = (blk[48 + (j / 2) * 32 + l] >> (4 * (j % 2))) & 15; bits[l] = (blk[16 + l] >> j) & 1; }
      gguf_pack_words(codes, true, (device uint *)out0);
      uint w = 0;
      for (uint l = 0; l < 32; ++l) w |= uint(bits[l]) << (((l & 1) ? 16 : 0) + 4 * (l / 8) + (l % 8) / 2);
      *(device uint *)out1 = w;
      if (j == 0) for (uint i = 0; i < 16; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_IQ4XS: {
      for (uint l = 0; l < 16; ++l) { const uchar q = blk[8 + 16 * j + l]; codes[l] = q & 15; codes[16 + l] = q >> 4; }
      gguf_pack_words(codes, false, (device uint *)out0);
      if (j == 0) for (uint i = 0; i < 8; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_IQ4NL: {
      for (uint l = 0; l < 16; ++l) { const uchar q = blk[2 + l]; codes[l] = q & 15; codes[16 + l] = q >> 4; }
      gguf_pack_words(codes, true, (device uint *)out0);
      meta[0] = blk[0]; meta[1] = blk[1];
      break;
    }
    case GGUF_FMT_Q6K: {
      const uint hb = j / 4, quarter = j % 4;
      for (uint l = 0; l < 32; ++l) {
        codes[l] = (blk[64 * hb + 32 * (quarter & 1) + l] >> (4 * (quarter >> 1))) & 15;
        bits[l] = (blk[128 + 32 * hb + l] >> (2 * quarter)) & 3;
      }
      gguf_pack_words(codes, true, (device uint *)out0);
      ((device uint *)out1)[0] = gguf_pair_word(bits, 2);
      ((device uint *)out1)[1] = gguf_pair_word(bits + 16, 2);
      if (j == 0) { for (uint i = 0; i < 16; ++i) meta[i] = blk[192 + i]; meta[16] = blk[208]; meta[17] = blk[209]; meta[18] = 0; meta[19] = 0; }
      break;
    }
    case GGUF_FMT_Q3K: {
      const uint hb = j / 4, jj = j % 4;
      for (uint l = 0; l < 32; ++l) { codes[l] = (blk[32 + 32 * hb + l] >> (2 * jj)) & 3; bits[l] = (blk[l] >> j) & 1; }
      ((device uint *)out0)[0] = gguf_pair_word(codes, 2);
      ((device uint *)out0)[1] = gguf_pair_word(codes + 16, 2);
      uint w = 0;
      for (uint l = 0; l < 32; ++l) w |= uint(bits[l]) << (((l & 1) ? 16 : 0) + l / 2);
      *(device uint *)out1 = w;
      if (j == 0) { meta[0] = blk[108]; meta[1] = blk[109]; meta[2] = 0; meta[3] = 0; for (uint i = 0; i < 12; ++i) meta[4 + i] = blk[96 + i]; }
      break;
    }
    case GGUF_FMT_Q80: {
      for (uint i = 0; i < 32; ++i) out0[i] = blk[2 + i];
      meta[0] = blk[0]; meta[1] = blk[1];
      break;
    }
    default: {  // IQ3_S: qs(8) | signs(4) | qh(1) | 4-bit scale (1) | 0 0
      for (uint i = 0; i < 8; ++i) out0[i] = blk[2 + 8 * j + i];
      for (uint i = 0; i < 4; ++i) out0[8 + i] = blk[74 + 4 * j + i];
      out0[12] = blk[66 + j];
      out0[13] = (blk[106 + j / 2] >> (4 * (j % 2))) & 15;
      out0[14] = 0; out0[15] = 0;
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
  }
}
// byte copy for native embedding rows: one thread per 16 bytes
kernel void gguf_copy(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                    constant GgufCopyParams &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  const uint begin = t * 16;
  if (begin >= p.bytes) return;
  if (begin + 16 <= p.bytes) *(device uint4 *)(dst + p.dst_offset + begin) = *(device const uint4 *)(src + p.src_offset + begin);
  else for (uint i = begin; i < p.bytes; ++i) dst[p.dst_offset + i] = src[p.src_offset + i];
}

// dependency kernel for serialized profiling: touching the output forces the next dispatch to wait
kernel void gguf_touch(device bfloat *y [[buffer(0)]], uint tid [[thread_position_in_grid]]) { if (tid == 0) y[0] = bfloat(float(y[0]) + 0.0f); }
