#pragma once

// Deterministic synthetic weights and inputs (h3 bench --synthetic, dev/tests/synthetic_check.sh): no weight files,
// the same shapes and memory as the converted VDN-H3 DiT, values from the hash of metal/abi/SyntheticABI.h. Tensor
// seeds: syntheticSeed(group, id), group = block index for block tensors, kIoGroup for the io weights, kInputGroup
// for the request inputs. Value ranges keep the 50-block forward finite (unit-gain linears, gates ~0.1).
#import <Metal/Metal.h>

#include "video/h3/engine/Status.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "metal/abi/video/SyntheticABI.h"

#include <cstdint>
#include <vector>

namespace h3::model {

inline constexpr int kSyntheticIoGroup = 1000, kSyntheticInputGroup = 2000;

uint64_t syntheticSeed(int group, int id);
float syntheticFloat(uint64_t seed, uint64_t i, float scale, float offset);
// Host arrays: fp32 / bf16 bits
void syntheticF32(uint64_t seed, size_t n, float scale, float offset, float *out);
void syntheticBf16(uint64_t seed, size_t n, float scale, float offset, uint16_t *out);

// GPU fills batched into one command buffer.
class SyntheticFiller {
public:
  explicit SyntheticFiller(metal::MetalBackend &backend) : b_(backend) {}
  Status allocate(size_t count, int type, uint64_t seed, float scale, float offset, __strong id<MTLBuffer> &out,
                  const char *what);
  void fill(id<MTLBuffer> buffer, size_t offsetBytes, size_t count, int type, uint64_t seed, float scale,
            float offset);
  Status run();
  [[nodiscard]] size_t bytes() const { return bytes_; }

private:
  struct Job {
    id<MTLBuffer> buffer;
    size_t offset;
    SyntheticFill fill;
  };
  metal::MetalBackend &b_;
  std::vector<Job> jobs_;
  size_t bytes_ = 0;
};

} // namespace h3::model
