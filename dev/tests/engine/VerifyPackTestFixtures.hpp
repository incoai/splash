#pragma once
#include "GgufFormatReference.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
using namespace splash;
using namespace gguf_reference;
using namespace splash::test;
using metal::CommandGraph;
using metal::MetalBackend;
using metal::MetalBuffer;
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
static uint64_t hash(const MetalBuffer &b) {
  auto p = static_cast<const uint8_t *>(b.contents());
  uint64_t h = 14695981039346656037ull;
  for (uint64_t i = 0; i < b.sizeBytes(); ++i)
    h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
struct Guarded {
  MetalBuffer backing, view;
  uint64_t bytes;
  Guarded(MetalBackend &b, uint64_t n, uint32_t fill) : bytes(n) {
    backing = b.allocateBuffer(n + 256, metal::BufferStorage::Shared, "M128 guarded");
    view = b.view(backing, 0, n);
    std::fill_n(static_cast<uint32_t *>(view.contents()), n / 4, fill);
    std::memset(static_cast<uint8_t *>(backing.contents()) + n, 0xa5, 256);
  }
  bool intact() const {
    auto p = static_cast<const uint8_t *>(backing.contents()) + bytes;
    return std::all_of(p, p + 256, [](uint8_t v) { return v == 0xa5; });
  }
};
struct Weight {
  uint32_t N, K;
  std::vector<uint8_t> native;
  MetalBuffer codes, metadata;
  uint64_t hc, hm;
};
static Weight weight(MetalBackend &b, uint32_t N, uint32_t K, std::mt19937 &rng,
                     const std::vector<uint32_t> &bank) {
  const auto fmt = Fmt(GGUF_FMT_AF4G64);
  Weight w{N, K, makeNative(fmt, N, K, rng), {}, {}, 0, 0};
  check(kQuantFormats[fmt].block_bytes == 36, "AF4 ABI");
  for (uint64_t u = 0; u < w.native.size() / 36; ++u) {
    uint32_t m = bank[(u * 37) % bank.size()];
    std::memcpy(w.native.data() + u * 36, &m, 4);
  }
  auto p = repack(fmt, w.native, N, K, nullptr);
  auto upload = [&](const auto &v) {
    auto x = b.allocateBuffer(v.size(), metal::BufferStorage::Shared, "M128 weight");
    std::memcpy(x.contents(), v.data(), v.size());
    return x;
  };
  w.codes = upload(p.w0);
  w.metadata = upload(p.meta);
  w.hc = hash(w.codes);
  w.hm = hash(w.metadata);
  return w;
}

// Model-free affine metadata for the synthetic operator comparisons.
static std::vector<uint32_t> verifyPackMetadata() {
  std::mt19937 rng(2026110732u);
  auto native = makeNative(Fmt(GGUF_FMT_AF4G64), 256, 4096, rng);
  std::vector<uint32_t> bank(native.size() / 36);
  for (size_t i = 0; i < bank.size(); ++i)
    std::memcpy(&bank[i], native.data() + i * 36, 4);
  return bank;
}
