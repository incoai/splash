// The GGUF load kernels (gguf_repack, gguf_copy) of the production metallib
// and the image planner's CPU-built alpha/beta tensor against the CPU
// reference, and that reference's values against hashes of upstream GGML's
// dequantization.
//   gguf-repack --cpu        golden hashes and the alpha/beta tensor
//   gguf-repack <metallib>   also the kernels
// With SPLASH_GGML_ORACLE=<libggml-base.dylib> the reference is also compared
// with GGML directly and GGML's hashes are printed; a build of llama.cpp
// 7ab4ee7 regenerates kGolden.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "GgufFormatReference.hpp"
#include "metal/abi/Gguf.h"
#include "model/GgufImage.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace gguf_reference;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("%-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  failures += !ok;
}

std::string sha256(const void *data, size_t bytes) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data, static_cast<CC_LONG>(bytes), digest);
  std::string hex;
  for (unsigned char byte : digest) {
    char text[3];
    std::snprintf(text, sizeof text, "%02x", byte);
    hex += text;
  }
  return hex;
}

// Every byte random and every half scale a random finite half: either sign,
// zero and subnormal included.
std::vector<uint8_t> fixture(Fmt f, uint32_t rows, uint32_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  return makeNative(f, rows, K, rng, [&] {
    uint16_t h;
    do h = static_cast<uint16_t>(rng());
    while ((h & 0x7C00) == 0x7C00);
    return h;
  });
}

// SHA-256 of GGML's fp32 dequantization of fixture(f, 256, 1024, kGoldenSeed + f).
constexpr uint32_t kGoldenRows = 256, kGoldenK = 1024, kGoldenSeed = 7;
const char *const kGolden[FMT_COUNT] = {
    "36905e2a1a87522067f1dd4a59e5fa658390d01338bb2639df70d622b97804f4", // q4k
    "39f9536d2c5efdfc8f566c176f9ffe8b2eefcc395caa9b2ef9111eb5fc76b453", // iq4xs
    "efe2505213961362b93459a7eb511d7bee39487a2d7f870c0fa972912d93df29", // iq4nl
    "8c2c2b9caf1e1831d516610e0bddf3acc7d736d2884bc1549dd9e579fdd61921", // q5k
    "6478ac742b3e4010323642d0f99b91ba9f62513838cdfc775ec564062d86f117", // q6k
    "8ecc6c113faf9fb8cc77d438dd595a146933c6c75394651cab377c41a55e5677", // q3k
    "1f11dbe883e04db7367fbc5df5c7d2c34a14bae5d772501687a9072f9237b605", // q80
    "dacc281a47cc2940e8352e3cd890283f69c8126d2894768ab7f179cf8fda34e8", // iq3s
};

void checkGoldens(void *ggml) {
  for (int f = 0; f < FMT_COUNT; ++f) {
    const std::vector<uint8_t> native = fixture(Fmt(f), kGoldenRows, kGoldenK, kGoldenSeed + f);
    std::vector<float> values;
    repack(Fmt(f), native, kGoldenRows, kGoldenK, &values);
    check(sha256(values.data(), values.size() * sizeof(float)) == kGolden[f],
          std::string("CPU reference matches the GGML golden hash: ") + fmtName(f));
    if (!ggml) continue;
    std::vector<float> official;
    std::string error;
    const bool loaded = ggmlDequantize(ggml, Fmt(f), native, official, error);
    check(loaded && official.size() == values.size() &&
              !memcmp(official.data(), values.data(), values.size() * sizeof(float)),
          std::string("CPU reference matches GGML: ") + fmtName(f) + (loaded ? "" : " (" + error + ")"));
    if (loaded)
      std::printf("GGML %s %s\n", fmtName(f), sha256(official.data(), official.size() * sizeof(float)).c_str());
  }
}

// Rows [from, rows) of the image come from llama.cpp's tiled value-head order:
// destination head h reads source head (h % groups) * groupHeads + h / groups.
uint32_t sourceRow(uint32_t n, uint32_t from, uint32_t headRows, uint32_t groupHeads, uint32_t groups) {
  if (n < from) return n;
  const uint32_t head = (n - from) / headRows;
  return from + ((head % groups) * groupHeads + head / groups) * headRows + (n - from) % headRows;
}

template <class T> void append(std::vector<uint8_t> &out, T value) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof value);
}

void appendString(std::vector<uint8_t> &out, const std::string &value) {
  append<uint64_t>(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

struct Tensor {
  std::string name;
  std::vector<uint64_t> dims; // dims[0] is the row length
  uint32_t type;
  std::vector<uint8_t> data;
};

// A version 3 qwen35 GGUF of the tensors, in order and 32-byte aligned.
std::vector<uint8_t> ggufFile(const std::vector<Tensor> &tensors, uint32_t blocks, uint32_t hidden) {
  constexpr uint32_t kString = 8, kUint32 = 4, kAlignment = 32;
  std::vector<uint8_t> out{'G', 'G', 'U', 'F'};
  append<uint32_t>(out, 3);
  append<uint64_t>(out, tensors.size());
  append<uint64_t>(out, 3);
  appendString(out, "general.architecture");
  append(out, kString);
  appendString(out, "qwen35");
  appendString(out, "qwen35.block_count");
  append(out, kUint32);
  append(out, blocks);
  appendString(out, "qwen35.embedding_length");
  append(out, kUint32);
  append(out, hidden);
  uint64_t offset = 0;
  for (const Tensor &tensor : tensors) {
    appendString(out, tensor.name);
    append<uint32_t>(out, tensor.dims.size());
    for (uint64_t dim : tensor.dims) append(out, dim);
    append(out, tensor.type);
    append(out, offset);
    offset += (tensor.data.size() + kAlignment - 1) / kAlignment * kAlignment;
  }
  for (const Tensor &tensor : tensors) {
    out.resize((out.size() + kAlignment - 1) / kAlignment * kAlignment, 0);
    out.insert(out.end(), tensor.data.begin(), tensor.data.end());
  }
  return out;
}

// The planner builds the GDN alpha/beta projection on the CPU: beta rows, alpha
// rows, then zero rows up to one 256-row tile, as one Q8_0 tensor with rows in
// grouped head order. Its plane0 and meta must equal the reference's repack of
// those native rows. The layer's other tensors are zero; only their shapes and
// types matter to the planner.
void checkAlphaBeta() {
  namespace model = splash::model;
  using namespace model::ggml;
  model::gguf::TargetGeometry geometry;
  geometry.layers = 1;
  geometry.hiddenSize = 512;
  geometry.vocabularySize = 256;
  geometry.intermediateSize = 256;
  geometry.gdnKeyHeads = 4;
  geometry.gdnValueHeads = 12;
  geometry.gdnHeadDimension = 64;
  geometry.convolutionDimension = 1280; // q and k of 4 heads, v of 12
  const uint32_t hidden = geometry.hiddenSize, heads = geometry.gdnValueHeads;
  const uint32_t groupHeads = geometry.gdnKeyHeads, groups = heads / groupHeads;
  const uint64_t valueRows = uint64_t{heads} * geometry.gdnHeadDimension;
  const std::vector<uint8_t> beta = fixture(Q80, heads, hidden, 200), alpha = fixture(Q80, heads, hidden, 201);

  std::vector<Tensor> tensors;
  auto add = [&](std::string name, std::vector<uint64_t> dims, uint32_t type, std::vector<uint8_t> data = {}) {
    if (data.empty()) {
      const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(type);
      uint64_t elements = 1;
      for (uint64_t dim : dims) elements *= dim;
      data.assign(elements / traits.blockElements * traits.blockBytes, 0);
    }
    tensors.push_back({std::move(name), std::move(dims), type, std::move(data)});
  };
  add("blk.0.attn_norm.weight", {hidden}, kF32);
  add("blk.0.attn_qkv.weight", {hidden, geometry.convolutionDimension}, kQ4_K);
  add("blk.0.attn_gate.weight", {hidden, valueRows}, kQ4_K);
  add("blk.0.ssm_beta.weight", {hidden, heads}, kQ8_0, beta);
  add("blk.0.ssm_alpha.weight", {hidden, heads}, kQ8_0, alpha);
  add("blk.0.ssm_conv1d.weight", {4, geometry.convolutionDimension}, kF32);
  add("blk.0.ssm_a", {heads}, kF32);
  add("blk.0.ssm_dt.bias", {heads}, kF32);
  add("blk.0.ssm_norm.weight", {geometry.gdnHeadDimension}, kF32);
  add("blk.0.ssm_out.weight", {valueRows, hidden}, kQ4_K);
  add("blk.0.post_attention_norm.weight", {hidden}, kF32);
  add("blk.0.ffn_gate.weight", {hidden, geometry.intermediateSize}, kQ4_K);
  add("blk.0.ffn_up.weight", {hidden, geometry.intermediateSize}, kQ4_K);
  add("blk.0.ffn_down.weight", {geometry.intermediateSize, hidden}, kQ4_K);
  add("output.weight", {hidden, geometry.vocabularySize}, kQ4_K);
  add("token_embd.weight", {hidden, geometry.vocabularySize}, kQ4_K);

  const uint32_t stride = rowBytes(Q80, hidden);
  std::vector<uint8_t> rows(size_t{256} * stride, 0);
  for (uint32_t n = 0; n < 2 * heads; ++n)
    std::memcpy(rows.data() + size_t{n} * stride,
                (n < heads ? beta : alpha).data() + size_t{sourceRow(n % heads, 0, 1, groupHeads, groups)} * stride, stride);
  const Packed expected = repack(Q80, rows, 256, hidden, nullptr);

  char directory[] = "/tmp/splash-gguf-repack-XXXXXX";
  if (!mkdtemp(directory)) {
    check(false, "create a temporary directory");
    return;
  }
  const std::filesystem::path path = std::filesystem::path(directory) / "alpha-beta.gguf";
  const std::vector<uint8_t> file = ggufFile(tensors, geometry.layers, hidden);
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(file.data()), file.size());
  bool matches = false;
  try {
    const model::GgufFile gguf(path);
    const model::gguf::Image image = model::gguf::ImagePlanner(gguf, geometry).layer(0);
    // The layer's only Q8_0 descriptor is alpha/beta's; its plane0 and meta fills follow it.
    for (size_t i = 0; i + 2 < image.fills.size(); ++i) {
      const std::vector<uint8_t> &descriptor = image.fills[i].bytes;
      uint32_t type = 0;
      if (descriptor.size() == 64) std::memcpy(&type, descriptor.data(), sizeof type);
      if (type != kQ8_0) continue;
      matches = image.fills[i + 1].bytes == expected.w0 && image.fills[i + 2].bytes == expected.meta;
      break;
    }
  } catch (const model::GgufError &error) {
    std::fprintf(stderr, "%s\n", error.what());
  }
  std::filesystem::remove_all(directory);
  check(matches, "planner alpha/beta tensor matches the CPU reference");
}

constexpr uint64_t kSection = 16384; // image section alignment, as the planner lays out images
constexpr uint32_t kSourceOffset = 96; // tensor data offset inside the mapped source window
constexpr uint8_t kPoison = 0xA5;
constexpr uint32_t kNoPermute = 0xFFFFFFFFu;

uint64_t alignUp(uint64_t value) { return (value + kSection - 1) / kSection * kSection; }

struct Gpu {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> repack, copy;
};

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, const char *name) {
  id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
  NSError *error = nil;
  id<MTLComputePipelineState> state = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
  if (!state) std::fprintf(stderr, "no pipeline %s\n", name);
  return state;
}

id<MTLBuffer> buffer(id<MTLDevice> device, uint64_t bytes, uint8_t fill) {
  id<MTLBuffer> result = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  std::memset(result.contents, fill, bytes);
  return result;
}

template <class Params>
bool run(const Gpu &gpu, id<MTLComputePipelineState> state, id<MTLBuffer> source, id<MTLBuffer> image,
         const Params &params, uint64_t threads) {
  id<MTLCommandBuffer> command = [gpu.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setBuffer:source offset:0 atIndex:0];
  [encoder setBuffer:image offset:0 atIndex:1];
  [encoder setBytes:&params length:sizeof params atIndex:2];
  [encoder dispatchThreadgroups:MTLSizeMake((threads + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.error) std::fprintf(stderr, "GPU error: %s\n", command.error.localizedDescription.UTF8String);
  return !command.error;
}

// Bytes [begin, end) of the image equal expected, bytes outside every such range stay poison.
struct Expected {
  uint64_t begin;
  const std::vector<uint8_t> *bytes;
};
bool imageMatches(id<MTLBuffer> image, const std::vector<Expected> &sections) {
  const auto *data = static_cast<const uint8_t *>(image.contents);
  std::vector<bool> covered(image.length, false);
  for (const Expected &section : sections) {
    if (memcmp(data + section.begin, section.bytes->data(), section.bytes->size())) return false;
    std::fill_n(covered.begin() + section.begin, section.bytes->size(), true);
  }
  for (uint64_t i = 0; i < image.length; ++i)
    if (!covered[i] && data[i] != kPoison) return false;
  return true;
}

// Row tiles are whole (the planner requires rows and K to be multiples of 256);
// the shapes cover several tiles, super-blocks and groups, and a permuted row range.
struct Shape {
  uint32_t rows, K, permuteFrom, headRows, groupHeads, groups;
};

void checkRepack(const Gpu &gpu, Fmt f, const Shape &shape, uint32_t seed) {
  const QuantFormat &layout = kQuantFormats[f];
  const uint32_t rows = shape.rows, K = shape.K, G = K / 32, stride = rowBytes(f, K);
  const std::vector<uint8_t> native = fixture(f, rows, K, seed);
  std::vector<uint8_t> ordered(native.size());
  for (uint32_t n = 0; n < rows; ++n)
    std::memcpy(ordered.data() + uint64_t{n} * stride,
                native.data() + uint64_t{sourceRow(n, shape.permuteFrom, shape.headRows, shape.groupHeads, shape.groups)} * stride,
                stride);
  const Packed expected = repack(f, ordered, rows, K, nullptr);

  const uint64_t plane0 = kSection, plane1 = alignUp(plane0 + expected.w0.size());
  const uint64_t meta = layout.plane1_bytes ? alignUp(plane1 + expected.w1.size()) : plane1;
  const uint64_t bytes = alignUp(meta + expected.meta.size()) + kSection;
  id<MTLBuffer> source = buffer(gpu.device, kSourceOffset + native.size(), 0);
  std::memcpy(static_cast<uint8_t *>(source.contents) + kSourceOffset, native.data(), native.size());
  id<MTLBuffer> image = buffer(gpu.device, bytes, kPoison);
  const GgufRepackParams params{rows, K, uint32_t(f), kSourceOffset, stride,
                                uint32_t(plane0), layout.plane1_bytes ? uint32_t(plane1) : 0u, uint32_t(meta),
                                shape.permuteFrom, shape.headRows, shape.groupHeads, shape.groups};
  const bool ran = run(gpu, gpu.repack, source, image, params, uint64_t{rows} * G);
  std::vector<Expected> sections{{plane0, &expected.w0}, {meta, &expected.meta}};
  if (layout.plane1_bytes) sections.push_back({plane1, &expected.w1});
  char what[96];
  std::snprintf(what, sizeof what, "gguf_repack %s rows=%u K=%u%s", fmtName(f), rows, K,
                shape.permuteFrom == kNoPermute ? "" : " permuted");
  check(ran && imageMatches(image, sections), what);
}

void checkCopy(const Gpu &gpu, uint32_t bytes) {
  std::mt19937 rng(bytes);
  std::vector<uint8_t> rows(bytes);
  for (uint8_t &byte : rows) byte = static_cast<uint8_t>(rng());
  id<MTLBuffer> source = buffer(gpu.device, kSourceOffset + bytes, 0);
  std::memcpy(static_cast<uint8_t *>(source.contents) + kSourceOffset, rows.data(), bytes);
  id<MTLBuffer> image = buffer(gpu.device, alignUp(kSection + bytes) + kSection, kPoison);
  const GgufCopyParams params{kSourceOffset, uint32_t(kSection), bytes};
  const bool ran = run(gpu, gpu.copy, source, image, params, (uint64_t{bytes} + 15) / 16);
  check(ran && imageMatches(image, {{kSection, &rows}}), "gguf_copy bytes=" + std::to_string(bytes));
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::fprintf(stderr, "usage: gguf-repack --cpu | <metallib>\n");
      return 2;
    }
    void *ggml = nullptr;
    if (const char *oracle = std::getenv("SPLASH_GGML_ORACLE")) {
      ggml = dlopen(oracle, RTLD_NOW | RTLD_LOCAL);
      check(ggml, std::string("load ") + oracle + (ggml ? "" : std::string(": ") + dlerror()));
    }
    checkGoldens(ggml);
    checkAlphaBeta();
    if (std::string(argv[1]) != "--cpu") {
      Gpu gpu{MTLCreateSystemDefaultDevice(), nil, nil, nil};
      gpu.queue = [gpu.device newCommandQueue];
      NSError *error = nil;
      id<MTLLibrary> library = [gpu.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                       error:&error];
      if (!library) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
      }
      gpu.repack = pipeline(gpu.device, library, "gguf_repack");
      gpu.copy = pipeline(gpu.device, library, "gguf_copy");
      if (!gpu.repack || !gpu.copy) return 1;
      const Shape shapes[] = {{512, 1024, kNoPermute, 0, 0, 0}, {768, 1280, 256, 16, 8, 4}};
      for (int s = 0; s < 2; ++s)
        for (int f = 0; f < FMT_COUNT; ++f) checkRepack(gpu, Fmt(f), shapes[s], 100 + 8 * s + f);
      // 768 whole 16-byte chunks and a 5-byte tail, then an exact multiple.
      checkCopy(gpu, 768 * 16 + 5);
      checkCopy(gpu, 1024 * 16);
    }
    std::printf("%s (%d failures)\n", failures ? "GGUF repack tests FAILED" : "GGUF repack tests passed", failures);
    return failures ? 1 : 0;
  }
}
