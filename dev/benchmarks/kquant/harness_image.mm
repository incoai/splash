// Builds MDKQ0001 images straight from a GGUF with the load-time repack kernels and
// compares them byte for byte with convert_gguf_to_splash.py output.
//   python3 dev/benchmarks/kquant/inline_metal.py metal/kernels/shared/kquant.metal > /tmp/kquant.metal
//   clang++ -std=c++20 -fobjc-arc -O2 -framework Foundation -framework Metal -I runtime \
//       dev/benchmarks/kquant/harness_image.mm runtime/model/GgufFile.cpp runtime/model/GgufImage.cpp -o /tmp/harness_image
//   /tmp/harness_image /tmp/kquant.metal model.gguf converted/target layer-0 layer-3 head embedding
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"

using namespace splash::model;

static std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: harness_image <inlined.metal> <gguf> <compare-dir> <layer-N|head|embedding>...\n"); return 2; }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = (MTLLanguageVersion)(4 << 16);
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr, "compile failed: %s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> repack = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kq_repack"] error:&error];
    id<MTLComputePipelineState> copy = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kq_copy"] error:&error];
    if (!repack || !copy) { fprintf(stderr, "pipelines failed\n"); return 1; }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    GgufFile file(argv[2]);
    gguf::ImagePlanner planner(file, gguf::TargetGeometry{});
    const std::string compareDir = argv[3];
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    const long page = sysconf(_SC_PAGESIZE);
    int failures = 0;
    for (int a = 4; a < argc; ++a) {
      const std::string which = argv[a];
      gguf::Image image = which == "head" ? planner.head() : which == "embedding" ? planner.embedding()
                          : planner.layer((uint32_t)std::stoul(which.substr(which.find('-') + 1)));
      auto started = std::chrono::steady_clock::now();
      id<MTLBuffer> out = [device newBufferWithLength:image.bytes options:MTLResourceStorageModeShared];
      memset(out.contents, 0, image.bytes);
      for (const gguf::Fill &fill : image.fills) memcpy((uint8_t *)out.contents + fill.offset, fill.bytes.data(), fill.bytes.size());
      id<MTLBuffer> src = nil; void *mapping = nullptr; size_t mapBytes = 0; uint64_t mapBase = 0;
      if (image.sourceEnd > image.sourceBegin) {
        mapBase = image.sourceBegin / page * page;
        mapBytes = ((image.sourceEnd - mapBase) + page - 1) / page * page;
        mapping = mmap(nullptr, mapBytes, PROT_READ, MAP_SHARED, fd, (off_t)mapBase);
        if (mapping == MAP_FAILED) { perror("mmap"); return 1; }
        src = [device newBufferWithBytesNoCopy:mapping length:mapBytes options:MTLResourceStorageModeShared deallocator:nil];
      }
      id<MTLCommandBuffer> commands = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
      for (gguf::Repack r : image.repacks) {
        r.params.src_offset = (uint32_t)(r.sourceOffset - mapBase);
        [encoder setComputePipelineState:repack];
        [encoder setBuffer:src offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&r.params length:sizeof r.params atIndex:2];
        const uint64_t threads = uint64_t(r.params.rows) * (r.params.input_size / 32);
        [encoder dispatchThreadgroups:MTLSizeMake((threads + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      for (gguf::Copy c : image.copies) {
        c.params.src_offset = (uint32_t)(c.sourceOffset - mapBase);
        [encoder setComputePipelineState:copy];
        [encoder setBuffer:src offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&c.params length:sizeof c.params atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((c.params.bytes / 16 + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [encoder endEncoding];
      [commands commit];
      [commands waitUntilCompleted];
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      std::vector<uint8_t> expected = readFile(compareDir + "/" + image.name);
      const uint8_t *got = (const uint8_t *)out.contents;
      std::string verdict;
      if (expected.empty()) verdict = "no reference file";
      else if (expected.size() != image.bytes) verdict = "SIZE MISMATCH reference " + std::to_string(expected.size());
      else {
        size_t first = 0; while (first < expected.size() && expected[first] == got[first]) ++first;
        if (first == expected.size()) verdict = "identical";
        else { size_t count = 0; for (size_t i = 0; i < expected.size(); ++i) count += expected[i] != got[i]; verdict = "MISMATCH first at " + std::to_string(first) + ", " + std::to_string(count) + " bytes differ"; }
      }
      if (verdict != "identical") ++failures;
      printf("%-14s %8.1f MB  %zu repacks %zu copies  %.3f s  %s\n", image.name.c_str(), image.bytes / 1e6, image.repacks.size(), image.copies.size(), seconds, verdict.c_str());
      if (mapping) munmap(mapping, mapBytes);
    }
    close(fd);
    printf(failures ? "FAILED\n" : "all images identical to converter output\n");
    return failures ? 1 : 0;
  }
}
