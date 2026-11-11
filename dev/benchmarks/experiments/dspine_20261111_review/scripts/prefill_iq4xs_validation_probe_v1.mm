// Diagnostic only: reuse the unchanged GGUF oracle helpers, but execute one
// plain IQ4_XS prefill (1024x1024, 168 live rows) instead of the full suite.
// Compile against each tree's own headers and host archive; do not mix their
// LinearConfig ABI. The supplied library may be production or production-and-test.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>

#define main unused_full_gguf_projection_suite
#include "gguf_projection_test.mm"
#undef main

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--inspect-only")) {
      std::cerr << "usage: prefill-iq4xs-probe <metallib> [--inspect-only]\n";
      return 2;
    }
    try {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("no Metal device");
      NSError *error = nil;
      NSString *path = [NSString stringWithUTF8String:argv[1]];
      id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
      if (!library)
        throw std::runtime_error("cannot load supplied library: " + std::string(error.localizedDescription.UTF8String));
      id<MTLFunction> function = [library newFunctionWithName:@"gguf_prefill_iq4xs_a"];
      if (!function) throw std::runtime_error("missing gguf_prefill_iq4xs_a");
      // Match MetalBackend::newPipeline: the same device API and no descriptor options.
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline)
        throw std::runtime_error("cannot create prefill pipeline: " + std::string(error.localizedDescription.UTF8String));
      const auto flag = [](const char *name) {
        const char *value = std::getenv(name);
        return value ? [NSString stringWithUTF8String:value] : @"(unset)";
      };
      NSDictionary *resources = @{
        @"kind": @"pipeline_resources", @"device": device.name,
        @"kernel": @"gguf_prefill_iq4xs_a",
        @"pipeline_static_threadgroup_bytes": @(pipeline.staticThreadgroupMemoryLength),
        @"device_max_threadgroup_bytes": @(device.maxThreadgroupMemoryLength),
        @"source_stage_bytes": @(2ULL * GGUF_TILE_COLUMNS * GGUF_PREFILL_STEP * sizeof(uint16_t)),
        @"source_codebook_bytes": @(256ULL * sizeof(uint32_t)),
        @"MTL_DEBUG_LAYER": flag("MTL_DEBUG_LAYER"),
        @"MTL_SHADER_VALIDATION": flag("MTL_SHADER_VALIDATION")
      };
      NSData *json = [NSJSONSerialization dataWithJSONObject:resources options:NSJSONWritingSortedKeys error:&error];
      if (!json) throw std::runtime_error("cannot serialize pipeline resources");
      std::fwrite(json.bytes, 1, json.length, stderr);
      std::fputc('\n', stderr);
      std::fflush(stderr); // Preserve the resource record if API validation aborts at dispatch.
      if (argc == 3) return 0;

      MetalBackend backend(argv[1]);
      const Linear linear(backend.capabilities());
      constexpr uint32_t n = 1024, k = 1024, rows = 168;
      const Tensor weights = tensor(backend, IQ4XS, n, k);
      const Projection projectionWeights = projection({&weights}, n);
      const LinearWorkload workload{{n, k}, rows, LinearPhase::Prefill, LinearEpilogue::None};
      const LinearPlan plan = Linear::plan(workload, {.tile = LinearTile::GgufPrefill}, FloatOutput::BFloat16);
      const auto inputs = activations(Inputs::Dense, plan.storageRows(), k);
      const Outcome out = run(backend, linear, plan, projectionWeights, nullptr,
                              storageRows(inputs, k, rows, plan.storageRows()), {}, kPoisonNaN, n,
                              "minimal IQ4_XS prefill");
      checkValues(out, products({&weights}, n, inputs, rows), {}, {}, workload, rows, n, true,
                  "minimal IQ4_XS prefill");
      section("minimal IQ4_XS prefill: one plain dispatch, independent fp64 and buffer guards");
      return failures ? 1 : 0;
    } catch (const std::exception &failure) {
      std::cerr << "prefill-iq4xs-probe: " << failure.what() << '\n';
      return 1;
    }
  }
}
