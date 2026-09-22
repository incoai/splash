// Keep the same independent scalar reference and checked fixtures as the
// canonical GDN oracle. Renaming its entrypoint lets this standalone driver
// compare staged candidates without duplicating reference math.
#import <Metal/Metal.h>
#define main splash_flash_gdn_reference_entrypoint
#include "flash_gdn_metal_test.mm"
#undef main

#include "flash/FlashGDNStaged.hpp"

namespace {

constexpr std::array<std::pair<FlashGDNStageTile, const char *>, 4> stageTiles{
    {{FlashGDNStageTile::Values8Time16, "v8-t16"},
     {FlashGDNStageTile::Values8Time32, "v8-t32"},
     {FlashGDNStageTile::Values16Time16, "v16-t16"},
     {FlashGDNStageTile::Values16Time32, "v16-t32"}}};

std::array<bool, 4> stageSupported{};

void preflightStages(const char *metallib) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(metallib)]
                                               error:&error];
    require(device && library, "staged GDN resource preflight could not load Metal library");
    const std::array<const char *, 4> names{
        "flash_gdn_staged_v8_t16", "flash_gdn_staged_v8_t32",
        "flash_gdn_staged_v16_t16", "flash_gdn_staged_v16_t32"};
    uint32_t supportedCount = 0;
    for (size_t i = 0; i < names.size(); ++i) {
      id<MTLFunction> function = [library newFunctionWithName:@(names[i])];
      require(function, "staged GDN resource preflight missing function");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      require(pipeline, "staged GDN resource preflight could not create pipeline");
      const uint32_t threads = i < 2 ? 256 : 512;
      stageSupported[i] =
          pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength &&
          threads <= pipeline.maxTotalThreadsPerThreadgroup &&
          pipeline.threadExecutionWidth == 32;
      supportedCount += stageSupported[i];
      std::cout << "resource_tile=" << stageTiles[i].second
                << " pipeline=" << names[i]
                << " static_threadgroup_memory_bytes=" << pipeline.staticThreadgroupMemoryLength
                << " device_threadgroup_memory_limit=" << device.maxThreadgroupMemoryLength
                << " threads=" << threads
                << " pipeline_max_threads=" << pipeline.maxTotalThreadsPerThreadgroup
                << " supported=" << stageSupported[i] << '\n';
    }
    require(supportedCount > 0, "no staged GDN tile fits this runtime's pipeline resources");
  }
}

void compareStageBytes(const Fixture &candidate, const Fixture &control) {
  for (const auto &[got, expected] :
       std::array<std::pair<MetalBuffer, MetalBuffer>, 7>{
           {{candidate.buffers.mixed, control.buffers.mixed},
            {candidate.buffers.decay, control.buffers.decay},
            {candidate.buffers.beta, control.buffers.beta},
            {candidate.buffers.recurrentRows, control.buffers.recurrentRows},
            {candidate.buffers.output, control.buffers.output},
            {candidate.state.convolution, control.state.convolution},
            {candidate.state.recurrent, control.state.recurrent}}})
    require(std::memcmp(got.contents(), expected.contents(), got.sizeBytes()) == 0,
            "staged GDN changed canonical intermediate/output/state/padding bytes");
  require(*static_cast<const uint32_t *>(candidate.buffers.diagnostics.contents()) == 0,
          "staged GDN set diagnostics for valid inputs");
}

void qualifyStaged(MetalBackend &backend, uint32_t maximum) {
  for (uint32_t rows : {1u, 7u, 15u, 16u, 17u, 31u, 32u, 33u, 128u, 257u,
                        512u, 2048u}) {
    if (rows > maximum) continue;
    for (uint32_t lanes : {1u, 2u})
      for (bool cold : {true, false})
        for (const auto &[tile, name] : stageTiles) {
          if (!stageSupported[static_cast<size_t>(tile)]) continue;
          Reference seed(rows, lanes, cold);
          Fixture candidate(backend, seed), control(backend, seed);
          CommandGraph staged, canonical;
          addGDNStagedPrefill(staged, candidate.weights(), candidate.buffers,
                              candidate.state, rows, lanes, tile);
          addGDN(canonical, control.weights(), control.buffers, control.state,
                   rows, lanes);
          require(staged.dispatches().size() == 4,
                  "staged GDN changed prefill dispatch count");
          static_cast<void>(backend.submitCommand(canonical.dispatches()));
          static_cast<void>(backend.submitCommand(staged.dispatches()));
          compareStageBytes(candidate, control);
          // Repeat the exact same projected sequence on the newly carried
          // state, exercising both F32 recurrence and BF16 history causality.
          static_cast<void>(backend.submitCommand(canonical.dispatches()));
          static_cast<void>(backend.submitCommand(staged.dispatches()));
          compareStageBytes(candidate, control);
          std::cout << "tile=" << name << " rows=" << rows << " lanes=" << lanes
                    << " cold=" << cold << " two_sequences=exact\n";
        }
  }
}

void benchmarkStaged(MetalBackend &backend, uint32_t maximum) {
  for (uint32_t rows : {128u, 512u, 2048u}) {
    if (rows > maximum) continue;
    Reference seed(rows, 1, false);
    for (const auto &[tile, name] : stageTiles) {
      if (!stageSupported[static_cast<size_t>(tile)]) continue;
      Fixture candidate(backend, seed), control(backend, seed);
      CommandGraph staged, prepared;
      addGDNStagedPrefill(staged, candidate.weights(), candidate.buffers,
                          candidate.state, rows, 1, tile);
      // Compare to the current prepared-prefill route, isolating recurrence.
      addGDNFused(prepared, control.weights(), control.buffers, control.state,
                       rows, 1, FlashGDNFusion::Prepare);
      auto run = [&](Fixture &fixture, CommandGraph &graph) {
        std::memcpy(fixture.state.convolution.contents(), seed.history.data(),
                    flashGDNConvolutionLaneBytes());
        std::memcpy(fixture.state.recurrent.contents(), seed.recurrent.data(),
                    flashGDNRecurrentLaneBytes());
        const auto timing = backend.submitCommand(graph.dispatches());
        require(*static_cast<const uint32_t *>(fixture.buffers.diagnostics.contents()) == 0,
                "staged benchmark set numeric diagnostics");
        return timing;
      };
      for (uint32_t warmup = 0; warmup < 3; ++warmup) {
        static_cast<void>(run(control, prepared));
        static_cast<void>(run(candidate, staged));
      }
      constexpr uint32_t pairs = 12;
      double originalGPU = 0.0, stagedGPU = 0.0;
      double originalWall = 0.0, stagedWall = 0.0;
      for (uint32_t pair = 0; pair < pairs; ++pair) {
        splash::metal::CommandTiming original, experiment;
        if (pair % 2 == 0) {
          original = run(control, prepared); experiment = run(candidate, staged);
        } else {
          experiment = run(candidate, staged); original = run(control, prepared);
        }
        originalGPU += original.gpuSeconds; stagedGPU += experiment.gpuSeconds;
        originalWall += original.wallSeconds; stagedWall += experiment.wallSeconds;
      }
      compareStageBytes(candidate, control);
      std::cout << std::setprecision(8) << "bench_rows=" << rows << " tile=" << name
                << " pairs=" << pairs << " original_gpu_ms=" << originalGPU / pairs * 1000
                << " staged_gpu_ms=" << stagedGPU / pairs * 1000
                << " gpu_speedup_percent=" << (originalGPU / stagedGPU - 1) * 100
                << " original_command_wall_ms=" << originalWall / pairs * 1000
                << " staged_command_wall_ms=" << stagedWall / pairs * 1000 << '\n';
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument("usage: gdn-staged METALLIB [MAX_ROWS] [bench] | --cpu-only");
    cpuOnly();
    if (std::string(argv[1]) == "--cpu-only") {
      std::cout << "flash_gdn_staged_cpu_reference: PASS\n";
      return 0;
    }
    const uint32_t maximum = argc >= 3 ? static_cast<uint32_t>(std::stoul(argv[2])) : 128;
    // Shader validation may change the compiled static allocation. Query the
    // same creation API used by MetalBackend, under the current environment,
    // before any command queue submissions; never estimate fit with sizeof.
    preflightStages(argv[1]);
    if (argc == 4 && std::string(argv[3]) == "resources") return 0;
    MetalBackend backend(argv[1]);
    if (argc == 4 && std::string(argv[3]) == "bench") {
      benchmarkStaged(backend, maximum);
      return 0;
    }
    qualifyStaged(backend, maximum);
    std::cout << "flash_gdn_staged_metal_test: PASS numerical_policy="
              << kFlashGDNStagedNumericalPolicy << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_gdn_staged_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
