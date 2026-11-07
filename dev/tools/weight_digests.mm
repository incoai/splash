// Loads the weight images of an installed model as the engine does and
// prints, one per line of a JSON array in load order, the component, size and
// SHA-256 of each: the release check records them (verify-models) and the
// regression benchmarks compare them between builds (dev/benchmarks/weights.py).
//
//   weight-digests METALLIB MODEL_DIRECTORY [--expert-memory SIZE] [--ple-memory SIZE]
//
// The options stream a MoE target's routed experts and a Qwen3.8-Flash-Next
// target's PLE table from their files, as the runtime oracle's do
// (tools/LoadOptions.hpp), for a Mac whose memory cannot hold every image or
// whose largest Metal buffer cannot hold one. What streams has no image: the
// layers' images leave out their routed experts and a table that streams has
// none (ple.bin), so the digests cover neither; the engine reads their bytes
// from the source files. Any SIZE the stores take loads the same images. A
// load whose images and stores the engine's Metal memory budget cannot hold
// is refused before any image loads, naming the options.
#import <Foundation/Foundation.h>

#include "engine/MemoryPlan.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "model/SsdReaders.hpp"
#include "model/WeightImages.hpp"
#include "tools/LoadOptions.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;

int main(int argc, char **argv) {
  @autoreleasepool {
    const auto usage = [] {
      std::fprintf(stderr,
                   "usage: weight-digests METALLIB MODEL_DIRECTORY [--expert-memory SIZE] [--ple-memory SIZE]\n");
      return 2;
    };
    // Options come as option-value pairs.
    if (argc < 3 || argc % 2 == 0) return usage();
    try {
      model::ModelLoadOptions options;
      for (int index = 3; index < argc; index += 2)
        if (!tools::takeLoadOption(argv[index], argv[index + 1], options)) return usage();
      // Streamed weights read on a pool of SSD readers, as the engine's do.
      options.readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path root = std::filesystem::canonical(argv[2]);
      const model::ModelDescriptor descriptor = model::inspectModelRoot(root);
      // No start of the engine loads weights past its Metal memory budget,
      // which --max-memory only lowers, and a load past it can fail in a Metal
      // command for memory: refuse it before any image loads.
      const uint64_t budget =
          engine::EngineMemoryPolicy::hardBudgetBytes(backend.capabilities().recommendedMaxWorkingSetBytes, 0);
      if (const uint64_t bytes = tools::weightBytes(root, descriptor, options); bytes > budget)
        throw std::runtime_error("the weights need " + std::to_string(bytes / engine::kMiB) +
                                 " MiB, and the Metal memory budget is " + std::to_string(budget / engine::kMiB) +
                                 " MiB; --expert-memory SIZE streams a MoE target's routed experts and "
                                 "--ple-memory SIZE a Qwen3.8-Flash-Next target's PLE table");
      const model::LoadedModel loaded = model::loadModel(backend, root, descriptor, options);
      const auto images = loaded.images->contents();
      std::vector<std::string> digests(images.size());
      model::parallelFor(images.size(), [&](size_t index, unsigned) {
        digests[index] = model::weightDigest(images[index].bytes);
      });
      std::cout << '[';
      for (size_t index = 0; index < images.size(); ++index)
        std::cout << (index ? ",\n" : "\n") << "{\"component\":\"" << images[index].component
                  << "\",\"bytes\":" << images[index].bytes.size() << ",\"sha256\":\"" << digests[index] << "\"}";
      std::cout << "\n]\n";
    } catch (const std::exception &error) {
      std::fprintf(stderr, "error: %s\n", error.what());
      return 1;
    }
  }
  return 0;
}
