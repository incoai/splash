// Loads every weight image of an installed model as the engine does and
// prints, one per line of a JSON array in load order, the component, size and
// SHA-256 of each: the release check records them (verify-models) and the
// regression benchmarks compare them between builds (dev/benchmarks/weights.py).
//
//   weight-digests METALLIB MODEL_DIRECTORY
#import <Foundation/Foundation.h>

#include "metal/MetalBackend.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "model/WeightImages.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace splash;

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) {
      std::fprintf(stderr, "usage: weight-digests METALLIB MODEL_DIRECTORY\n");
      return 2;
    }
    try {
      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path root = std::filesystem::canonical(argv[2]);
      const model::LoadedModel loaded =
          model::loadModel(backend, root, model::inspectModelRoot(root));
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
