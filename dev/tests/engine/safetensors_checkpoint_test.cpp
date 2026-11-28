#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightStore.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;
namespace {
void shard(const std::filesystem::path &path, std::string_view header, size_t bytes = 16) {
  const uint64_t length = header.size();
  std::vector<uint8_t> file(sizeof length);
  std::memcpy(file.data(), &length, sizeof length);
  file.insert(file.end(), header.begin(), header.end());
  for (size_t i = 0; i < bytes; ++i) file.push_back(static_cast<uint8_t>(i + 1));
  splash::test::writeFile(path, file);
}
constexpr auto valid = R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]}})";
}
int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-safetensors-checkpoint");
    const std::filesystem::path &root = directory.path();
    shard(root / "model.safetensors", valid);
    SafetensorsCheckpoint source(root);
    require(source.require("a").shape == std::vector<uint64_t>({2, 2}), "shape changed");
    std::array<uint8_t, 4> data{};
    source.require("a").read(7, data);
    require(data == std::array<uint8_t, 4>{8, 9, 10, 11}, "bounded slice differs");
    rejects([&] { source.require("a").read(14, data); }, "source tensor read is out of bounds",
            "out-of-bounds read accepted");
    rejects([&] { (void)source.require("missing"); }, "missing source tensor: missing", "missing tensor accepted");
    // Rewriting a shard in place, even with the same content, writes the
    // file the checkpoint holds.
    shard(root / "model.safetensors", valid);
    rejects([&] { source.checkUnchanged(); }, "written while the model is loaded", "written source accepted");
    shard(root / "extra.safetensors", valid);
    rejects([&] { SafetensorsCheckpoint invalid(root); }, "duplicate source tensor: a",
            "duplicate tensor accepted");
    std::filesystem::remove(root / "extra.safetensors");
    for (const auto &[header, error] : std::initializer_list<std::pair<std::string_view, std::string_view>>{
             {R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,15]}})", "safetensors data range is invalid"},
             {R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[1,17]}})", "safetensors data range is invalid"},
             {R"({"a":{"dtype":"U32","shape":[true,4],"data_offsets":[0,16]}})",
              "safetensors metadata requires an integer"},
             {R"({"a":{"dtype":"U32","shape":[1,1,1,1,1,1,1,1,4],"data_offsets":[0,16]}})",
              "source tensor metadata exceeds bounds"},
             {R"({"a":{"dtype":"U32","shape":[4503599627370496,4096],"data_offsets":[0,16]}})",
              "source tensor size overflows"},
             {R"({"a":{"dtype":"U32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"U32","shape":[2],"data_offsets":[8,16]}})",
              "overlapping source tensors"},
             {R"({"a":{"dtype":"FP4","shape":[4],"data_offsets":[0,16]}})", "unsupported safetensors dtype: FP4"}}) {
      shard(root / "model.safetensors", header);
      rejects([&] { SafetensorsCheckpoint invalid(root); }, error,
              "malformed tensor accepted: " + std::string(header));
    }
    shard(root / "model.safetensors", valid, 8);
    rejects([&] { SafetensorsCheckpoint invalid(root); }, "safetensors data range is invalid",
            "truncated tensor accepted");
    // A part of an installed model that is absent, or holds no shard, is a
    // weight error that names its directory, never the filesystem's.
    std::filesystem::create_directories(root / "empty");
    for (const auto &[part, error] : std::initializer_list<std::pair<std::filesystem::path, std::string>>{
             {root / "vision", "no safetensors weights in " + (root / "vision").string() + ": "},
             {root / "empty", "no safetensors weights in " + (root / "empty").string()}}) {
      try {
        SafetensorsCheckpoint missing(part);
        throw std::runtime_error("a source without shards was read: " + part.string());
      } catch (const WeightStoreError &refused) {
        require(std::string_view(refused.what()).find(error) != std::string_view::npos,
                "a source without shards was refused as \"" + std::string(refused.what()) + "\"");
      }
    }
    std::cout << "safetensors checkpoint: bounded reads, metadata, written and malformed sources PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
