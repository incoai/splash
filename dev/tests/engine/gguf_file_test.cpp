#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestGguf.hpp"
#include "model/GgufFile.hpp"

#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace gguf = splash::test::gguf;
using splash::test::require;
using splash::model::ggml::kF32;

constexpr uint32_t kQ4_K = 12;
static_assert(std::string_view(splash::model::ggmlTypeTraits(kQ4_K)->name) == "Q4_K");
constexpr uint64_t kQ4KBlockBytes = 144;

// A file of one tensor, "weight", with one Q4_K block of data.
gguf::Bytes model(const std::vector<uint64_t> &dims, uint64_t displacement = 0, uint32_t type = kQ4_K) {
  return gguf::file({}, {{"weight", dims, type, gguf::Bytes(kQ4KBlockBytes), displacement}});
}

// A file whose only key, "unused", is an array: its element type and count,
// then the encoded elements.
gguf::Bytes array(const gguf::Bytes &value) { return gguf::file({{"unused", gguf::kArray, value}}, {}); }

// An array that declares count uint64 elements and holds none.
gguf::Bytes emptyUint64Array(uint64_t count) {
  gguf::Bytes value;
  gguf::append(value, gguf::kUint64);
  gguf::append(value, count);
  return array(value);
}

// An F32 tensor of `count` values whose every byte is `fill`, which tells its
// bytes from another tensor's.
gguf::Tensor filled(std::string name, uint64_t count, uint8_t fill) {
  return {std::move(name), {count}, kF32, gguf::Bytes(count * sizeof(float), fill)};
}

// A part of a split GGUF: the keys, the split keys gguf-split writes into
// every part, and the part's tensors.
gguf::Bytes part(std::vector<gguf::Key> keys, uint16_t number, uint16_t count, int32_t tensors,
                 const std::vector<gguf::Tensor> &held) {
  for (gguf::Key &key : gguf::splitKeys(number, count, tensors)) keys.push_back(std::move(key));
  return gguf::file(keys, held);
}

// A GGUF split in two as llama.cpp's gguf-split writes it, tensors in both
// parts, the metadata in the first: each tensor is read from its part, at
// its offset in that part's data, which starts after that part's header.
// Parts out of split order, a tensor in both parts and a split.tensors.count
// the parts do not hold are refused.
void checkSplit(const std::filesystem::path &directory) {
  const std::vector<gguf::Tensor> first{filled("a", 8, 0xa1), filled("b", 16, 0xb2)},
      second{filled("c", 8, 0xc3), filled("d", 24, 0xd4)};
  const std::vector<std::filesystem::path> paths{directory / "split-00001-of-00002.gguf",
                                                 directory / "split-00002-of-00002.gguf"};
  const auto write = [&](int32_t tensors, const std::vector<gguf::Tensor> &later) {
    splash::test::writeFile(paths[0], part({gguf::stringKey("general.architecture", "fixture")}, 0, 2, tensors, first));
    splash::test::writeFile(paths[1], part({}, 1, 2, tensors, later));
  };
  write(4, second);
  {
    const splash::model::OpenGguf split(paths);
    const splash::model::GgufFile &file = split.file();
    require(file.files() == 2 && file.tensors().size() == 4 && file.metadata().architecture() == "fixture" &&
                file.metadata().unsignedValue("split.count") == 2,
            "a split GGUF's tensors and its first part's metadata changed");
    // The parts' headers differ in size, so a tensor read at the other
    // part's data offset reads other bytes.
    require(file.source(0).dataOffset() != file.source(1).dataOffset(), "the split parts' data offsets are the same");
    require(file.require("c").offset == 0 && file.require("d").offset == second[0].data.size(),
            "a split part's tensor offsets are not in its own data");
    for (uint32_t number = 0; number < 2; ++number)
      for (const gguf::Tensor &expected : number ? second : first) {
        const splash::model::GgufTensor &tensor = file.require(expected.name);
        gguf::Bytes read(tensor.bytes);
        file.source(tensor.file).readData(tensor.offset, read);
        require(tensor.file == number && read == expected.data,
                "split GGUF tensor " + expected.name + " is not read from its part");
      }
  }
  splash::test::rejects([&] { splash::model::OpenGguf reversed({paths[1], paths[0]}); },
                        "GGUF split file " + paths[0].string() + " is out of order",
                        "accepted a split GGUF's parts out of order");
  write(3, {first[0]});
  splash::test::rejects([&] { splash::model::OpenGguf split(paths); }, "duplicate GGUF tensor: a",
                        "accepted a tensor in two split parts");
  write(5, second);
  splash::test::rejects([&] { splash::model::OpenGguf split(paths); },
                        "the GGUF's split files hold another number of tensors than split.tensors.count",
                        "accepted a split.tensors.count the parts do not hold");
}
} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-gguf-file");
    const auto path = directory.path() / "test.gguf";
    // Parsing data must fail with the expected error.
    const auto rejects = [&](const char *name, const gguf::Bytes &data, std::string_view expected) {
      splash::test::writeFile(path, data);
      splash::test::rejects([&] {
        splash::model::WeightSource source(path);
        splash::model::GgufFile file(source);
      }, expected, std::string("accepted invalid GGUF: ") + name);
    };
    splash::test::writeFile(path, gguf::file({gguf::stringKey("general.architecture", "fixture"),
                                              gguf::uint32Key("fixture.block_count", 2)},
                                             {{"weight", {256, 1}, kQ4_K, gguf::Bytes(kQ4KBlockBytes)}}));
    splash::model::WeightSource source(path);
    splash::model::GgufFile valid(source);
    const splash::model::GgufMetadata &metadata = valid.metadata();
    require(metadata.architecture() == "fixture" && metadata.unsignedValue("fixture.block_count") == 2,
            "metadata values changed");
    const auto &weight = valid.require("weight");
    require(weight.bytes == kQ4KBlockBytes && weight.elements() == 256 && weight.rows() == 1,
            "valid Q4_K shape/size changed");
    require(source.dataOffset() + weight.offset + weight.bytes == source.bytes(),
            "valid data must end exactly at EOF");

    constexpr std::string_view overflow = "GGUF size overflows", pastEnd = "runs past the end of the file";
    rejects("shape overflow", model({256, uint64_t{1} << 60}), overflow);
    rejects("row product overflow", model({256, uint64_t{1} << 63, 2}), overflow);
    rejects("byte size overflow", model({uint64_t{1} << 62}, 0, kF32), overflow);
    rejects("offset addition overflow", model({256, 1}, std::numeric_limits<uint64_t>::max() - 31), pastEnd);
    rejects("offset past EOF", model({256, 1}, 160), pastEnd);
    rejects("zero dimension", model({256, 0}), "zero tensor dimension");
    rejects("unaligned block", model({255, 1}), "not block aligned");
    rejects("unaligned offset", model({256, 1}, 1), "tensor data is misaligned");
    auto truncated = model({256, 1});
    truncated.pop_back();
    rejects("truncated tensor", truncated, pastEnd);
    const gguf::Tensor block{"weight", {256, 1}, kQ4_K, gguf::Bytes(kQ4KBlockBytes)};
    rejects("duplicate name", gguf::file({}, {block, block}), "duplicate GGUF tensor: weight");
    // The first part of a split GGUF read alone, whose split.count names the
    // others.
    rejects("lone split part", gguf::file({gguf::uint32Key("split.count", 3)}, {block}),
            "the GGUF's split.count does not match its 1 file");
    // Whatever the types of the two values, and for an array too.
    rejects("duplicate key",
            gguf::file({gguf::uint32Key("fixture.block_count", 2), gguf::stringKey("fixture.block_count", "2")}, {}),
            "duplicate GGUF metadata key: fixture.block_count");
    rejects("duplicate array key",
            gguf::file({gguf::int32ArrayKey("fixture.widths", {1}), gguf::int32ArrayKey("fixture.widths", {2})}, {}),
            "duplicate GGUF metadata key: fixture.widths");
    rejects("overflowing array size", emptyUint64Array(uint64_t{1} << 62), overflow);
    rejects("truncated array", emptyUint64Array(100), "GGUF header is truncated");
    gguf::Bytes nested;
    for (int i = 0; i < 18; ++i) {
      gguf::append(nested, gguf::kArray);
      gguf::append<uint64_t>(nested, 1);
    }
    rejects("excessive nesting", array(nested), "nesting is too deep");
    checkSplit(directory.path());
    std::cout << "GGUF parser bounds tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
