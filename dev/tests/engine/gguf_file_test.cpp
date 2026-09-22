#include "model/GgufFile.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <unistd.h>

namespace {
using Bytes = std::vector<char>;

template <class T> void append(Bytes &out, T value) {
  const auto *p = reinterpret_cast<const char *>(&value);
  out.insert(out.end(), p, p + sizeof value);
}

void string(Bytes &out, const std::string &value) {
  append<uint64_t>(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

Bytes header(uint64_t tensors, uint64_t keys = 0) {
  Bytes out{'G', 'G', 'U', 'F'};
  append<uint32_t>(out, 3);
  append(out, tensors);
  append(out, keys);
  return out;
}

void tensor(Bytes &out, const std::vector<uint64_t> &dims, uint64_t offset = 0,
            uint32_t type = splash::model::ggml::kQ4_K) {
  string(out, "weight");
  append<uint32_t>(out, dims.size());
  for (auto dim : dims) append(out, dim);
  append(out, type);
  append(out, offset);
}

Bytes model(const std::vector<uint64_t> &dims, uint64_t offset = 0,
            uint32_t type = splash::model::ggml::kQ4_K) {
  auto out = header(1);
  tensor(out, dims, offset, type);
  out.resize((out.size() + 31) / 32 * 32 + 144);
  return out;
}

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
} // namespace

int main() {
  char directory[] = "/tmp/splash-gguf-test-XXXXXX";
  if (!mkdtemp(directory)) return 1;
  const auto path = std::filesystem::path(directory) / "test.gguf";
  const auto write = [&](const Bytes &data) {
    std::ofstream file(path, std::ios::binary);
    file.write(data.data(), data.size());
  };
  const auto rejects = [&](const char *name, const Bytes &data) {
    write(data);
    try { splash::model::GgufFile file(path); }
    catch (const splash::model::GgufError &) { return; }
    throw std::runtime_error(std::string("accepted invalid GGUF: ") + name);
  };
  try {
    write(model({256, 1}));
    splash::model::GgufFile valid(path);
    const auto &weight = valid.require("weight");
    require(weight.bytes == 144 && weight.elements() == 256 && weight.rows() == 1,
            "valid Q4_K shape/size changed");
    require(valid.absoluteOffset(weight) + weight.bytes == valid.fileBytes(),
            "valid data must end exactly at EOF");

    rejects("shape overflow", model({256, uint64_t{1} << 60}));
    rejects("row product overflow", model({256, uint64_t{1} << 63, 2}));
    rejects("byte size overflow", model({uint64_t{1} << 62}, 0, splash::model::ggml::kF32));
    rejects("offset addition overflow", model({256, 1}, std::numeric_limits<uint64_t>::max() - 31));
    rejects("offset past EOF", model({256, 1}, 160));
    rejects("zero dimension", model({256, 0}));
    rejects("unaligned block", model({255, 1}));
    rejects("unaligned offset", model({256, 1}, 1));
    auto truncated = model({256, 1});
    truncated.pop_back();
    rejects("truncated tensor", truncated);
    auto duplicate = header(2);
    tensor(duplicate, {256, 1});
    tensor(duplicate, {256, 1});
    duplicate.resize((duplicate.size() + 31) / 32 * 32 + 144);
    rejects("duplicate name", duplicate);
    for (auto count : {uint64_t{1} << 62, uint64_t{100}}) {
      auto metadata = header(0, 1);
      string(metadata, "unused");
      append<uint32_t>(metadata, 9); // array
      append<uint32_t>(metadata, 10); // uint64
      append(metadata, count);
      rejects("overflowing/truncated skipped array", metadata);
    }
    auto nested = header(0, 1);
    string(nested, "unused");
    append<uint32_t>(nested, 9);
    for (int i = 0; i < 18; ++i) {
      append<uint32_t>(nested, 9);
      append<uint64_t>(nested, 1);
    }
    rejects("excessive nesting", nested);
    std::filesystem::remove_all(directory);
    std::cout << "GGUF parser bounds tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::filesystem::remove_all(directory);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
