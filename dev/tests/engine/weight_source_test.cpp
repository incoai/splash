// A weight source reads the file it opened for as long as the model is loaded:
// a write to that file fails the next image written from it, while a new
// file at its path changes nothing. Its failures name it.
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/WeightSource.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;

namespace {

const std::vector<uint8_t> kBytes{1, 2, 3, 4, 5, 6, 7, 8};

void readsTensorDataAfterItsOffset(const std::filesystem::path &path) {
  WeightSource source(path);
  source.setDataOffset(2);
  std::array<uint8_t, 3> bytes{};
  source.readData(1, bytes);
  require(bytes == std::array<uint8_t, 3>{4, 5, 6}, "tensor data was not read after its offset");
  const SourceTensor tensor{&source, "U8", {4}, 2, 4};
  rejects([&] { tensor.read(2, bytes); }, "out of bounds", "a read past a tensor was accepted");
  rejects([&] { source.setDataOffset(kBytes.size() + 1); }, "starts past its end", "data past the file was accepted");
}

void aWriteToTheOpenFileIsRejected(const std::filesystem::path &path) {
  const WeightSource source(path);
  source.checkUnchanged();
  const int file = open(path.c_str(), O_WRONLY);
  const uint8_t value = 9;
  require(file >= 0 && pwrite(file, &value, 1, 0) == 1, "write the source fixture");
  close(file);
  rejects([&] { source.checkUnchanged(); }, "written while the model is loaded", "a written source was accepted");
}

void aNewFileAtThePathChangesNothing(const std::filesystem::path &path) {
  const WeightSource source(path);
  const auto replacement = path.parent_path() / "replacement";
  splash::test::writeFile(replacement, std::vector<uint8_t>(kBytes.size(), 0));
  std::filesystem::rename(replacement, path);
  source.checkUnchanged();
  std::array<uint8_t, 1> byte{};
  source.readData(1, byte);
  require(byte[0] == kBytes[1], "a source read the file that replaced it");
}

void aFailingSourceIsNamed(const std::filesystem::path &path) {
  const auto missing = path.parent_path() / "missing.gguf";
  rejects([&] { static_cast<void>(WeightSource(missing)); }, "unable to open " + missing.string(),
          "a missing source was reported without its path");
  rejects([&] { static_cast<void>(WeightSource(path.parent_path())); }, "not a regular file",
          "a directory was opened as a source");
  const WeightSource source(path);
  std::filesystem::resize_file(path, 2);
  std::array<uint8_t, 4> bytes{};
  rejects([&] { source.readData(0, bytes); }, "truncated weight file: " + path.string(),
          "a failed read was reported without its path");
}

} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-weight-source");
    const auto path = directory.path() / "source.bin";
    splash::test::writeFile(path, kBytes);
    readsTensorDataAfterItsOffset(path);
    aWriteToTheOpenFileIsRejected(path);
    splash::test::writeFile(path, kBytes);
    aNewFileAtThePathChangesNothing(path);
    aFailingSourceIsNamed(path);
    std::cout << "weight source: reads, writes to the open file, replacement and naming PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
