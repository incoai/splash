#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace splash::model {

// Bytes [offset, offset + size) of the file open at descriptor.
void readWeightBytes(int descriptor, uint64_t offset, std::span<uint8_t> bytes);

// A source file of a model's weights, open until the model is unloaded: its
// images are written from it again whenever their memory is restored
// (WeightImages). Its parser reads the metadata through descriptor() and
// sets where the tensor data starts; tensor offsets are relative to it.
// checkUnchanged throws once the open file was written since it was opened,
// so no image mixes its old and new bytes; replacing it at its path does not
// change the open file.
class WeightSource final {
public:
  explicit WeightSource(const std::filesystem::path &path);
  ~WeightSource();
  WeightSource(const WeightSource &) = delete;
  WeightSource &operator=(const WeightSource &) = delete;
  [[nodiscard]] const std::filesystem::path &path() const noexcept;
  [[nodiscard]] int descriptor() const noexcept;
  // The file's size when it was opened.
  [[nodiscard]] uint64_t bytes() const noexcept;
  void setDataOffset(uint64_t offset);
  [[nodiscard]] uint64_t dataOffset() const noexcept;
  // Bytes [offset, offset + size) of the tensor data.
  void readData(uint64_t offset, std::span<uint8_t> bytes) const;
  void checkUnchanged() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A tensor of a source file: dtype and shape as the file names them, and
// where its bytes are in the file's tensor data.
struct SourceTensor final {
  const WeightSource *file = nullptr;
  std::string dtype;
  std::vector<uint64_t> shape;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  void read(uint64_t at, std::span<uint8_t> destination) const;
};

} // namespace splash::model
