#pragma once

// .npy / .npz files: the prep directory of h3_prep_write and the hidden states and latents of the engine process's
// --hidden-states and --keep (VideoRuntime.mm). Writes the exact bytes numpy writes (np.save format 1.0; np.savez:
// stored zip entries with a zip64 local header). Failures throw std::runtime_error.
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace h3::model {

struct NpyArray {
  std::string dtype;  // numpy descr: "<f4", "<f8", "<u2", "|u1", "<i8", "<i4"
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;

  [[nodiscard]] size_t count() const;
  [[nodiscard]] size_t itemSize() const;
  template <class T>
  [[nodiscard]] const T *as() const { return reinterpret_cast<const T *>(data.data()); }

  static NpyArray of(const std::string &dtype, std::vector<int64_t> shape, const void *data, size_t bytes);
};

[[nodiscard]] std::map<std::string, NpyArray> readNpz(const std::string &path);
void writeNpy(const std::string &path, const NpyArray &array);
void writeNpz(const std::string &path, const std::vector<std::pair<std::string, const NpyArray *>> &arrays);

} // namespace h3::model
