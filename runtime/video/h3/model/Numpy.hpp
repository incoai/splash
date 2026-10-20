#pragma once

// .npy / .npz files of the prepared inputs (python -m h3 prep) and of the native CLI's outputs. Writes the exact
// bytes numpy writes (np.save format 1.0; np.savez: stored zip entries with a zip64 local header).
#include "video/h3/engine/Status.hpp"

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
  // Elements converted to float32 / int64 (f4, f8, i4, i8, u1, u2).
  Status toFloat32(std::vector<float> &out) const;
  Status toInt64(int64_t &out) const;  // 0-d or single-element arrays

  static NpyArray of(const std::string &dtype, std::vector<int64_t> shape, const void *data, size_t bytes);
};

Status readNpy(const std::string &path, NpyArray &out);
Status readNpz(const std::string &path, std::map<std::string, NpyArray> &out);
Status writeNpy(const std::string &path, const NpyArray &array);
Status writeNpz(const std::string &path, const std::vector<std::pair<std::string, const NpyArray *>> &arrays);

// Typed access to a member of a loaded npz: dtype and element count checked.
Status member(const std::map<std::string, NpyArray> &npz, const std::string &name, const char *dtype, size_t count,
              const NpyArray *&out);

} // namespace h3::model
