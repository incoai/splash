#include "video/h3/model/Safetensors.hpp"

#include "video/h3/model/Json.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <utility>

namespace h3::model {

Status readFully(int fd, uint64_t offset, size_t bytes, void *dst, const std::string &path) {
  auto *out = static_cast<uint8_t *>(dst);
  size_t done = 0;
  while (done < bytes) {
    const size_t want = std::min<size_t>(bytes - done, size_t(1) << 26);
    const ssize_t got = pread(fd, out + done, want, static_cast<off_t>(offset + done));
    if (got <= 0) return error(Code::Io, "short read of %s at %llu", path.c_str(), (unsigned long long)(offset + done));
    done += static_cast<size_t>(got);
  }
  return {};
}

SafetensorsFile::SafetensorsFile(SafetensorsFile &&other) noexcept
    : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)), tensors_(std::move(other.tensors_)) {}

SafetensorsFile &SafetensorsFile::operator=(SafetensorsFile &&other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) ::close(fd_);
    path_ = std::move(other.path_);
    fd_ = std::exchange(other.fd_, -1);
    tensors_ = std::move(other.tensors_);
  }
  return *this;
}

SafetensorsFile::~SafetensorsFile() {
  if (fd_ >= 0) ::close(fd_);
}

Status SafetensorsFile::open(const std::string &path, SafetensorsFile &out) {
  SafetensorsFile file;
  file.path_ = path;
  file.fd_ = ::open(path.c_str(), O_RDONLY);
  if (file.fd_ < 0) return error(Code::Io, "cannot open %s", path.c_str());
  uint64_t headerBytes = 0;
  H3_TRY(readFully(file.fd_, 0, 8, &headerBytes, path));
  if (headerBytes == 0 || headerBytes > (100u << 20)) return error(Code::Io, "%s: bad safetensors header", path.c_str());
  std::string header(headerBytes, '\0');
  H3_TRY(readFully(file.fd_, 8, headerBytes, header.data(), path));
  Json root;
  Status parsed = Json::parse(header, root);
  if (!parsed.ok()) return error(Code::Io, "%s: %s", path.c_str(), parsed.message().c_str());
  const uint64_t base = 8 + headerBytes;
  for (const auto &[name, entry] : root.object()) {
    if (name == "__metadata__") continue;
    const Json *dtype = entry.get("dtype"), *shape = entry.get("shape"), *offsets = entry.get("data_offsets");
    if (!dtype || !shape || !offsets || offsets->size() != 2)
      return error(Code::Io, "%s: bad entry %s", path.c_str(), name.c_str());
    TensorInfo info;
    info.dtype = dtype->string();
    for (const Json &d : shape->array()) info.shape.push_back(d.integer());
    info.offset = base + static_cast<uint64_t>(offsets->array()[0].integer());
    info.bytes = static_cast<uint64_t>(offsets->array()[1].integer() - offsets->array()[0].integer());
    file.tensors_[name] = std::move(info);
  }
  out = std::move(file);
  return {};
}

Status SafetensorsFile::find(const std::string &name, const char *dtype, const std::vector<int64_t> &shape,
                             TensorInfo &out) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) return error(Code::Io, "%s: no tensor %s", path_.c_str(), name.c_str());
  if (dtype && it->second.dtype != dtype)
    return error(Code::Io, "%s: %s is %s, expected %s", path_.c_str(), name.c_str(), it->second.dtype.c_str(), dtype);
  if (!shape.empty() && it->second.shape != shape)
    return error(Code::Io, "%s: %s has an unexpected shape", path_.c_str(), name.c_str());
  out = it->second;
  return {};
}

Status SafetensorsFile::read(const TensorInfo &info, void *dst) const {
  return readFully(fd_, info.offset, info.bytes, dst, path_);
}

Status SafetensorsFile::read(const std::string &name, const char *dtype, const std::vector<int64_t> &shape, void *dst,
                             size_t capacity) const {
  TensorInfo info;
  H3_TRY(find(name, dtype, shape, info));
  if (info.bytes > capacity) return error(Code::Io, "%s: %s larger than its buffer", path_.c_str(), name.c_str());
  return read(info, dst);
}

Status SafetensorsFile::readRange(uint64_t offset, size_t bytes, void *dst) const {
  return readFully(fd_, offset, bytes, dst, path_);
}

} // namespace h3::model
