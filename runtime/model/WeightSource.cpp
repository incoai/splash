#include "model/WeightSource.hpp"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace splash::model {
namespace {

// A source is a model's own file, so its failures name it.
[[noreturn]] void fail(const char *operation, const std::filesystem::path &path) {
  throw std::system_error(errno, std::generic_category(), std::string(operation) + " " + path.string());
}

} // namespace

void readWeightBytes(int fd, uint64_t offset, std::span<uint8_t> bytes) {
  if (bytes.size() > uint64_t(std::numeric_limits<off_t>::max()) ||
      offset > uint64_t(std::numeric_limits<off_t>::max()) - bytes.size())
    throw std::overflow_error("weight read offset overflow");
  while (!bytes.empty()) {
    const ssize_t count = pread(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) throw std::system_error(errno, std::generic_category(), "read weight bytes");
    if (!count) throw std::runtime_error("truncated weight file");
    offset += count;
    bytes = bytes.subspan(count);
  }
}

struct WeightSource::Impl {
  std::filesystem::path path;
  int file = -1;
  struct stat state{};
  uint64_t dataOffset = 0;
  explicit Impl(const std::filesystem::path &path) : path(path) {
    file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file < 0) fail("unable to open", path);
    // Loading reads each tensor once into memory: through the page cache it
    // would leave another copy of the model there.
    if (fstat(file, &state) || fcntl(file, F_NOCACHE, 1)) {
      const int error = errno;
      close(file);
      errno = error;
      fail("unable to open", path);
    }
    if (!S_ISREG(state.st_mode)) {
      close(file);
      throw std::runtime_error("weight source is not a regular file: " + path.string());
    }
  }
  ~Impl() { close(file); }
};

WeightSource::WeightSource(const std::filesystem::path &path) : impl_(std::make_unique<Impl>(path)) {}
WeightSource::~WeightSource() = default;
const std::filesystem::path &WeightSource::path() const noexcept { return impl_->path; }
int WeightSource::descriptor() const noexcept { return impl_->file; }
uint64_t WeightSource::bytes() const noexcept { return uint64_t(impl_->state.st_size); }
void WeightSource::setDataOffset(uint64_t offset) {
  if (offset > bytes()) throw std::runtime_error("weight source data starts past its end: " + impl_->path.string());
  impl_->dataOffset = offset;
}
uint64_t WeightSource::dataOffset() const noexcept { return impl_->dataOffset; }
void WeightSource::readData(uint64_t offset, std::span<uint8_t> bytes) const {
  if (offset > std::numeric_limits<uint64_t>::max() - impl_->dataOffset)
    throw std::overflow_error("weight read offset overflow");
  try {
    readWeightBytes(impl_->file, impl_->dataOffset + offset, bytes);
  } catch (const std::exception &error) {
    throw std::runtime_error(std::string(error.what()) + ": " + impl_->path.string());
  }
}
void WeightSource::checkUnchanged() const {
  struct stat current{};
  if (fstat(impl_->file, &current)) fail("unable to stat", impl_->path);
  const struct stat &opened = impl_->state;
  if (current.st_size != opened.st_size || current.st_mtimespec.tv_sec != opened.st_mtimespec.tv_sec ||
      current.st_mtimespec.tv_nsec != opened.st_mtimespec.tv_nsec)
    throw std::runtime_error("weight source was written while the model is loaded: " + impl_->path.string());
}

void SourceTensor::read(uint64_t at, std::span<uint8_t> destination) const {
  if (at > bytes || destination.size() > bytes - at) throw std::out_of_range("source tensor read is out of bounds");
  file->readData(offset + at, destination);
}

} // namespace splash::model
