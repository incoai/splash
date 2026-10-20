#include "video/h3/model/Numpy.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace h3::model {
namespace {

constexpr int64_t kGrowthAxisMaxDigits = 21;  // numpy.lib.format: spare header space for the first axis

Status readFile(const std::string &path, std::vector<uint8_t> &out) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return error(Code::Io, "cannot open %s", path.c_str());
  const std::streamsize size = file.tellg();
  file.seekg(0);
  out.resize(static_cast<size_t>(size));
  if (size && !file.read(reinterpret_cast<char *>(out.data()), size)) return error(Code::Io, "cannot read %s", path.c_str());
  return {};
}

Status writeFile(const std::string &path, const std::vector<uint8_t> &bytes) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) return error(Code::Io, "cannot write %s", path.c_str());
  const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
  if (std::fclose(f) != 0 || !ok) return error(Code::Io, "cannot write %s", path.c_str());
  return {};
}

template <class T>
T load(const uint8_t *p) {
  T v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

template <class T>
void put(std::vector<uint8_t> &out, T v) {
  const auto *p = reinterpret_cast<const uint8_t *>(&v);
  out.insert(out.end(), p, p + sizeof v);
}

uint32_t crc32(const uint8_t *data, size_t n) {
  uint32_t table[256];
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    table[i] = c;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

Status parseNpy(const uint8_t *p, size_t n, const std::string &what, NpyArray &out) {
  if (n < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) return error(Code::Io, "%s: not a .npy array", what.c_str());
  const uint8_t major = p[6];
  size_t headerLen = 0, start = 0;
  if (major == 1) {
    headerLen = load<uint16_t>(p + 8);
    start = 10;
  } else {
    headerLen = load<uint32_t>(p + 8);
    start = 12;
  }
  if (start + headerLen > n) return error(Code::Io, "%s: truncated header", what.c_str());
  const std::string header(reinterpret_cast<const char *>(p + start), headerLen);
  auto field = [&](const char *key) -> std::string {
    const size_t k = header.find(std::string("'") + key + "'");
    if (k == std::string::npos) return {};
    return header.substr(header.find(':', k) + 1);
  };
  NpyArray a;
  std::string descr = field("descr");
  const size_t q0 = descr.find('\''), q1 = descr.find('\'', q0 + 1);
  if (q0 == std::string::npos || q1 == std::string::npos) return error(Code::Io, "%s: bad descr", what.c_str());
  a.dtype = descr.substr(q0 + 1, q1 - q0 - 1);
  if (header.find("'fortran_order': True") != std::string::npos) return error(Code::Io, "%s: Fortran order", what.c_str());
  const std::string shape = field("shape");
  const size_t s0 = shape.find('('), s1 = shape.find(')');
  std::stringstream dims(shape.substr(s0 + 1, s1 - s0 - 1));
  std::string d;
  while (std::getline(dims, d, ','))
    if (d.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stoll(d));
  const size_t bytes = a.count() * a.itemSize();
  if (a.itemSize() == 0 || start + headerLen + bytes > n) return error(Code::Io, "%s: unsupported or short data", what.c_str());
  a.data.assign(p + start + headerLen, p + start + headerLen + bytes);
  out = std::move(a);
  return {};
}

std::vector<uint8_t> serializeNpy(const NpyArray &a) {
  std::string shape = "(";
  for (size_t i = 0; i < a.shape.size(); ++i) shape += std::to_string(a.shape[i]) + (a.shape.size() == 1 ? ",)" : i + 1 < a.shape.size() ? ", " : ")");
  if (a.shape.empty()) shape = "()";
  std::string header = "{'descr': '" + a.dtype + "', 'fortran_order': False, 'shape': " + shape + ", }";
  if (!a.shape.empty()) header += std::string(kGrowthAxisMaxDigits - std::to_string(a.shape[0]).size(), ' ');
  const size_t hlen = header.size() + 1, padlen = 64 - ((6 + 2 + 2 + hlen) % 64);
  header += std::string(padlen, ' ') + "\n";
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  put<uint16_t>(out, static_cast<uint16_t>(header.size()));
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), a.data.begin(), a.data.end());
  return out;
}

} // namespace

size_t NpyArray::count() const {
  size_t n = 1;
  for (int64_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

size_t NpyArray::itemSize() const {
  if (dtype == "<f4" || dtype == "<i4" || dtype == "<u4") return 4;
  if (dtype == "<f8" || dtype == "<i8" || dtype == "<u8") return 8;
  if (dtype == "<u2" || dtype == "<i2" || dtype == "<f2") return 2;
  if (dtype == "|u1" || dtype == "|i1" || dtype == "|b1") return 1;
  return 0;
}

Status NpyArray::toFloat32(std::vector<float> &out) const {
  const size_t n = count();
  out.resize(n);
  if (dtype == "<f4") std::memcpy(out.data(), data.data(), n * 4);
  else if (dtype == "<f8") for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(as<double>()[i]);
  else return error(Code::Io, "expected a float array, got %s", dtype.c_str());
  return {};
}

Status NpyArray::toInt64(int64_t &out) const {
  if (count() != 1) return error(Code::Io, "expected a scalar");
  if (dtype == "<i8") out = as<int64_t>()[0];
  else if (dtype == "<i4") out = as<int32_t>()[0];
  else return error(Code::Io, "expected an integer, got %s", dtype.c_str());
  return {};
}

NpyArray NpyArray::of(const std::string &dtype, std::vector<int64_t> shape, const void *data, size_t bytes) {
  NpyArray a;
  a.dtype = dtype;
  a.shape = std::move(shape);
  const auto *p = static_cast<const uint8_t *>(data);
  a.data.assign(p, p + bytes);
  return a;
}

Status readNpy(const std::string &path, NpyArray &out) {
  std::vector<uint8_t> bytes;
  H3_TRY(readFile(path, bytes));
  return parseNpy(bytes.data(), bytes.size(), path, out);
}

Status readNpz(const std::string &path, std::map<std::string, NpyArray> &out) {
  std::vector<uint8_t> z;
  H3_TRY(readFile(path, z));
  int64_t eocd = -1;
  for (int64_t i = static_cast<int64_t>(z.size()) - 22; i >= 0; --i)
    if (load<uint32_t>(&z[static_cast<size_t>(i)]) == 0x06054b50u) {
      eocd = i;
      break;
    }
  if (eocd < 0) return error(Code::Io, "%s: not a zip file", path.c_str());
  const uint16_t entries = load<uint16_t>(&z[eocd + 10]);
  size_t cd = load<uint32_t>(&z[eocd + 16]);
  out.clear();
  for (uint16_t e = 0; e < entries; ++e) {
    if (cd + 46 > z.size() || load<uint32_t>(&z[cd]) != 0x02014b50u) return error(Code::Io, "%s: bad central directory", path.c_str());
    const uint16_t method = load<uint16_t>(&z[cd + 10]);
    uint64_t size = load<uint32_t>(&z[cd + 24]);
    const uint16_t nameLen = load<uint16_t>(&z[cd + 28]), extraLen = load<uint16_t>(&z[cd + 30]),
                   commentLen = load<uint16_t>(&z[cd + 32]);
    uint64_t local = load<uint32_t>(&z[cd + 42]);
    const std::string name(reinterpret_cast<const char *>(&z[cd + 46]), nameLen);
    if (method != 0) return error(Code::Io, "%s: compressed member %s", path.c_str(), name.c_str());
    for (size_t x = cd + 46 + nameLen; x + 4 <= cd + 46 + nameLen + extraLen;) {  // zip64 extra
      const uint16_t id = load<uint16_t>(&z[x]), len = load<uint16_t>(&z[x + 2]);
      if (id == 1) {
        size_t f = x + 4;
        if (size == 0xFFFFFFFFu) { size = load<uint64_t>(&z[f]); f += 8; }
        if (load<uint32_t>(&z[cd + 20]) == 0xFFFFFFFFu) f += 8;
        if (local == 0xFFFFFFFFu) local = load<uint64_t>(&z[f]);
      }
      x += 4 + len;
    }
    if (local + 30 > z.size()) return error(Code::Io, "%s: bad local header", path.c_str());
    const size_t data = local + 30 + load<uint16_t>(&z[local + 26]) + load<uint16_t>(&z[local + 28]);
    if (data + size > z.size()) return error(Code::Io, "%s: truncated member %s", path.c_str(), name.c_str());
    std::string key = name.size() > 4 && name.substr(name.size() - 4) == ".npy" ? name.substr(0, name.size() - 4) : name;
    H3_TRY(parseNpy(&z[data], size, path + ":" + name, out[key]));
    cd += 46 + nameLen + extraLen + commentLen;
  }
  return {};
}

Status writeNpy(const std::string &path, const NpyArray &array) { return writeFile(path, serializeNpy(array)); }

Status writeNpz(const std::string &path, const std::vector<std::pair<std::string, const NpyArray *>> &arrays) {
  std::vector<uint8_t> z, central;
  for (const auto &[key, array] : arrays) {
    const std::vector<uint8_t> member = serializeNpy(*array);
    const std::string name = key + ".npy";
    const uint32_t crc = crc32(member.data(), member.size()), offset = static_cast<uint32_t>(z.size());
    // local header (zip64 sizes, as zipfile writes with force_zip64), DOS date 1980-01-01
    put<uint32_t>(z, 0x04034b50u);
    put<uint16_t>(z, 45);
    put<uint16_t>(z, 0);
    put<uint16_t>(z, 0);
    put<uint16_t>(z, 0);
    put<uint16_t>(z, 0x21);
    put<uint32_t>(z, crc);
    put<uint32_t>(z, 0xFFFFFFFFu);
    put<uint32_t>(z, 0xFFFFFFFFu);
    put<uint16_t>(z, static_cast<uint16_t>(name.size()));
    put<uint16_t>(z, 20);
    z.insert(z.end(), name.begin(), name.end());
    put<uint16_t>(z, 1);
    put<uint16_t>(z, 16);
    put<uint64_t>(z, member.size());
    put<uint64_t>(z, member.size());
    z.insert(z.end(), member.begin(), member.end());
    // central directory record: made by unix / 4.5, -rw------- (0o600 << 16)
    put<uint32_t>(central, 0x02014b50u);
    put<uint16_t>(central, 0x032d);
    put<uint16_t>(central, 45);
    put<uint16_t>(central, 0);
    put<uint16_t>(central, 0);
    put<uint16_t>(central, 0);
    put<uint16_t>(central, 0x21);
    put<uint32_t>(central, crc);
    put<uint32_t>(central, static_cast<uint32_t>(member.size()));
    put<uint32_t>(central, static_cast<uint32_t>(member.size()));
    put<uint16_t>(central, static_cast<uint16_t>(name.size()));
    put<uint16_t>(central, 0);
    put<uint16_t>(central, 0);
    put<uint16_t>(central, 0);
    put<uint16_t>(central, 0);
    put<uint32_t>(central, 0600u << 16);
    put<uint32_t>(central, offset);
    central.insert(central.end(), name.begin(), name.end());
  }
  const uint32_t cdOffset = static_cast<uint32_t>(z.size()), cdSize = static_cast<uint32_t>(central.size());
  z.insert(z.end(), central.begin(), central.end());
  put<uint32_t>(z, 0x06054b50u);
  put<uint16_t>(z, 0);
  put<uint16_t>(z, 0);
  put<uint16_t>(z, static_cast<uint16_t>(arrays.size()));
  put<uint16_t>(z, static_cast<uint16_t>(arrays.size()));
  put<uint32_t>(z, cdSize);
  put<uint32_t>(z, cdOffset);
  put<uint16_t>(z, 0);
  return writeFile(path, z);
}

Status member(const std::map<std::string, NpyArray> &npz, const std::string &name, const char *dtype, size_t count,
              const NpyArray *&out) {
  auto it = npz.find(name);
  if (it == npz.end()) return error(Code::Io, "npz member %s missing", name.c_str());
  if (dtype && it->second.dtype != dtype)
    return error(Code::Io, "npz member %s is %s, expected %s", name.c_str(), it->second.dtype.c_str(), dtype);
  if (count && it->second.count() != count)
    return error(Code::Io, "npz member %s has %zu elements, expected %zu", name.c_str(), it->second.count(), count);
  out = &it->second;
  return {};
}

} // namespace h3::model
