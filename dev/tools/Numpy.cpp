#include "Numpy.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace splash::dev {
namespace {

constexpr int64_t kGrowthAxisMaxDigits = 21;  // numpy.lib.format: spare header space for the first axis

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) fail("cannot open " + path);
  const std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint8_t> out(static_cast<size_t>(size));
  if (size && !file.read(reinterpret_cast<char *>(out.data()), size)) fail("cannot read " + path);
  return out;
}

void writeFile(const std::string &path, const std::vector<uint8_t> &bytes) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) fail("cannot write " + path);
  const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
  if (std::fclose(f) != 0 || !ok) fail("cannot write " + path);
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

NpyArray parseNpy(const uint8_t *p, size_t n, const std::string &what) {
  if (n < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) fail(what + ": not a .npy array");
  const uint8_t major = p[6];
  size_t headerLen = 0, start = 0;
  if (major == 1) {
    headerLen = load<uint16_t>(p + 8);
    start = 10;
  } else {
    headerLen = load<uint32_t>(p + 8);
    start = 12;
  }
  if (start + headerLen > n) fail(what + ": truncated header");
  const std::string header(reinterpret_cast<const char *>(p + start), headerLen);
  auto field = [&](const char *key) -> std::string {
    const size_t k = header.find(std::string("'") + key + "'");
    if (k == std::string::npos) return {};
    return header.substr(header.find(':', k) + 1);
  };
  NpyArray a;
  std::string descr = field("descr");
  const size_t q0 = descr.find('\''), q1 = descr.find('\'', q0 + 1);
  if (q0 == std::string::npos || q1 == std::string::npos) fail(what + ": bad descr");
  a.dtype = descr.substr(q0 + 1, q1 - q0 - 1);
  if (header.find("'fortran_order': True") != std::string::npos) fail(what + ": Fortran order");
  const std::string shape = field("shape");
  const size_t s0 = shape.find('('), s1 = shape.find(')');
  std::stringstream dims(shape.substr(s0 + 1, s1 - s0 - 1));
  std::string d;
  while (std::getline(dims, d, ','))
    if (d.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stoll(d));
  const size_t bytes = a.count() * a.itemSize();
  if (a.itemSize() == 0 || start + headerLen + bytes > n) fail(what + ": unsupported or short data");
  a.data.assign(p + start + headerLen, p + start + headerLen + bytes);
  return a;
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

NpyArray NpyArray::of(const std::string &dtype, std::vector<int64_t> shape, const void *data, size_t bytes) {
  NpyArray a;
  a.dtype = dtype;
  a.shape = std::move(shape);
  const auto *p = static_cast<const uint8_t *>(data);
  a.data.assign(p, p + bytes);
  return a;
}

std::map<std::string, NpyArray> readNpz(const std::string &path) {
  const std::vector<uint8_t> z = readFile(path);
  int64_t eocd = -1;
  for (int64_t i = static_cast<int64_t>(z.size()) - 22; i >= 0; --i)
    if (load<uint32_t>(&z[static_cast<size_t>(i)]) == 0x06054b50u) {
      eocd = i;
      break;
    }
  if (eocd < 0) fail(path + ": not a zip file");
  const uint16_t entries = load<uint16_t>(&z[eocd + 10]);
  size_t cd = load<uint32_t>(&z[eocd + 16]);
  std::map<std::string, NpyArray> out;
  for (uint16_t e = 0; e < entries; ++e) {
    if (cd + 46 > z.size() || load<uint32_t>(&z[cd]) != 0x02014b50u) fail(path + ": bad central directory");
    const uint16_t method = load<uint16_t>(&z[cd + 10]);
    uint64_t size = load<uint32_t>(&z[cd + 24]);
    const uint16_t nameLen = load<uint16_t>(&z[cd + 28]), extraLen = load<uint16_t>(&z[cd + 30]),
                   commentLen = load<uint16_t>(&z[cd + 32]);
    uint64_t local = load<uint32_t>(&z[cd + 42]);
    const std::string name(reinterpret_cast<const char *>(&z[cd + 46]), nameLen);
    if (method != 0) fail(path + ": compressed member " + name);
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
    if (local + 30 > z.size()) fail(path + ": bad local header");
    const size_t data = local + 30 + load<uint16_t>(&z[local + 26]) + load<uint16_t>(&z[local + 28]);
    if (data + size > z.size()) fail(path + ": truncated member " + name);
    std::string key = name.size() > 4 && name.substr(name.size() - 4) == ".npy" ? name.substr(0, name.size() - 4) : name;
    out[key] = parseNpy(&z[data], size, path + ":" + name);
    cd += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

void writeNpz(const std::string &path, const std::vector<std::pair<std::string, const NpyArray *>> &arrays) {
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
  writeFile(path, z);
}

} // namespace splash::dev
