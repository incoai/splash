#pragma once

#include "engine/Json.hpp"

#include <string>

namespace splash::test {

// `value` as a document of its own, through its writeJson.
template <class T> [[nodiscard]] std::string jsonOf(const T &value) {
  json::Writer writer;
  writeJson(writer, value);
  return writer.str();
}

} // namespace splash::test
