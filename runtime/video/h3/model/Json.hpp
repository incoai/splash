#pragma once

// Minimal JSON reader for the weight manifests and safetensors headers (objects, arrays, strings, numbers, booleans,
// null). Numbers are kept as double and as the exact integer when they are integral.
#include "video/h3/engine/Status.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace h3::model {

class Json {
public:
  enum class Kind { Null, Bool, Number, String, Array, Object };

  static Status parse(const std::string &text, Json &out);
  static Status parseFile(const std::string &path, Json &out);

  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] bool isObject() const { return kind_ == Kind::Object; }
  // Member of an object, or nullptr.
  [[nodiscard]] const Json *get(const std::string &key) const;
  [[nodiscard]] size_t size() const { return kind_ == Kind::Array ? array_.size() : object_.size(); }
  [[nodiscard]] int64_t integer() const { return integer_; }
  [[nodiscard]] double number() const { return number_; }
  [[nodiscard]] const std::string &string() const { return string_; }
  [[nodiscard]] bool boolean() const { return boolean_; }
  [[nodiscard]] const std::map<std::string, Json> &object() const { return object_; }
  [[nodiscard]] const std::vector<Json> &array() const { return array_; }

private:
  friend class JsonParser;
  Kind kind_ = Kind::Null;
  bool boolean_ = false;
  double number_ = 0;
  int64_t integer_ = 0;
  std::string string_;
  std::vector<Json> array_;
  std::map<std::string, Json> object_;
};

} // namespace h3::model
