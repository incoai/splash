#include "video/h3/model/Json.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace h3::model {

class JsonParser {
public:
  explicit JsonParser(const std::string &text) : s_(text) {}

  Status document(Json &out) {
    H3_TRY(value(out, 0));
    skip();
    if (i_ != s_.size()) return fail("trailing characters");
    return {};
  }

private:
  Status fail(const char *what) const { return error(Code::Io, "JSON: %s at offset %zu", what, i_); }

  void skip() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }

  bool literal(const char *word) {
    const size_t n = std::strlen(word);
    if (s_.compare(i_, n, word) != 0) return false;
    i_ += n;
    return true;
  }

  Status value(Json &out, int depth) {
    if (depth > 64) return fail("nesting too deep");
    skip();
    if (i_ >= s_.size()) return fail("unexpected end");
    const char c = s_[i_];
    if (c == '{') return object(out, depth);
    if (c == '[') return array(out, depth);
    if (c == '"') {
      out.kind_ = Json::Kind::String;
      return string(out.string_);
    }
    if (literal("true")) {
      out.kind_ = Json::Kind::Bool;
      out.boolean_ = true;
      return {};
    }
    if (literal("false")) {
      out.kind_ = Json::Kind::Bool;
      return {};
    }
    if (literal("null")) return {};
    return number(out);
  }

  Status object(Json &out, int depth) {
    out.kind_ = Json::Kind::Object;
    ++i_;
    skip();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return {};
    }
    for (;;) {
      skip();
      if (i_ >= s_.size() || s_[i_] != '"') return fail("expected a key");
      std::string key;
      H3_TRY(string(key));
      skip();
      if (i_ >= s_.size() || s_[i_] != ':') return fail("expected ':'");
      ++i_;
      H3_TRY(value(out.object_[key], depth + 1));
      skip();
      if (i_ < s_.size() && s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (i_ < s_.size() && s_[i_] == '}') {
        ++i_;
        return {};
      }
      return fail("expected ',' or '}'");
    }
  }

  Status array(Json &out, int depth) {
    out.kind_ = Json::Kind::Array;
    ++i_;
    skip();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      return {};
    }
    for (;;) {
      out.array_.emplace_back();
      H3_TRY(value(out.array_.back(), depth + 1));
      skip();
      if (i_ < s_.size() && s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (i_ < s_.size() && s_[i_] == ']') {
        ++i_;
        return {};
      }
      return fail("expected ',' or ']'");
    }
  }

  Status string(std::string &out) {
    ++i_;  // opening quote
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (i_ >= s_.size()) return fail("bad escape");
      const char e = s_[i_++];
      switch (e) {
      case 'n': out.push_back('\n'); break;
      case 't': out.push_back('\t'); break;
      case 'r': out.push_back('\r'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'u': {
        if (i_ + 4 > s_.size()) return fail("bad \\u escape");
        const unsigned code = static_cast<unsigned>(std::strtoul(s_.substr(i_, 4).c_str(), nullptr, 16));
        i_ += 4;
        if (code < 0x80) {
          out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
          out.push_back(static_cast<char>(0xC0 | (code >> 6)));
          out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
          out.push_back(static_cast<char>(0xE0 | (code >> 12)));
          out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
          out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
        break;
      }
      default: out.push_back(e);
      }
    }
    if (i_ >= s_.size()) return fail("unterminated string");
    ++i_;
    return {};
  }

  Status number(Json &out) {
    const char *begin = s_.c_str() + i_;
    char *end = nullptr;
    const double v = std::strtod(begin, &end);
    if (end == begin) return fail("unexpected character");
    out.kind_ = Json::Kind::Number;
    out.number_ = v;
    const std::string token(begin, static_cast<size_t>(end - begin));
    if (token.find_first_of(".eE") == std::string::npos) out.integer_ = std::strtoll(token.c_str(), nullptr, 10);
    else out.integer_ = static_cast<int64_t>(v);
    i_ += static_cast<size_t>(end - begin);
    return {};
  }

  const std::string &s_;
  size_t i_ = 0;
};

Status Json::parse(const std::string &text, Json &out) {
  out = Json();
  return JsonParser(text).document(out);
}

Status Json::parseFile(const std::string &path, Json &out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return error(Code::Io, "cannot open %s", path.c_str());
  std::stringstream text;
  text << file.rdbuf();
  Status s = parse(text.str(), out);
  if (!s.ok()) return error(Code::Io, "%s: %s", path.c_str(), s.message().c_str());
  return {};
}

const Json *Json::get(const std::string &key) const {
  auto it = object_.find(key);
  return it == object_.end() ? nullptr : &it->second;
}

} // namespace h3::model
