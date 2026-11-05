#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::json {

// An object member's name. A name spelled in the source is written as it is:
// its constructor runs at compile time and refuses anything but a string
// literal free of quotes, backslashes and control characters, so a character
// array filled at run time does not compile as one. A name read at run time
// is escaped.
class Key final {
public:
  template <size_t N> consteval Key(const char (&name)[N]) : text_(name, N - 1), plain_(true) {
    if (name[N - 1] != '\0')
      throw "a JSON key is not a string literal";
    for (const char character : text_)
      if (character == '"' || character == '\\' || static_cast<unsigned char>(character) < 0x20)
        throw "a literal JSON key needs escaping";
  }
  Key(std::string_view name) noexcept : text_(name) {}
  Key(const std::string &name) noexcept : text_(name) {}

  [[nodiscard]] std::string_view text() const noexcept { return text_; }
  [[nodiscard]] bool plain() const noexcept { return plain_; }

private:
  std::string_view text_;
  bool plain_ = false;
};

// Writes one JSON document and places every comma and colon: an object's
// members are key() then a value, an array's are values. A double prints to
// 10 significant digits; exact() prints one to the digits that read back as
// it; one that is not finite, which JSON cannot spell, prints as null. str()
// throws std::logic_error for a document that is empty, unclosed or holds a
// key without a value, as a value without a key in an object or a key outside
// one throws at once.
class Writer final {
public:
  Writer() { out_.precision(10); }

  Writer &key(Key name) {
    if (!beforeKey())
      put(',');
    if (name.plain()) {
      put('"');
      put(name.text());
      put('"');
    } else {
      string(name.text());
    }
    put(':');
    return *this;
  }

  template <std::same_as<bool> T> Writer &value(T flag) {
    beforeValue();
    put(flag ? std::string_view("true") : std::string_view("false"));
    return *this;
  }
  // Integers print as numbers, a uint8_t too; char is text, not a number.
  template <std::integral T>
    requires(!std::same_as<T, bool> && !std::same_as<T, char>)
  Writer &value(T number) {
    beforeValue();
    out_ << +number;
    return *this;
  }
  template <std::floating_point T> Writer &value(T number) {
    beforeValue();
    floating(number);
    return *this;
  }
  Writer &value(const char *text) { return value(std::string_view(text)); }
  Writer &value(const std::string &text) { return value(std::string_view(text)); }
  Writer &value(std::string_view text) {
    beforeValue();
    string(text);
    return *this;
  }
  Writer &null() {
    beforeValue();
    put("null");
    return *this;
  }
  Writer &exact(double number) {
    beforeValue();
    const std::streamsize precision = out_.precision(std::numeric_limits<double>::max_digits10);
    floating(number);
    out_.precision(precision);
    return *this;
  }

  template <class T> Writer &field(Key name, T &&member) {
    key(name);
    return value(std::forward<T>(member));
  }

  template <class Body> Writer &object(Body &&body) {
    open(Frame::Object, '{');
    body();
    return close(Frame::Object, '}');
  }
  template <class Body> Writer &object(Key name, Body &&body) {
    key(name);
    return object(std::forward<Body>(body));
  }
  template <class Body> Writer &array(Body &&body) {
    open(Frame::Array, '[');
    body();
    return close(Frame::Array, ']');
  }
  template <class Body> Writer &array(Key name, Body &&body) {
    key(name);
    return array(std::forward<Body>(body));
  }

  [[nodiscard]] std::string str() const {
    if (!complete_ || !frames_.empty())
      throw std::logic_error("an unfinished JSON document");
    return out_.str();
  }

private:
  enum class Frame : unsigned char { Object, Array };
  struct Open {
    Frame frame;
    bool empty = true;
    bool keyed = false;
  };

  // Whether the key is its object's first member, which takes no comma.
  bool beforeKey() {
    if (frames_.empty() || frames_.back().frame != Frame::Object || frames_.back().keyed)
      throw std::logic_error("a JSON key outside an object or after another key");
    Open &top = frames_.back();
    const bool first = top.empty;
    top.empty = false;
    top.keyed = true;
    return first;
  }
  void beforeValue() {
    if (frames_.empty()) {
      if (complete_)
        throw std::logic_error("a second JSON document");
      complete_ = true;
      return;
    }
    Open &top = frames_.back();
    if (top.frame == Frame::Object) {
      if (!top.keyed)
        throw std::logic_error("a JSON object member without a key");
      top.keyed = false;
      return;
    }
    if (!top.empty)
      put(',');
    top.empty = false;
  }
  void open(Frame frame, char bracket) {
    beforeValue();
    frames_.push_back({frame});
    put(bracket);
  }
  Writer &close(Frame frame, char bracket) {
    if (frames_.empty() || frames_.back().frame != frame || frames_.back().keyed)
      throw std::logic_error("a JSON container closed with a key left without a value");
    frames_.pop_back();
    put(bracket);
    return *this;
  }
  // The one escaping rule: runs of plain characters as they are, the rest
  // as JSON escapes.
  void string(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    put('"');
    size_t run = 0;
    for (size_t index = 0; index < text.size(); ++index) {
      const auto character = static_cast<unsigned char>(text[index]);
      const char control[] = {'\\', 'u', '0', '0', hex[character >> 4], hex[character & 0x0f]};
      std::string_view escaped;
      switch (character) {
      case '"': escaped = "\\\""; break;
      case '\\': escaped = "\\\\"; break;
      case '\b': escaped = "\\b"; break;
      case '\f': escaped = "\\f"; break;
      case '\n': escaped = "\\n"; break;
      case '\r': escaped = "\\r"; break;
      case '\t': escaped = "\\t"; break;
      default:
        if (character >= 0x20)
          continue;
        escaped = {control, sizeof(control)};
      }
      put(text.substr(run, index - run));
      put(escaped);
      run = index + 1;
    }
    put(text.substr(run));
    put('"');
  }
  // The one rule for a floating value: a finite one at the stream's
  // precision, and null for an infinity or NaN, which JSON has no number for.
  template <std::floating_point T> void floating(T number) {
    if (std::isfinite(number))
      out_ << number;
    else
      put("null");
  }
  // Punctuation, keys and strings go straight to the stream's buffer;
  // numbers go through the stream for its precision.
  void put(char character) { out_.rdbuf()->sputc(character); }
  void put(std::string_view text) { out_.rdbuf()->sputn(text.data(), static_cast<std::streamsize>(text.size())); }

  std::ostringstream out_;
  std::vector<Open> frames_;
  bool complete_ = false;
};

} // namespace splash::json
