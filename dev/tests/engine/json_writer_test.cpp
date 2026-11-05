// json::Writer places every comma and colon of a document, in nested and
// empty containers; escapes strings and non-literal keys; prints a string,
// literal or not, as a string, small integers as numbers, a double to 10
// digits and exact() to the digits that read back, and a floating value that
// is not finite as null; and refuses a value
// without a key in an object, a key in an array, and an unfinished document.
#include "TestChecks.hpp"
#include "engine/Json.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

using namespace splash;
using splash::test::rejects;
using splash::test::require;

namespace {

void testCommas() {
  json::Writer writer;
  writer.object([&] {
    writer.field("a", 1).object("empty", [] {}).array("none", [] {});
    writer.array("list", [&] {
      writer.value(1).value(2).object([&] { writer.field("b", true).field("c", false); }).array([] {});
    });
    writer.field("d", 0.5);
  });
  require(writer.str() == R"({"a":1,"empty":{},"none":[],"list":[1,2,{"b":true,"c":false},[]],"d":0.5})",
          "commas or colons misplaced: " + writer.str());
}

void testStrings() {
  const char *pointer = "text";
  const std::string owned = "owned";
  json::Writer writer;
  writer.object([&] {
    writer.field("literal", "token_major").field("pointer", pointer).field("owned", owned);
    writer.field("escaped", std::string_view("q\"b\\n\n\t\x01"));
    writer.key(std::string("dynamic\"key")).value(1);
  });
  require(writer.str() == R"({"literal":"token_major","pointer":"text","owned":"owned",)"
                          R"("escaped":"q\"b\\n\n\t\u0001","dynamic\"key":1})",
          "a string printed as something else or was not escaped: " + writer.str());
}

void testNumbers() {
  json::Writer writer;
  writer.array([&] {
    writer.value(uint8_t{200}).value(int8_t{-5}).value(uint64_t{18446744073709551615ULL});
    writer.value(1.0 / 3.0).exact(3.814697265625).exact(1.0 / 3.0).value(2.0 / 3.0);
    writer.null();
  });
  require(writer.str() == "[200,-5,18446744073709551615,0.3333333333,3.814697265625,0.33333333333333331,"
                          "0.6666666667,null]",
          "a number printed wrongly or exact() kept its precision: " + writer.str());
}

// JSON has no infinity or NaN: each is null, from value() or exact(), as a
// float or a double, and the document around it stays JSON.
void testNonFiniteNumbers() {
  constexpr double infinity = std::numeric_limits<double>::infinity();
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  json::Writer writer;
  writer.object([&] {
    writer.field("inf", infinity).field("minus_inf", -infinity).field("nan", nan).field("minus_nan", -nan);
    writer.field("float_inf", std::numeric_limits<float>::infinity());
    writer.field("float_nan", std::numeric_limits<float>::quiet_NaN());
    writer.key("exact_inf").exact(infinity).key("exact_nan").exact(nan).key("after").exact(0.1);
    writer.field("next", 0.5);
  });
  require(writer.str() == R"({"inf":null,"minus_inf":null,"nan":null,"minus_nan":null,"float_inf":null,)"
                          R"("float_nan":null,"exact_inf":null,"exact_nan":null,"after":0.10000000000000001,)"
                          R"("next":0.5})",
          "a value that is not finite printed as something other than null: " + writer.str());
}

void testMisuse() {
  rejects(
      [] {
        json::Writer writer;
        writer.object([&] { writer.value(1); });
      },
      "without a key", "an object member without a key was written");
  rejects(
      [] {
        json::Writer writer;
        writer.array([&] { writer.key("a"); });
      },
      "outside an object", "a key inside an array was written");
  rejects(
      [] {
        json::Writer writer;
        writer.object([&] { writer.key("a").key("b"); });
      },
      "after another key", "a key after a key was written");
  rejects(
      [] {
        json::Writer writer;
        writer.object([&] { writer.key("a"); });
      },
      "left without a value", "an object closed with a key left without a value");
  rejects(
      [] {
        json::Writer writer;
        writer.object([&] { static_cast<void>(writer.field("a", 1).str()); });
      },
      "unfinished", "an unclosed document was taken");
  rejects([] { static_cast<void>(json::Writer().str()); }, "unfinished", "an empty document was taken");
  rejects(
      [] {
        json::Writer writer;
        writer.value(1).value(2);
      },
      "second", "two documents were written as one");
}

} // namespace

int main() {
  try {
    testCommas();
    testStrings();
    testNumbers();
    testNonFiniteNumbers();
    testMisuse();
    std::cout << "json writer tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "json writer tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
