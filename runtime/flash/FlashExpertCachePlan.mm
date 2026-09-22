#include "flash/FlashExpertCachePlan.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <sys/stat.h>

namespace splash::flash {
namespace {
constexpr uint64_t kExpertBytes = 9830400;
constexpr uint64_t kMapBytes = 48 * 16384;
constexpr std::string_view kPolicy = "F32 q*SF+bias once-roundedBF16";

[[noreturn]] void fail(std::string_view message) {
  throw std::invalid_argument("Flash expert plan: " + std::string(message));
}
std::string text(id value) {
  if (![value isKindOfClass:[NSString class]]) fail("expected string metadata");
  NSString *string = value;
  const char *bytes = string.UTF8String;
  if (!bytes) fail("invalid UTF-8 string metadata");
  std::string result(bytes, [string lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
  if (result.find('\0') != std::string::npos) fail("embedded NUL in metadata");
  return result;
}
NSDictionary *object(id value) {
  if (![value isKindOfClass:[NSDictionary class]]) fail("expected object metadata");
  return value;
}
NSArray *array(id value) {
  if (![value isKindOfClass:[NSArray class]]) fail("expected array metadata");
  return value;
}
void keys(NSDictionary *value, std::initializer_list<const char *> expected) {
  if (value.count != expected.size()) fail("unexpected or missing metadata fields");
  for (const char *key : expected)
    if (!value[[NSString stringWithUTF8String:key]]) fail("missing metadata field");
}
uint64_t integer(id value) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() ||
      CFNumberIsFloatType((__bridge CFNumberRef)value))
    fail("expected integer metadata; booleans/floats are invalid");
  NSNumber *number = value;
  const char type = number.objCType[0];
  if (std::string_view("cCsSiIlLqQ").find(type) == std::string_view::npos ||
      (std::string_view("csilq").find(type) != std::string_view::npos && number.longLongValue < 0))
    fail("integer metadata is negative or fractional");
  return number.unsignedLongLongValue;
}
double real(id value) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) fail("expected numeric rate");
  const double result = [static_cast<NSNumber *>(value) doubleValue];
  if (!std::isfinite(result)) fail("non-finite rate");
  return result;
}
void equal(uint64_t actual, uint64_t expected) {
  if (actual != expected) fail("metadata count or geometry mismatch");
}
uint64_t add(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) fail("count overflow");
  return left + right;
}
uint64_t multiply(uint64_t left, uint64_t right) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left) fail("count overflow");
  return left * right;
}
void ratio(id value, uint64_t selected, uint64_t observed) {
  if (!observed || selected > observed ||
      std::abs(real(value) - double(selected) / double(observed)) > 1e-12)
    fail("frequency hit rate mismatch");
}
void digest(std::string_view value) {
  if (value.size() != 64 || value.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    fail("invalid lowercase SHA256 metadata");
}
std::string sha256(NSData *data) {
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> output{};
  CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), output.data());
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : output) { result += alphabet[byte >> 4]; result += alphabet[byte & 15]; }
  return result;
}

// Foundation may canonicalize whole-valued floating NSNumber values. Inspect
// the raw number tokens too: only the schema's hit_rate/coefficients_gib scalar
// fields permit decimal/exponent syntax; IDs and counts must be integer tokens.
void integerTokens(std::span<const uint8_t> bytes) {
  const std::string_view json(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  std::string_view key;
  for (size_t cursor = 0; cursor < json.size();) {
    if (json[cursor] == '"') {
      const size_t began = ++cursor;
      while (cursor < json.size() && json[cursor] != '"') {
        if (json[cursor] == '\\') ++cursor;
        ++cursor;
      }
      if (cursor == json.size()) fail("unterminated metadata string");
      const auto string = json.substr(began, cursor - began);
      size_t next = ++cursor;
      while (next < json.size() && std::string_view(" \r\n\t").find(json[next]) != std::string_view::npos) ++next;
      if (next < json.size() && json[next] == ':') key = string;
    } else if (json[cursor] == '-' || (json[cursor] >= '0' && json[cursor] <= '9')) {
      const size_t began = cursor++;
      while (cursor < json.size() && std::string_view("0123456789.eE+-").find(json[cursor]) != std::string_view::npos) ++cursor;
      if (json.substr(began, cursor - began).find_first_of(".eE") != std::string_view::npos &&
          key != "hit_rate" && key != "coefficients_gib") fail("floating token in integer metadata");
    } else ++cursor;
  }
}
} // namespace

FlashExpertCachePlan loadFlashExpertCachePlan(const std::filesystem::path &path,
                                             std::string_view expectedSourceIdentity) {
  @autoreleasepool {
    digest(expectedSourceIdentity);
    struct stat status{};
    if (::stat(path.c_str(), &status) || !S_ISREG(status.st_mode) || status.st_size <= 0 || status.st_size > (4 << 20))
      fail("plan is not a bounded regular JSON file");
    NSString *nativePath = [[NSString alloc] initWithBytes:path.c_str() length:path.string().size() encoding:NSUTF8StringEncoding];
    if (!nativePath) fail("invalid UTF-8 plan path");
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:nativePath options:0 error:&error];
    if (!data || data.length != uint64_t(status.st_size) || data.length > (4 << 20)) fail("could not read bounded plan");
    integerTokens({static_cast<const uint8_t *>(data.bytes), data.length});
    NSDictionary *root = object([NSJSONSerialization JSONObjectWithData:data options:0 error:&error]);
    if (error) fail("invalid plan JSON");
    keys(root, {"schema", "source_identity", "requested_limit", "quantization_policy", "selected_experts",
                "captures", "scope", "layer_stats", "aggregate", "memory_estimate"});
    if (text(root[@"schema"]) != "splash-flash-hot-expert-plan-v1" || text(root[@"quantization_policy"]) != kPolicy)
      fail("unsupported schema or BF16 operand policy");
    FlashExpertCachePlan result;
    result.sourceIdentity = text(root[@"source_identity"]);
    if (result.sourceIdentity != expectedSourceIdentity) fail("source identity mismatch");
    const uint64_t limit = integer(root[@"requested_limit"]);
    if (limit != 32 && limit != 64 && limit != 128 && limit != 256) fail("unsupported requested limit");
    result.requestedLimit = static_cast<uint32_t>(limit);
    NSArray *selected = array(root[@"selected_experts"]), *stats = array(root[@"layer_stats"]);
    equal(selected.count, 48); equal(stats.count, 48);
    uint64_t totalSelected = 0;
    for (uint32_t layer = 0; layer < 48; ++layer) {
      NSArray *ids = array(selected[layer]);
      if (!ids.count || ids.count > limit) fail("selected layer count exceeds limit or is empty");
      auto &target = result.selectedExperts[layer];
      for (id value in ids) {
        const uint64_t expert = integer(value);
        if (expert >= 512 || (!target.empty() && expert <= target.back())) fail("selected IDs must be sorted unique integers in [0,511]");
        target.push_back(static_cast<uint32_t>(expert));
      }
      NSDictionary *stat = object(stats[layer]);
      keys(stat, {"layer", "observed_assignments", "selected_assignments", "hit_rate", "observed_experts", "selected_count"});
      equal(integer(stat[@"layer"]), layer); equal(integer(stat[@"selected_count"]), ids.count);
      const uint64_t observed = integer(stat[@"observed_assignments"]), hits = integer(stat[@"selected_assignments"]);
      const uint64_t distinct = integer(stat[@"observed_experts"]);
      if (distinct < ids.count || distinct > 512 || distinct > observed || hits < ids.count) fail("invalid layer frequency counts");
      ratio(stat[@"hit_rate"], hits, observed);
      result.observedAssignments = add(result.observedAssignments, observed);
      result.selectedAssignments = add(result.selectedAssignments, hits);
      totalSelected = add(totalSelected, ids.count);
    }
    NSDictionary *scope = object(root[@"scope"]);
    keys(scope, {"phase_filter", "validated_records", "included_records", "excluded_records", "included_rows",
                 "excluded_rows", "rows_by_phase", "count_unit", "selection_policy", "capture_validation"});
    result.phaseFilter = text(scope[@"phase_filter"]);
    if (result.phaseFilter != "all" && result.phaseFilter != "prefill" && result.phaseFilter != "decode") fail("invalid phase scope");
    if (text(scope[@"count_unit"]) != "routed expert assignment: 10 selections per row per target layer" ||
        text(scope[@"selection_policy"]) != "positive counts only; count descending then ID ascending; selected maps sorted by ID" ||
        text(scope[@"capture_validation"]) != "all records validated before phase exclusion; source files unchanged during reading") fail("unsupported frequency scope");
    const uint64_t validated = integer(scope[@"validated_records"]), included = integer(scope[@"included_records"]), excluded = integer(scope[@"excluded_records"]);
    const uint64_t includedRows = integer(scope[@"included_rows"]), excludedRows = integer(scope[@"excluded_rows"]);
    equal(validated, add(included, excluded));
    if (!included || includedRows < included || includedRows > multiply(included, 2048) ||
        excludedRows < excluded || excludedRows > multiply(excluded, 2048)) fail("invalid record/row scope");
    NSDictionary *phases = object(scope[@"rows_by_phase"]); keys(phases, {"prefill", "decode"});
    const uint64_t prefill = integer(phases[@"prefill"]), decode = integer(phases[@"decode"]);
    equal(add(prefill, decode), includedRows);
    if ((result.phaseFilter == "all" && excluded) || (result.phaseFilter == "decode" && prefill) || (result.phaseFilter == "prefill" && decode)) fail("phase exclusion mismatch");
    for (id stat in stats) equal(integer(object(stat)[@"observed_assignments"]), multiply(includedRows, 10));
    NSArray *captures = array(root[@"captures"]);
    if (!captures.count) fail("no source capture provenance");
    uint64_t sourceRecords = 0, sourceIncluded = 0, sourceExcluded = 0;
    for (id entry in captures) {
      NSDictionary *capture = object(entry);
      keys(capture, {"path", "bytes", "records", "included_records", "excluded_records", "sha256"});
      if (text(capture[@"path"]).empty()) fail("empty capture path");
      digest(text(capture[@"sha256"]));
      const uint64_t records = integer(capture[@"records"]), in = integer(capture[@"included_records"]), out = integer(capture[@"excluded_records"]);
      const uint64_t bytes = integer(capture[@"bytes"]);
      equal(records, add(in, out));
      if (bytes < records || (!records && bytes)) fail("invalid capture byte scope");
      sourceRecords = add(sourceRecords, records); sourceIncluded = add(sourceIncluded, in); sourceExcluded = add(sourceExcluded, out);
    }
    equal(sourceRecords, validated); equal(sourceIncluded, included); equal(sourceExcluded, excluded);
    NSDictionary *aggregate = object(root[@"aggregate"]);
    keys(aggregate, {"observed_assignments", "selected_assignments", "hit_rate", "selected_count"});
    equal(integer(aggregate[@"observed_assignments"]), result.observedAssignments);
    equal(integer(aggregate[@"selected_assignments"]), result.selectedAssignments);
    equal(integer(aggregate[@"selected_count"]), totalSelected);
    ratio(aggregate[@"hit_rate"], result.selectedAssignments, result.observedAssignments);
    result.observedHitRate = real(aggregate[@"hit_rate"]);
    NSDictionary *memory = object(root[@"memory_estimate"]);
    keys(memory, {"scope", "assumed_geometry", "coefficients_bytes_per_expert", "coefficients_bytes_per_layer", "coefficients_bytes", "coefficients_gib",
                  "requested_limit_coefficients_bytes_per_layer", "requested_limit_coefficients_bytes", "id_map_allocated_bytes", "diagnostics_allocated_bytes",
                  "total_estimated_allocated_bytes", "allocation_assumption", "native_geometry_and_allocation_validation_required"});
    if (text(memory[@"scope"]) != "estimated additional cache allocations; original weights and runtime workspace excluded" ||
        text(memory[@"allocation_assumption"]) != "separate per-layer int32[512] ID maps and uint32 diagnostics, each rounded to 16KiB" ||
        CFGetTypeID((__bridge CFTypeRef)memory[@"native_geometry_and_allocation_validation_required"]) != CFBooleanGetTypeID() ||
        ![memory[@"native_geometry_and_allocation_validation_required"] boolValue]) fail("invalid allocation estimate scope");
    NSDictionary *geometry = object(memory[@"assumed_geometry"]);
    keys(geometry, {"layers", "experts_per_layer", "top_k", "hidden_size", "intermediate_size", "projections_per_expert", "coefficient_dtype", "allocation_alignment_bytes"});
    equal(integer(geometry[@"layers"]), 48); equal(integer(geometry[@"experts_per_layer"]), 512);
    equal(integer(geometry[@"top_k"]), 10); equal(integer(geometry[@"hidden_size"]), 2560);
    equal(integer(geometry[@"intermediate_size"]), 640); equal(integer(geometry[@"projections_per_expert"]), 3);
    equal(integer(geometry[@"allocation_alignment_bytes"]), 16384);
    if (text(geometry[@"coefficient_dtype"]) != "BF16") fail("invalid coefficient dtype estimate");
    equal(integer(memory[@"coefficients_bytes_per_expert"]), kExpertBytes);
    NSArray *layerBytes = array(memory[@"coefficients_bytes_per_layer"]); equal(layerBytes.count, 48);
    for (uint32_t layer = 0; layer < 48; ++layer) equal(integer(layerBytes[layer]), multiply(result.selectedExperts[layer].size(), kExpertBytes));
    result.estimatedCoefficientBytes = multiply(totalSelected, kExpertBytes);
    equal(integer(memory[@"coefficients_bytes"]), result.estimatedCoefficientBytes);
    if (std::abs(real(memory[@"coefficients_gib"]) - double(result.estimatedCoefficientBytes) / double(1ULL << 30)) > 1e-12) fail("coefficient GiB estimate mismatch");
    equal(integer(memory[@"requested_limit_coefficients_bytes_per_layer"]), multiply(limit, kExpertBytes));
    equal(integer(memory[@"requested_limit_coefficients_bytes"]), multiply(48, multiply(limit, kExpertBytes)));
    equal(integer(memory[@"id_map_allocated_bytes"]), kMapBytes); equal(integer(memory[@"diagnostics_allocated_bytes"]), kMapBytes);
    equal(integer(memory[@"total_estimated_allocated_bytes"]), add(result.estimatedCoefficientBytes, 2 * kMapBytes));
    result.planSha256 = sha256(data);
    return result;
  }
}
} // namespace splash::flash
