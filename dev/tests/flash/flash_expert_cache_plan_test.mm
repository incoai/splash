#include "flash/FlashExpertCachePlan.hpp"

#import <Foundation/Foundation.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
std::string encoded(id value) {
  NSError *error = nil;
  NSData *data = [NSJSONSerialization dataWithJSONObject:value options:0 error:&error];
  require(data && !error, "test fixture JSON serialization failed");
  return {static_cast<const char *>(data.bytes), data.length};
}
NSMutableDictionary *decoded(const std::string &json) {
  NSData *data = [NSData dataWithBytes:json.data() length:json.size()];
  NSError *error = nil;
  id value = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:&error];
  require([value isKindOfClass:[NSMutableDictionary class]] && !error, "test fixture parse failed");
  return value;
}
std::string firstID(std::string json, std::string_view suffix) {
  size_t at = json.find("\"selected_experts\"");
  require(at != std::string::npos, "selected IDs field missing");
  at = json.find('[', at); at = json.find('[', at + 1) + 1;
  while (json[at] == ' ' || json[at] == '\n') ++at;
  size_t end = at;
  while (json[end] >= '0' && json[end] <= '9') ++end;
  require(end > at, "first expert ID missing");
  json.insert(end, suffix);
  return json;
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: flash_expert_cache_plan_test VALID_PLAN_JSON");
      std::ifstream input(argv[1]);
      const std::string baseline((std::istreambuf_iterator<char>(input)), {});
      const std::string source = [decoded(baseline)[@"source_identity"] UTF8String];
      const auto plan = splash::flash::loadFlashExpertCachePlan(argv[1], source);
      require(plan.planSha256.size() == 64 && plan.sourceIdentity == source, "valid plan identities missing");
      require(plan.observedAssignments > 0 && plan.selectedAssignments <= plan.observedAssignments, "valid plan counts wrong");
      for (const auto &layer : plan.selectedExperts) require(!layer.empty() && layer.size() <= plan.requestedLimit, "valid plan list wrong");
      NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString];
      [[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:nil];
      const std::filesystem::path path = std::string(directory.UTF8String) + "/plan.json";
      size_t rejected = 0;
      auto rejects = [&](const std::string &json) {
        { std::ofstream output(path); output << json; }
        try { (void)splash::flash::loadFlashExpertCachePlan(path, source); }
        catch (const std::invalid_argument &) { ++rejected; return; }
        throw std::runtime_error("malformed plan was accepted");
      };
      rejects(firstID(baseline, ".0"));
      rejects(firstID(baseline, "e0"));
      for (int kind = 0; kind < 13; ++kind) {
        NSMutableDictionary *root = decoded(baseline);
        NSMutableArray *selected = root[@"selected_experts"];
        NSMutableDictionary *scope = root[@"scope"], *memory = root[@"memory_estimate"], *aggregate = root[@"aggregate"];
        switch (kind) {
          case 0: selected[0][0] = @YES; break;
          case 1: selected[0][0] = @512; break;
          case 2: selected[0][1] = selected[0][0]; break;
          case 3: selected[0] = [NSMutableArray array]; break;
          case 4: [selected removeLastObject]; break;
          case 5: root[@"requested_limit"] = @YES; break;
          case 6: root[@"requested_limit"] = @16; break;
          case 7: root[@"quantization_policy"] = @"incorrect-policy"; break;
          case 8: root[@"source_identity"] = @"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"; break;
          case 9: scope[@"included_rows"] = @YES; break;
          case 10: aggregate[@"hit_rate"] = @YES; break;
          case 11: memory[@"native_geometry_and_allocation_validation_required"] = @1; break;
          case 12: memory[@"coefficients_bytes"] = @0; break;
        }
        rejects(encoded(root));
      }
      std::filesystem::remove_all(path.parent_path());
      std::cout << "PASS CPU expert-plan loader: valid plan + " << rejected << " malformed numeric/schema/count cases\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
  }
}
