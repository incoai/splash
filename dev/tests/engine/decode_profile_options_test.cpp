#include "../../tools/DecodeProfileOptions.hpp"
#include <iostream>
#include <unistd.h>
using splash::tools::DecodeProfileOptions;
int main() {
  unsigned checks = 0;
  auto parse = [](std::initializer_list<std::string_view> a) { return DecodeProfileOptions::parse(a); };
  auto check = [&](bool value) { ++checks; if (!value) throw std::runtime_error("profile option assertion failed"); };
  auto rejects = [&](std::initializer_list<std::string_view> a) {
    ++checks;
    try { (void)parse(a); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid profile arguments accepted");
  };
  const auto defaults = parse({});
  check(defaults.activations == "auto" && defaults.state == "fp32" && !defaults.configOnly);
  check(defaults.json().find("\"http_ttft_s\":null") != std::string::npos);
  check(defaults.json().find("\"fusion_enabled\":false") != std::string::npos);
  check(parse({"--activations", "bf16"}).activations == "bf16");
  check(parse({"--max-memory", "96G"}).maxMemory == (96ull << 30));
  check(parse({"--config-only"}).configOnly);
  check(parse({"--verify-rows", "all"}).verify == "all");
  rejects({"--activations", "int8"}); rejects({"--state-format", "auto"});
  rejects({"--kv-format", "fp32"}); rejects({"--verify-rows", "guess"});
  rejects({"--max-memory", "0"}); rejects({"--max-memory", "18446744073709551615G"});
  rejects({"--cycles", "0"}); rejects({"--cycles", "-1"}); rejects({"--cycles", "4294967296"});
  rejects({"--prompt-tokens", "4294967296"}); rejects({"--max-context", "0"});
  rejects({"--prompt-tokens", "262144"}); rejects({"--cycles", "100000"});
  rejects({"--activations"}); rejects({"--unknown", "yes"});
  rejects({"--config-only", "--config-only"}); rejects({"--activations", "auto", "--activations", "bf16"});
  rejects({"--images", "1x1", "--max-memory", "96G"}); rejects({"--require-flash-next"});
  rejects({"--input-tokens", "/no/such/token-fixture"});
  std::string temporary = (std::filesystem::temp_directory_path() / "splash-profile-options-XXXXXX").string();
  const int fd = mkstemp(temporary.data());
  if (fd < 0) throw std::runtime_error("temporary fixture creation failed");
  close(fd);
  const std::filesystem::path file = temporary;
  struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove(path); } } cleanup{file};
  const std::string path = file.string();
  auto write = [&](const char *s) { std::ofstream(file) << s; };
  write("0 1\n4294967295");
  const auto tokens = parse({"--input-tokens", path});
  check(tokens.tokens == std::vector<uint32_t>({0, 1, UINT32_MAX}) && tokens.promptTokens == 3);
  check(tokens.json().find(path) == std::string::npos);
  rejects({"--prompt-tokens", "3", "--input-tokens", path});
  check(parse({"--require-flash-next", "--input-tokens", path, "--activations", "bf16", "--state-format", "bf16", "--kv-format", "int8", "--max-memory", "96G", "--max-context", "262144"}).requireFlashNext);
  rejects({"--require-flash-next", "--input-tokens", path, "--activations", "auto", "--state-format", "bf16", "--max-memory", "96G"});
  rejects({"--report", ""}); rejects({"--dispatch-list", ""}); rejects({"--input-tokens", ""});
  rejects({"--report", path}); rejects({"--dispatch-list", path});
  rejects({"--report", "new-file", "--dispatch-list", "./new-file"});
  rejects({"--images", ""}); rejects({"--max-cache-disk", "512G"});
  check(tokens.tokenDigest().size() == 66);
  const auto digest = tokens.tokenDigest();
  write("0\n1 4294967295\n");
  check(parse({"--input-tokens", path}).tokenDigest() == digest);
  write(""); rejects({"--input-tokens", path});
  write("-1"); rejects({"--input-tokens", path});
  write("4294967296"); rejects({"--input-tokens", path});
  write("1x"); rejects({"--input-tokens", path});
  write("1 2 invalid"); rejects({"--input-tokens", path});
  check(splash::tools::profileJsonString("a\n\"b\\") == "\"a\\u000a\\\"b\\\\\"");
  check(splash::tools::profileNumber(0.000000005) != "0.000000");
  for (double invalid : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    ++checks;
    try { (void)splash::tools::profileNumber(invalid); throw std::runtime_error("invalid observation accepted"); }
    catch (const std::invalid_argument &) {}
  }
  const auto output = path + ".report";
  splash::tools::writePrivateProfileOutput(output, "{\"scope\":\"unit_fixture\"}\n");
  Cleanup cleanOutput{output};
  std::ifstream written(output); std::string record; std::getline(written, record);
  check(record == "{\"scope\":\"unit_fixture\"}");
  check((std::filesystem::status(output).permissions() & std::filesystem::perms::group_read) == std::filesystem::perms::none);
  ++checks;
  try { splash::tools::writePrivateProfileOutput(output, "overwrite"); throw std::runtime_error("output clobbered"); }
  catch (const std::runtime_error &e) { if (std::string(e.what()) == "output clobbered") throw; }
  std::ifstream unchanged(output); std::getline(unchanged, record); check(record == "{\"scope\":\"unit_fixture\"}");
  std::cout << checks << " profile option/input checks passed (no Metal or model).\n";
}
