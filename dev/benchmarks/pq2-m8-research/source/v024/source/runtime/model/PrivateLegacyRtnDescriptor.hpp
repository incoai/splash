#pragma once
// PRIVATE R194 control only; not a stock-source or numerical-equivalence claim.
#include "ModelDescriptor.hpp"
#include "PrivateLegacyRtnPins.hpp"
#import <Foundation/Foundation.h>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <tuple>

namespace splash::model::private_r194 {
inline void require(bool ok, const char *why) {
  if (!ok) throw std::invalid_argument(std::string("PRIVATE R194: ") + why);
}
inline auto capabilityTuple(const ModelCapabilities &c) {
  return std::tie(c.vocabularySize,c.maximumContextTokens,c.maximumBatchWidth,
    c.prefillTokenBudget,c.draftQueryRows,c.draftProposalTokens,
    c.targetVerifyRows,c.draftContextTokens);
}
inline ModelDescriptor combine(const ModelDescriptor &p,const ModelDescriptor &s) {
  require(p.valid() && s.valid(),"invalid input descriptor");
  require(p.targetSource==TargetSource::Packed && p.draftSource==DraftSource::Packed &&
          p.visionSource==VisionSource::Packed && s.targetSource==TargetSource::Mlx &&
          s.draftSource==DraftSource::Checkpoint && s.visionSource==VisionSource::None,
          "source role mismatch");
  require(p.target==s.target && p.draft==s.draft &&
          capabilityTuple(p.capabilities)==capabilityTuple(s.capabilities) &&
          p.targetKvLayout==s.targetKvLayout && p.stateLayout==s.stateLayout,
          "target/draft/capability/KV/state/token interface mismatch");
  auto result=s;
  result.name="1.1.0 inference + legacy packed MLX RTN4 + new draft / private descriptor adapter";
  result.targetSource=TargetSource::Packed;
  require(result.valid(),"combined descriptor invalid");
  return result;
}
inline NSDictionary *object(const std::filesystem::path &p) {
  NSData *d=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:p.c_str()]];
  require(d!=nil,"missing private metadata");
  id o=[NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
  require([o isKindOfClass:[NSDictionary class]],"malformed private metadata");
  return o;
}
inline void validateRevisions(NSDictionary *p,NSDictionary *s) {
  require([p[@"upstream"][@"target"][@"revision"] isEqual:@"3e6447f082e89cc7f0bc6e5441afd38dfce760ff"] &&
    [p[@"upstream"][@"target"][@"repo_id"] isEqual:@"mlx-community/Qwen3.8-27B-4bit"] &&
    [p[@"upstream"][@"tokenizer"][@"revision"] isEqual:@"3e6447f082e89cc7f0bc6e5441afd38dfce760ff"] &&
    [s[@"sources"][@"target"][@"revision"] isEqual:@"3e6447f082e89cc7f0bc6e5441afd38dfce760ff"] &&
    [s[@"sources"][@"target"][@"repo"] isEqual:@"mlx-community/Qwen3.8-27B-4bit"] &&
    [s[@"sources"][@"draft"][@"revision"] isEqual:@"015e795645c74b1a0eeef3b570031fb62e769bc5"] &&
    [s[@"sources"][@"draft"][@"repo"] isEqual:@"incoai/Qwen3.8-27B-DFlash2"] &&
    [s[@"target_format"] isEqual:@"mlx-affine"] && [s[@"vision_format"] isEqual:@"none"],
    "revision/source identity mismatch");
}
inline void validatePins() {
  for(const auto &p:pins) {
    struct stat st{};
    require(::stat(p.path,&st)==0,"missing pinned input");
    require(std::filesystem::canonical(p.path)==p.resolved &&
      uint64_t(st.st_dev)==p.dev && uint64_t(st.st_ino)==p.ino && uint64_t(st.st_size)==p.size &&
      int64_t(st.st_mtimespec.tv_sec)*1000000000+st.st_mtimespec.tv_nsec==p.mt &&
      int64_t(st.st_ctimespec.tv_sec)*1000000000+st.st_ctimespec.tv_nsec==p.ct,
      "substituted or changed input");
    if(*p.hex) {
      std::ifstream f(p.path,std::ios::binary);
      std::string b((std::istreambuf_iterator<char>(f)),{}),h;
      constexpr char digits[]="0123456789abcdef";
      for(unsigned char c:b){h+=digits[c>>4];h+=digits[c&15];}
      require(h==p.hex,"small metadata content drift");
    }
  }
}
inline void validateRoot(const std::filesystem::path &root) {
  namespace fs=std::filesystem;
  require(fs::canonical(root)==assembly && !fs::is_symlink(root),"private canonical root mismatch");
  require(!fs::exists(root/"model.json") && !fs::exists(root/"manifest.json") &&
    !fs::exists(fs::path(packed)/"model.json"),"packed manifest shadowing");
  for(const char *role:{"target","draft"}) {
    auto dir=root/role;
    require(fs::is_directory(dir) && !fs::is_symlink(dir) && fs::canonical(dir).parent_path()==root,
      "real canonical sibling directories required");
    size_t n=0;
    for(const auto &e:fs::directory_iterator(dir)) {
      require(e.is_symlink() && fs::is_regular_file(e.path()),"file symlinks only"); ++n;
    }
    require(n==(std::string_view(role)=="target"?66:2),"unexpected assembly entries");
  }
}
inline ModelDescriptor inspect(const std::filesystem::path &root,const char *optIn) {
  require(optIn && std::string_view(optIn)=="R194-legacy-rtn4-new015e-private-v1","missing/malformed private opt-in");
  validateRoot(root);validatePins();
  auto p=inspectModelPackage(packed);
  auto s=inspectModelPackage(metadata);
  validateRevisions(object(std::filesystem::path(packed)/"manifest.json"),object(std::filesystem::path(metadata)/"model.json"));
  NSDictionary *pc=object(std::filesystem::path(packed)/"tokenizer/config.json");
  NSDictionary *sc=object(std::filesystem::path(metadata)/"config.json");
  for(NSString *key in @[@"bos_token_id",@"eos_token_id"])
    require([pc[@"text_config"][key] isEqual:sc[@"text_config"][key]],"token-ID mismatch");
  require([sc[@"text_config"][@"bos_token_id"] isEqual:@248044] &&
          [sc[@"text_config"][@"eos_token_id"] isEqual:@248044],"fixed token IDs changed");
  auto result=combine(p,s);validatePins();return result;
}
inline ModelDescriptor inspectSelected(const std::filesystem::path &root) {
  const char *opt=std::getenv("SPLASH_PRIVATE_R194_DESCRIPTOR");
  return opt?inspect(root,opt):inspectModelPackage(root);
}
} // namespace splash::model::private_r194
