#include "flash/FlashMTPQMVPolicy.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace {
uint64_t checks = 0;
void require(bool value, const char *reason) {
  ++checks;
  if (!value) throw std::runtime_error(reason);
}
template<class F> void rejects(F operation) {
  bool caught = false;
  try { operation(); } catch (const std::invalid_argument &) { caught = true; }
  require(caught, "invalid opt-in flag or dependency was accepted");
}
struct Geometry {
  uint32_t experts=1, outputs=2560, inputs=2560, bits=4, group=64;
  uint64_t weights=1280, parameters=80;
  bool eligible() const {
    return splash::flash::flashMTPQMVF32Geometry(
        experts,outputs,inputs,bits,group,weights,parameters);
  }
};
}

int main() {
  using namespace splash::flash;
  try {
    for (bool global : {false,true}) {
      require(!flashMTPQMVF32Switch(nullptr,global),"absent opt-in changed default");
      require(!flashMTPQMVF32Switch("0",global),"explicit opt-out changed default");
      for (const char *malformed : {"","true","false","yes","on","01","2","-1"," 1","1 ","\t1"})
        rejects([&]{(void)flashMTPQMVF32Switch(malformed,global);});
    }
    rejects([]{(void)flashMTPQMVF32Switch("1",false);});
    require(flashMTPQMVF32Switch("1",true),"explicit dependency-qualified opt-in was rejected");

    const Geometry original;
    require(original.eligible(),"original contiguous trained Q4/G64 geometry was rejected");
    for(uint32_t value : {0u,2u,512u}) {
      auto changed=original;changed.experts=value;
      require(!changed.eligible(),"unsupported expert geometry selected trained projection");
    }
    for(uint32_t value : {0u,640u,2559u,2561u,248320u,UINT32_MAX}) {
      auto changed=original;changed.outputs=value;
      require(!changed.eligible(),"unsupported output shape selected trained projection");
      changed=original;changed.inputs=value;
      require(!changed.eligible(),"unsupported input shape selected trained projection");
    }
    for(uint32_t value : {0u,3u,5u,6u,8u,UINT32_MAX}) {
      auto changed=original;changed.bits=value;
      require(!changed.eligible(),"a different source code format selected trained projection");
    }
    for(uint32_t value : {0u,32u,128u,UINT32_MAX}) {
      auto changed=original;changed.group=value;
      require(!changed.eligible(),"a different coefficient group selected trained projection");
    }
    for(uint64_t value : {uint64_t{0},uint64_t{1279},uint64_t{1281},UINT64_MAX}) {
      auto changed=original;changed.weights=value;
      require(!changed.eligible(),"padded/short/overflowing weight stride selected trained projection");
    }
    for(uint64_t value : {uint64_t{0},uint64_t{79},uint64_t{81},UINT64_MAX}) {
      auto changed=original;changed.parameters=value;
      require(!changed.eligible(),"padded/short/odd/overflowing parameter stride selected trained projection");
    }

    // Derive physical rows from actual real-pair contexts: four streams per
    // pair, across every supported homogeneous one-to-four-lane cohort.
    for(uint32_t lanes=1;lanes<=4;++lanes) for(uint32_t logical=1;logical<=8;++logical) {
      const uint32_t physical=4*lanes*logical;
      require(flashMTPQMVF32Window(physical,logical),
              "supported real-pair cohort failed physical/logical eligibility");
    }
    for(const auto &[physical,logical] :
        std::array<std::pair<uint32_t,uint32_t>,15>{{
          {0,1},{1,1},{3,1},{5,1},{127,8},{129,8},{132,8},
          {4,0},{4,9},{36,9},{64,16},{512,128},
          {4,2},{16,8},{UINT32_MAX,UINT32_MAX}}})
      require(!flashMTPQMVF32Window(physical,logical),
              "unsupported/priming/incomplete-stream window was accepted");
    for(const auto &counts : {std::array<uint32_t,4>{1,2,4,8},
                              std::array<uint32_t,4>{8,8,8,8},
                              std::array<uint32_t,4>{2,4,6,8}}) {
      uint32_t realRows=0;
      for(uint32_t value:counts) realRows+=value;
      require(flashMTPQMVF32Window(realRows*4,*std::min_element(counts.begin(),counts.end())),
              "bounded ragged real-pair cohort was rejected");
    }
    require(std::string_view(kFlashMTPQMVF32ProposalSemantics).find("proposal")!=std::string_view::npos,
            "route identity omitted proposal-only numerical scope");
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
              << ",\"gpu_commands\":0,\"model_loaded\":false}\n";
    return 0;
  } catch(const std::exception &error) {
    std::cerr << "MTP QMV policy CPU failure: " << error.what() << '\n';
    return 1;
  }
}
