#include "HyperCaptureContract.hpp"
#include <iostream>
int main() {
  using namespace splash::test;
  HyperStreamMeanParams p{8, 2560, 4, 2, 10240, 2560};
  std::array<CaptureRegion,4> regions{{{1,16,327680,327712},{2,16,40960,40992},{3,16,128,160},{4,16,204800,204832}}};
  unsigned count=0;
  auto accepts=[&](auto params, auto r) { validateHyperCapture(params,r); ++count; };
  auto rejects=[&](auto params, auto r) {
    try { validateHyperCapture(params,r); } catch(const std::invalid_argument &) { ++count; return; }
    throw std::runtime_error("unsafe fusion contract accepted");
  };
  accepts(p,regions);
  for (size_t i=0;i<4;++i) {
    auto r=regions;r[i].bytes-=i==1||i==3?2:4;rejects(p,r);
    r=regions;r[i].offset+=1;rejects(p,r);
    r=regions;r[i].identity=0;rejects(p,r);
    r=regions;r[i].offset=r[i].capacity+1;rejects(p,r);
  }
  for(size_t i=1;i<4;++i) {auto r=regions;r[i].identity=1;r[i].capacity=regions[0].capacity;rejects(p,r);}
  auto r=regions;r[2].identity=2;r[2].capacity=r[1].capacity;accepts(p,r); // Read-only alias is safe.
  r=regions;r[3].identity=1;r[3].offset=400000;r[0].capacity=r[3].capacity=700000;accepts(p,r);
  auto bad=p;bad.rows=0;rejects(bad,regions);
  bad=p;bad.hidden=0;rejects(bad,regions);
  bad=p;bad.streams=0;rejects(bad,regions);
  bad=p;bad.streams=5;rejects(bad,regions);
  bad=p;bad.column=UINT32_MAX;rejects(bad,regions);
  bad=p;bad.row=UINT32_MAX;rejects(bad,regions);
  bad=p;bad.rows=bad.hidden=bad.stride=UINT32_MAX;bad.column=0;rejects(bad,regions);
  std::cout<<count<<" buffer/alias/geometry contract checks passed (no Metal or model).\n";
}
