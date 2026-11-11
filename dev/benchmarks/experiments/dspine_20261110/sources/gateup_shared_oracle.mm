#include "GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace splash;
using namespace gguf_reference;

struct Guarded {
  metal::MetalBuffer base, view;
  uint64_t bytes;
  Guarded(metal::MetalBackend &backend, uint64_t size, uint8_t fill=0):bytes(size) {
    base=backend.allocateBuffer(size+256,metal::BufferStorage::Shared,"gate/up oracle guarded");
    std::memset(base.contents(),fill,size);std::memset(static_cast<uint8_t *>(base.contents())+size,0xa5,256);
    view=backend.view(base,0,size);
  }
  void check()const {
    const auto *p=static_cast<const uint8_t *>(base.contents())+bytes;
    if(!std::all_of(p,p+256,[](uint8_t b){return b==0xa5;}))throw std::runtime_error("guard overwritten");
  }
};
struct Weight {
  metal::MetalBuffer code, other, meta;
  std::vector<float> reference;
};
Weight weight(metal::MetalBackend &backend,uint32_t n,uint32_t k,std::mt19937 &rng,bool reference) {
  const auto native=makeNative(Fmt(GGUF_FMT_AF4G64),n,k,rng);
  Weight w;
  const auto packed=repack(Fmt(GGUF_FMT_AF4G64),native,n,k,reference?&w.reference:nullptr);
  const auto upload=[&](const auto &v){auto b=backend.allocateBuffer(v.size(),metal::BufferStorage::Shared,"oracle planes");std::memcpy(b.contents(),v.data(),v.size());return b;};
  w.code=upload(packed.w0);w.other=upload(packed.w1);w.meta=upload(packed.meta);return w;
}
double silu(double g){return g/(1+std::exp(-g));}
float rounded(double x){return test::bf16ToFloat(test::floatToBf16(float(x)));}
void run(metal::MetalBackend &backend,uint32_t rows,uint32_t live,uint32_t n,uint32_t k,bool benchmark) {
  std::mt19937 rng(1789+rows+k);auto gate=weight(backend,n,k,rng,!benchmark),up=weight(backend,n,k,rng,!benchmark);
  Guarded input(backend,uint64_t(rows)*k*2),output(backend,uint64_t(rows)*n*2),gateOutput(backend,uint64_t(rows)*n*2),partials(backend,uint64_t(16)*rows*n*2*4,0xff),counters(backend,n/32*4);
  auto *x=static_cast<uint16_t *>(input.view.contents());std::vector<float> xf(uint64_t(rows)*k);
  std::uniform_real_distribution<float> dist(-0.2f,0.2f);
  for(uint32_t row=0;row<rows;++row)for(uint32_t col=0;col<k;++col){const auto at=uint64_t(row)*k+col;x[at]=row<live?test::floatToBf16(dist(rng)):0x7fc0;xf[at]=test::bf16ToFloat(x[at]);}
  std::vector<uint16_t> old;
  uint32_t comparisons=0,cpuChecks=0;
  const std::vector<uint32_t> splits=benchmark?std::vector<uint32_t>{1,2}:k==512?std::vector<uint32_t>{1,2,4,8,16}:std::vector<uint32_t>{1,2,4};
  for(bool spread:{false,true})for(uint32_t split:splits) {
    std::vector<uint32_t> groups=rows==64?std::vector<uint32_t>{1,2,102,4,104,0}:std::vector<uint32_t>{1,2,102,0};
    for(uint32_t mode:groups) {
      const bool sharedStage=mode>=100;const uint32_t groupCount=sharedStage?mode-100:mode;
      const std::string name="gguf_decode_af4g64_m"+std::to_string(rows)+"_gate_up"+(groupCount>1?"_sg"+std::to_string(groupCount)+(sharedStage?"_shared":""):"");
      metal::CommandGraph graph;
      const uint32_t repeats=benchmark?50:1;
      for(uint32_t i=0;i<repeats;++i) {
        if(groupCount==0) {
          const auto prefix="gguf_decode_af4g64_m"+std::to_string(rows);
          graph.add(prefix+"_a",{input.view,gate.code,gate.other,gate.meta,gateOutput.view,partials.view,counters.view,gateOutput.view},
                     GgufDecodeParams{k,split,n,0,spread},{n/64,split,1},{64,1,1});
          graph.add(prefix+"_g",{input.view,up.code,up.other,up.meta,output.view,partials.view,counters.view,gateOutput.view},
                     GgufDecodeParams{k,split,n,0,spread},{n/64,split,1},{64,1,1});
        } else graph.add(name,{input.view,gate.code,gate.other,gate.meta,up.code,up.other,up.meta,output.view,partials.view,counters.view},
                          GgufDecodeParams{k,split,n,0,spread},{n/32,split,1},{32*groupCount,1,1});
      }
      const auto preparationStart=std::chrono::steady_clock::now();
      backend.preparePipelines(graph.dispatches());
      const double preparationMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-preparationStart).count();
      const double seconds=backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
      input.check();output.check();gateOutput.check();partials.check();counters.check();
      const auto *count=static_cast<const uint32_t *>(counters.view.contents());
      if(!std::all_of(count,count+n/32,[](uint32_t v){return v==0;}))throw std::runtime_error("split counter not released");
      const auto *out=static_cast<const uint16_t *>(output.view.contents());
      if(groupCount==1)old.assign(out,out+uint64_t(rows)*n);
      else for(uint64_t i=0;i<uint64_t(live)*n;++i){++comparisons;if(out[i]!=old[i])throw std::runtime_error("row subtile differs bitwise from original kernel");}
      for(uint32_t row=live;row<rows;++row)for(uint32_t col=0;col<n;++col)
        if(!std::isnan(test::bf16ToFloat(out[uint64_t(row)*n+col])))throw std::runtime_error("NaN padding was not isolated");
      if(!benchmark)for(uint32_t row:{0u,live/2,live-1})for(uint32_t col:{0u,n/2,n-1}) {
        const auto g=dot(xf.data()+uint64_t(row)*k,gate.reference.data()+uint64_t(col)*k,k);
        const auto u=dot(xf.data()+uint64_t(row)*k,up.reference.data()+uint64_t(col)*k,k);
        const double gb=projectionBound(g,true),ub=projectionBound(u,true),center=rounded(g.value);
        const double radius=std::max(std::fabs(rounded(g.value-gb)-center),std::fabs(rounded(g.value+gb)-center));
        // |SiLU'| <= 1+|x|exp(-|x|) < 2. Include native fast-exp error.
        const double t=std::max(std::fabs(rounded(g.value-gb)),std::fabs(rounded(g.value+gb)))*1.4426950408889634;
        const double rel=(2*(3+std::floor(2*t))+2*std::log(2.0)*t+6)*0x1p-24;
        const double s=silu(center),e=2*radius+std::fabs(s)*rel;
        double lo=INFINITY,hi=-INFINITY;
        for(double uv:{double(rounded(u.value-ub)),double(rounded(u.value+ub))})for(double sv:{s-e,s+e}){lo=std::min(lo,uv*sv);hi=std::max(hi,uv*sv);}
        const float actual=test::bf16ToFloat(out[uint64_t(row)*n+col]);++cpuChecks;
        if(!std::isfinite(actual)||actual<rounded(lo)||actual>rounded(hi))throw std::runtime_error("gate/up outside independent fp64 interval");
      }
      std::cout<<"{\"rows\":"<<rows<<",\"live_rows\":"<<live<<",\"outputs\":"<<n<<",\"inputs\":"<<k<<",\"splits\":"<<split<<",\"spread\":"<<(spread?"true":"false")<<",\"simdgroups\":"<<groupCount<<",\"shared_weight_stage\":"<<(sharedStage?"true":"false")<<",\"dispatches_per_projection\":"<<(groupCount?1:2)<<",\"repeats\":"<<repeats<<",\"pipeline_prepare_wall_ms\":"<<preparationMs<<",\"gpu_ms_per_projection\":"<<seconds*1000/repeats<<"}\n";
    }
  }
  std::cout<<"{\"case_passed\":true,\"rows\":"<<rows<<",\"live_rows\":"<<live<<",\"bitwise_comparisons\":"<<comparisons<<",\"independent_fp64_checks\":"<<cpuChecks<<"}\n";
}
int main(int argc,char **argv) {
  try {
    if(argc<2||argc>3)throw std::invalid_argument("usage: gateup-tile-oracle METALLIB [--benchmark]");
    const bool benchmark=argc==3&&std::string(argv[2])=="--benchmark";
    if(argc==3&&!benchmark)throw std::invalid_argument("unknown argument");
    metal::MetalBackend backend(argv[1]);
    if(benchmark){run(backend,32,32,17408,5120,true);run(backend,64,64,17408,5120,true);}
    else for(uint32_t rows:{32u,64u})for(uint32_t k:{512u,1152u})for(uint32_t live:{rows,rows*3/4})run(backend,rows,live,128,k,false);
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
