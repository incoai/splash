#include "metal/CommandGraph.hpp"
#include "LinearNumerics.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace splash;
int main(int argc,char **argv) {
  try {
    if(argc!=2)throw std::invalid_argument("usage: bridge-norm-oracle METALLIB");
    metal::MetalBackend backend(argv[1]);
    constexpr uint32_t H=5120,Rows=4;
    auto input=backend.allocateBuffer(H*Rows*2,metal::BufferStorage::Shared,"oracle input");
    auto weight=backend.allocateBuffer(H*2,metal::BufferStorage::Shared,"oracle gamma");
    auto output=backend.allocateBuffer(H*Rows*2,metal::BufferStorage::Shared,"oracle output");
    auto *x=static_cast<uint16_t *>(input.contents()),*w=static_cast<uint16_t *>(weight.contents());
    for(uint32_t h=0;h<H;++h)w[h]=test::floatToBf16(0.5f+float(h%128)/64);
    for(uint32_t row=0;row<Rows;++row)for(uint32_t h=0;h<H;++h)
      x[row*H+h]=test::floatToBf16((h%2?1.5f:0.5f)*std::ldexp(1.0f,int(row)-2));
    metal::CommandGraph graph;
    graph.add("dspine_bridge_norm",{input,weight,output},H,{Rows,1,1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    uint32_t mismatches=0,distinguishing=0;
    auto *actual=static_cast<const uint16_t *>(output.contents());
    for(uint32_t row=0;row<Rows;++row) {
      const float scale=std::ldexp(1.0f,int(row)-2);
      const float inverse=1/std::sqrt(1.25f*scale*scale+1e-6f);
      for(uint32_t h=0;h<H;++h) {
        const float normalized=test::bf16ToFloat(x[row*H+h])*inverse;
        const float gamma=test::bf16ToFloat(w[h]);
        const uint16_t expected=test::floatToBf16(test::bf16ToFloat(test::floatToBf16(normalized))*gamma);
        const uint16_t single=test::floatToBf16(normalized*gamma);
        distinguishing+=single!=expected;
        if(actual[row*H+h]!=expected) {
          if(mismatches++<4)std::cerr<<row<<","<<h<<": "<<actual[row*H+h]<<" vs "<<expected<<"\n";
        }
      }
    }
    std::cout<<"{\"elements\":"<<H*Rows<<",\"mismatches\":"<<mismatches<<",\"elements_distinguishing_single_round\":"<<distinguishing<<"}\n";
    if(mismatches||distinguishing<1000)throw std::runtime_error("bridge norm contract failed");
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
