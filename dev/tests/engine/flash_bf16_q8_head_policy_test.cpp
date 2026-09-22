#include "flash/FlashBF16Q8Head.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace splash::flash;
int main(int argc,char **argv){try {
  const std::string mode=argc>1?argv[1]:"unset";
  if(mode=="unset")unsetenv("SPLASH_FLASH_MTP_Q8_BF16_REGISTER");
  else setenv("SPLASH_FLASH_MTP_Q8_BF16_REGISTER",mode.c_str(),1);
  bool rejected=false,enabled=false;try{enabled=flashBF16Q8HeadEnabled();}catch(const std::invalid_argument &){rejected=true;}
  if(rejected!=(mode!="unset"&&mode!="0"&&mode!="1")||(!rejected&&enabled!=(mode=="1")))
    throw std::runtime_error("strict route switch mismatch");
  FlashAffineProjection p;p.experts=1;p.inputSize=2560;p.outputSize=248320;p.bits=8;p.groupSize=64;p.weightRowStrideBytes=2560;p.parameterRowStrideBytes=80;
  uint64_t valid=0,invalid=0;
  for(uint32_t rows=0;rows<257;++rows){if(flashBF16Q8HeadGeometry(p,rows)!=(rows>=2&&rows<=4))throw std::runtime_error("row eligibility mismatch");++valid;}
  for(uint32_t row:{2u,3u,4u}){
    for(uint32_t k:{0u,32u,2559u,2561u,10240u}){auto q=p;q.inputSize=k;if(flashBF16Q8HeadGeometry(q,row))throw std::runtime_error("invalidKaccepted");++invalid;}
    for(uint32_t n:{0u,32u,248319u,248321u}){auto q=p;q.outputSize=n;if(flashBF16Q8HeadGeometry(q,row))throw std::runtime_error("invalidNaccepted");++invalid;}
    for(uint32_t bits:{0u,4u,5u,6u,7u,9u}){auto q=p;q.bits=bits;if(flashBF16Q8HeadGeometry(q,row))throw std::runtime_error("invalidbitsaccepted");++invalid;}
    for(uint32_t group:{0u,32u,63u,65u,128u}){auto q=p;q.groupSize=group;if(flashBF16Q8HeadGeometry(q,row))throw std::runtime_error("invalidgroupaccepted");++invalid;}
    for(uint32_t experts:{0u,2u,512u}){auto q=p;q.experts=experts;if(flashBF16Q8HeadGeometry(q,row))throw std::runtime_error("invalidexpertsaccepted");++invalid;}
    for(uint64_t stride:{uint64_t{0},uint64_t{2559},UINT64_MAX}){auto q=p;q.weightRowStrideBytes=stride;if(flashBF16Q8HeadGeometry(q,row))throw std::runtime_error("invalidweightstrideaccepted");++invalid;}
    for(uint64_t stride:{uint64_t{0},uint64_t{79},uint64_t{81},UINT64_MAX-1}){auto q=p;q.parameterRowStrideBytes=stride;if(flashBF16Q8HeadGeometry(q,row))throw std::runtime_error("invalidparamstrideaccepted");++invalid;}
  }
  if(FlashBF16Q8Head::plannedBytes()!=0)throw std::runtime_error("routeaddedplannedallocation");
  std::cout<<"{\"pass\":true,\"mode\":\""<<mode<<"\",\"valid_checks\":"<<valid<<",\"invalid_checks\":"<<invalid<<",\"gpu_commands\":0,\"allocation_delta\":0}\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
