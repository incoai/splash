#include "FlashMTPGPUChainFourPolicy.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace p=splash::flash::mtp_gpu_chain_four_policy;
namespace {
uint64_t checks=0;
void require(bool value,const char *message) {
  ++checks;if(!value)throw std::runtime_error(message);
}
constexpr int64_t kTokenSentinel=-0x1777177717771777LL;
constexpr FlashGreedyGPURowResult kRecordSentinel{0xa7a7a7a7,0xb8b8b8b8,0xc9c9c9c9,0xdadadada};
struct Slot {
  int64_t token=kTokenSentinel;
  FlashGreedyGPURowResult record=kRecordSentinel;
  std::array<uint16_t,32> hidden;
  Slot(){hidden.fill(0x7fc1);}
};
FlashGreedyGPURowResult proposal(uint32_t position,uint32_t eosAt,uint32_t eosToken,
                                uint32_t badAt,uint32_t errorKind) {
  auto r=FlashGreedyGPURowResult{17+position,0xbf80,0,0};
  if(position==eosAt)r.token=eosToken;
  if(position==badAt) {
    if(errorKind==1)r.errors=kFlashGreedyGPUErrorNonfinite;
    if(errorKind==2)r.token=248320;
    if(errorKind==3)r.rank=0x7fff;
    if(errorKind==4)r.reserved=1;
    if(errorKind==5)r.errors=kFlashGreedyGPUErrorInputToken;
  }
  return r;
}
struct Expected {uint32_t proposals=0,consumed=0,error=0;bool eos=false;};
// Independent scalar controller: reserve the bonus row; select one seed;
// consume one real pair for each subsequent proposal; stop at its first event.
Expected scalar(uint32_t requested,uint32_t budget,uint32_t available,uint32_t eosAt,
                uint32_t badAt,uint32_t diagnosticAt,uint32_t kind) {
  Expected out;
  const uint32_t depth=requested<budget?requested:budget-1;
  if(!depth)return out;
  if(diagnosticAt==1){out.error=4;return out;}
  if(badAt==1){out.error=kind==1?1:2;return out;}
  out.proposals=1;out.eos=eosAt==1;
  for(uint32_t position=2;position<=depth&&!out.eos;++position) {
    if(out.consumed==available){out.error=8;break;}
    ++out.consumed;
    if(diagnosticAt==position){out.error=4;break;}
    if(badAt==position){out.error=kind==1?1:2;break;}
    ++out.proposals;out.eos=eosAt==position;
  }
  return out;
}
void exercise(uint32_t requested,uint32_t budget,uint32_t modulo,uint32_t available,
              uint32_t eosAt=0,uint32_t eosToken=248044,uint32_t badAt=0,
              uint32_t diagnosticAt=0,uint32_t kind=2) {
  const uint32_t initial=16+modulo;
  p::Control control;
  std::array<Slot,3> suffix;
  const auto seed=proposal(1,eosAt,eosToken,badAt,kind);
  for(uint32_t body=0;body<3;++body) {
    p::Params params{248320,requested,budget,initial,initial+available,
                     body<available?97u:0u,body,0};
    const auto before=suffix[body];
    const bool seedRead=p::beginStep(control,params,seed,diagnosticAt==1?4:0);
    require(!body||!seedRead,"later body reread an external seed record");
    for(p::Groups groups:{p::Groups{1,1,1},p::Groups{320,1,1},
                          p::Groups{8,4,2},p::Groups{UINT32_MAX,7,3}})
      require(p::indirectGroups(control,groups)==(control.bodyEnabled?groups:p::Groups{0,0,0}),
              "indirect triplets changed active dimensions or failed to zero all skipped axes");
    p::writeNextToken(control,params,suffix[body].token);
    if(control.bodyEnabled) {
      require(suffix[body].token==int64_t(control.proposals[body]),"active proposal did not feed I64 ABI");
      suffix[body].record=proposal(body+2,eosAt,eosToken,badAt,kind);
      suffix[body].hidden.fill(uint16_t(0x3f80+body));
      const bool recordRead=p::finishStep(control,params,suffix[body].record,
          diagnosticAt==body+2?4:0);
      require(recordRead==(diagnosticAt!=body+2),"diagnostics did not precede suffix-record validation");
    } else {
      require(!p::finishStep(control,params,kRecordSentinel,UINT32_MAX),
              "skipped finish read an unwritten record or diagnostics");
      require(suffix[body].token==before.token&&
              std::memcmp(&suffix[body].record,&before.record,sizeof(before.record))==0&&
              suffix[body].hidden==before.hidden,"skipped token/argmax/hidden suffix was modified");
    }
    const auto pending=p::publish(control,initial,false);
    require(!pending.ready&&!pending.success&&!pending.poisonHead&&pending.logicalLength==initial,
            "an unfinished whole command published logical state");
  }
  const auto expected=scalar(requested,budget,available,eosAt,badAt,diagnosticAt,kind);
  require(control.proposalCount==expected.proposals&&control.consumedPairs==expected.consumed&&
          control.errors==expected.error&&bool(control.finishedEOS)==expected.eos,
          "four-body reference disagrees with independent scalar EOS/quota/error/capacity controller");
  for(uint32_t index=0;index<control.proposalCount;++index)
    require(control.proposals[index]==proposal(index+1,eosAt,eosToken,0,0).token,
            "published proposal prefix ID differs");
  for(uint32_t index=control.proposalCount;index<4;++index)
    require(control.proposals[index]==UINT32_MAX,"unpublished proposal suffix was read/written");
  const auto published=p::publish(control,initial,true);
  require(published.ready&&published.success==(expected.error==0)&&
          published.logicalLength==(expected.error?initial:initial+expected.consumed)&&
          published.rollbackLength==initial&&
          published.poisonHead==bool(expected.consumed&&expected.error&&expected.error!=8),
          "whole-command logical publication/poison/rollback contract differs");
  const auto failed=p::publish(control,initial,true,false);
  require(!failed.success&&failed.logicalLength==initial&&
          failed.poisonHead==bool(control.consumedPairs||control.bodyEnabled),
          "failed command advanced healthy state or missed possible execution");
  require(p::newlyCompletedPools(initial,expected.consumed)==
          (initial+expected.consumed)/4-initial/4,"cumulative pairs crossed wrong pooling boundary");
}
}
int main() {
  try {
    static_assert(sizeof(p::Params)==32&&sizeof(p::Control)==48);
    for(uint32_t budget=1;budget<=17;++budget)for(uint32_t depth=0;depth<=4;++depth)
      for(uint32_t modulo=0;modulo<4;++modulo)for(uint32_t available=0;available<=3;++available) {
        require(p::boundedDepth(depth,budget)==std::min(depth,budget-1),"budget failed to reserve correction/bonus row");
        exercise(depth,budget,modulo,available);
        for(uint32_t eosAt=1;eosAt<=4;++eosAt)for(uint32_t eos:{248044u,248046u})
          exercise(depth,budget,modulo,available,eosAt,eos);
      }
    for(uint32_t budget=1;budget<=17;++budget)for(uint32_t depth=0;depth<=4;++depth)
      for(uint32_t position=1;position<=4;++position)for(uint32_t kind=1;kind<=5;++kind) {
        exercise(depth,budget,position%4,3,0,248044,position,0,kind);
        exercise(depth,budget,position%4,3,0,248044,position,position,kind);
      }
    for(uint32_t position=1;position<=4;++position)
      for(uint32_t modulo=0;modulo<4;++modulo)
        exercise(4,17,modulo,3,0,248046,0,position);
    for(uint32_t field=0;field<9;++field) {
      p::Params bad;
      if(field==0)bad.vocabulary=0;
      if(field==1)bad.vocabulary=248321;
      if(field==2)bad.requestedDepth=5;
      if(field==3)bad.remaining=0;
      if(field==4)bad.capacity=0;
      if(field==5)bad.capacity=262145;
      if(field==6)bad.begin=bad.capacity+1;
      if(field==7)bad.dispatchCount=257;
      if(field==8)bad.reserved=1;
      p::Control control;
      require(!p::beginStep(control,bad,kRecordSentinel)&&control.errors==16&&
              !control.proposalCount&&!control.consumedPairs&&!control.bodyEnabled,
              "malformed parameters read seed or mutated head eligibility");
    }
    p::Params badBody;badBody.bodyIndex=3;p::Control badControl;
    require(!p::beginStep(badControl,badBody,kRecordSentinel)&&badControl.errors==16,
            "bodyIndex beyond three bodies was accepted");
    // A valid successor cannot be fabricated without an actually consumed predecessor.
    p::Params successor;successor.bodyIndex=1;p::Control absent;
    require(!p::beginStep(absent,successor,kRecordSentinel)&&absent.errors==16&&!absent.bodyEnabled,
            "a missing predecessor revived the chain");
    for(uint32_t consumed=0;consumed<=3;++consumed) {
      const uint32_t begin=262144-consumed;
      require(p::newlyCompletedPools(begin,consumed)==262144/4-begin/4,
              "capacity-edge pooling cursor overflowed");
    }
    std::cout<<"{\"pass\":true,\"cpu_checks\":"<<checks
             <<",\"gpu_commands\":0,\"model_loaded\":false}\n";
    return 0;
  }catch(const std::exception &error){
    std::cerr<<"four-proposal CPU policy failure: "<<error.what()<<'\n';return 1;
  }
}
