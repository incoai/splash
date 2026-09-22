#include "FlashMTPGPUChainPolicy.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace policy = splash::flash::mtp_gpu_chain_policy;
namespace {
uint64_t checks = 0;
constexpr uint32_t kFirstSentinel = 0xa7a7a7a7, kSecondSentinel = 0xb8b8b8b8;
constexpr int64_t kI64Sentinel = -0x1777177717771777LL;
constexpr FlashGreedyGPURowResult kGreedySentinel{0xa7a7a7a7,0xb8b8b8b8,0xc9c9c9c9,0xdadadada};
void require(bool value, const char *reason) {
  ++checks;
  if (!value) throw std::runtime_error(reason);
}
bool same(const FlashGreedyGPURowResult &a, const FlashGreedyGPURowResult &b) {
  return std::memcmp(&a,&b,sizeof(a)) == 0;
}
FlashGreedyGPURowResult valid(uint32_t token=17) {
  return {token,0xbf80,0,0};
}
struct Storage {
  int64_t nextI64 = kI64Sentinel;
  FlashGreedyGPURowResult argmaxSuffix = kGreedySentinel;
  std::array<uint16_t,32> hiddenSuffix;
  Storage() { hiddenSuffix.fill(0x7fc1); }
};
void skippedSuffix(const policy::Prepared &prepared) {
  Storage storage;
  const auto original = storage;
  policy::writeNextToken(prepared,storage.nextI64);
  for(policy::Groups groups : {policy::Groups{1,1,1},policy::Groups{320,1,1},
                              policy::Groups{8,4,2},policy::Groups{UINT32_MAX,17,3}})
    require(policy::indirectGroups(prepared,groups) == policy::Groups{0,0,0},
            "skipped body did not zero the entire indirect triplet");
  require(storage.nextI64 == original.nextI64 &&
          same(storage.argmaxSuffix,original.argmaxSuffix) &&
          storage.hiddenSuffix == original.hiddenSuffix,
          "skipped body modified I64/argmax/hidden suffix sentinels");
}
}

int main() {
  try {
    static_assert(sizeof(policy::Params)==32 && sizeof(policy::Control)==40);
    for(uint32_t word=0;word<65536;++word) {
      const bool finite=(word&0x7f80)!=0x7f80;
      const uint32_t rank=(word&0x7fff)==0?0x8000:
          (word&0x8000)?((~word)&0xffff):(word|0x8000);
      const auto record=FlashGreedyGPURowResult{17,rank,0,0};
      require((policy::recordError(record,248320)==0)==finite,
              "greedy rank contract disagrees with a real BF16 finite/nonfinite word");
    }
    for(uint32_t rank : {0u,0x7fu,0x7fffu,0xff80u,0xffffu,UINT32_MAX}) {
      auto record=valid();record.rank=rank;
      require(policy::recordError(record,248320)==policy::kErrorInvalidGreedy,
              "invalid greedy rank was accepted");
    }
    for(uint32_t rank : {0x80u,0x7ffeu,0x8000u,0xff7fu}) {
      auto record=valid();record.rank=rank;
      require(policy::recordToken(record,248320)==17,"legal greedy rank boundary was rejected");
    }
    std::array<FlashGreedyGPURowResult,9> malformed{
        FlashGreedyGPURowResult{248320,0xbf80,0,0},
        FlashGreedyGPURowResult{UINT32_MAX,0xbf80,0,0},
        FlashGreedyGPURowResult{17,0x7fff,0,0},
        FlashGreedyGPURowResult{17,0xbf80,0,1},
        FlashGreedyGPURowResult{17,0xbf80,kFlashGreedyGPUErrorNonfinite,0},
        FlashGreedyGPURowResult{17,0xbf80,kFlashGreedyGPUErrorInputToken,0},
        FlashGreedyGPURowResult{17,0xbf80,kFlashGreedyGPUErrorBudget,0},
        FlashGreedyGPURowResult{17,0xbf80,0x80000000,0},
        FlashGreedyGPURowResult{UINT32_MAX,0,kFlashGreedyGPUErrorNonfinite,1}};
    for(const auto &record:malformed) {
      const auto expected=record.errors&kFlashGreedyGPUErrorNonfinite?
          policy::kErrorNonfinite:policy::kErrorInvalidGreedy;
      require(policy::recordError(record,248320)==expected,"greedy error priority changed");
      bool threw=false;
      try{(void)policy::recordToken(record,248320);}
      catch(const std::runtime_error &error){
        threw=true;
        require(std::string(error.what())==(expected==policy::kErrorNonfinite?
            "non-finite Flash vocabulary logit":
            "Flash MTP GPU greedy result has invalid status or extent"),"greedy exception contract changed");
      }
      require(threw,"malformed greedy record did not raise its production-compatible exception");
    }

    for(uint32_t budget=1;budget<=17;++budget) for(uint32_t requested=0;requested<=2;++requested)
      for(uint32_t begin=0;begin<16;++begin) {
        const uint32_t expectedDepth=requested<budget?requested:budget-1;
        require(policy::boundedDepth(requested,budget)==expectedDepth,"depth exceeded quota-minus-bonus bound");
        policy::Params params{248320,requested,budget,begin,32,61,{0,0}};
        const auto prepared=policy::prepare(params,valid(),0,{kFirstSentinel,kSecondSentinel});
        require(prepared.depth==expectedDepth && prepared.control.consumedPairs==0,
                "preparation consumed a head pair before execution");
        require(prepared.control.proposalCount==(expectedDepth?1u:0u),
                "seed selection count disagrees with bounded depth");
        require(prepared.control.proposals[1]==kSecondSentinel,"preparation wrote the second proposal suffix");
        require(bool(prepared.control.bodyEnabled)==(expectedDepth==2),"second head body eligibility differs");
        require(prepared.seedRead==bool(expectedDepth),"unneeded seed was inspected");
        const auto pending=policy::complete(prepared,malformed.back(),UINT32_MAX,false);
        require(!pending.ready && !pending.poisonHead && pending.logicalLength==begin &&
                pending.control.consumedPairs==0,"pending command published head mutation");
        if(!prepared.control.bodyEnabled) {
          skippedSuffix(prepared);
          const auto completed=policy::complete(prepared,malformed.back(),UINT32_MAX,true);
          require(completed.ready && !completed.poisonHead && completed.logicalLength==begin &&
                  completed.control.proposalCount==(expectedDepth?1u:0u) &&
                  completed.control.errors==0,"skipped body read or validated poisoned suffix data");
        } else {
          Storage storage;
          policy::writeNextToken(prepared,storage.nextI64);
          require(storage.nextI64==17,"valid U32 token was not converted to the I64 embedding ABI");
          for(policy::Groups groups:{policy::Groups{1,1,1},policy::Groups{320,1,1},policy::Groups{8,4,2}})
            require(policy::indirectGroups(prepared,groups)==groups,"active body changed static dispatch dimensions");
          for(uint32_t token:{19u,248044u,248046u}) {
            const auto completed=policy::complete(prepared,valid(token),0,true);
            require(completed.ready && !completed.poisonHead && completed.control.errors==0 &&
                    completed.control.proposalCount==2 && completed.control.consumedPairs==1 &&
                    completed.logicalLength==begin+1 && completed.rollbackLength==begin &&
                    completed.control.proposals[1]==token && bool(completed.control.finishedEOS)==policy::stopToken(token),
                    "completed valid body counts/state/EOS/rollback differ");
            require(policy::newlyCompletedPools(begin,completed.control.consumedPairs)==(begin%4==3),
                    "single consumed pair used a wrong four-token pooling boundary");
          }
          const auto diagnosed=policy::complete(prepared,valid(19),4,true);
          require(diagnosed.poisonHead && diagnosed.logicalLength==begin &&
                  diagnosed.control.consumedPairs==1 && diagnosed.control.errors==policy::kErrorDiagnostics &&
                  diagnosed.control.proposals[1]==kSecondSentinel,
                  "actual diagnosed body published state/second token or failed to poison");
          for(const auto &second:malformed) {
            const auto rejected=policy::complete(prepared,second,0,true);
            require(rejected.poisonHead && rejected.logicalLength==begin &&
                    rejected.control.consumedPairs==1 && rejected.control.errors!=0 &&
                    rejected.control.proposals[1]==kSecondSentinel,
                    "invalid executed suffix advanced healthy head state");
          }
          const auto failedCommand=policy::complete(prepared,valid(19),0,true,false);
          require(failedCommand.poisonHead && failedCommand.logicalLength==begin &&
                  failedCommand.control.proposalCount==0,"failed command published state/proposals");
        }

        for(uint32_t token:{248044u,248046u}) {
          const auto stopped=policy::prepare(params,valid(token),0,{kFirstSentinel,kSecondSentinel});
          require(!stopped.control.bodyEnabled && stopped.control.proposalCount==(expectedDepth?1u:0u) &&
                  bool(stopped.control.finishedEOS)==bool(expectedDepth) &&
                  stopped.control.proposals[1]==kSecondSentinel,"first EOS entered another head pair");
          skippedSuffix(stopped);
        }
        for(const auto &seed:malformed) {
          const auto badSeed=policy::prepare(params,seed,0,{kFirstSentinel,kSecondSentinel});
          require(!badSeed.control.bodyEnabled && badSeed.control.proposalCount==0 &&
                  badSeed.control.proposals[0]==kFirstSentinel && badSeed.control.proposals[1]==kSecondSentinel &&
                  (expectedDepth?badSeed.control.errors!=0:badSeed.control.errors==0),
                  "invalid or unneeded seed modified proposal/head eligibility");
          skippedSuffix(badSeed);
          const auto rejected=policy::complete(badSeed,malformed.back(),4,true);
          require(!rejected.poisonHead && rejected.logicalLength==begin &&
                  rejected.control.consumedPairs==0,"rejected seed poisoned or advanced an unexecuted head");
        }
        const auto priorFailure=policy::prepare(params,valid(),4,{kFirstSentinel,kSecondSentinel});
        require(!priorFailure.control.bodyEnabled &&
                priorFailure.control.errors==(expectedDepth?policy::kErrorDiagnostics:0),
                "prior diagnostics were ignored or unnecessary zero-work diagnostics were read");
        skippedSuffix(priorFailure);
      }

    for(uint32_t requested=0;requested<=2;++requested) {
      policy::Params full{248320,requested,17,32,32,0,{0,0}};
      const auto prepared=policy::prepare(full,valid(),0,{kFirstSentinel,kSecondSentinel});
      require(!prepared.control.bodyEnabled &&
              prepared.control.errors==(requested==2?policy::kErrorCapacity:0),
              "full context advanced a pair or rejected a legal zero-body seed");
      skippedSuffix(prepared);
      for(uint32_t stop:{248044u,248046u}) {
        const auto stopped=policy::prepare(full,valid(stop));
        require(!stopped.control.bodyEnabled && stopped.control.errors==0,"full-context EOS required a body");
      }
    }
    policy::Params noDispatch;
    noDispatch.dispatchCount=0;
    const auto missingBody=policy::prepare(noDispatch,valid());
    require(!missingBody.control.bodyEnabled && missingBody.control.errors==policy::kErrorMalformedParams,
            "requested second proposal accepted an absent body dispatch list");
    for(uint32_t index=0;index<8;++index) {
      policy::Params bad;
      if(index==0)bad.requestedDepth=3;
      if(index==1)bad.remaining=0;
      if(index==2)bad.vocabulary=0;
      if(index==3)bad.vocabulary=248321;
      if(index==4)bad.capacity=0;
      if(index==5)bad.capacity=262145;
      if(index==6)bad.begin=bad.capacity+1;
      if(index==7)bad.reserved[1]=1;
      const auto rejected=policy::prepare(bad,valid(),0,{kFirstSentinel,kSecondSentinel});
      require(rejected.control.errors==policy::kErrorMalformedParams && !rejected.control.bodyEnabled &&
              rejected.control.proposalCount==0 && !rejected.seedRead &&
              rejected.control.proposals[0]==kFirstSentinel && rejected.control.proposals[1]==kSecondSentinel,
              "malformed parameters inspected/wrote proposal state");
      skippedSuffix(rejected);
    }
    for(uint32_t begin:{0u,1u,2u,3u,4u,7u,8u,262143u})
      require(policy::poolBegin(begin)==begin/4 &&
              policy::newlyCompletedPools(begin,0)==0 &&
              policy::newlyCompletedPools(begin,1)==(begin%4==3),
              "pooling cursor overflowed or crossed a wrong boundary");
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
              << ",\"gpu_commands\":0,\"model_loaded\":false}\n";
    return 0;
  } catch(const std::exception &error) {
    std::cerr << "GPU chain CPU reference failure: " << error.what() << '\n';
    return 1;
  }
}
