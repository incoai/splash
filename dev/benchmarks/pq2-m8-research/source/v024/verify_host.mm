// R277 next-load-after-decode vs unchanged R274 gather; source-derived plain fused QKV. Build does not authorize execution.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "metal/abi/Gguf.h"
#include "metal/abi/GgufRepack.h"

#include <iomanip>
using Clock=std::chrono::steady_clock;
static std::string root,prep;
static const std::string controlRoot="/Users/zaiyang/bonsai-pq2-research/bonsai2-study/";

static void check(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static void initializePaths(){uint32_t n=0;_NSGetExecutablePath(nullptr,&n);std::vector<char>b(n);check(_NSGetExecutablePath(b.data(),&n)==0,"package executable");char actual[PATH_MAX];check(realpath(b.data(),actual)!=nullptr,"package executable realpath");std::string path(actual);auto slash=path.find_last_of('/');check(slash!=std::string::npos,"package parent");prep=path.substr(0,slash+1);root=prep+"fixtures/";}
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static double epoch(){return NSDate.date.timeIntervalSince1970;}
static NSString* ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
static std::string str(NSString*s){check([s isKindOfClass:NSString.class],"string field");return s.UTF8String;}
static NSString* hash(NSData*b){check(b!=nil,"hash input");unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b.bytes,CC_LONG(b.length),d);NSMutableString*s=[NSMutableString string];for(auto x:d)[s appendFormat:@"%02x",x];return s;}
static NSData* read(const std::string&p){NSData*b=[NSData dataWithContentsOfFile:ns(p)];check(b!=nil,"read file");return b;}
static NSDictionary* json(const std::string&p){NSError*e=nil;id x=[NSJSONSerialization JSONObjectWithData:read(p) options:0 error:&e];check(!e&&[x isKindOfClass:NSDictionary.class],"JSON dictionary");return x;}
static void expected(const std::string&p,NSString*h){check(h.length==64&&[hash(read(p)) isEqual:h],"SHA256 pin");}
static std::string leasePath,leaseIdentity,outdir,runName,runToken,targetAck;
static pid_t owner=0;
struct Control {std::string path;dev_t dev;ino_t ino;off_t size;timespec mt,ct;};
static std::vector<Control> controls;
static std::string immutable(NSDictionary*d){NSMutableDictionary*x=[d mutableCopy];[x removeObjectForKey:@"heartbeat_unix"];[x removeObjectForKey:@"target_identity_ack"];NSError*e=nil;NSData*b=[NSJSONSerialization dataWithJSONObject:x options:NSJSONWritingSortedKeys error:&e];check(!e,"lease identity");return str(hash(b));}
static void guard(bool initial=false){
 NSDictionary*d=json(leasePath);const double now=epoch();
 check([d[@"version"] intValue]==1&&[d[@"phase"] isEqual:@"v002-target-forward"]&&[d[@"gpu_authorized"] boolValue],"separate root GPU authorization");
 check([d[@"token"] isKindOfClass:NSString.class]&&[d[@"token"] length]>=32,"lease token");
 check([d[@"heartbeat_unix"] doubleValue]<=now+1&&now-[d[@"heartbeat_unix"] doubleValue]<=3,"live controller heartbeat");
 check([d[@"deadline_unix"] doubleValue]>now&&[d[@"deadline_unix"] doubleValue]<=now+900,"finite future deadline");
 pid_t p=[d[@"owner_pid"] intValue];check(p>1&&p!=getpid()&&kill(p,0)==0,"live owner");
 if(initial){owner=p;leaseIdentity=immutable(d);runToken=str(d[@"token"]);}else check(owner==p&&leaseIdentity==immutable(d),"immutable authorization");
 if(!targetAck.empty()){NSDictionary*a=d[@"target_identity_ack"];check([a isKindOfClass:NSDictionary.class]&&immutable(a)==targetAck,"bound target acknowledgement unchanged");}
 NSArray*locks=d[@"locks"];check([locks isKindOfClass:NSArray.class]&&locks.count==5,"five actual reservations");
 for(NSDictionary*l in locks){int fd=open(str(l[@"path"]).c_str(),O_RDONLY|O_CLOEXEC);check(fd>=0,"lock open");struct stat st{};int sr=fstat(fd,&st);struct flock q{};q.l_type=F_WRLCK;q.l_whence=SEEK_SET;q.l_start=0;q.l_len=0;int rc=fcntl(fd,F_GETLK,&q);int flags=fcntl(fd,F_GETFD);close(fd);check(sr==0&&uint64_t(st.st_dev)==[l[@"dev"] unsignedLongLongValue]&&st.st_ino==[l[@"ino"] unsignedLongLongValue],"lock inode");check(rc==0&&q.l_type==F_WRLCK&&q.l_pid==owner&&q.l_start==0&&q.l_len==0&&(flags&FD_CLOEXEC),"independent F_GETLK owner/noninheritance");}
 if(initial){
  uint32_t n=0;_NSGetExecutablePath(nullptr,&n);std::vector<char>b(n);check(_NSGetExecutablePath(b.data(),&n)==0,"own executable");char actual[PATH_MAX];check(realpath(b.data(),actual)!=nullptr,"own realpath");expected(actual,d[@"host_sha256"]);
  expected(prep+"input-pins.json",d[@"input_manifest_sha256"]);
  NSDictionary*review=d[@"review"];expected(str(review[@"path"]),review[@"sha256"]);
  NSArray*c=d[@"controls"];check([c isKindOfClass:NSArray.class]&&c.count>=4,"controls");
  bool stop=false,feedback=false,direction=false,iteration=false;
  for(NSDictionary*v in c){std::string path=str(v[@"path"]);expected(path,v[@"sha256"]);struct stat st{};check(stat(path.c_str(),&st)==0,"control stat");controls.push_back({path,st.st_dev,st.st_ino,st.st_size,st.st_mtimespec,st.st_ctimespec});stop|=path==controlRoot+"worker/STOP";feedback|=path==controlRoot+"worker/FEEDBACK.md";direction|=path==controlRoot+"DIRECTION.md";iteration|=path==controlRoot+"ITERATION.md";}
  check(stop&&feedback&&direction&&iteration,"required historical stop/livefeedback controls");
  NSDictionary*pins=json(prep+"input-pins.json");for(NSString*k in pins){NSDictionary*v=pins[k];expected(root+str(k),v[@"sha256"]);check(read(root+str(k)).length==[v[@"bytes"] unsignedLongLongValue],"input byte size");}
 }else for(auto&c:controls){struct stat st{};check(stat(c.path.c_str(),&st)==0&&st.st_dev==c.dev&&st.st_ino==c.ino&&st.st_size==c.size&&st.st_mtimespec.tv_sec==c.mt.tv_sec&&st.st_mtimespec.tv_nsec==c.mt.tv_nsec&&st.st_ctimespec.tv_sec==c.ct.tv_sec&&st.st_ctimespec.tv_nsec==c.ct.tv_nsec,"live control changed");}
}
static void awaitIdentityAck(){
 auto started=Clock::now();
 while(seconds(started)<3){
  guard();NSDictionary*d=json(leasePath);NSDictionary*a=d[@"target_identity_ack"];
  if([a isKindOfClass:NSDictionary.class]&&[a[@"pid"] intValue]==getpid()&&[a[@"pgid"] intValue]==getpgrp()&&[a[@"run_dir"] isEqual:ns(outdir)]&&[a[@"token"] isEqual:ns(runToken)]){targetAck=immutable(a);return;}
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
 }
 throw std::runtime_error("independent target identity acknowledgement timeout before device");
}

#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include "model/Runtime.hpp"
#include "model/V024ExpectedGraph.hpp"
#include "model/QwenState.hpp"
#include "model/PrivateLegacyRtnDescriptor.hpp"
#include "ops/PageStorage.hpp"
#include "ops/Vision.hpp"
#include <filesystem>
#include <limits>
using namespace splash;using namespace splash::engine;
static void bytesFile(const std::string&name,const std::vector<uint8_t>&b){std::ofstream f(outdir+"/"+name,std::ios::binary);f.write((const char*)b.data(),b.size());check(bool(f),"write raw output");}
static std::string normalized(const std::string& x,size_t ordinal){
 check(ordinal<v024_expected::originalGraph.size(),"V024 graph ordinal");
 const std::string original(v024_expected::originalGraph[ordinal]);if(x==original)return original;
 for(const auto &s:v024_expected::substitutions)if(s.ordinal==ordinal){
  const std::string expected=std::string(s.candidate)+original.substr(s.original.size());
  if(x==expected)return original;
 }
 throw std::runtime_error("V024 unexpected name/ordinal/geometry/binding-size graph change");
}
static void equivalent(const model::Runtime::V002Result&a,const model::Runtime::V002Result&b){
 check(a.graph.size()==b.graph.size()&&a.bindings.size()==b.bindings.size()&&a.params==b.params,"full graph/parameter identity");
 for(size_t i=0;i<a.graph.size();++i)check(normalized(a.graph[i],i)==normalized(b.graph[i],i),"graph differs beyond selected kernel names");
 for(size_t i=0;i<a.bindings.size();++i)check(a.bindings[i].sameView(b.bindings[i]),"actual buffer view differs");
 check(a.outputs==b.outputs,"complete target output byte identity");
}
static void dump(const std::string&label,const model::Runtime::V002Result&r){
 NSMutableArray*entries=[NSMutableArray array];for(const auto &v:r.outputs){bytesFile(label+"-"+v.first,v.second);[entries addObject:@{@"name":ns(v.first),@"bytes":@(v.second.size()),@"sha256":hash([NSData dataWithBytesNoCopy:(void*)v.second.data() length:v.second.size() freeWhenDone:NO])}];}
 NSMutableArray*graph=[NSMutableArray array];for(const auto&v:r.graph)[graph addObject:ns(v)];NSDictionary*d=@{@"outputs":entries,@"graph":graph,@"selected_r233_dispatches":@(r.selected),@"selected_down_mapping_dispatches":@(r.selectedDown),@"ordinary_target_cb_gpu_seconds":@(r.gpuSeconds),@"ordinary_target_cb_wall_seconds":@(r.wallSeconds)};
 check([[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingPrettyPrinted error:nil] writeToFile:ns(outdir+"/"+label+".json") atomically:YES],"write qualification metadata");
}
static void runV002(){
 NSDictionary*lease=json(leasePath);std::string route=str(lease[@"route"]),mode=str(lease[@"mode"]);bool pq2=route=="pq2";check(pq2,"V024 fixed PQ2 complete-forward route");check(mode=="qualify"||mode=="screen","finite diagnostic mode");
 if(mode=="screen"){NSDictionary*q=lease[@"numeric_qualification"];expected(str(q[@"path"]),q[@"sha256"]);auto review=json(str(q[@"path"]));check([review[@"route"] isEqual:lease[@"route"]]&&[review[@"status"] isEqual:@"ACCEPTED_V024_FULL_TARGET_NUMERIC"]&&[review[@"host_sha256"] isEqual:lease[@"host_sha256"]]&&[review[@"input_manifest_sha256"] isEqual:lease[@"input_manifest_sha256"]],"root full-target numerical qualification");}
 printf("V024_PACKAGE ORIGINAL_vs_R233_HALF_gate_up_plus_fixed_S2_down_mapping; half_field=combined_candidate\n");
 const std::string B="/Users/zaiyang/bonsai-pq2-research/",S=B+"bonsai2-study/";
 const std::filesystem::path modelRoot=pq2?B+"splash-pq2/install/models/.resolved/7c5f9b68d81bd11dd98231f11d733aaf4d8fa7ed80b5e8c14e4db00547373b3a":S+"raw/task-r194-legacy-rtn-adapter-prep/assembly";
 guard();metal::MetalBackend backend(root+"combined.metallib");backend.setOperationGuard([]{guard();});backend.setDispatchProfiling(false);
 if(const auto e=backend.capabilities().validationError())throw std::runtime_error(*e);
 model::ModelDescriptor descriptor;
 if(pq2)descriptor=model::inspectModelPackage(modelRoot);else{
  // Same accepted descriptor combination; new canonical paths are separately stat-pinned by controller.
  const auto p=model::inspectModelPackage(B+"splash-pr114-bench/pkg/official");const auto s=model::inspectModelPackage(S+"raw/task-r194-legacy-rtn-adapter-prep/source-metadata");descriptor=model::private_r194::combine(p,s);
  model::private_r194::validateRevisions(model::private_r194::object(B+"splash-pr114-bench/pkg/official/manifest.json"),model::private_r194::object(S+"raw/task-r194-legacy-rtn-adapter-prep/source-metadata/model.json"));
 }
 const uint64_t reserve=EngineMemoryPolicy::hostAvailableReserveBytes(backend.capabilities().physicalMemoryBytes);auto available=queryHostAvailableMemory();check(available&&*available>reserve,"host memory reserve");
 check(model::preparedModelWeightBytes(modelRoot,descriptor)<*available-reserve,"weights above protected host reserve");
 auto package=model::loadModelPackage(backend,modelRoot,descriptor,[]{throw std::runtime_error("V002 refuses any cache preparation/miss");});
 ops::ExecutionPlans operators(backend.capabilities());auto planned=model::plannedRuntimeMemory(backend.capabilities(),package,operators,kv::Format::Int8);
 ModelMemoryFootprint footprint{package.targetActualAllocatedBytes(),package.draft.actualAllocatedBytes,package.vision.actualAllocatedBytes,package.stateLayout().activeCellBytes(),planned.sharedPrefillPlannedAllocatedBytes,planned.sharedDecodePlannedAllocatedBytes,planned.pipelineReserveBytes,planned.runtimeOverheadReserveBytes};
 ModelMemoryProfile profile{package.name(),package.maximumContextTokens(),package.targetKvLayout(kv::Format::Int8),footprint};auto plan=requireEngineMemoryPlan(backend.capabilities(),profile);const auto&budget=plan.breakdown();
 MemoryGovernor governor(backend,budget.hardBudgetBytes-budget.pipelineReserveBytes-budget.runtimeOverheadReserveBytes,reserve);auto admission=governor.allocationAdmission();
 const auto pageCount=std::max(128U,package.targetKvLayout(kv::Format::Int8).sparseMappingBatchPages());kv::PageStorage pages(backend,admission,package.targetKvLayout(kv::Format::Int8),pageCount);model::QwenStateStorage states(backend,admission,package.stateLayout());
 model::RuntimeContext context{backend,admission,package,pages,states,operators,ops::kMaximumImagePatches,budget.pipelineReserveBytes,budget.runtimeOverheadReserveBytes};
 auto reservation=governor.tryReserve(planned.sharedPrefillPlannedAllocatedBytes+planned.sharedDecodePlannedAllocatedBytes);check(reservation.has_value(),"runtime arena reservation");model::Runtime runtime(context);reservation->commit();
 std::vector<uint32_t>prompt(128),tokens(8),pageTable{0,1,2,3,4};for(uint32_t i=0;i<128;++i)prompt[i]=i+1;for(uint32_t i=0;i<8;++i)tokens[i]=i+129;
 for(auto p:pageTable)check(bool(pages.ensureResident(p)),"five KV pages");ModelRequest request;request.id=9001;request.prompt=prompt;request.maxNewTokens=64;runtime.beginColdRequest(request,0);
 BatchPlan pp{WorkKind::Prefill,BatchCohort::Greedy,{{9001,128}},DecodeStage::Regular};ModelBatchItem pi{9001,0,0,0,128,pageTable};pi.inputTokens=prompt;
 static_cast<void>(runtime.prefill(pp,{&pi,1}));runtime.v002FreezeKV();ModelBatchItem item{9001,0,128,0,0,pageTable};
 auto base=runtime.v002Forward(item,tokens,false);dump("qualification-stock",base);uint32_t forwards=1;
 if(pq2){auto half=runtime.v002Forward(item,tokens,true,true);++forwards;dump("qualification-candidate",half);equivalent(base,half);}
 auto repeat=runtime.v002Forward(item,tokens,false);++forwards;dump("qualification-repeat",repeat);equivalent(base,repeat);
 if(mode=="screen"){
  for(int i=0;i<2;++i){auto result=runtime.v002Forward(item,tokens,pq2&&i==1,pq2&&i==1);equivalent(base,result);++forwards;}
  // Eight ABBA/BAAB blocks; RTN has the same 32 same-route samples as an A/A stability control.
  for(int block=0;block<8;++block)for(int pos=0;pos<4;++pos){bool half=pq2&&((block%2==0)?(pos==1||pos==2):(pos==0||pos==3));auto result=runtime.v002Forward(item,tokens,half,half);equivalent(base,result);++forwards;
   printf("V024_SAMPLE {\"model\":\"%s\",\"block\":%d,\"position\":%d,\"half\":%s,\"gpu_seconds\":%.12g,\"wall_seconds\":%.12g,\"target_dispatches\":%zu,\"selected\":%u,\"thermal\":%ld}\n",route.c_str(),block,pos,half?"true":"false",result.gpuSeconds,result.wallSeconds,result.graph.size(),result.selected,(long)NSProcessInfo.processInfo.thermalState);
  }
 }
 runtime.end(9001);guard();printf("R219_COMPLETE {\"target_forwards\":%u,\"checks_passed\":true,\"mode\":\"%s\",\"route\":\"%s\",\"prefix_tokens\":128,\"verify_rows\":8,\"prefill_timed\":false,\"draft_timed\":false,\"profiling\":false}\n",forwards,mode.c_str(),route.c_str());
}
int main(int argc,char**argv){@autoreleasepool{try{
 setbuf(stdout,nullptr);check(argc==2,"one fresh output directory");initializePaths();outdir=argv[1];check(std::filesystem::is_directory(outdir),"fresh directory created by controller");const char*l=getenv("R219_LEASE");check(l,"explicit separate lease");leasePath=l;guard(true);
 printf("R219_OWNER {\"pid\":%d,\"ppid\":%d,\"pgid\":%d,\"run_dir\":\"%s\",\"token\":\"%s\"}\n",getpid(),getppid(),getpgrp(),outdir.c_str(),runToken.c_str());awaitIdentityAck();runV002();return 0;
 }catch(const std::exception&e){fprintf(stderr,"V002_FAILURE %s\n",e.what());return 1;}}}
