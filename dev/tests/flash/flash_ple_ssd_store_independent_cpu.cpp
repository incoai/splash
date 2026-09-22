// Independent byte-level reference for the production SSD row reader/cache.
// Only temporary small files are created; no checkpoint or GPU is touched.
#include "flash/FlashPLESSDStore.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using Store = splash::flash::FlashPLESSDStore;
namespace {
uint64_t checks = 0, compared = 0, randomizedBatches = 0;
void require(bool b, const char *why) { ++checks; if (!b) throw std::runtime_error(why); }
struct Fixture {
  std::filesystem::path root;
  std::array<std::array<uint8_t,100>,200> expected{};
  std::vector<Store::Source> sources;
  std::vector<Store::Part> parts;
  explicit Fixture(const std::filesystem::path &path) : root(path) {
    std::filesystem::create_directories(root);
    std::array<std::vector<uint8_t>,3> planes;
    planes[0].resize(200*80); planes[1].resize(200*10); planes[2].resize(200*10);
    for (unsigned row = 0; row < 200; ++row) {
      for (unsigned column = 0; column < 100; ++column)
        expected[row][column] = uint8_t(row*67 + column*29 + (row ^ column)*11);
      std::copy_n(expected[row].data(),80,planes[0].data()+row*80);
      std::copy_n(expected[row].data()+80,10,planes[1].data()+row*10);
      std::copy_n(expected[row].data()+90,10,planes[2].data()+row*10);
    }
    for (unsigned p = 0; p < 3; ++p) {
      const auto file = root / ("plane" + std::to_string(p) + ".bin");
      std::ofstream output(file,std::ios::binary);
      output.write(reinterpret_cast<const char *>(planes[p].data()),planes[p].size());
      output.close();
      sources.push_back({file,planes[p].size()});
    }
    for (unsigned part = 0; part < 4; ++part)
      parts.push_back({50,{0,part*50*80,80},{1,part*50*10,10},{2,part*50*10,10}});
  }
  std::unique_ptr<Store> store(uint64_t cache, uint64_t scratch, bool noCache=true) {
    return std::make_unique<Store>(sources,parts,Store::Options{cache,scratch,noCache});
  }
  void compare(Store &store, const std::vector<int64_t> &ids) {
    constexpr unsigned guard=64;
    std::vector<uint8_t> out(ids.size()*100+2*guard,0xa5);
    store.lookupRows(ids,std::span<uint8_t>(out.data()+guard,ids.size()*100));
    for (unsigned i=0;i<guard;++i)
      require(out[i]==0xa5&&out[out.size()-1-i]==0xa5,"store overwrote output guard");
    for (size_t row=0;row<ids.size();++row) {
      for (unsigned column=0;column<100;++column)
        require(out[guard+row*100+column]==expected[ids[row]][column],"store differs from independent row bytes");
      compared+=100;
    }
    const auto stats=store.statistics();
    require(stats.cacheAccountedBytes<=stats.cacheBudgetBytes,"cache exceeded admitted memory");
    require(stats.completedReadBytes==stats.requestedReadBytes,"successful reads did not complete");
    require(!stats.poisoned,"valid lookup poisoned store");
  }
};

void randomized(Fixture &f,uint64_t cache,uint64_t scratch) {
  auto store=f.store(cache,scratch);
  require(store->tableRows()==200&&store->shardRows()==50&&store->partCount()==4,"store geometry differs");
  f.compare(*store,{0,49,50,99,100,149,150,199,0,199,50,50});
  auto before=store->statistics();
  f.compare(*store,{50,50,199,0,150,149,100,99,50,49});
  auto after=store->statistics();
  if(cache>=1024*1024)
    require(after.requestedReadBytes==before.requestedReadBytes,"warm rows caused reads");
  std::mt19937_64 rng(200100+cache+scratch);
  for(unsigned batch=0;batch<400;++batch) {
    const unsigned rows=unsigned(rng()%193);
    std::vector<int64_t> ids(rows);
    for(unsigned row=0;row<rows;++row)
      ids[row]=rng()%5==0&&row?ids[row-1]:int64_t(rng()%200);
    if(batch%37==0)store->clearCache();
    f.compare(*store,ids);++randomizedBatches;
  }
  auto stats=store->statistics();
  require(stats.duplicateMissRows>0,"duplicates were not deduplicated");
  if(cache==0)require(stats.cacheAccountedBytes==0&&stats.cachedRows==0&&stats.cacheHitRows==0,"zero cache cached rows");
  if(cache>0&&cache<4096)require(stats.cacheEvictions>0,"small cache did not evict under churn");
}

void measuredDedup(Fixture &f) {
  auto store=f.store(0,1024);
  f.compare(*store,{199,0,0,1,1,1,198,199,0});
  auto stats=store->statistics();
  require(stats.requestedRows==9&&stats.uniqueMissRows==4&&stats.duplicateMissRows==5,"miss sort/dedup accounting wrong");
  require(stats.requestedReadBytes==400&&stats.completedReadBytes==400,"dedup did not read exactly four 100B rows");
  // Two adjacent runs per plane (0..1 and198..199) => six reads.
  require(stats.readRequests==6,"coalescing did not preserve independent plane adjacency");
  f.compare(*store,{199,0,0,1,1,1,198,199,0});
  stats=store->statistics();
  require(stats.requestedReadBytes==800&&stats.uniqueMissRows==8,"zero cache second batch accounting wrong");
}

void invalidRows(Fixture &f) {
  auto store=f.store(1024*1024,128);
  f.compare(*store,{0,199});
  for(int64_t invalid:{INT64_MIN,int64_t{-1},int64_t{200},INT64_MAX}) {
    const auto before=store->statistics();
    std::vector<int64_t> ids{0,invalid,199};
    std::vector<uint8_t> output(300,0x39);
    bool caught=false;
    try{store->lookupRows(ids,output);}catch(const std::runtime_error&){caught=true;}
    require(caught,"invalid source row accepted");
    require(std::all_of(output.begin(),output.end(),[](auto x){return x==0x39;}),"invalid source row wrote partial output");
    const auto after=store->statistics();
    require(!after.poisoned&&after.failedBatches==before.failedBatches+1&&
            after.requestedReadBytes==before.requestedReadBytes,"invalid source row poisoned or read source");
    f.compare(*store,{199,0});
  }
  const std::vector<int64_t> ids{0};
  std::vector<uint8_t> output(99,0x39);
  bool caught=false;
  try{store->lookupRows(ids,output);}catch(const std::runtime_error&){caught=true;}
  require(caught&&!store->statistics().poisoned,"bad output extent did not reject cleanly");
}

void concurrency(Fixture &f) {
  auto store=f.store(4096,128);
  std::array<std::exception_ptr,4> errors;
  std::array<std::thread,4> threads;
  // These workers use independent output buffers, one shared immutable source
  // and bounded cache. Each checks bytes locally; main owns global counters.
  for(unsigned lane=0;lane<4;++lane)threads[lane]=std::thread([&,lane]{
    try {
      std::mt19937_64 rng(lane+91);
      for(unsigned batch=0;batch<100;++batch){
        std::vector<int64_t>ids(32);for(auto &id:ids)id=rng()%200;
        std::vector<uint8_t>out(3200);store->lookupRows(ids,out);
        for(size_t row=0;row<32;++row)
          if(!std::equal(f.expected[ids[row]].begin(),f.expected[ids[row]].end(),out.begin()+row*100))
            throw std::runtime_error("concurrent store output differs");
      }
    }catch(...){errors[lane]=std::current_exception();}
  });
  for(auto &thread:threads)thread.join();
  for(auto &error:errors)require(!error,"concurrent store worker failed");
  const auto stats=store->statistics();
  require(stats.preparedBatches==400&&stats.requestedRows==12800&&!stats.poisoned,
          "concurrent store accounting differs");
  compared+=12800*100;
}
} // namespace

int main(int argc,char**argv){
  try {
    if(argc!=2)throw std::invalid_argument("usage: store-oracle TEMP_DIRECTORY");
    Fixture f(argv[1]);
    measuredDedup(f);invalidRows(f);
    for(uint64_t cache:{0ULL,1024ULL,4096ULL,1048576ULL})
      for(uint64_t scratch:{80ULL,128ULL,1024ULL})randomized(f,cache,scratch);
    concurrency(f);
    std::cout<<"{\"valid\":true,\"gpu_executed\":false,\"fixture_rows\":200,\"row_bytes\":100,\"fixture_total_bytes\":20000,\"checks\":"
      <<checks<<",\"compared_bytes\":"<<compared<<",\"randomized_batches\":"<<randomizedBatches
      <<",\"concurrent_batches\":400,\"cache_budgets\":[0,1024,4096,1048576],\"read_scratch_limits\":[80,128,1024]}\n";
    return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
