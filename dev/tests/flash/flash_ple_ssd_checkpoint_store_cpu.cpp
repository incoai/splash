// Bounded actual checkpoint proof. No mapping, model execution or GPU.
#include "flash/FlashPLESSDStore.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Store=splash::flash::FlashPLESSDStore;
template<class T>T read(std::ifstream&input){T v{};input.read(reinterpret_cast<char*>(&v),sizeof(v));if(!input)throw std::runtime_error("short fixture");return v;}
Store::Plane plane(std::ifstream&i){return{read<uint32_t>(i),read<uint64_t>(i),read<uint64_t>(i)};}
int main(int argc,char**argv){try{
  if(argc!=3)throw std::invalid_argument("usage: checkpoint-store-oracle FIXTURE OUTPUT");
  std::ifstream input(argv[1],std::ios::binary);
  const auto n=read<uint32_t>(input);
  std::vector<Store::Source>sources;
  for(unsigned i=0;i<n;++i){const auto bytes=read<uint64_t>(input),length=read<uint64_t>(input);std::string path(length,'\0');input.read(path.data(),length);sources.push_back({path,bytes});}
  const auto parts=read<uint32_t>(input);std::vector<Store::Part>table;
  for(unsigned i=0;i<parts;++i){const auto rows=read<uint64_t>(input);const auto w=plane(input),s=plane(input),b=plane(input);table.push_back({rows,w,s,b});}
  const auto count=read<uint32_t>(input);std::vector<int64_t>ids(count);for(auto&id:ids)id=read<int64_t>(input);
  Store store(sources,table,{64ULL*1024*1024,1024,true});std::vector<uint8_t>output(count*100);store.lookupRows(ids,output);
  const auto first=store.statistics();store.lookupRows(ids,output);const auto second=store.statistics();
  if(first.requestedReadBytes!=second.requestedReadBytes||second.cacheHitRows!=count||second.poisoned)throw std::runtime_error("checkpoint row warm cache failed");
  std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<const char*>(output.data()),output.size());if(!out)throw std::runtime_error("output write failed");
  std::cout<<"{\"valid\":true,\"gpu_executed\":false,\"rows\":"<<count<<",\"table_rows\":"<<store.tableRows()<<",\"read_requests\":"<<first.readRequests<<",\"read_bytes\":"<<first.completedReadBytes<<",\"warm_cache_hit_rows\":"<<second.cacheHitRows<<",\"cache_budget_bytes\":"<<second.cacheBudgetBytes<<",\"cache_accounted_bytes\":"<<second.cacheAccountedBytes<<",\"cached_rows\":"<<second.cachedRows<<",\"file_cache_bypass\":"<<(second.fileCacheBypassEnabled?"true":"false")<<"}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
