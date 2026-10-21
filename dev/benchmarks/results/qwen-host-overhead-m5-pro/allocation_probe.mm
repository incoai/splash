#include "metal/CommandGraph.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
thread_local bool counting = false;
thread_local unsigned long allocations = 0;
void *operator new(std::size_t n) { if (counting) ++allocations; if (void *p=std::malloc(n)) return p; throw std::bad_alloc(); }
void operator delete(void *p) noexcept { std::free(p); }
void build() {
  splash::metal::CommandGraph graph;
  for (unsigned i=0;i<2048;++i) {
    struct Params { unsigned index, rows, stride, reserved; } params{i,8,4096,0};
    graph.add("synthetic_graph_dispatch", {{},{},{},{},{},{}}, params, {16,1,1});
  }
  if (graph.dispatches().size()!=2048) std::abort();
}
int main() {
  build(); allocations=0; counting=true;
  const auto begin=std::chrono::steady_clock::now();
  for (unsigned i=0;i<100;++i) build();
  const auto end=std::chrono::steady_clock::now(); counting=false;
  std::printf("{\"dispatches\":204800,\"allocations\":%lu,\"seconds\":%.9f}\n",allocations,std::chrono::duration<double>(end-begin).count());
}
