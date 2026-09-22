#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <sstream>
int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    std::cout << "device: " << dev.name.UTF8String << " metal4=" << [dev supportsFamily:(MTLGPUFamily)5002] << " apple9=" << [dev supportsFamily:MTLGPUFamilyApple9] << "\n";
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf();
    MTLCompileOptions *opt = [MTLCompileOptions new];
    opt.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);   // MTLLanguageVersion4_0
    opt.mathMode = MTLMathModeFast;
    NSError *err = nil;
    NSDate *t0 = [NSDate date];
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:ss.str().c_str()] options:opt error:&err];
    if (!lib) { std::cout << "COMPILE FAILED: " << err.localizedDescription.UTF8String << "\n"; return 1; }
    std::cout << "compiled in " << -[t0 timeIntervalSinceNow] << " s; functions: " << lib.functionNames.count << "\n";
    for (NSString *n in lib.functionNames) {
      id<MTLFunction> fn = [lib newFunctionWithName:n];
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!p) { std::cout << "  pipeline FAILED " << n.UTF8String << ": " << err.localizedDescription.UTF8String << "\n"; continue; }
      std::cout << "  ok " << n.UTF8String << " maxThreads=" << p.maxTotalThreadsPerThreadgroup << " tgmem=" << p.staticThreadgroupMemoryLength << "\n";
    }
  }
  return 0;
}
