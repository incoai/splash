#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if(argc<3)throw std::invalid_argument("usage: pipeline-resources METALLIB KERNEL...");
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
      NSError *error=nil;
      id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
      if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
      for(int i=2;i<argc;++i) {
        NSString *name=@(argv[i]);id<MTLFunction> fn=[library newFunctionWithName:name];
        if(!fn)throw std::runtime_error("missing kernel: "+std::string(argv[i]));
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:fn error:&error];
        if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
        std::cout<<"{\"kernel\":\""<<argv[i]<<"\",\"static_threadgroup_bytes\":"<<pipeline.staticThreadgroupMemoryLength
                 <<",\"maximum_threads\":"<<pipeline.maxTotalThreadsPerThreadgroup
                 <<",\"execution_width\":"<<pipeline.threadExecutionWidth
                 <<",\"device_threadgroup_limit_bytes\":"<<device.maxThreadgroupMemoryLength<<"}\n";
      }
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
  }
}
