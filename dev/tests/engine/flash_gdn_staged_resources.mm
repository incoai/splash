// Pipeline metadata only: this program never creates a command queue,
// command buffer, encoder, or GPU dispatch.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])]
                                               error:&error];
    if (!device || !library) {
      std::cerr << (error.localizedDescription.UTF8String ?: "no Metal device/library") << '\n';
      return 1;
    }
    NSMutableArray *pipelines = [NSMutableArray array];
    for (NSString *name in @[@"flash_gdn_staged_v8_t16", @"flash_gdn_staged_v8_t32",
                             @"flash_gdn_staged_v16_t16", @"flash_gdn_staged_v16_t32"]) {
      id<MTLFunction> function = [library newFunctionWithName:name];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline) {
        [pipelines addObject:@{@"name": name, @"error": error.localizedDescription ?: @"creation failed"}];
        continue;
      }
      [pipelines addObject:@{@"name": name,
          @"static_threadgroup_memory_bytes": @(pipeline.staticThreadgroupMemoryLength),
          @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
          @"execution_width": @(pipeline.threadExecutionWidth),
          @"memory_supported": @(pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength)}];
    }
    NSDictionary *report = @{@"device": device.name,
        @"max_threadgroup_memory_bytes": @(device.maxThreadgroupMemoryLength),
        @"pipelines": pipelines, @"gpu_commands": @0};
    NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                  options:NSJSONWritingPrettyPrinted error:&error];
    std::cout.write(static_cast<const char *>(json.bytes), json.length);
    std::cout << '\n';
  }
  return 0;
}
