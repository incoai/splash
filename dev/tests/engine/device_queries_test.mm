#include "metal/DeviceQueries.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

@interface SparseDevice : NSObject
@property BOOL supportsPlacementSparse;
@end

@implementation SparseDevice
@end

@interface ForwardingDevice : NSObject
- (BOOL)supportsPlacementSparse;
@end

@implementation ForwardingDevice
- (BOOL)supportsPlacementSparse {
    id<SplashPlacementSparseDevice> backing = (id<SplashPlacementSparseDevice>)[NSObject new];
    return backing.supportsPlacementSparse;
}
@end

@interface FailingDevice : NSObject
- (BOOL)supportsPlacementSparse;
@end

@implementation FailingDevice
- (BOOL)supportsPlacementSparse {
    @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                  reason:@"driver query failed"
                                userInfo:nil];
}
@end

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void requireQueryError(id device, const char *name, const char *reason) {
    require([device respondsToSelector:@selector(supportsPlacementSparse)],
            "failing device must advertise the selector");
    try {
        splash::metal::queryPlacementSparseSupport((id<MTLDevice>)device);
    } catch (const splash::metal::MetalBackendError &error) {
        const std::string message = error.what();
        require(message.find("supportsPlacementSparse") != std::string::npos,
                "query error must identify the capability");
        require(message.find(name) != std::string::npos &&
                    message.find(reason) != std::string::npos,
                "query error must preserve the driver exception");
        return;
    }
    require(false, "query failure must become a MetalBackendError");
}

} // namespace

int main() {
    @autoreleasepool {
        using splash::metal::queryPlacementSparseSupport;
        require(!queryPlacementSparseSupport((id<MTLDevice>)[NSObject new]),
                "a missing selector must report unsupported");
        SparseDevice *device = [SparseDevice new];
        require(!queryPlacementSparseSupport((id<MTLDevice>)device),
                "a false capability must report unsupported");
        device.supportsPlacementSparse = YES;
        require(queryPlacementSparseSupport((id<MTLDevice>)device),
                "a supported capability must remain supported");
        requireQueryError([ForwardingDevice new],
                          NSInvalidArgumentException.UTF8String,
                          "unrecognized selector");
        requireQueryError([FailingDevice new],
                          NSInternalInconsistencyException.UTF8String,
                          "driver query failed");
    }
    std::cout << "device queries: PASS\n";
}
