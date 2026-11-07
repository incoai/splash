#pragma once

// Method replacements through which tests observe and steer what Metal
// objects do, and the recorders of command queues' residency sets and of
// compute encoders' resource declarations built on them. Objective-C++ only.

#include "TestChecks.hpp"

#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <string>
#include <vector>

namespace splash::test {

// Replaces a method of the class of `object`, every object's of that class,
// while it lives.
class MethodReplacement final {
public:
    MethodReplacement(id object, SEL selector, IMP replacement)
        : method_(class_getInstanceMethod(object_getClass(object), selector)) {
        require(method_ != nullptr, "replacement method is missing");
        original = method_setImplementation(method_, replacement);
    }
    ~MethodReplacement() { method_setImplementation(method_, original); }
    MethodReplacement(const MethodReplacement &) = delete;
    MethodReplacement &operator=(const MethodReplacement &) = delete;

    IMP original = nullptr;

private:
    Method method_ = nullptr;
};

// Records the residency sets added to command queues while it lives: Metal
// offers no way to read a queue's sets back. Arm it before the backend it
// observes.
class QueueSets final {
public:
    QueueSets() {
        original = replacement_.original;
        added.clear();
    }

    // Whether a set added holds the buffer labeled `label`.
    [[nodiscard]] bool holds(NSString *label) const {
        for (id<MTLResidencySet> set : added)
            for (id<MTLAllocation> allocation in set.allAllocations)
                if ([((id<MTLBuffer>)allocation).label isEqualToString:label]) return true;
        return false;
    }

    static inline std::vector<id<MTLResidencySet>> added;

private:
    static inline IMP original = nullptr;

    static void recordAdd(id queue, SEL selector, id<MTLResidencySet> set) {
        added.push_back(set);
        reinterpret_cast<void (*)(id, SEL, id)>(original)(queue, selector, set);
    }

    id<MTLCommandQueue> queue_ = [MTLCreateSystemDefaultDevice() newCommandQueue];
    MethodReplacement replacement_{queue_, @selector(addResidencySet:), reinterpret_cast<IMP>(recordAdd)};
};

// Records the resources compute encoders make resident for their dispatches
// (useResources:count:usage:) while it lives, call by call: Metal offers no
// way to read them back. Arm it before the commands it observes.
class EncoderUses final {
public:
    struct Use {
        NSUInteger count;
        MTLResourceUsage usage;
        std::vector<std::string> labels;
    };

    EncoderUses() {
        original = replacement_.original;
        uses.clear();
    }

    // Whether any call recorded declared the resource labeled `label`.
    [[nodiscard]] bool declares(NSString *label) const {
        for (const Use &use : uses)
            for (const std::string &each : use.labels)
                if ([label isEqualToString:@(each.c_str())]) return true;
        return false;
    }

    static inline std::vector<Use> uses;

private:
    using UseResources = void (*)(id, SEL, const __unsafe_unretained id<MTLResource> *, NSUInteger,
                                  MTLResourceUsage);
    static inline IMP original = nullptr;

    static void recordUse(id encoder, SEL selector, const __unsafe_unretained id<MTLResource> *resources,
                          NSUInteger count, MTLResourceUsage usage) {
        Use &use = uses.emplace_back(Use{count, usage, {}});
        for (NSUInteger index = 0; index < count; ++index)
            use.labels.emplace_back(resources[index].label ? resources[index].label.UTF8String : "");
        reinterpret_cast<UseResources>(original)(encoder, selector, resources, count, usage);
    }

    // An encoder of the class the backend's are, which Metal picks per
    // device and validation setting.
    static id<MTLComputeCommandEncoder> probe() {
        id<MTLCommandBuffer> command = [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder endEncoding];
        return encoder;
    }

    MethodReplacement replacement_{probe(), @selector(useResources:count:usage:), reinterpret_cast<IMP>(recordUse)};
};

} // namespace splash::test
