#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::kv;

namespace {

id<MTLBuffer> buffer(id<MTLDevice> device, uint64_t bytes) {
    id<MTLBuffer> result = [device
        newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal buffer allocation failed");
    return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     const char *name) {
    NSString *functionName = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function) throw std::runtime_error(std::string("missing kernel: ") + name);
    NSError *error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        throw std::runtime_error(error.localizedDescription.UTF8String);
    }
    return result;
}

void finish(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        std::string message = "Metal command failed";
        if (command.error) {
            message += ": ";
            message += command.error.localizedDescription.UTF8String;
        }
        throw std::runtime_error(message);
    }
}

void requireEqual(const void *left, const void *right, uint64_t bytes,
                  const char *name) {
    const auto *observed = static_cast<const uint8_t *>(left);
    const auto *expected = static_cast<const uint8_t *>(right);
    for (uint64_t index = 0; index < bytes; ++index) {
        if (observed[index] != expected[index]) {
            throw std::runtime_error(
                std::string(name) + " differs from CPU reference at byte " +
                std::to_string(index) + " (Metal=" +
                std::to_string(observed[index]) + ", CPU=" +
                std::to_string(expected[index]) + ", Metal-u16=" +
                std::to_string(static_cast<const uint16_t *>(left)[index / 2]) +
                ", CPU-u16=" +
                std::to_string(static_cast<const uint16_t *>(right)[index / 2]) +
                ")");
        }
    }
}

void requireFloatClose(const float *left, const float *right, uint64_t count,
                       const char *name) {
    for (uint64_t index = 0; index < count; ++index) {
        float tolerance = std::max(1.0e-8f, std::abs(right[index]) * 2.0e-6f);
        if (!std::isfinite(left[index]) ||
            std::abs(left[index] - right[index]) > tolerance) {
            throw std::runtime_error(
                std::string(name) + " differs from CPU reference at " +
                std::to_string(index));
        }
    }
}

void run(const char *libraryPath) {
    constexpr uint32_t validTokens = 17;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device unavailable");
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:
        [NSString stringWithUTF8String:libraryPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (!library) {
        throw std::runtime_error(error.localizedDescription.UTF8String);
    }
    auto quantize = pipeline(
        device, library, "splash_q8_quantize_kv_page");
    auto dequantize = pipeline(
        device, library, "splash_q8_dequantize_kv_page");
    auto gather = pipeline(
        device, library, "splash_q8_gather_logical_kv_page");

    std::vector<float> logicalKeys(
        uint64_t{validTokens} * kKvHeads * kHeadDimension);
    std::vector<float> logicalValues(logicalKeys.size());
    std::vector<BFloat16Bits> physicalKeys(kElementsPerLayerPage);
    std::vector<BFloat16Bits> physicalValues(kElementsPerLayerPage);
    for (uint32_t token = 0; token < validTokens; ++token) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            for (uint32_t dimension = 0; dimension < kHeadDimension;
                 ++dimension) {
                uint64_t logical = logicalIndex(token, head, dimension);
                float key = float(int((token * 17 + head * 23 + dimension * 7) %
                                      1009) - 504) / 113.0f;
                float value = float(int((token * 29 + head * 13 + dimension * 11) %
                                        1013) - 506) / 97.0f;
                BFloat16Bits keyBits = floatToBFloat16(key);
                BFloat16Bits valueBits = floatToBFloat16(value);
                logicalKeys[logical] = bfloat16ToFloat(keyBits);
                logicalValues[logical] = bfloat16ToFloat(valueBits);
                physicalKeys[splash_kv_key_element(head, token, dimension)] =
                    keyBits;
                physicalValues[splash_kv_value_element(head, token, dimension)] =
                    valueBits;
            }
        }
    }

    auto reference = std::make_unique<Q8LayerPage>();
    quantizeLayerPage(
        logicalKeys, logicalValues, validTokens, *reference);

    id<MTLBuffer> sourceKeys = buffer(
        device, kElementsPerLayerPage * sizeof(BFloat16Bits));
    id<MTLBuffer> sourceValues = buffer(
        device, kElementsPerLayerPage * sizeof(BFloat16Bits));
    id<MTLBuffer> q8Keys = buffer(device, kKeyDataBytesPerLayerPage);
    id<MTLBuffer> keyScales = buffer(device, kKeyScaleBytesPerLayerPage);
    id<MTLBuffer> q8Values = buffer(device, kValueDataBytesPerLayerPage);
    id<MTLBuffer> valueScales = buffer(device, kValueScaleBytesPerLayerPage);
    std::memcpy(sourceKeys.contents, physicalKeys.data(), sourceKeys.length);
    std::memcpy(sourceValues.contents, physicalValues.data(), sourceValues.length);

    Q8KVMetalPageParams params{0, 0, validTokens, 0};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> quantizeCommand = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [quantizeCommand computeCommandEncoder];
    [encoder setComputePipelineState:quantize];
    [encoder setBuffer:sourceKeys offset:0 atIndex:0];
    [encoder setBuffer:sourceValues offset:0 atIndex:1];
    [encoder setBuffer:q8Keys offset:0 atIndex:2];
    [encoder setBuffer:keyScales offset:0 atIndex:3];
    [encoder setBuffer:q8Values offset:0 atIndex:4];
    [encoder setBuffer:valueScales offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreadgroups:
                 MTLSizeMake(2 * kScalesPerTensorLayerPage, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kHeadDimension, 1, 1)];
    [encoder endEncoding];
    finish(quantizeCommand);

    requireEqual(q8Keys.contents, reference->keys.data(), q8Keys.length,
                 "Q8 keys");
    requireFloatClose(static_cast<const float *>(keyScales.contents),
                      reference->keyScales.data(),
                      kScalesPerTensorLayerPage, "Q8 key scales");
    requireFloatClose(static_cast<const float *>(valueScales.contents),
                      reference->valueScales.data(),
                      kScalesPerTensorLayerPage, "Q8 value scales");
    requireEqual(q8Values.contents, reference->values.data(), q8Values.length,
                 "Q8 values");

    id<MTLBuffer> decodedKeys = buffer(device, sourceKeys.length);
    id<MTLBuffer> decodedValues = buffer(device, sourceValues.length);
    id<MTLBuffer> logicalGatherKeys = buffer(device, sourceKeys.length);
    id<MTLBuffer> logicalGatherValues = buffer(device, sourceValues.length);
    id<MTLCommandBuffer> decodeCommand = [queue commandBuffer];
    encoder = [decodeCommand computeCommandEncoder];
    [encoder setComputePipelineState:dequantize];
    [encoder setBuffer:q8Keys offset:0 atIndex:0];
    [encoder setBuffer:keyScales offset:0 atIndex:1];
    [encoder setBuffer:q8Values offset:0 atIndex:2];
    [encoder setBuffer:valueScales offset:0 atIndex:3];
    [encoder setBuffer:decodedKeys offset:0 atIndex:4];
    [encoder setBuffer:decodedValues offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(2 * kElementsPerLayerPage, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:gather];
    [encoder setBuffer:q8Keys offset:0 atIndex:0];
    [encoder setBuffer:keyScales offset:0 atIndex:1];
    [encoder setBuffer:q8Values offset:0 atIndex:2];
    [encoder setBuffer:valueScales offset:0 atIndex:3];
    [encoder setBuffer:logicalGatherKeys offset:0 atIndex:4];
    [encoder setBuffer:logicalGatherValues offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(kElementsPerLayerPage, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    finish(decodeCommand);

    std::vector<float> decodedLogicalKeys(kElementsPerLayerPage);
    std::vector<float> decodedLogicalValues(kElementsPerLayerPage);
    dequantizeLayerPage(*reference, validTokens,
                        decodedLogicalKeys, decodedLogicalValues);
    std::vector<BFloat16Bits> expectedLogicalKeys(kElementsPerLayerPage);
    std::vector<BFloat16Bits> expectedLogicalValues(kElementsPerLayerPage);
    std::vector<BFloat16Bits> expectedPhysicalKeys(kElementsPerLayerPage);
    std::vector<BFloat16Bits> expectedPhysicalValues(kElementsPerLayerPage);
    for (uint32_t token = 0; token < kPageTokens; ++token) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            for (uint32_t dimension = 0; dimension < kHeadDimension;
                 ++dimension) {
                uint64_t logical = logicalIndex(token, head, dimension);
                BFloat16Bits key = floatToBFloat16(decodedLogicalKeys[logical]);
                BFloat16Bits value =
                    floatToBFloat16(decodedLogicalValues[logical]);
                expectedLogicalKeys[logical] = key;
                expectedLogicalValues[logical] = value;
                expectedPhysicalKeys[
                    splash_kv_key_element(head, token, dimension)] = key;
                expectedPhysicalValues[
                    splash_kv_value_element(head, token, dimension)] = value;
            }
        }
    }
    requireEqual(decodedKeys.contents, expectedPhysicalKeys.data(),
                 decodedKeys.length, "dequantized physical keys");
    requireEqual(decodedValues.contents, expectedPhysicalValues.data(),
                 decodedValues.length, "dequantized physical values");
    requireEqual(logicalGatherKeys.contents, expectedLogicalKeys.data(),
                 logicalGatherKeys.length, "gathered logical keys");
    requireEqual(logicalGatherValues.contents, expectedLogicalValues.data(),
                 logicalGatherValues.length, "gathered logical values");

    // A uniform tensor checks exact CPU/Metal agreement for the common
    // per-token/head scale and saturation path.
    std::fill(logicalValues.begin(), logicalValues.end(), 1.0f);
    std::fill(physicalValues.begin(), physicalValues.end(), BFloat16Bits{0});
    for (uint32_t token = 0; token < validTokens; ++token) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            for (uint32_t dimension = 0; dimension < kHeadDimension;
                 ++dimension) {
                physicalValues[splash_kv_value_element(head, token, dimension)] =
                    floatToBFloat16(1.0f);
            }
        }
    }
    quantizeLayerPage(logicalKeys, logicalValues, validTokens, *reference);
    std::memcpy(sourceValues.contents, physicalValues.data(), sourceValues.length);
    quantizeCommand = [queue commandBuffer];
    encoder = [quantizeCommand computeCommandEncoder];
    [encoder setComputePipelineState:quantize];
    [encoder setBuffer:sourceKeys offset:0 atIndex:0];
    [encoder setBuffer:sourceValues offset:0 atIndex:1];
    [encoder setBuffer:q8Keys offset:0 atIndex:2];
    [encoder setBuffer:keyScales offset:0 atIndex:3];
    [encoder setBuffer:q8Values offset:0 atIndex:4];
    [encoder setBuffer:valueScales offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreadgroups:
                 MTLSizeMake(2 * kScalesPerTensorLayerPage, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kHeadDimension, 1, 1)];
    [encoder endEncoding];
    finish(quantizeCommand);
    requireEqual(q8Values.contents, reference->values.data(), q8Values.length,
                 "same-sign maximum Q8 values");
    requireFloatClose(static_cast<const float *>(valueScales.contents),
                      reference->valueScales.data(),
                      kScalesPerTensorLayerPage,
                      "same-sign maximum Q8 value scales");
}

}  // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 2) {
                std::cerr << "usage: q8_paged_kv_metal_test METALLIB\n";
                return 2;
            }
            run(argv[1]);
            std::cout << "q8_paged_kv_metal_test: ok\n";
        } catch (const std::exception &error) {
            std::cerr << "q8_paged_kv_metal_test: " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
