#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace splash::benchmark {

// Chat requests whose greedy answers move with any change in the target's
// arithmetic, unlike the counting the decode throughput prompt asks for:
// backend_benchmark's transcripts scenario runs them, and
// dev/benchmarks/backend_regression.py holds their outputs identical between
// builds. Each is the decode throughput prompt's template, one user message
// with thinking off, in the tokens Qwen3.6's and Qwen3.8's tokenizers give
// alike, for:
//   0 Write a Python function that merges overlapping intervals, then explain its time complexity.
//   1 Implement an LRU cache in C++ with O(1) get and put, and show a short usage example.
//   2 Write a Go HTTP handler that accepts a JSON body with a name field and replies with a JSON greeting.
//   3 用中文解释TCP三次握手的过程，并说明为什么不能只用两次握手。
//   4 请写一首关于秋天的七言绝句，然后逐句解释其中的意象。
//   5 A train leaves at 9:40 and covers 210 km at 84 km/h. When does it arrive? Explain step by step.
inline constexpr uint32_t kTranscriptPrompt0[] = {
    248045, 846, 198, 7734, 264, 12654, 709, 421, 78161, 48142, 26126, 11, 1179, 10033, 1141, 854,
    22373, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
};
inline constexpr uint32_t kTranscriptPrompt1[] = {
    248045, 846, 198, 60856, 449, 436, 34810, 6297, 303, 351, 992, 440, 496, 7, 16, 8, 615, 321,
    2113, 11, 321, 1420, 264, 2716, 10121, 3010, 13, 248046, 198, 248045, 74455, 198, 248068, 271,
    248069, 271
};
inline constexpr uint32_t kTranscriptPrompt2[] = {
    248045, 846, 198, 7734, 264, 5805, 9830, 6795, 421, 25503, 264, 4566, 2407, 440, 264, 803, 2002,
    321, 29636, 440, 264, 4566, 40719, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
};
inline constexpr uint32_t kTranscriptPrompt3[] = {
    248045, 846, 198, 95826, 99986, 98682, 48218, 110114, 119587, 108796, 3709, 96172, 98419, 98832,
    96849, 123940, 109039, 119587, 1710, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
};
inline constexpr uint32_t kTranscriptPrompt4[] = {
    248045, 846, 198, 96220, 96674, 112064, 97785, 118999, 96822, 96393, 96923, 96804, 3709, 98118,
    97577, 96804, 98682, 109439, 125093, 1710, 248046, 198, 248045, 74455, 198, 248068, 271, 248069,
    271
};
inline constexpr uint32_t kTranscriptPrompt5[] = {
    248045, 846, 198, 32, 5257, 10583, 506, 220, 24, 25, 19, 15, 321, 14103, 220, 17, 16, 15, 12754,
    506, 220, 23, 19, 12754, 7294, 13, 3095, 1503, 424, 16821, 30, 79091, 2923, 539, 2923, 13,
    248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
};
inline constexpr std::array<std::span<const uint32_t>, 6> kTranscriptPrompts{
    kTranscriptPrompt0, kTranscriptPrompt1, kTranscriptPrompt2, kTranscriptPrompt3, kTranscriptPrompt4, kTranscriptPrompt5};

} // namespace splash::benchmark
