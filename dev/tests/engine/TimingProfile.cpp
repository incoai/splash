// The fitted profiles and the records they are fitted to. Every number is a
// measurement or a fit to measurements, with its source beside it; the fits
// are least squares over the model prefillTimes() computes (relative wall
// error, the recorded host waits weighted 0.25 on the M6 and 0.05 on the
// Studio), with the turnaround at 60 ms, the middle of the three a test
// runs (5, 60, 150 ms) until I-model measures it.
//
// What the status logs settled (2-s samples of the churn runs, 2026-10-31 and
// 11-01; windows between consecutive prefill completions):
// - Experts are read once per command, not per 2048-row chunk: a command of
//   one to four chunks loads each of its 48 layers once (prefill loads per
//   command / 48 = 0.90-1.06 on the M6 in mMNa, mMNb, pFa, m150a, fn15b, and
//   0.88-1.05 on the Studio in sMNa, sMNb, s150a, sIPa, smA). So the M6's
//   extra cost per chunk is compute: its chunk cost is 934.5 ms against the
//   Studio's 0, and its cost per row 0.84 ms against 0.33.
// - Routing is far from uniform: a layer of a Flash-Next command routes 302
//   distinct experts at 300 rows (14.5K of 24,576 over the 48 layers) and 362
//   at 8192, where uniform routing gives 510-512 (routed per load against
//   rows per load, all the runs above).
// - Resident shares are the stores' slots over 24,576 experts: M6 5721 (the
//   commands' run) and 4856-5722 in churn; Studio 18G 5672 (main) and 6681
//   (1.5.0); Studio default memory 11,768-13,216.
// - Studio 18G: of the expert store's read value (r150, 6681 slots, against
//   main, 5672: decode 68.3 against 77.6 MB a token, prefill hit 0.318 against
//   0.295 of about 4.2 TB routed a run), prefill reads are about 40 % (M6: 11 %,
//   amend/V.md §0.3).

#include "TimingProfile.hpp"

namespace splash::test {

namespace {

// Flash-Next Q2_0 (ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF): 48 layers,
// full attention every 4th, block-sparse (blocks of 4 tokens, 512 selected,
// so 2048 keys a row past 2048), 512 experts, 10 a token, an expert of
// 4,915,200 weights at 2.25 bits (the store's 1,382,407-byte slots), the
// distinct experts a layer routes by rows (above), a Full page of the 12
// attention layers' K/V of 423,936 bytes (status kv_page_bytes).
constexpr TimingProfile kFlashNext{
    .layers = 48,
    .attentionEvery = 4,
    .reach = {.denseHistoryTokens = 2048, .selectedTokens = 2048, .blockTokens = 4},
    .turnaroundMilliseconds = 60.0,
    .experts = {.experts = 512,
                .topK = 10,
                .expertBytes = 1382400,
                .routing = {{{300, 302.0}, {1536, 325.0}, {4096, 345.0}, {8192, 362.0}}}},
    // A prompt of token 0 routes the same ten experts in every row.
    .warmupDistinctExperts = 10,
    .spilledPageBytes = 423936,
};

// Decode tokens a lane commits per command: 7.93-7.98 in every run above.
constexpr double kTokensPerLaneCommand = 7.94;

// The Studio (M5 Max, 32 GPU cores) with Flash-Next at --max-memory 18G:
// the four commands' walls (DATA.md, kStudio18GFlashNextQ2Commands); no host
// wait of them was recorded, and the host floor of the 48 host steps is the
// M6's (their per-command host waits from the churn windows, 1040-1780 ms,
// weigh 0.05). Worst 8.3 %, mean 4.3 %.
constexpr TimingProfile studio(TimingProfile profile) {
  profile.chunkMilliseconds = 0.0;
  profile.rowMilliseconds = 0.3272;
  profile.expertMilliseconds = 0.05718;
  profile.weightBytesPerMillisecond = profile.spilledKvBytesPerMillisecond = 12.4e6;
  profile.hostFloorMilliseconds = 720.0;
  profile.hiddenShare = 0.05;
  // Decode's expert reads: the width-1 command against bytes a command at
  // 5672, 6681 and 13,216 slots (117.0, 102.1 and 45.6 ms; 77.6, 68.3 and
  // 22.5 MB a token), least squares.
  profile.decodeWeightBytesPerMillisecond = 6.22e6;
  return profile;
}

} // namespace

// The 24 GB M6 (12 GPU cores) with Flash-Next and its store at 5721 slots:
// the eleven commands of kM6FlashNextQ2Commands, worst 12.3 %, mean 4.1 %
// (the warm-up +7.4 %); hosts within 14 % but the 2272-row command's (−30 %).
const TimingProfile kM6FlashNextQ2 = [] {
  TimingProfile profile = kFlashNext;
  profile.name = "m6-flash-next-q2";
  profile.chunkMilliseconds = 934.5;
  profile.rowMilliseconds = 0.8413;
  profile.expertMilliseconds = 0.02610;
  profile.weightBytesPerMillisecond = profile.spilledKvBytesPerMillisecond = 5.0e6;
  profile.hostFloorMilliseconds = 720.0;
  profile.hiddenShare = 0.8;
  // Width 1-3 the medians of m150a, m150a2, m150b, m150b2 (5722 slots);
  // width 4 one more step of width 2 to 3 (no churn run decodes four lanes).
  profile.decodeMilliseconds = {149.5, 286.8, 475.9, 665.0};
  // Width 1 at 4856 slots (mMNa, mMNa2, mMNb: median 177.3 ms) against 5722:
  // 27.8 ms for 26.2 MB a token more.
  profile.decodeWeightBytesPerMillisecond = 7.49e6;
  // amend/M.md D3: a Spill lane's decode, median +1.0 ms a token.
  profile.spillDecodeMilliseconds = 1.0 * kTokensPerLaneCommand;
  // amend/V.md §0.3: main at the floor against r150's fixed store.
  profile.curve = {.decodeBytesPerToken = {{{4856, 125.7e6}, {5722, 99.5e6}}},
                   .prefillHit = {{{4856, 0.255}, {5722, 0.288}}}};
  profile.measuredResidentShare = 5721.0 / 24576;
  return profile;
}();

const TimingProfile kStudio18GFlashNextQ2 = [] {
  TimingProfile profile = studio(kFlashNext);
  profile.name = "studio-18g-flash-next-q2";
  // Widths 1 and 3 the medians of sMNa, sMNa2, sMNb, sB (5672 slots), width 2
  // sB's (the only run with more than one), width 4 one more step of 2 to 3.
  profile.decodeMilliseconds = {117.0, 229.7, 292.4, 355.1};
  // amend/M.md D3: Studio 18G, median +1.9 ms a token.
  profile.spillDecodeMilliseconds = 1.9 * kTokensPerLaneCommand;
  // main (sMNa, sMNb) and 1.5.0 (s150a).
  profile.curve = {.decodeBytesPerToken = {{{5672, 77.6e6}, {6681, 68.3e6}}},
                   .prefillHit = {{{5672, 0.295}, {6681, 0.318}}}};
  profile.measuredResidentShare = 5672.0 / 24576;
  return profile;
}();

// The Studio at its default memory: the same Mac and target as at 18G, the
// store at 13,216 slots (sA). At its prefill hit (0.638) a command of
// 300-8192 rows reads 0.56-0.67 s of experts compute does not hide, under
// the host floor, so the floor binds here: its 720 ms is the M6's (studio()).
const TimingProfile kStudio36FlashNextQ2 = [] {
  TimingProfile profile = studio(kFlashNext);
  profile.name = "studio-36g-flash-next-q2";
  // Width 1 sA's median; widths 2-4 by the 18G profile's ratios to width 1.
  profile.decodeMilliseconds = {45.6, 89.5, 114.0, 138.4};
  // amend/M.md D3: Studio default memory, median +1.7 ms a token.
  profile.spillDecodeMilliseconds = 1.7 * kTokensPerLaneCommand;
  // sFFa (11,768 slots) and sA (13,216).
  profile.curve = {.decodeBytesPerToken = {{{11768, 30.6e6}, {13216, 22.5e6}}},
                   .prefillHit = {{{11768, 0.570}, {13216, 0.638}}}};
  profile.measuredResidentShare = 13216.0 / 24576;
  return profile;
}();

// The Studio with every Flash-Next expert resident and its K/V spilling
// (--expert-memory off --kv-spill always): nothing read on demand but a
// streaming lane's history, at the Studio's SSD rate. It streams no experts,
// so its commands run no host steps (no host floor) and carry none of the
// streamed fit's expert term, which on the Studio stands in for the cost of
// streaming a command, collinear with its fixed part: a command costs its
// turnaround, its rows at the fit's rate and its unhidden history reads.
// No Mac the tracks time on runs it: Flash-Next Q2_0's 24,576 routed experts
// take 32,400 MiB, and the Studio's plan leaves its store 18,552 MiB (13,731
// slots, sA's start), so its commands are not measured.
const TimingProfile kStoreLessSpillFlashNext = [] {
  TimingProfile profile = studio(kFlashNext);
  profile.name = "store-less-spill-flash-next";
  profile.expertMilliseconds = 0.0;
  profile.hostFloorMilliseconds = 0.0;
  // kStudio36FlashNextQ2's width-1 command less its 22.5 MB a token of expert
  // reads (8 rows at 6.22 GB/s: 28.9 ms), widths 2-4 by its ratios.
  profile.decodeMilliseconds = {16.7, 32.8, 41.7, 50.7};
  profile.spillDecodeMilliseconds = 1.7 * kTokensPerLaneCommand;
  profile.curve = {.decodeBytesPerToken = {{{24576, 0.0}, {24576, 0.0}}},
                   .prefillHit = {{{24576, 1.0}, {24576, 1.0}}}};
  return profile;
}();

// Qwen3.8-27B UD-IQ3_XXS on the M6, dense: 64 layers, full attention every
// 4th. The backend benchmark's prompts (kDense27BPrompts) as commands of 2048
// rows, solved exactly; decode the benchmark's GPU time a step by width
// (round1001/intB backend-regression.json, baseline round 1, medians).
const TimingProfile kDense27B{
    .name = "m6-dense-27b",
    .layers = 64,
    .attentionEvery = 4,
    .turnaroundMilliseconds = 60.0,
    .chunkMilliseconds = 71.7,
    .rowMilliseconds = 2.9725,
    .megapairMilliseconds = 1.8532,
    .decodeMilliseconds = {114.61, 114.49, 140.14, 145.34},
};

// unsloth/flint-GGUF UD-Q4_K_XL on the Studio: 49 attending layers, a
// 513-key window but every fourth from layer 0 full (13 of them), its
// experts resident. Fitted to the cold prompts' TTFT of kFlintPrompts
// (persist-flint-a, persist2-flint-a; 0.1 s resolution, all within 30 ms):
// those do not resolve a cost per row from the cost per command, and the fit
// puts it at 0. Decode: width 1 from those runs (tokens a second over tokens a
// cycle, median 17.3 ms); widths 2-4 by k35BStudio18G's ratios, the other
// small-expert MoE target.
const TimingProfile kFlint{
    .name = "studio-flint",
    .layers = 49,
    .attentionEvery = 1,
    .fullAttentionEvery = 4,
    .reach = {.windowTokens = 513},
    .turnaroundMilliseconds = 60.0,
    .chunkMilliseconds = 136.6,
    .megapairMilliseconds = 0.8873,
    .decodeMilliseconds = {17.3, 20.3, 31.1, 41.9},
};

// Qwen3.6-35B-A3B on the Studio at --max-memory 18G, streaming its experts
// without HiSparse (m35M, m35L, m35L2, st35): 40 layers, full attention every
// 4th, 256 experts, 8 a token, 1,911,603-byte slots (12,693,045,248 bytes of
// arena in 6640); the store's cap 6640 of 10,240 slots, its floor 0.856 of
// that. Prefill from 703 command windows: 486.6 ms a command and 0.3713 ms per
// 1e6 pairs (the commands' depth from the lanes' processed prompt), of which
// the median host wait, 370 ms, is the floor; the windows' errors are wide
// (p10 −50 %, p90 +89 %), so it is a structure, not a fit.
const TimingProfile k35BStudio18G = [] {
  TimingProfile profile{
      .name = "studio-18g-35b",
      .layers = 40,
      .attentionEvery = 4,
      .turnaroundMilliseconds = 60.0,
      .chunkMilliseconds = 486.6 - 60.0 - 370.0,
      .megapairMilliseconds = 0.3713,
      .weightBytesPerMillisecond = 12.4e6,
      .hostFloorMilliseconds = 370.0,
      // Widths 1-3 the medians of m35M, m35L, m35L2; width 4 one more step.
      .decodeMilliseconds = {31.4, 36.9, 56.4, 75.9},
      .decodeWeightBytesPerMillisecond = 6.22e6,
      .experts = {.experts = 256,
                  .topK = 8,
                  .expertBytes = 1911603,
                  .routing = {{{183, 108.0}, {1024, 192.0}, {4096, 197.0}, {8192, 203.0}}}},
      .warmupDistinctExperts = 8,
      .measuredResidentShare = 5643.0 / 10240,
  };
  // At 5643 slots the pooled runs (m35M, m35L2, st35): 8.77 MB a token, prefill
  // hit 0.712. The runs nearer the cap (fx35, m35L) do not resolve its value
  // from run-to-run noise, so the cap's point follows the uniform marginal
  // (amend/M.md B8): misses scale with the share not resident.
  profile.curve = {.decodeBytesPerToken = {{{5643, 8.77e6}, {6640, 6.87e6}}},
                   .prefillHit = {{{5643, 0.712}, {6640, 0.774}}}};
  return profile;
}();

// Flash-Next Q2_0 on the 24 GB M6: rows, wall and host milliseconds
// (ragged_scheduler_test.cpp, testPrefillEstimateFitsCommandsOfDifferentRows):
// the 2048-row warm-up, two cold 8K prompts' groups, then agent turns of a few
// hundred rows past a cached prefix.
const std::array<MeasuredCommand, 11> kM6FlashNextQ2Commands{{{2048, 3213, 724, true},
                                                              {6144, 9172, 653, false},
                                                              {2431, 5263, 839, false},
                                                              {6144, 9192, 666, false},
                                                              {2782, 5536, 684, false},
                                                              {8188, 11745, 668, false},
                                                              {6144, 9254, 708, false},
                                                              {2272, 5270, 1027, false},
                                                              {305, 2987, 1693, false},
                                                              {336, 2892, 1620, false},
                                                              {724, 3588, 1628, false}}};

// Flash-Next Q2_0 on the Studio at 18G (DATA.md): walls only.
const std::array<MeasuredCommand, 4> kStudio18GFlashNextQ2Commands{
    {{1024, 2230, 0, false}, {2048, 2970, 0, false}, {4096, 3680, 0, false}, {8192, 4970, 0, false}}};

// The backend benchmark's partial scenario on the M6, prefill GPU time, means
// of three: 14,096 tokens cold, their 10,000-token prefix cold, and the 4,112
// past its last whole page (9984) as a cache hit.
const std::array<MeasuredPrompt, 3> kDense27BPrompts{{{0, 14096, 45768.0}, {0, 10000, 31865.7}, {9984, 4112, 14086.3}}};

// Flint's cold prompts on the Studio, TTFT.
const std::array<MeasuredPrompt, 5> kFlintPrompts{
    {{0, 1512, 200.0}, {0, 3012, 500.0}, {0, 5012, 800.0}, {0, 7012, 1200.0}, {0, 12012, 2200.0}}};

} // namespace splash::test
