# DSpine with engine 1.6.1

The [qualification release](https://github.com/linson007/splash/releases/tag/dspine-integration-review-20261111-v1) contains the source-specific report, registered comparisons, raw observations, validation outcomes and failed gates for the upstream-integrated implementation. This directory records its immutable source and build identities. The release evidence is separate from the [frozen pre-integration review](../dspine_20261111_review/README.md); its throughput ranges and sampling screen do not transfer to this source.

The integration is `5855e24b8a903f0c1ff9af6b1d41467cfcdaba55`, incorporating upstream engine 1.6.1 at `767c646ed91eb45a1e13a59525da7373123ac4cf`. Its runtime is `src-7e9d0622992dd15b5d3731bac0f4770a7239fe48bd9624cf6e905e3114a419a7`. The normal PR checkout produced the same native executable and both Metal libraries byte-for-byte. [Artifact witness](main-integrated-bundle-witness-v1.json), [source checks](main-source-checks-v1.json), [control/source identities](SOURCE.json).

Upstream introduces activation quantization, warmup and resource-lifecycle changes. The integrated target keeps its device-selected activation policy; DSpine's draft, bridge and context projections retain BF16 activations. `--activations bf16` selects the target's explicit BF16 policy. Each comparison names its policy; this is a runtime configuration policy, not a claim that every operator uses the same precision. Requested sixteen-row geometry remains per loaded model, with its own scratch, recurrent tape and persistent-cache identity.

The review corrected longest-verify warmup to respect a served context bound and preserved the canonical cache-layout serialization in model-owned helpers. CPU fixtures cover both activation policies, both widths, every lane count and very small context limits. A window-only cache-namespace fixture protects the distinction between 2K and 8K draft histories.

The clean upstream DFlash2 control uses the same shared measurement driver and frozen prompt tokens as the integrated arm. The separate M64 control changes one automatic M48 selection field, retaining the QKV correction, shaders, precision, splits and separate gate/up passes. Benchmark evidence keeps exact build, model-layout, output, operating, noise and effect checks. Server-cache checks and full-vocabulary probes are functional evidence with their own scopes.

The completed engine 1.6.1 comparisons retain every registered ABBA group and failed gate. Native rows below measure aggregate decode throughput; HTTP rows measure returned tokens over request-phase wall time. Ranges are observations across three groups, not confidence intervals. All cohorts are reused confirmation sets.

| Comparison | Observed throughput change | Qualification and scope |
| --- | --- | --- |
| Matched M64 → M48, DSpine16 B3 | **+11.04–12.29%**; 121.67–123.69 → 136.63–138.12 tok/s | Qualified; three prompts, three cloned lanes, 256-token caps; all 324 arrays match |
| Clean upstream → integrated DFlash2-8 | −0.158% to −0.151% | Registered nonregression passes; three short prompts, all 108 arrays match |
| DFlash2-8 → DSpine16, agent12 | −4.40% to −3.47% | Whole cohort unqualified: first-group noise/effect gates fail; all 144 arrays match. Short category independently qualifies 5.92–6.01% slower |
| Same modes, historical ALL31 | +19.46–22.95% | Whole cohort unqualified: per-case noise fails groups one and three; all 372 arrays match. Eight cases independently qualify gains |
| Same modes, mixed cold HTTP | −1.96% to −1.36% | Unqualified across all groups: first two effects do not clear their bounds; cold responses match |
| Same modes, mixed warm HTTP | −33.65% to −33.26% | Unqualified: output identity and cold/warm parity fail; no serving-speed claim |

The [full report](https://github.com/linson007/splash/releases/download/dspine-integration-review-20261111-v1/REPORT.md), [all-case comparison](https://github.com/linson007/splash/releases/download/dspine-integration-review-20261111-v1/CASE_RESULTS.md), plots and raw evidence are in the qualification release. These results do not establish a fastest mode for most production traffic. The agent cycle audit finds 5.89% more decode tokens per cycle but 9.69–10.76% greater wall cost per cycle for DSpine16; existing counters cannot allocate that extra cost to individual components. Reducing serial cycle cost and resolving warm-cache behavior are the next priorities. Extending the existing rotated-PQ2_0 int8 kernel to M48 would not accelerate this unrotated AF4 model.

DFlash2 remains the default drafter; DSpine's default execution width remains eight. Select sixteen rows explicitly and compare representative prompt types, output caps, activation policy and concurrency. Context length alone did not select the fastest drafter in the frozen review, and a cloned three-lane gain is not a general serving-speed claim.
