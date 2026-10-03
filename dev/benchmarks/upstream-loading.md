# Upstream model loading

Measurements behind serving MLX and GGUF upstream models directly, taken in
September 2026 while upstream loading was built; each section names what was
measured and how to repeat it. Devices: M3 Max 40 GPU cores (Apple9), M5 Pro 16
and 20 GPU cores (Apple10), AC power, one GPU job at a time. Build the tools
below with `make all build/engine-tests/<tool>`.

## Sources

| Source | Revision or content SHA-256 |
| --- | --- |
| `mlx-community/Qwen3.8-27B-4bit` | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | `38740b847e4cb78f352aba30aa41c76e08e6eb46` |
| `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |
| `unsloth/Qwen3.6-35B-A3B-GGUF` | `a483e9e6cbd595906af30beda3187c2663a1118c` |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` |
| 27B `mmproj-BF16.gguf` | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| 35B `mmproj-BF16.gguf` | `356dfaa3111376a4f7165e32e8749713378d1700b37cf52e0c50d9f23322334d` |

Drafts are prepared from each family's DFlash2 repository (`families.FAMILIES`);
the ones measured here were `incoai/Qwen3.8-27B-DFlash2` at `015e7956` and
`incoai/Qwen3.6-35B-A3B-DFlash2` at `51ef7b69`. The prepared files are
byte-identical to the drafts of the released Qwen3.8-27B and Qwen3.6-35B-A3B
packages.

## Prepared bytes

Preparation changes layout, never values:

- Affine 35B: every byte of all 40 layers, head and embedding matches the
  released package.
- Affine 27B: all packed weights, parameters, other tensors and padding across
  64 layers, head and embedding match. The GDN decay vector of each of the 48
  GDN layers differs by at most one float ULP from the package's MLX
  exponential; the adapter computes `float(-exp(double(A_log)))`, and the
  source oracle allows at most two ULP in that section.
- GGUF: the bounded repack of all eight supported formats, with multiple row
  tiles, wide rows, head permutations and offsets above 4 GiB, matches the CPU
  reference bytewise on the M3 Max and the M5 Pro.
- Vision: every source prepares the packed `vision/model.bin`, padding included.

| Vision source | Bytes | SHA-256 |
| --- | ---: | --- |
| 27B MLX and 27B mmproj-BF16 | 930,250,752 | `8973858e75ec3464f626d5f9111dbceda58f3bac5737ea5693767686003c65df` |
| 35B MLX and 35B mmproj-BF16 | 901,939,200 | `20ba816b644c7221d33d1b188612d096dae3e96600be2dc3d4bd1600c0d30d00` |

Each Unsloth mmproj holds 334 tensors: 110 BF16 matrices and 224 F32 tensors
(1-D tensors, both patch-embedding frames and the position table). None of the
F32 values has non-zero low 16 bits (0 of 4,833,008 for 27B, 0 of 4,829,936 for
35B), so they convert to BF16 exactly.

The affine comparison is `affine-source-oracle` (DEVELOPMENT.md, Validate),
which prints `packed_exact=true` and the decay's `decay_max_ulp` per file; give
it a scratch `SPLASH_WEIGHT_CACHE`. The GGUF repack check is `gguf-preparation`
in `make test-engine-metal`.

```sh
SPLASH_WEIGHT_CACHE=$(mktemp -d) build/engine-tests/affine-source-oracle build/splash.metallib \
  install/models/mlx-community/Qwen3.6-35B-A3B-4bit/target install/models/incoai/Qwen3.6-35B-A3B-Splash
```

A prepared vision file is the `weights` file of the cache entry whose `source`
names `component vision/model.bin` (`grep -l '^component vision/model.bin'
~/Library/Caches/Splash/weights/*/source`).

The vision fixture embedding is unchanged, with Metal shader validation:
`7946f077435ef45d0a596461a9d9a234ff458c805c007bb3ea1af7296bd230f9` for 35B on
both M5 Pro devices, `011d9121bf52f85a26e7eff28e9c9459a48eb0bb36d585ff8e7ef7621cd3a37b`
for 27B on the M3 Max. The 35B mmproj and MLX sources give the packed embedding.
`make test-real MODEL=...` prints it (`embedding SHA-256`) with the vision
parity check.

## GGUF tokenizer

Both GGUFs declare `gpt2/qwen35`: 248,320 vocabulary entries, 247,587 merges,
27 control tokens, six user-defined tokens and 243 unused entries. The 27B file
declares 65 blocks including one MTP layer; the derived configuration describes
64 target layers.

Each tokenizer built from GGUF metadata was compared with the MLX Qwen tokenizer,
which shares vocabulary, merges and pre-tokenizer (used only as a test reference):

- all 248,077 shared vocabulary IDs match;
- 1,042 text cases per model match in input IDs and character offsets, covering
  CJK, combining marks, emoji, whitespace, control characters, code, JSON,
  numbers, every added token and 1,000 seeded random combinations;
- decoding with special tokens matches; with special tokens skipped it follows
  the GGUF's control-token flags, which differ from the HF tokenizer for some FIM
  tokens;
- the embedded template is kept byte for byte, and 16 rendering cases per model
  (system messages, tool calls and results, image placeholders, tools and
  thinking on and off) match the original template and the reference token IDs.

That comparison script is not in the repository. The committed checks are
`python -m unittest dev.tests.test_gguf_metadata` (synthetic metadata) and the
chat-template probe tests over the embedded templates in
`dev/tests/fixtures/chat_templates/`.

## Preparation cost

35B UD-Q4_K_M GGUF: 42 artifacts, 22,143,172,608 bytes, prepared by the
batched executor of `2ca5691`; later preparation changes did not repeat this
timing.

| Device | Cold preparation | Reuse of prepared files |
| --- | ---: | ---: |
| M3 Max | 29.74 s | 0.641 s |
| M5 Pro 16 cores | 24.30–25.87 s | 0.533–0.539 s |

Cold means an empty preparation cache and no source hash proof. The times cover
backend creation, source verification, planning, conversion, output hashing and
publication, not tokenizer, server startup or warmup. Peak process RSS is about
65–68 MB; staging is bounded to 32 MiB inside a 64 MiB admission reserve and does
not grow with tensor, layer or expert count. No run increased the system swap
counters. Reopening the prepared affine 35B files takes 0.064 s on the M5 Pro 20
(`affine-source-oracle ... --load-only`, which prints `seconds=`). The GGUF
timing harness is not in the repository; a cold start logs each artifact's
`Prepared <component> in N s`.

## Decode throughput

Baseline: PR 114 (`d25020e`), serving the Splash packages. Candidate: its child
`f72b411`, which introduced source preparation, serving the packages
("packed"), the MLX checkpoints with the packages' drafts ("prepared"), and the
Unsloth GGUFs through the GGUF packages of that time ("GGUF"). 64 tokens per lane at B1–B4, run
baseline/candidate/candidate/baseline; ratios are baseline median GPU time over
candidate median GPU time, so 1.000 is unchanged.

```sh
build/engine-tests/backend-benchmark build/splash.metallib MODEL_ROOT \
  --scenario decode --samples 3
```

| Device | Target | B1 | B2 | B3 | B4 |
| --- | --- | ---: | ---: | ---: | ---: |
| M3 Max | packed affine 27B | 0.999 | 1.000 | 0.999 | 0.999 |
| M3 Max | packed affine 35B | 1.004 | 0.997 | 0.998 | 0.996 |
| M3 Max | GGUF 27B | 1.003 | 1.001 | 1.000 | 1.000 |
| M3 Max | GGUF 35B | 0.999 | 1.000 | 1.000 | 0.998 |
| M5 Pro 16 | packed affine 27B | 0.995 | 1.000 | 1.003 | 0.997 |
| M5 Pro 16 | prepared affine 27B | 1.000 | 0.997 | 1.000 | 0.998 |
| M5 Pro 16 | prepared affine 35B | 0.999 | 1.004 | 1.000 | 0.998 |
| M5 Pro 16 | GGUF 35B | 1.005 | 1.008 | 1.001 | 1.000 |

Output hashes, accepted draft counts and decode batch counts matched the baseline
in every run. The M3 Max prepared-affine 27B round drifted strongly within and
across runs (B2–B4 1.04–1.07) and is not claimed as a gain. A repeated 2048-token
prefill probe on the M5 Pro 16 (GGUF 35B, seven repetitions per process, ABBA)
gave 683.9 ms baseline and 683.5 ms candidate; the probe, derived from
`decode_profile.mm`, is not in the repository.

## End to end

`splash serve` installs of all four upstream models on the M5 Pro 16, with the
draft fetched from the shared draft repository, produce the same answers as the
released packages for two text and two image prompts (greedy decoding, same
binary), and their prepared vision files match the hashes above. With
`HF_HUB_OFFLINE=1`, the installer's check of an installed model takes 0.1 s;
engine startup follows it. This comparison was run by hand, with prompts not in
the repository.
