# splash: load Unsloth GGUF directly, repack in memory (design, 2026-09-21)

Decision (Yesheng): no second on-disk copy of the target weights; the GGUF is the only weight file.
The kernel-friendly K-quant layout (`MDKQ0001` images: descriptor + plane0 + plane1 + meta, 16 KiB sections)
is built in anonymous Metal memory at load time and never written to disk. Anonymous memory means macOS
swaps/compresses under pressure instead of dropping file pages; accepted.

## 1. Checkpoint structure

Hugging Face repo `incoai/Qwen3.8-27B-Splash-GGUF` ("shell", ~1.3 GB): `manifest.json`, `draft/` (DFlash 2,
splash-packed-q4), `vision/model.bin`, `tokenizer/`. No `target/` weights. Manifest (schema_version 4):

```json
"format": {"name": "gguf", "draft_layer_magic": "MDFD0004", "vision_magic": "MDFV0001", "section_alignment_bytes": 16384},
"target": {
  "architecture": "qwen35", "block_count": 65, "nextn_predict_layers": 1,
  "gguf": {"repo_id": "unsloth/Qwen3.8-27B-GGUF", "revision": "<sha>",
           "default": "Qwen3.8-27B-UD-Q4_K_M.gguf",
           "variants": {"Qwen3.8-27B-UD-Q4_K_M.gguf": {"size": 16464440224, "sha256": "..."},
                        "Qwen3.8-27B-UD-Q4_K_S.gguf": {...}, "Qwen3.8-27B-UD-IQ4_XS.gguf": {...}}}
}
```
`artifacts`/`artifact_set_sha256` cover only shell files. The GGUF's integrity is the HF download's job; the
loader validates structure (architecture, block_count, tensor names, shapes, types), not a hash, so any GGUF of
the architecture whose tensor types are all supported loads (variant list is documentation + default).

Model root on disk (assembled by the launcher, real directories + per-file symlinks, same shape as the HF cache):
```
<root>/manifest.json            -> shell snapshot
<root>/draft/*.bin, vision/, tokenizer/   (symlinks into the shell snapshot)
<root>/target/<name>.gguf       -> HF cache blob (hf_hub_download of the one file) or a user-supplied local path
```
Native CLI unchanged: `serve-native <root>/target <root>/draft <ctx> <mem>`. `inspectModelPackage` sees format
"gguf" and requires target/ to hold exactly one GGUF (or one complete shard set `-0000N-of-0000M.gguf`).
Launcher: `splash ... --model incoai/Qwen3.8-27B-Splash-GGUF [--gguf Qwen3.8-27B-UD-Q4_K_S.gguf | --gguf-path /x/y.gguf]`.
The `gguf-kquant` package format + convert_gguf_to_splash.py are removed; MDKQ0001 becomes an in-memory image only.

## 2. Loading logic

Pipeline per process start (target only; draft/vision/tokenizer load as today):

1. **Parse** GGUF header(s): magic/version 3, KVs (`general.architecture`, `general.alignment` default 32,
   `qwen35.*`, `general.split.*`), tensor infos (name, dims, type, offset). Build name -> (shard, offset, bytes,
   type, N, K). Ignore `blk.64.nextn.*` (MTP) tensors. Validate the complete expected tensor set for 64 layers:
   names, shapes (e.g. attn_qkv 10240x5120, ffn_* 17408/5120, output 248320x5120), types in
   {Q4_K,Q5_K,Q6_K,Q3_K,IQ4_XS,IQ4_NL,Q8_0,IQ3_S} for linears, F32 for norms/conv/ssm_a/dt_bias, Q8_0 for
   alpha/beta, Q4_K for token_embd. One error listing every offending tensor.
2. **Plan** the image of each layer file (layer-N, head, embedding) exactly as the converter did:
   16-byte header, then sections (norms bf16, KQ desc 64 B + plane0 + plane1 + meta, ...), 16 KiB aligned.
   Sizes derive from (type, N, K). Sum = weight bytes reported to the memory governor before any allocation.
3. **Allocate** one anonymous Shared MTLBuffer per image (`allocateBuffer`, label `target/layer-N.bin`), wrapped
   in `WeightFile` through a new memory-backed Impl (same `section()`/`finish()` cursor semantics; readers unchanged).
4. **Stream** layers L = 0..63 with 2 in flight:
   - map the covering file range of layer L's tensors (blk.L.* are contiguous in Unsloth files; fallback to
     per-tensor mappings), page-aligned, `MAP_SHARED` read-only, `wrapSharedMemory` -> source buffer;
   - a reader thread prefetches layer L+1's range (`madvise(MADV_WILLNEED)` + touch) so SSD reads overlap GPU work;
   - CPU writes header, descriptors, norms (f32->bf16, keep GGUF 1+w form), conv1d (bf16, rows un-reordered),
     decay f32 and dt_bias bf16 (un-reordered), gdn-ab = [beta 48 | alpha 48 | 0] Q8_0 rows straight into the image;
   - GPU: one `kq_repack_<fmt>` dispatch per (tensor, plane): thread per (row n, 32-group g) -> 16 B (8/32 B for
     Q3_K/Q8_0) at tile order ((n/256)*G+g)*256+(n%256); meta thread per (n, superblock). Row permutation for
     the GDN head order folded into source-row addressing (rows >= 4096 of qkv, all rows of z: grouped h_out ->
     tiled h_in = (h_out%3)*16 + h_out/3, head = 128 rows). Same for head.bin (Q6_K logits) and a plain copy
     for embedding.bin (native rows);
   - commit; when L-1's command buffer completes, munmap its source range (pages stay clean in the page cache).
5. **Finish**: wait for the last command buffer, close fds, `finish()` every WeightFile (image fully consumed).

Cost model on the M5 Pro (48 GB), UD-Q4_K_M:

| item | value |
|---|---|
| GGUF file | 16.46 GB (866 tensors, data 32-B aligned) |
| anonymous images | layers ~14.4 GB + head 1.05 GB + embedding 0.72 GB = ~16.2 GB |
| in-flight source mappings | <= 3 layers, ~0.7 GB, clean/reclaimable |
| GPU repack | ~0.2-0.3 s total (15 GB read + write at memory bandwidth) |
| SSD read, cold / warm page cache | ~3 s / ~1.5 s (today's lazy mmap pays the same read during warmup) |
| target load time | <= 4 s cold, <= 2.5 s warm |

## 3. Code changes (splash)

- `runtime/model/GgufFile.{hpp,cpp}` (new): header/KV/tensor-info parser, shards, type->KQ_FMT_* map, range queries.
- `runtime/model/GgufTargetLoader.{hpp,mm}` (new): plan, allocate, stream, prefetch thread, CPU fills, dispatches.
- `runtime/metal/kernels/shared/kquant_repack.metal` (new) + `KQRepackParams` in `metal/abi/KQuant.h`:
  eight `kq_repack_<fmt>_{plane0,plane1,meta}` kernels (port of `repack()` in the converter; the harness's C++ CPU
  repack stays as the validation reference).
- `WeightStore`: `WeightFile` gets a memory-backed constructor (owned MetalBuffer, declaredBytes = image size).
- `ModelDescriptor.mm`: format "gguf" (replaces "gguf-kquant"), target/ GGUF discovery, manifest `target.gguf`.
- `QwenTarget.hpp` `loadQwenTargetWeights`: when `descriptor.gguf`, take WeightFiles from the GgufTargetLoader
  instead of opening `target/layer-N.bin`; readers (`readKQuantSegment/Projection/Embedding`) unchanged.
- `splash` launcher + `server/server.py`: download shell + selected GGUF (`hf_hub_download`), assemble the root,
  `--gguf` / `--gguf-path` options; `--max-memory auto` uses the planned image bytes.
- Dev tool `dev/tools/gguf-image-hash`: builds the images and prints per-section sha256 (compare with the
  converter's layer files -> byte-identical), plus negative tests (unsupported type, wrong arch, missing tensor,
  truncated shard).

Validation: (1) per-section hashes equal the converter output for UD-Q4_K_M (all 8 types exercised);
(2) greedy decode tokens identical to the package-based run on the prompt mix; (3) load time and peak footprint
measured on the M5 Pro; (4) UD-Q4_K_S / UD-IQ4_XS headers accepted or rejected with a clear message.

Out of scope: Qwen3.6-35B-A3B MoE (3-D expert tensors), llama.cpp mmproj vision, GGUF drafts (DFlash has none).
