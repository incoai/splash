# End-to-end: UD-Q4_K_M (GGUF K-quants) vs splash's packed-q4 package, same splash build, M5 Pro, 2026-09-21

Setup: splash main e8fffde + K-quant integration (splash-kquant-integration.diff, kquant.metal, KQuant.h), built with Xcode 27's
Metal toolchain. Package B = `convert_gguf_to_splash.py` output of unsloth/Qwen3.8-27B-UD-Q4_K_M (16.2 GB target; draft, vision and
tokenizer taken unchanged from incoai/Qwen3.8-27B-Splash). Package A = incoai/Qwen3.8-27B-Splash (15.4 GB target). One HTTP
server at a time (server/server.py -> build/splash serve-native), DFlash speculative decoding on for both, temperature 0,
order A, 60 s idle, B, 60 s, A, 60 s, B. Numbers from the server's own request metrics (start_to_first_token, first_token_to_done).

| metric | splash package | K-quant package | ratio |
|---|---|---|---|
| decode, 1 stream, 256 new tokens | 45.6 tok/s (45.6..45.8, n=6) | 33.1 tok/s (33.0..33.2, n=6) | 0.725 |
| decode, 4 concurrent streams, aggregate | 85.9 tok/s (80.7..93.3) | 82.6 tok/s (76.6..86.4) | 0.961 |
| prefill, 2062-token prompt | 424 tok/s (413..434) | 407 tok/s (403..415) | 0.960 |
| time to first token, short prompt | 237 ms | 235 ms | 0.994 |

Why single-stream decode is 0.73 and not the ~0.85 the per-kernel numbers predict: bytes +6.4% (4.79 vs 4.5 bits/element, lm_head
Q6_K = +0.3 GB/step); M=8 kernel efficiency 0.83-0.96 on the wide matrices; and dispatch fragmentation: the GGUF keeps
qkv/z/alpha-beta and q/k/v as separate tensors with different types, so a GDN layer issues ~13 linear dispatches (segments, split-K
partial + reduce, out_proj head permute) where splash issues 4 fused ones; ~560 extra dispatches and ~350 MB of fp32 partial traffic
per step. Batched decode (M=32) and prefill are compute-bound and land within 4% of splash. Outputs are coherent (sample answers
checked); accuracy of the served model was not re-measured here (see the KLD study for UD-Q4_K_M vs BF16).
