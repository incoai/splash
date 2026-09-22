"""Compare splash and ggml GEMM outputs on the same tensor/activations against the fp64 references."""
import numpy as np, sys, json, os
d = sys.argv[1]
fmt, N, K, M = open(f"{d}/meta.txt").read().split(); N, K, M = int(N), int(K), int(M)
ref = np.fromfile(f"{d}/Y_ref_ggml.f64", dtype=np.float64).reshape(M, N)
ref_native = np.fromfile(f"{d}/Y_ref_native.f64", dtype=np.float64).reshape(M, N)
ref_staged = np.fromfile(f"{d}/Y_ref_staged.f64", dtype=np.float64).reshape(M, N)
ys = {k: np.fromfile(f"{d}/{f}", dtype=np.float32).reshape(M, N).astype(np.float64) for k, f in (("splash", "Y_splash.f32"), ("ggml-metal", "Y_ggml_metal.f32"), ("ggml-cpu", "Y_ggml_cpu.f32"))}
scale = np.abs(ref).mean()
def rel(a, b): return float(np.abs(a - b).mean() / scale)
def bf16_ulp_rate(y):  # share of outputs within one bf16 ulp of the fp64 reference (splash writes bf16 outputs)
    ulp = np.abs(ref) * 2.0**-7; return float((np.abs(y - ref) <= ulp).mean())
out = {"fmt": fmt, "N": N, "K": K, "M": M, "ref_dequant_agreement_native_vs_ggml": rel(ref_native, ref), "ref_staged_vs_ggml": rel(ref_staged, ref)}
for k, y in ys.items():
    e = np.abs(y - ref) / scale
    out[k] = {"mean_abs_err/mean_abs_ref": rel(y, ref), "median": float(np.median(e)), "p99": float(np.percentile(e, 99)), "max_abs_err": float(np.abs(y - ref).max()), "within_1_bf16_ulp": bf16_ulp_rate(y)}
out["splash_vs_ggml_metal"] = rel(ys["splash"], ys["ggml-metal"]); out["ggml_metal_vs_ggml_cpu"] = rel(ys["ggml-metal"], ys["ggml-cpu"])
# splash emits bf16: compare bit-for-bit with llama.cpp's Metal output rounded to bf16, and count the ulp distance
def to_bf16_bits(a):
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64); return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.int64)
sb, gb = to_bf16_bits(ys["splash"]), to_bf16_bits(ys["ggml-metal"])
ulps = np.abs(sb - gb)
out["splash_vs_bf16(ggml_metal)"] = {"identical": float((ulps == 0).mean()), "within_1_ulp": float((ulps <= 1).mean()), "within_2_ulp": float((ulps <= 2).mean()), "max_ulp": int(ulps.max())}
print(json.dumps(out))
