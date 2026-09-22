// Same-tensor GEMM comparison: llama.cpp's official ggml Metal/CPU mul_mat on N rows of a real GGUF tensor
// with deterministic bf16-representable activations. Writes the inputs for splash's harness_prod "file" mode
// and the ggml outputs + an fp64 reference from ggml's own dequantization.
//   ggml_gemm_parity <gguf> <tensor> <N rows> <M activations> <seed> <outdir>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

extern "C" ggml_backend_t ggml_backend_metal_init(void);

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16; return (uint16_t)u; }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static void dump(const std::string &path, const void *p, size_t n) { std::ofstream o(path, std::ios::binary); o.write((const char *)p, n); }

static const char *fmtName(ggml_type t) {
  switch (t) { case GGML_TYPE_Q4_K: return "q4k"; case GGML_TYPE_IQ4_XS: return "iq4xs"; case GGML_TYPE_IQ4_NL: return "iq4nl"; case GGML_TYPE_Q5_K: return "q5k";
    case GGML_TYPE_Q6_K: return "q6k"; case GGML_TYPE_Q3_K: return "q3k"; case GGML_TYPE_Q8_0: return "q80"; case GGML_TYPE_IQ3_S: return "iq3s"; default: return nullptr; }
}

static std::vector<float> compute(ggml_backend_t backend, ggml_type type, int64_t K, int64_t N, int64_t M,
                                  const std::vector<uint8_t> &w, const std::vector<float> &x, double *ms) {
  ggml_init_params ip{ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *W = ggml_new_tensor_2d(ctx, type, K, N);
  ggml_tensor *X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
  ggml_tensor *Y = ggml_mul_mat(ctx, W, X);   // [N, M]
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  ggml_backend_tensor_set(W, w.data(), 0, w.size());
  ggml_backend_tensor_set(X, x.data(), 0, x.size() * 4);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, Y);
  ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_alloc_graph(galloc, gf);
  ggml_backend_graph_compute(backend, gf);   // warm (pipeline creation)
  const int64_t t0 = ggml_time_us();
  for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(backend, gf);
  ggml_backend_synchronize(backend);
  *ms = (ggml_time_us() - t0) / 1e3 / 5;
  std::vector<float> y((size_t)N * M);
  ggml_backend_tensor_get(Y, y.data(), 0, y.size() * 4);
  ggml_gallocr_free(galloc); ggml_backend_buffer_free(buf); ggml_free(ctx);
  return y;
}

int main(int argc, char **argv) {
  if (argc != 7) { fprintf(stderr, "usage: ggml_gemm_parity <gguf> <tensor> <N> <M> <seed> <outdir>\n"); return 2; }
  ggml_time_init();
  const std::string gguf = argv[1], name = argv[2], outdir = argv[6];
  const int64_t N = atol(argv[3]), M = atol(argv[4]); const unsigned seed = atoi(argv[5]);
  ggml_context *meta = nullptr;
  gguf_init_params gp{true, &meta};
  gguf_context *g = gguf_init_from_file(gguf.c_str(), gp);
  if (!g) { fprintf(stderr, "cannot read %s\n", gguf.c_str()); return 1; }
  const int64_t id = gguf_find_tensor(g, name.c_str());
  if (id < 0) { fprintf(stderr, "no tensor %s\n", name.c_str()); return 1; }
  ggml_tensor *info = ggml_get_tensor(meta, name.c_str());
  const ggml_type type = info->type; const int64_t K = info->ne[0], rowsTotal = info->ne[1];
  if (!fmtName(type) || N > rowsTotal) { fprintf(stderr, "unsupported type or too many rows\n"); return 1; }
  const size_t rowBytes = ggml_row_size(type, K);
  std::vector<uint8_t> w((size_t)N * rowBytes);
  { std::ifstream in(gguf, std::ios::binary); in.seekg((std::streamoff)(gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id))); in.read((char *)w.data(), w.size()); if (!in) { fprintf(stderr, "short read\n"); return 1; } }
  std::mt19937 rng(seed); std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<float> x((size_t)M * K); std::vector<uint16_t> xb(x.size());
  for (size_t i = 0; i < x.size(); ++i) { xb[i] = f2bf(u(rng)); x[i] = bf2f(xb[i]); }
  // fp64 reference from ggml's own dequantization (the values llama.cpp's kernels represent)
  const ggml_type_traits *tr = ggml_get_type_traits(type);
  std::vector<float> wf((size_t)N * K);
  for (int64_t n = 0; n < N; ++n) tr->to_float(w.data() + n * rowBytes, wf.data() + n * K, K);
  std::vector<double> ref((size_t)M * N);
  for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) { double acc = 0; for (int64_t k = 0; k < K; ++k) acc += (double)x[m * K + k] * wf[n * K + k]; ref[m * N + n] = acc; }
  double msMetal = 0, msCpu = 0;
  ggml_backend_t metal = ggml_backend_metal_init(); if (!metal) { fprintf(stderr, "no Metal backend\n"); return 1; }
  std::vector<float> yMetal = compute(metal, type, K, N, M, w, x, &msMetal);
  ggml_backend_t cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, 8);
  std::vector<float> yCpu = compute(cpu, type, K, N, M, w, x, &msCpu);
  auto stats = [&](const std::vector<float> &y, const char *label) { double maxabs = 0, sumrel = 0; for (size_t i = 0; i < y.size(); ++i) { maxabs = std::max(maxabs, std::fabs((double)y[i] - ref[i])); sumrel += std::fabs((double)y[i] - ref[i]) / (std::fabs(ref[i]) + 1e-3); }
    printf("%-12s %s N=%lld K=%lld M=%lld: max abs err vs fp64 %.4g, mean rel err %.3e\n", label, fmtName(type), (long long)N, (long long)K, (long long)M, maxabs, sumrel / y.size()); };
  stats(yMetal, "ggml-metal"); stats(yCpu, "ggml-cpu");
  printf("timing: ggml-metal %.3f ms, ggml-cpu %.3f ms per mul_mat\n", msMetal, msCpu);
  dump(outdir + "/W.native", w.data(), w.size()); dump(outdir + "/X.bf16", xb.data(), xb.size() * 2);
  dump(outdir + "/Y_ggml_metal.f32", yMetal.data(), yMetal.size() * 4); dump(outdir + "/Y_ggml_cpu.f32", yCpu.data(), yCpu.size() * 4); dump(outdir + "/Y_ref_ggml.f64", ref.data(), ref.size() * 8);
  { FILE *m = fopen((outdir + "/meta.txt").c_str(), "w"); fprintf(m, "%s %lld %lld %lld\n", fmtName(type), (long long)N, (long long)K, (long long)M); fclose(m); }
  ggml_backend_free(metal); ggml_backend_free(cpu); gguf_free(g); ggml_free(meta);
  return 0;
}
