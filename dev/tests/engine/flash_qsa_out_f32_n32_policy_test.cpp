// CPU-only: no device, weights, cache construction or command submission.
#include "flash/FlashFloatDenseCache.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
using splash::flash::flashQSAOutF32N32Enabled;
using splash::flash::flashQSAOutF32N32Geometry;
namespace {
uint64_t checks = 0;
void expect(bool condition, const char *reason) {
  ++checks;
  if (!condition) throw std::runtime_error(reason);
}
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--flag") {
      const char *value = std::getenv("SPLASH_FLASH_QSA_OUT_F32_N32");
      const bool valid = !value || std::string_view(value) == "0" || std::string_view(value) == "1";
      bool threw = false, actual = false;
      try { actual = flashQSAOutF32N32Enabled(); }
      catch (const std::invalid_argument &) { threw = true; }
      expect(valid != threw, "strict environment parsing differs");
      if (valid) expect(actual == (value && std::string_view(value) == "1"), "flag state differs");
    } else {
      for (uint32_t layer = 0; layer < 50; ++layer)
        for (uint32_t rows = 0; rows <= 17; ++rows)
          for (uint32_t bits : {4u,5u,6u,8u}) {
            const auto prefix = "language_model.model.layers." + std::to_string(layer) + ".self_attn.o_proj";
            const bool expected = layer < 48 && layer % 4 == 3 && rows >= 4 && rows <= 16 && (bits == 5 || bits == 6);
            expect(flashQSAOutF32N32Geometry(prefix,rows,2560,6144,bits,64)==expected, "source/row/format geometry differs");
          }
      for (const char *prefix : {"language_model.lm_head","language_model.model.layers.3.self_attn.q_proj",
          "language_model.model.layers.3.linear_attn.out_proj","language_model.model.layers.3.mlp.shared_expert.down_proj",
          "language_model.model.layers.3.self_attn.o_proj.extra","language_model.model.layers.-1.self_attn.o_proj",
          "language_model.model.layers.003.self_attn.o_proj","language_model.model.layers.x.self_attn.o_proj",
          "language_model.mtp.layers.3.self_attn.o_proj","language_model.model.hyper_connection_mixer.input_mix_weight_up"})
        expect(!flashQSAOutF32N32Geometry(prefix,4,2560,6144,6,64), "unsupported role/prefix admitted");
      for (uint32_t n : {0u,32u,2559u,2561u,UINT32_MAX})
        expect(!flashQSAOutF32N32Geometry("language_model.model.layers.15.self_attn.o_proj",4,n,6144,6,64), "changed output geometry admitted");
      for (uint32_t k : {0u,32u,6143u,6145u,UINT32_MAX})
        expect(!flashQSAOutF32N32Geometry("language_model.model.layers.15.self_attn.o_proj",4,2560,k,6,64), "changed input geometry admitted");
      for (uint32_t group : {0u,32u,63u,65u,128u,UINT32_MAX})
        expect(!flashQSAOutF32N32Geometry("language_model.model.layers.15.self_attn.o_proj",4,2560,6144,6,group), "unsupported group admitted");
      expect(!flashQSAOutF32N32Geometry("language_model.model.layers.15.self_attn.o_proj",UINT32_MAX,2560,6144,6,64), "overflow rows admitted");
      // Existing measured policy is independent of the new route flag.
      expect(!splash::flash::flashFloatDenseSmallRowsPolicy("language_model.model.layers.3.self_attn.o_proj",4,2560,6144,4,64), "Q4 raw policy changed");
      expect(splash::flash::flashFloatDenseSmallRowsPolicy("language_model.model.layers.31.self_attn.o_proj",4,2560,6144,8,64)==splash::flash::FlashFloatDenseSmallRowsTile::M8N64, "Q8 policy changed");
    }
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
