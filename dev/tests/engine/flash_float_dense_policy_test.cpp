// CPU only: exercises policy selection without creating a Metal device,
// loading model weights, or constructing a dense cache.
#include "flash/FlashFloatDenseCache.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {
using splash::flash::FlashFloatDenseSmallRowsTile;
using splash::flash::flashFloatDenseSmallRowsPolicy;
using Tile = FlashFloatDenseSmallRowsTile;

static_assert(noexcept(flashFloatDenseSmallRowsPolicy(std::string_view{}, 0, 0, 0, 0, 0)));
static_assert(std::is_same_v<decltype(flashFloatDenseSmallRowsPolicy("", 0, 0, 0, 0, 0)),
                             std::optional<Tile>>);

struct Geometry {
  std::string_view prefix;
  uint32_t outputSize;
  uint32_t inputSize;
  uint32_t bits;
  uint32_t groupSize;
};

constexpr Geometry head{"language_model.lm_head", 248320, 2560, 8, 64};
constexpr Geometry qkv4{"language_model.model.layers.0.linear_attn.in_proj_qkv",
                        10240, 2560, 4, 64};
constexpr Geometry qkv5{"language_model.model.layers.0.linear_attn.in_proj_qkv",
                        10240, 2560, 5, 64};
constexpr Geometry z6{"language_model.model.layers.0.linear_attn.in_proj_z",
                      6144, 2560, 6, 64};
constexpr Geometry z5g128{"language_model.model.layers.0.linear_attn.in_proj_z",
                          6144, 2560, 5, 128};
constexpr Geometry q5{"language_model.model.layers.3.self_attn.q_proj",
                      12288, 2560, 5, 64};
constexpr Geometry pleValue4{"language_model.model.layers.1.ple.value_proj",
                             2560, 2560, 4, 64};
constexpr Geometry sharedDown8{"language_model.model.layers.0.mlp.shared_expert.down_proj",
                               2560, 640, 8, 128};

uint64_t checks = 0;

std::string describe(const Geometry &geometry, uint32_t rows) {
  return std::string(geometry.prefix) + " R" + std::to_string(rows) +
      " N" + std::to_string(geometry.outputSize) + " K" + std::to_string(geometry.inputSize) +
      " Q" + std::to_string(geometry.bits) + " G" + std::to_string(geometry.groupSize);
}

std::optional<Tile> select(const Geometry &geometry, uint32_t rows) {
  return flashFloatDenseSmallRowsPolicy(geometry.prefix, rows, geometry.outputSize,
                                       geometry.inputSize, geometry.bits, geometry.groupSize);
}

void expectRaw(const Geometry &geometry, uint32_t rows) {
  ++checks;
  if (select(geometry, rows))
    throw std::runtime_error(describe(geometry, rows) + " unexpectedly selected a float cache");
}

void expectCached(const Geometry &geometry, uint32_t rows,
                  std::optional<Tile> expectedTile = std::nullopt) {
  ++checks;
  const auto selected = select(geometry, rows);
  if (!selected)
    throw std::runtime_error(describe(geometry, rows) + " unexpectedly selected raw affine");
  if (expectedTile && selected != expectedTile)
    throw std::runtime_error(describe(geometry, rows) + " selected the wrong tile");
}

void headPolicy() {
  for (uint32_t rows = 2; rows <= 16; ++rows)
    expectCached(head, rows, rows <= 8 ? Tile::M8N64 : Tile::M16N64);
  for (uint32_t rows : {0u, 1u, 17u, std::numeric_limits<uint32_t>::max()})
    expectRaw(head, rows);

  for (uint32_t rows : {2u, 8u, 16u}) {
    for (uint32_t n : {0u, 64u, 248256u, 248321u, std::numeric_limits<uint32_t>::max()}) {
      auto altered = head; altered.outputSize = n; expectRaw(altered, rows);
    }
    for (uint32_t k : {0u, 2496u, 2624u, 2561u, std::numeric_limits<uint32_t>::max()}) {
      auto altered = head; altered.inputSize = k; expectRaw(altered, rows);
    }
    for (uint32_t bits : {0u, 4u, 5u, 6u, 7u, 9u, std::numeric_limits<uint32_t>::max()}) {
      auto altered = head; altered.bits = bits; expectRaw(altered, rows);
    }
    for (uint32_t group : {0u, 32u, 65u, 128u, std::numeric_limits<uint32_t>::max()}) {
      auto altered = head; altered.groupSize = group; expectRaw(altered, rows);
    }
  }
}

void alwaysRawProjections() {
  // HC down, shared gate/up, and the small QSA outputs lose even at R16.
  for (std::string_view prefix : {
      "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down",
      "language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_down",
      "language_model.model.hyper_connection_mixer.input_mix_weight_down"})
    for (uint32_t bits : {4u, 5u, 6u, 8u})
      for (uint32_t rows = 1; rows <= 16; ++rows)
        expectRaw({prefix, 320, 10240, bits, 64}, rows);

  for (std::string_view prefix : {
      "language_model.model.layers.0.mlp.shared_expert.gate_proj",
      "language_model.model.layers.0.mlp.shared_expert.up_proj"})
    for (uint32_t rows = 1; rows <= 16; ++rows)
      expectRaw({prefix, 640, 2560, 8, 128}, rows);

  constexpr std::array qsaRaw{
      Geometry{"language_model.model.layers.3.self_attn.k_proj", 512, 2560, 4, 64},
      Geometry{"language_model.model.layers.3.self_attn.v_proj", 512, 2560, 4, 64},
      Geometry{"language_model.model.layers.3.self_attn.indexer.index_qk_proj", 640, 2560, 4, 64},
      Geometry{"language_model.model.layers.3.self_attn.o_proj", 2560, 6144, 4, 64},
  };
  for (const auto &geometry : qsaRaw)
    for (uint32_t rows = 1; rows <= 16; ++rows) expectRaw(geometry, rows);
}

void measuredBodyDecisions() {
  // These expectations are independent explicit regressions for measured rows.
  // Exact R4 and intermediate tiles are checked in the padding regressions.
  struct Measured {
    Geometry geometry;
    bool row4;
    bool row8;
    bool row16;
  };
  constexpr std::array cases{
      Measured{qkv4, false, true, true},
      Measured{qkv5, false, false, true},
      Measured{z6, false, true, true},
      Measured{q5, true, true, true},
      Measured{pleValue4, true, true, true},
      Measured{sharedDown8, false, false, true},
  };
  for (const auto &entry : cases) {
    for (uint32_t rows : {0u, 1u, 2u, 3u, 17u, std::numeric_limits<uint32_t>::max()})
      expectRaw(entry.geometry, rows);
    for (const auto &[rows, cached] :
         std::array{std::pair{4u, entry.row4}, std::pair{8u, entry.row8}, std::pair{16u, entry.row16}}) {
      if (cached) expectCached(entry.geometry, rows);
      else expectRaw(entry.geometry, rows);
    }
  }

  // An R16 win cannot admit unmeasured intermediate rows when M16 lost at R8.
  for (const auto &geometry : {qkv5, sharedDown8, z5g128}) {
    for (uint32_t rows : {0u, 1u, 2u, 3u, 17u}) expectRaw(geometry, rows);
    for (uint32_t rows = 9; rows <= 15; ++rows) expectRaw(geometry, rows);
  }
}

void intermediateRowPolicy() {
  // R9..15 use M16 only when that padded tile already won the measured R8 case.
  constexpr std::array winners{
      qkv4,
      Geometry{qkv4.prefix, 10240, 2560, 6, 64},
      z6,
      Geometry{q5.prefix, 12288, 2560, 4, 64},
      q5,
      Geometry{q5.prefix, 12288, 2560, 6, 64},
      Geometry{q5.prefix, 12288, 2560, 8, 64},
      Geometry{"language_model.model.layers.3.self_attn.o_proj", 2560, 6144, 5, 64},
      Geometry{"language_model.model.layers.3.self_attn.o_proj", 2560, 6144, 6, 64},
      Geometry{"language_model.model.layers.3.self_attn.o_proj", 2560, 6144, 8, 64},
      Geometry{"language_model.model.layers.3.self_attn.k_proj", 512, 2560, 6, 64},
      Geometry{"language_model.model.layers.3.self_attn.indexer.index_qk_proj", 640, 2560, 5, 64},
      Geometry{"language_model.model.layers.3.self_attn.indexer.index_qk_proj", 640, 2560, 6, 64},
      Geometry{"language_model.model.layers.3.self_attn.indexer.index_qk_proj", 640, 2560, 8, 64},
      Geometry{"language_model.model.layers.1.ple.key_proj", 10240, 2560, 4, 64},
      pleValue4,
  };
  for (const auto &geometry : winners) {
    for (uint32_t rows : {0u, 1u, 2u, 3u, 17u}) expectRaw(geometry, rows);
    for (uint32_t rows = 9; rows <= 15; ++rows)
      expectCached(geometry, rows, Tile::M16N64);
  }
  for (std::string_view prefix : {
      "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up",
      "language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_up",
      "language_model.model.hyper_connection_mixer.input_mix_weight_up"})
    for (uint32_t bits : {4u, 5u, 6u, 8u}) {
      const Geometry geometry{prefix, 10240, 320, bits, 64};
      for (uint32_t rows : {0u, 1u, 2u, 3u, 17u}) expectRaw(geometry, rows);
      for (uint32_t rows = 9; rows <= 15; ++rows)
        expectCached(geometry, rows, Tile::M16N64);
    }

  constexpr std::array excluded{
      qkv5, z5g128, sharedDown8,
      Geometry{"language_model.model.layers.0.linear_attn.out_proj", 2560, 6144, 5, 128},
      Geometry{"language_model.model.layers.3.self_attn.k_proj", 512, 2560, 5, 64},
      Geometry{"language_model.model.layers.3.self_attn.k_proj", 512, 2560, 8, 64},
      Geometry{"language_model.model.layers.3.self_attn.v_proj", 512, 2560, 5, 64},
      Geometry{"language_model.model.layers.3.self_attn.v_proj", 512, 2560, 6, 128},
      Geometry{"language_model.model.layers.3.self_attn.v_proj", 512, 2560, 8, 64},
  };
  for (const auto &geometry : excluded)
    for (uint32_t rows = 9; rows <= 15; ++rows) expectRaw(geometry, rows);
}

void paddedR4RowsPolicy() {
  struct Qualified {
    Geometry geometry;
    Tile tile;
  };
  constexpr std::array qualified{
      Qualified{{qkv4.prefix, 10240, 2560, 6, 64}, Tile::M8N64},
      Qualified{q5, Tile::M8N64},
      Qualified{{q5.prefix, 12288, 2560, 6, 64}, Tile::M8N64},
      Qualified{{q5.prefix, 12288, 2560, 8, 64}, Tile::M16N64},
      Qualified{{"language_model.model.layers.3.self_attn.o_proj", 2560, 6144, 5, 64}, Tile::M16N64},
      Qualified{{"language_model.model.layers.3.self_attn.o_proj", 2560, 6144, 6, 64}, Tile::M16N64},
      Qualified{{"language_model.model.layers.3.self_attn.o_proj", 2560, 6144, 8, 64}, Tile::M8N64},
      Qualified{pleValue4, Tile::M8N64},
  };
  for (const auto &entry : qualified)
    for (uint32_t rows = 4; rows <= 7; ++rows)
      expectCached(entry.geometry, rows, entry.tile);
  for (const auto &geometry : {qkv4, qkv5})
    for (uint32_t rows = 4; rows <= 7; ++rows) expectRaw(geometry, rows);
  for (std::string_view prefix : {
      "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up",
      "language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_up",
      "language_model.model.hyper_connection_mixer.input_mix_weight_up"})
    for (uint32_t bits : {4u, 5u, 6u, 8u})
      for (uint32_t rows = 4; rows <= 7; ++rows)
        expectCached({prefix, 10240, 320, bits, 64}, rows, Tile::M8N64);
}

void singleRowsStayRaw() {
  for (auto geometry : {head, qkv4, qkv5, z6, z5g128, q5, pleValue4, sharedDown8})
    for (uint32_t bits : {4u, 5u, 6u, 8u})
      for (uint32_t group : {32u, 64u, 128u}) {
        geometry.bits = bits; geometry.groupSize = group;
        expectRaw(geometry, 1);
      }
}

void unknownPrefixesStayRaw() {
  constexpr std::array unknown{
      Geometry{"", 248320, 2560, 8, 64},
      Geometry{"unknown.lm_head", 248320, 2560, 8, 64},
      Geometry{"language_model.lm_head.extra", 248320, 2560, 8, 64},
      Geometry{"mtp.lm_head", 248320, 2560, 8, 64},
      Geometry{"unknown.linear_attn.in_proj_qkv", 10240, 2560, 4, 64},
      Geometry{"mtp.layers.0.linear_attn.in_proj_qkv", 10240, 2560, 4, 64},
      Geometry{"unknown.self_attn.q_proj", 12288, 2560, 5, 64},
      Geometry{"mtp.layers.0.self_attn.q_proj", 12288, 2560, 5, 64},
      Geometry{"language_model.model.layers.3.self_attn.unmeasured_proj", 12288, 2560, 5, 64},
      Geometry{"mtp.layers.0.ple.value_proj", 2560, 2560, 4, 64},
  };
  for (const auto &geometry : unknown)
    for (uint32_t rows = 0; rows <= 17; ++rows) expectRaw(geometry, rows);
}

void unmeasuredBodyGeometriesStayRaw() {
  for (const auto &geometry : {qkv4, z6, q5, pleValue4})
    for (uint32_t rows : {8u, 16u}) {
      for (uint32_t n : {0u, geometry.outputSize - 64, geometry.outputSize + 64,
                         std::numeric_limits<uint32_t>::max()}) {
        auto altered = geometry; altered.outputSize = n; expectRaw(altered, rows);
      }
      for (uint32_t k : {0u, geometry.inputSize - 128, geometry.inputSize + 128,
                         std::numeric_limits<uint32_t>::max()}) {
        auto altered = geometry; altered.inputSize = k; expectRaw(altered, rows);
      }
      for (uint32_t bits : {0u, 3u, 7u, 9u, std::numeric_limits<uint32_t>::max()}) {
        auto altered = geometry; altered.bits = bits; expectRaw(altered, rows);
      }
      for (uint32_t group : {0u, 32u, 63u, 96u, std::numeric_limits<uint32_t>::max()}) {
        auto altered = geometry; altered.groupSize = group; expectRaw(altered, rows);
      }
    }
}
} // namespace

int main() {
  try {
    headPolicy();
    alwaysRawProjections();
    measuredBodyDecisions();
    intermediateRowPolicy();
    paddedR4RowsPolicy();
    singleRowsStayRaw();
    unknownPrefixesStayRaw();
    unmeasuredBodyGeometriesStayRaw();
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_float_dense_policy_test: " << error.what() << '\n';
    return 1;
  }
}
