#pragma once

// Plans the affine images (AffinePreparation.hpp) of a safetensors checkpoint:
// a header block, then 16 KiB-aligned sections, each bound to the checkpoint
// tensors it reads. AffineTarget.cpp plans an MLX target with it,
// DraftCheckpoint.cpp a DFlash2 draft.

#include "Checked.hpp"
#include "model/AffinePreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model::affine {

// An image of a header block; append places each section after it.
[[nodiscard]] inline Image image(std::string name, std::string_view magic, uint32_t layer, uint32_t type) {
  return {std::move(name), std::string(magic), layer, type, kWeightFileAlignment, {}, {}};
}

inline void append(Image &image, Section section) {
  section.offset = image.bytes;
  image.bytes = alignWeightOffset(image.bytes + section.bytes);
  image.sections.push_back(std::move(section));
}

// A tensor copied as stored, BF16 or U32.
inline void copy(Image &image, const std::string &name, std::vector<uint64_t> shape,
                 const std::string &dtype = "BF16") {
  Section section;
  section.bytes = dtype == "U32" ? 4 : kBFloat16Bytes;
  for (uint64_t dimension : shape)
    section.bytes = checkedMultiply<WeightStoreError>(section.bytes, dimension, "affine tensor");
  section.input = {name, {dtype}, std::move(shape)};
  append(image, std::move(section));
}

// Binds every input of image to its checkpoint tensor, which must have one of
// the input's dtypes and its shape, once the checkpoint states the
// quantization of every affine module the image reads.
inline void bind(Image &image, const SafetensorsCheckpoint &source) {
  const auto bindInput = [&](Input &input) {
    const SourceTensor &tensor = source.require(input.name);
    if (std::find(input.dtypes.begin(), input.dtypes.end(), tensor.dtype) == input.dtypes.end() ||
        tensor.shape != input.shape)
      throw WeightStoreError("source tensor type or shape does not match: " + input.name);
    input.tensor = &tensor;
  };
  for (const auto &[module, bits] : image.quantized) source.requireQuantization(module, bits);
  for (Section &section : image.sections) {
    if (section.parts.empty()) bindInput(section.input);
    for (ProjectionPart &part : section.parts)
      for (Input &field : part.fields) bindInput(field);
  }
}

} // namespace splash::model::affine
