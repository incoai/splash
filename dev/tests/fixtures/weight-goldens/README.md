`goldens.json` holds every SHA-256 the weight tests compare against:

- `gguf_dequantization`: upstream GGML's fp32 dequantization of the reference
  fixture of each GGUF format, and of block_nvfp4 rows of llama.cpp's NVFP4
  (`ggml-nvfp4`, which the loader converts into NVFP4's native rows)
  (`gguf-reference`). They pin the CPU reference
  (`dev/tests/engine/GgufFormatReference.hpp`) to llama.cpp 7ab4ee7, and
  PQ2_0, which upstream GGML lacks, to PrismML-Eng/llama.cpp 01ae597, so no
  Splash change touches them.
- `gguf_images`: every image `gguf-preparation` writes from its dense and MoE
  GGUF fixtures.
- `draft_images`: every image `draft-preparation` writes from the DFlash2
  draft checkpoint `run_draft_preparation.py` writes.
- `vision_image`: the image written from every `run_vision_preparation.py`
  tower.

An image hash fails on any change of the image's bytes. When a change means
to change them:

1. Change the independent oracles to the intended bytes: the serialized
   `expected` images in `fixture()` of `run_vision_preparation.py`, the
   af4g64 rows `run_draft_preparation.py` quantizes and, for GGUF and the
   draft, the CPU reference planes (`GgufFormatReference.hpp`) and the
   expected sections `gguf_preparation_test.mm` and
   `draft_preparation_test.mm` build from their sources. Each driver
   compares every image with its oracle before it reports its hash, so
   until the oracle matches it fails without one.
2. Run the three drivers directly: in `make test-engine-cpu` and
   `test-engine-metal` each is one line of a recipe that stops at its first
   failing line, and the vision and draft drivers come first.

   ```sh
   make build/splash.metallib build/engine-tests/vision-preparation \
     build/engine-tests/draft-preparation build/engine-tests/gguf-preparation
   python3 dev/tests/engine/run_vision_preparation.py build/engine-tests/vision-preparation \
     dev/tests/fixtures/weight-goldens/goldens.json
   MTL_SHADER_VALIDATION=1 python3 dev/tests/engine/run_draft_preparation.py \
     build/engine-tests/draft-preparation build/splash.metallib dev/tests/fixtures/weight-goldens/goldens.json
   MTL_SHADER_VALIDATION=1 build/engine-tests/gguf-preparation build/splash.metallib \
     dev/tests/fixtures/weight-goldens/goldens.json dev/tests/fixtures/mlx-quantization/fixture.json
   ```

   `draft-preparation` and `gguf-preparation` print each mismatching image
   with its new hash, and the vision driver's assertion prints the hash it
   got.
3. Check that only the images the change should touch moved, then write the
   new hashes here in the same commit, which says which images changed and
   why.

The dequantization hashes change only with the fixture itself. Regenerate
them with a libggml-base built from llama.cpp 7ab4ee7 (PrismML-Eng/llama.cpp
01ae597 for PQ2_0; its other formats decode as upstream's):
`SPLASH_GGML_ORACLE=<libggml-base.dylib> build/engine-tests/gguf-reference dev/tests/fixtures/weight-goldens/goldens.json dev/tests/fixtures/mlx-quantization/fixture.json`
compares the reference with GGML and prints GGML's hash of each format.

The MLX formats have no GGML hash: `gguf-reference` and `gguf-preparation`
compare them with MLX's own values in
`dev/tests/fixtures/mlx-quantization/fixture.json`, which
`python dev/tools/mlx_quantization_fixture.py dev/tests/fixtures/mlx-quantization/fixture.json`
regenerates (it needs MLX: `pip install mlx`).
