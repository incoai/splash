`goldens.json` holds every SHA-256 the weight tests compare against:

- `gguf_dequantization`: upstream GGML's fp32 dequantization of the reference
  fixture of each GGUF format (`gguf-reference`). They pin the CPU reference
  (`dev/tests/engine/GgufFormatReference.hpp`) to llama.cpp 7ab4ee7, and
  PQ2_0, which upstream GGML lacks, to PrismML-Eng/llama.cpp 01ae597, so no
  Splash change touches them.
- `gguf_images`: every image `gguf-preparation` prepares from its dense and
  MoE GGUF fixtures.
- `affine_images`: every image `run_affine_preparation.py` prepares from its
  dense and MoE checkpoints and its DFlash2 draft checkpoint.
- `vision_image`: the file every `run_vision_preparation.py` tower prepares.

A prepared-image hash fails on any change of prepared bytes. When a change
means to change them:

1. Give the new bytes a new cache key, or caches of the old bytes are served.
   The key records the preparation identity, which fingerprints the files
   listed in `INPUTS` of `dev/tools/weight_preparation_identity.py`, and the
   image's plan: an edit to one of those files or a changed plan gives a new
   key. If the bytes changed through another file, add that file to `INPUTS`.
2. Change the independent oracles to the intended bytes: the serialized
   `expected` images in `fixture()` of `run_affine_preparation.py` and
   `run_vision_preparation.py`, and for GGUF the CPU reference planes
   (`GgufFormatReference.hpp`) and the expected sections
   `gguf_preparation_test.mm` builds from its sources. Both Python drivers
   compare every prepared file with their oracle before they report its hash,
   so until the oracle matches they fail without one.
3. Run the three drivers directly: in `make test-engine-cpu` and
   `test-engine-metal` each is one line of a recipe that stops at its first
   failing line, and the vision and affine drivers come first.

   ```sh
   make build/splash.metallib build/engine-tests/vision-preparation \
     build/engine-tests/affine-preparation build/engine-tests/gguf-preparation
   python3 dev/tests/engine/run_vision_preparation.py build/engine-tests/vision-preparation \
     dev/tests/fixtures/weight-goldens/goldens.json
   MTL_SHADER_VALIDATION=1 python3 dev/tests/engine/run_affine_preparation.py \
     build/engine-tests/affine-preparation build/splash.metallib dev/tests/fixtures/weight-goldens/goldens.json
   MTL_SHADER_VALIDATION=1 build/engine-tests/gguf-preparation build/splash.metallib \
     dev/tests/fixtures/weight-goldens/goldens.json
   ```

   `gguf-preparation` prints each mismatching image with its new hash, and
   the Python drivers' assertions print the hashes they got.
4. Check that only the images the change should touch moved, then write the
   new hashes here in the same commit, which says which images changed and
   why.

The dequantization hashes change only with the fixture itself. Regenerate
them with a libggml-base built from llama.cpp 7ab4ee7 (PrismML-Eng/llama.cpp
01ae597 for PQ2_0; its other formats decode as upstream's):
`SPLASH_GGML_ORACLE=<libggml-base.dylib> build/engine-tests/gguf-reference dev/tests/fixtures/weight-goldens/goldens.json`
compares the reference with GGML and prints GGML's hash of each format.
