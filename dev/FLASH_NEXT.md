# Experimental Qwen3.8 Flash-Next runtime

This path imports a compatible local Qwen3.8 Flash-Next MLX checkpoint into a
byte-preserving, 16 KiB aligned Splash bundle. The importer validates tensor
metadata and hashes the copied payload. It does not alter the original model.
The Flash worker is separate from the existing DFlash 2 engine.

On an Apple silicon Mac with Xcode and Python 3.12 or newer:

```sh
make -j4 flash-next BUILD=build/flash-next
./splash serve --model local/Qwen3.8-Flash-Next \
  --local-model /absolute/path/to/Flash-Next-MLX-checkpoint \
  --max-context 16K --no-webui
```

The first `--local-model` launch imports into `install/local-models/`. Later
launches can pass `--local-package` with that imported directory. Model files,
build products, and local profiles are excluded from Git. Flash-Next currently
accepts text input only; `/status` and `/v1/models` report this capability.

The M5 Ultra routes are opt-in. `SPLASH_FLASH_MTP_DRAFT_DEPTH=3` selects the
tested singleton draft depth for the source-built route. SSD n-gram streaming
can be selected with `--ple-ssd-streaming`; `--ple-ssd-cache-mb` bounds its row
cache. Bulk QSA prefill additionally requires
`SPLASH_FLASH_QSA_F32=1`, `SPLASH_FLASH_QSA_MPP=1`, and
`SPLASH_FLASH_QSA_ROW_TILES=1` before setting
`SPLASH_FLASH_QSA_BULK_PREFILL=1` and
`SPLASH_FLASH_QSA_BULK_PREFILL_SG8=1`. Its gate covers only a fresh 2,048-row
prefill, with the ordinary route retained for other shapes. These settings
must be qualified together for the checkpoint and hardware being served.

The local M5 Ultra measurements were made with a pinned checkpoint, profile,
and build. They do not establish throughput for this rebased source build.
There is no published Flash-Next Splash model package in this change, and no
general hardware default is selected.
