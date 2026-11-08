These clips test the engine's `video-frames` command and the server's video
preparation (`dev/tests/server/test_videos.py`). `generate.py` writes them with
FFmpeg; each frame of the `index` clips draws its index as a grid the tests read
back, so a test sees exactly which frame the decoder returned.

- `index.mp4`: H.264, 160x96, 60 frames at 30 fps, keyframes 12 frames apart
  and B-frames, so the stream is reordered and an edit list shifts it to start
  at 0.
- `index_hevc10.mov`: the same frames as 10-bit HEVC.
- `index_prores.mov`: the first 20 as ProRes 422 Proxy.
- `index_vfr.mp4`: the same frames without 30 to 44, the others at their
  times, so half a second is missing.
- `index_rot90.mp4`, `index_rot180.mp4`, `index_rot270.mp4`: `index.mp4` with
  a display matrix turning it counterclockwise.
- `index_truncated.mp4`: `index.mp4` with its index first, cut through its
  media data.
- `index_single.mp4`: one frame.
- `index_vp9.mp4`: VP9, which decodes in hardware only.
- `index_av1.mp4`: the first 20 frames as AV1, which decodes only on Macs
  with a hardware AV1 decoder.
- `index_hlg.mp4`: the first 12 as 10-bit HEVC tagged HLG, tone mapped to SDR.
- `index.ts`: the first 30 frames in an MPEG transport stream.
- `index_sar2.mp4`: four frames of pixels twice as wide as tall.
- `index.flv`: H.264 in FLV, a container AVFoundation does not read.
- `pattern_bgra.mov`, `pattern_bgra_rot90.mov`: four 64x48 frames of a sharp
  colour pattern, uncompressed BGRA, which decode exactly, so resizing compares
  byte for byte with Pillow's.
