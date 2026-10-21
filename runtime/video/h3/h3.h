/*
 * h3: MiniMax-H3 on-device engine (M5 Pro, macOS 27) -- its C ABI, which Splash's video engine process
 * (runtime/video/VideoRuntime.mm) drives; the engine runs on Splash's Metal backend and weight readers.
 *
 * One engine per request. The host supplies the request options and the text-encoder hidden states (Qwen3-VL-32B,
 * layer 50); the engine owns the device, the weights, the activations and every kernel choice, and returns latents,
 * frames, the waveform and the mp4. Phases:
 *
 *   h3_create            device, kernel library, request geometry
 *   h3_prep              text hidden states -> layout, RoPE, adaLN tables, refined text, schedule, noise
 *   h3_dit_load          DiT weights (resident or SSD-streamed), key-range table, activations; prepared layout
 *   h3_denoise           all steps: forward + Euler update, per-step callback      (or h3_dit_forward: one forward)
 *   h3_dit_release       frees the DiT memory (weights, slots, activations)
 *   h3_decode            video VAE: final latents -> uint8 frames
 *   h3_preview           TAEH3 preview decoder: final latents -> uint8 frames (+ yuv420p)
 *   h3_decode_audio      audio VAE (BigVGAN): final audio latents -> stereo waveform, 32 kHz
 *   h3_write_mp4         (no engine) uint8 frames + waveform -> H.264 / AAC mp4 (AVFoundation)
 *   h3_destroy
 *
 * Every call returns H3_OK or an error code; h3_error() has the message. Inputs are borrowed for the duration of the
 * call (h3_dit_load: until h3_dit_release). The engine reads no environment variable; its phase summaries and errors
 * go to stderr as Splash's log lines.
 */
#ifndef H3_H3_H
#define H3_H3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_ABI_VERSION 2
#define H3_MAX_QKFIX_BLOCKS 8

typedef enum h3_status {
  H3_OK = 0,
  H3_ERROR_ARGUMENT = 1, /* request or inputs inconsistent */
  H3_ERROR_IO = 2,       /* file missing, unreadable or malformed (weights, stream image) */
  H3_ERROR_METAL = 3,    /* device, kernel library, pipeline or allocation failure */
  H3_ERROR_GPU = 4,      /* a command buffer failed */
  H3_ERROR_STATE = 5,    /* phase called out of order */
} h3_status;

typedef enum h3_variant { H3_VARIANT_TURBO = 0, H3_VARIANT_DEFAULT = 1 } h3_variant;
typedef enum h3_vae_precision { H3_VAE_W8 = 0, H3_VAE_F16 = 1 } h3_vae_precision;

typedef struct h3_engine h3_engine;

/* Where the engine finds its files. */
typedef struct h3_config {
  const char *metallib; /* splash.metallib, whose video group holds the engine's kernels */
  const char *weights;  /* weights root: dit-turbo/, dit-default/, vae/, preview/ */
} h3_config;

/* User-facing request options. */
typedef struct h3_request {
  h3_variant variant;                         /* turbo: 8 steps, default: 50 (from the schedule) */
  int32_t frames;                             /* 17 n + 5 at 24 fps: 3 s = 73, 10 s = 243, 15 s = 362 */
  int32_t width, height;                      /* 1344 x 768 */
  int32_t window_chunk, window_radius;        /* window softmax: 5 / 1; anchor frames first and last */
  int32_t qkfix;                              /* fp16 Q / K in the QKFIX blocks */
  int32_t qkfix_blocks[H3_MAX_QKFIX_BLOCKS];  /* 45, 48, 49 */
  int32_t n_qkfix_blocks;
  int32_t stream;                             /* SSD weight streaming */
  int32_t stream_slots;                       /* block slots in the ring (8) */
  const char *stream_image;                   /* NULL: <weights>/dit-<variant>/stream.bin */
  h3_vae_precision vae_precision;
} h3_request;

/* Shapes that follow from the request (no engine needed). */
typedef struct h3_geometry {
  int32_t frames, width, height;
  int32_t latent_frames, latent_height, latent_width;
  int32_t video_rows, audio_rows; /* DiT latent rows: [video_rows][96], [audio_rows][32] */
  int32_t decode_frames;          /* frames of h3_decode */
  int32_t preview_frames;         /* frames of h3_preview */
  int32_t audio_samples;          /* samples per channel of h3_decode_audio (800 per audio latent) */
} h3_geometry;

/* Text conditioning of h3_prep: the hidden states of Qwen3-VL-32B-Instruct after decoder layer 50 of 64
 * (hidden_states[50], unnormalized), fp32 [text_rows][5120], all rows text (the prompt tokenized without special
 * tokens or chat template). */
typedef struct h3_prep_inputs {
  const float *hidden;
  int32_t text_rows; /* >= 2 */
  int32_t steps;     /* 0: the variant's (turbo 8, default 50) */
  uint64_t seed;     /* noise: MLX random.normal under mx.random.seed(seed) */
} h3_prep_inputs;

/* Outputs of h3_prep, owned by the engine (valid until the next h3_prep, h3_prep_release or h3_destroy). They are
 * bit-identical to the former Python prep and feed h3_dit_load (text_rows, tags, rope, text) and h3_denoise
 * (ada_blocks, ada_final, video_t, audio_t, video_sigmas, audio_sigmas, video_noise, audio_noise) directly. */
typedef struct h3_prepared {
  int32_t text_rows, rows, steps, distinct_t;
  const uint8_t *tags;                      /* [rows] 1 text, 2 audio, 0 video */
  const uint16_t *rope;                     /* bf16 [rows][96]: cos 48 | sin 48 */
  const float *pos;                         /* [rows][3] (t, h, w) */
  const uint16_t *text;                     /* bf16 [text_rows][5376] refined text */
  const uint16_t *ada_blocks;               /* bf16 [steps][50][3][6][5376] */
  const uint16_t *ada_final;                /* bf16 [steps][3][2][5376] */
  const float *video_t, *audio_t;           /* [steps] */
  const float *video_sigmas, *audio_sigmas; /* [steps + 1] */
  const float *table;                       /* [distinct_t] sorted distinct timesteps */
  const int64_t *video_index, *audio_index; /* [steps] into table */
  const float *video_noise;                 /* [video_rows][96] */
  const float *audio_noise;                 /* [audio_rows][32] */
  double seconds;
} h3_prepared;

/* Prepared layout of the packed DiT sequence [text L | audio | video] (h3_prep). */
typedef struct h3_dit_inputs {
  int32_t text_rows;     /* L */
  const uint8_t *tags;   /* [M] modality: 0 video, 1 text, 2 audio */
  const uint16_t *rope;  /* bf16 [M][96]: cos 48 | sin 48 */
  const uint16_t *text;  /* bf16 [L][5376] refined text */
} h3_dit_inputs;

/* Denoising schedule (rectified-flow Euler, eta 0, one forward per step). */
typedef struct h3_schedule {
  int32_t steps;
  const uint16_t *ada_blocks;               /* bf16 [steps][50][3][6][5376] */
  const uint16_t *ada_final;                /* bf16 [steps][3][2][5376] */
  const float *video_t, *audio_t;           /* [steps] */
  const float *video_sigmas, *audio_sigmas; /* [steps + 1] */
} h3_schedule;

typedef struct h3_forward_stats {
  double wall_ms, embed_ms, blocks_span_ms, blocks_gpu_ms, final_ms;
  double linear_ms, attention_ms, other_ms, branch_ms; /* 0: not measured (one command runs a block) */
  double gpu_busy_ms, gpu_idle_ms, gpu_gap_max_ms;      /* GPU time of the block commands, the rest of their span; 0 */
  int32_t gpu_gaps, command_buffers;                    /* 0; block commands */
  double stream_wait_ms, stream_wait_max_ms, stream_read_gb, stream_read_s, stream_read_max_ms;
  int32_t stream_waits, stream_blocks;
  double device_mb;
} h3_forward_stats;

typedef struct h3_dit_info {
  int32_t rows, padded_rows, text_rows, audio_rows, video_rows;
  int32_t window_items, chunked_groups, partial_slots;
  double density;           /* allowed (q, k) pairs / rows^2 */
  int32_t stream_slots;     /* 0: weights resident */
  double weights_mb;        /* resident weights + stream slots */
  double activations_mb;    /* MemoryPlan buffers */
  double device_mb;
} h3_dit_info;

typedef struct h3_decode_stats {
  double wall_s, gpu_s, host_copy_s, upload_s, gpu_span_s;
  int64_t nonfinite;
  double device_mb;
} h3_decode_stats;

typedef struct h3_preview_stats {
  double wall_s, gpu_s, host_copy_s;
  int64_t nonfinite;
  int32_t chunks, frames;
  double device_mb;
} h3_preview_stats;

typedef struct h3_audio_stats {
  double wall_s, gpu_s;
  int64_t samples, nonfinite; /* samples per channel; non-finite values in the waveform */
  double device_mb;
} h3_audio_stats;

typedef struct h3_mp4_stats {
  double wall_s;
  int32_t frames;
  int64_t audio_samples; /* per channel */
  double video_seconds, audio_seconds;
  int64_t bytes;
  int32_t video_bitrate; /* bits/s */
} h3_mp4_stats;

/* Called after every denoising step with the updated latents ([video_rows][96], [audio_rows][32]). */
typedef void (*h3_step_callback)(void *user, int32_t step, const h3_forward_stats *stats, const float *video,
                                 const float *audio);

h3_request h3_default_request(void);
h3_status h3_geometry_of(const h3_request *request, h3_geometry *out);
/* Key-range table of the window softmax for this request and text length: int32 words; returns the word count
 * (call with capacity 0 to size the buffer), or -1 on a bad request. */
int64_t h3_window_table(const h3_request *request, int32_t text_rows, int32_t *words, int64_t capacity);

h3_status h3_create(const h3_config *config, const h3_request *request, h3_engine **out);
void h3_destroy(h3_engine *engine);
const char *h3_error(const h3_engine *engine);

/* Request-constant DiT inputs (weights: <weights>/dit-<variant>/prep.safetensors, vdn-w8-blockNN / vdn-globals LoRA
 * factors, <weights>/fl2va/transformer adaLN bases). */
h3_status h3_prep(h3_engine *engine, const h3_prep_inputs *inputs, h3_prepared *out);
/* The last h3_prep's arrays as a prep directory (layout.npz, sched.npz, ada_blocks.npy, ada_final.npy, text_ref.npy,
 * noise.npz, prep.json; the files of the former Python prep). */
h3_status h3_prep_write(h3_engine *engine, const char *dir);
void h3_prep_release(h3_engine *engine);

h3_status h3_dit_load(h3_engine *engine, const h3_dit_inputs *inputs);
h3_status h3_dit_info_of(const h3_engine *engine, h3_dit_info *out);
h3_status h3_dit_forward(h3_engine *engine, const uint16_t *ada_blocks, const uint16_t *ada_final,
                         const float *video_in, const float *audio_in, float *video_out, float *audio_out,
                         h3_forward_stats *stats);
h3_status h3_denoise(h3_engine *engine, const h3_schedule *schedule, const float *video_noise,
                     const float *audio_noise, float *video_out, float *audio_out, h3_step_callback callback,
                     void *user);
void h3_dit_release(h3_engine *engine);

h3_status h3_decode(h3_engine *engine, const float *video_latents, uint8_t *frames, h3_decode_stats *stats);
h3_status h3_preview(h3_engine *engine, const float *video_latents, uint8_t *frames, uint8_t *yuv420,
                     h3_preview_stats *stats);
/* audio_latents: final audio rows [audio_rows][32] (channel-major, as h3_denoise returns them); wav: [2][audio_samples]
 * fp32 in [-1, 1], 32 kHz. */
h3_status h3_decode_audio(h3_engine *engine, const float *audio_latents, float *wav, h3_audio_stats *stats);

/* MP4 through AVFoundation (no engine needed): frames uint8 RGB [frame_count][height][width][3] at 24 fps -> H.264
 * High (19.8 Mb/s at 1344 x 768, >= the libx264 CRF 18 quality), wav (nullable) fp32 [2][samples] at 32 kHz -> AAC-LC
 * 192 kb/s. Every frame is kept; both tracks start at 0. */
h3_status h3_write_mp4(const char *path, const uint8_t *frames, int32_t frame_count, int32_t width, int32_t height,
                       const float *wav, int64_t samples, h3_mp4_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* H3_H3_H */
