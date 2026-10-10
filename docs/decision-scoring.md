# Decision scoring: `POST /v1/score`

Prefill-only scoring of fixed answers. A client that asks a model a yes/no or
multiple-choice question gets the probabilities of its answers from one
prefill of the prompt, with no decoding. The prompt is rendered and tokenized
exactly as `/v1/chat/completions` does. The answer position is the end of the
assistant turn's opening plus an optional `assistant_prefix` such as
`{"same":`. The labels are the single tokens that may follow there.

## Survey: where things live

| Concern | Where |
| --- | --- |
| HTTP routing, API-key auth, error bodies | `server/server.py` (`do_POST`: path allow-list, auth before body parsing, `_safe_error`/`_error`) |
| Chat request parsing and rendering | `server/frontend.py`: `_prepare_prompt` (model, `reasoning_effort` default, `chat_template_kwargs`, messages, tools) and `_render_prompt` (template, images, tokenization, generation-prompt probe) |
| Reasoning effort `none` | `template_options` sets `enable_thinking=False`. Qwen templates then render the generation prompt as `<|im_start|>assistant\n<think>\n\n</think>\n\n`, an empty closed think block. `_generation_prompt` reports whether the prompt leaves a think block open (`RenderedPrompt.thinking`). |
| Tokenizer | `server/tokenization.py` `PromptTokenizer.encode` (the HF fast tokenizer with `add_special_tokens=False`, reusing tokens at `<|im_end|>` boundaries). Prompts with images go through `_render_image_tokens` (offset mapping) and `_expand_image_pads`. |
| Scheduler | `runtime/engine/Scheduler.cpp`. Phases: queued, waiting_resources, waiting_prefix, prefill, decode, waiting_mask. Priorities: foreground, normal, background. A ragged prefill takes the shortest remaining sequences up to a 2048-row budget. Decode batches up to `maximumBatchWidth` = 4 lanes. `waiting_resources` holds requests without a state cell or KV pages, up to a 30 s resource wait. |
| Prefill and final-position logits | `runtime/model/Runtime.cpp`: ragged prefill, then each finished prompt's final row is copied to row 0 of its decode lane (`ops::RowCopy`) and `encodeInitialSelections` runs the target head into `DecodeTensor::Logits` (fp32, whole vocabulary). Generation runs the policy (sampler) there. A score request (`scoreTokens`) reads the row instead. |
| Sampler | `runtime/ops/Sampling.cpp` (greedy argmax, top-k/top-p, DFlash/MTP acceptance). It does not run for score requests. |
| Prefix cache | `runtime/engine/Cache.cpp`, `KvCache`, `StateCache`. KV is cached in 32-token pages. Qwen-Next GDN layers keep recurrent state, which the engine checkpoints at page boundaries (rolling checkpoints every 4096 tokens, junctions where requests diverge, the state before the generation prompt). A later request resumes from the deepest cached boundary of its prefix. |
| KV page lifecycle | Pages are allocated at admission, written by prefill and verify, and published to the prefix cache when a request finishes. They are released to the pool or kept by the cache under LRU. |
| Images | The frontend resizes images and expands placeholders into image spans. The engine runs the vision tower during prefill and injects image rows at the span positions. The prefix cache keys image blocks by content digest. |
| `/status` counters | Native JSON (`runtime/engine/Status.cpp`: scheduler, requests, metrics) plus frontend sections (`server/frontend.py` `status`). |

## Design

Splash already has a score-only native request: `EngineRequest.scoreTokens`.
The prompt is prefilled to its end, no token is generated, and the raw fp32
logits at the requested token ids are returned in `DoneEvent.optionLogits`.
`/v1/judgments` and `/v1/systemone` use it with a fixed prompt shape.
`/v1/score` reuses it with Chat's own rendering and adds a full-vocabulary
readout.

1. **Native readout.** When a score request's last prompt chunk completes,
   `Runtime.cpp` reads the fp32 logits row of the final position. From it,
   besides the label logits, it computes:
   - the log-sum-exp over the whole vocabulary (248 320 tokens, fp64
     accumulation),
   - the 20 highest logits, with ties broken toward the lower id.

   The `DoneEvent` carries both. Native protocol version 9 appends
   `f64 log_normalizer, u32 top_count, u32 ids[], f32 logits[]` after the
   option logits, only when there are option logits; generation Done events
   are byte-identical to version 8's. CPU cost is about 1 ms per score request.
2. **Rendering.** `prepare_score` calls `_prepare_prompt` and `_render_prompt`,
   the same calls Chat uses, so the template, reasoning default, images and
   tokens are identical. A prompt that opens a think block is refused, because
   the answer would sit inside it. The assistant prefix is appended to the
   rendered text, and the result must extend Chat's tokens unchanged.
3. **Labels.** Each label must be exactly one token in its position: the
   prompt text plus prefix plus label must tokenize as the prompt tokens plus
   one token. The same check is used for `/v1/judgments` answer slots. A
   failing label gets a 400 with its token breakdown. `label_token_ids`
   bypasses the text check.
4. **Batch invariance.** A prefill command that contains a score request runs
   its projections split-free (`QwenTargetPrefillBuffers::splitFree`,
   `ops::LinearWorkload::splitFree`). Chunks of up to 32 rows use the decode
   tile with a single K pass, whose outputs equal the large prefill tile's bit
   for bit. Without this, a K split reorders fp32 sums in small chunks. Small
   chunks come from ragged prefills of several requests and from cache-resumed
   suffixes, and in a MoE model a reordered sum can flip an expert choice.
   Measured on 35B, the same request alone and in a ragged prefill with others
   then differed by up to 0.63 in a label logit (0.09 in probability).
   Split-free, its logits are bit-identical alone, with up to 24 concurrent
   others, and cold or warm in the prefix cache. The command also runs on the
   GPU alone: the Neural Engine's fp16 share of a dense FFN would make the
   logits depend on its split. Prefill commands without a score request keep
   their plans, so Chat's numerics do not change.
5. **Scheduling.** A score job goes through the same scheduler as any prefill:
   priorities, ragged prefills and decode share. It never decodes and never
   drafts, so MTP/DFlash do not run. It releases its lane when its prefill
   ends.
6. **Images.** Images are allowed. Score requests were text-only before;
   prefill encodes images the same way for scoring as for generation.
7. **Cache.** The job's `generation_prompt_tokens` is Chat's generation prompt
   plus the prefix tokens. Its reusable state therefore stays before the
   assistant turn, where a Chat request with the same messages or another
   score request with a different prefix resumes.
8. **Readout math (server).**
   - `logprob` = logit − log_normalizer: full-vocabulary log-softmax at T = 1,
     no penalties.
   - `prob` = softmax over the labels' logits / `temperature`.
   - `label_mass` = Σ exp(logprob) over the labels.
   - `argmax` = the label with the highest logit.
   - `top_k` = the first `return_top_k` (at most 20) of the native top tokens.
9. **Counters.** `/status` gains `score.score_requests`,
   `score.score_prompt_tokens` and `score.score_cached_tokens`, and
   `latency.score_request` is a latency histogram.

Chat and the other existing endpoints are unchanged. Done events for
generation keep their bytes, and only score jobs read the new fields.

## API

### `POST /v1/score`

```bash
curl -s http://127.0.0.1:8001/v1/score \
  -H "Authorization: Bearer $SPLASH_API_KEY" -H 'Content-Type: application/json' -d '{
  "model": "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
  "messages": [
    {"role": "system", "content": "Compare two products. Answer with JSON {\"same\": true|false, \"difference\": \"...\"}."},
    {"role": "user", "content": "A: Bosch GSR 12V-15, 2x2Ah\nB: Bosch GSR 12V-15 Professional, 2 batteries 2.0 Ah"}
  ],
  "assistant_prefix": "{\"same\":",
  "labels": [" true", " false"],
  "return_top_k": 5
}'
```

```json
{
  "object": "score",
  "model": "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
  "labels": [
    {"label": " true", "token_id": 830, "logprob": -0.021, "prob": 0.979, "logit": 31.2},
    {"label": " false", "token_id": 895, "logprob": -3.87, "prob": 0.021, "logit": 27.4}
  ],
  "label_mass": 0.9998,
  "argmax": " true",
  "top_k": [{"token_id": 830, "token": " true", "logprob": -0.021}],
  "usage": {"prompt_tokens": 96, "cached_tokens": 64},
  "timing": {"queue_ms": 0.4, "prefill_ms": 61.0, "total_ms": 61.4}
}
```

Fields:
- `messages`: as in Chat, including images.
- `tools`: optional, as in Chat. The template renders them into the prompt, so
  a score request with an agent's system prompt and tools starts as that
  agent's requests do.
- `reasoning_effort` and `chat_template_kwargs`: as in Chat. The server's
  `--default-reasoning-effort` applies, and thinking must end up disabled.
- `assistant_prefix`: optional text after the assistant turn's opening.
- `labels`: two or more labels, each one token in that position, or
  `label_token_ids`, which overrides the text labels.
- `temperature`: optional, default 1. It divides only the label logits used
  for `prob`.
- `return_top_k`: optional, 0 to 20.
- `priority` and `timeout`: as in Chat.

Whether a label carries a leading space depends on the tokenizer and on what
the model writes. For Qwen3.6/3.8, `{"same":` followed by ` true` is the
tokenization of the model's own output `{"same": true`. A label that is not
one token gets a 400 that lists its tokens:

```json
{"error": {"message": "each label must be exactly one token after the assistant prefix",
  "type": "invalid_request_error", "code": "invalid_request_error",
  "details": {"labels": [{"label": "true false", "tokens": [{"id": 1866, "text": "true"}, {"id": 895, "text": " false"}]}]}}}
```

### Warming an agent's prompt

A score request prefills its prompt and decodes nothing, and the prefix cache
keeps that prompt as it keeps any other. One that carries an agent's system
prompt and tools therefore warms the agent's next requests: they resume from
the cache up to where their own messages begin. Its labels only have to be
valid; their probabilities do not matter here.

```bash
curl -s http://127.0.0.1:8001/v1/score -H "Authorization: Bearer $SPLASH_API_KEY" \
  -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "system", "content": "You are a coding agent. ..."},
               {"role": "user", "content": "ping"}],
  "tools": [{"type": "function", "function": {"name": "read_file",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
  "labels": [" yes", " no"],
  "reasoning_effort": "none"
}'
```

### Choice by letter

```bash
curl -s http://127.0.0.1:8001/v1/score -H "Authorization: Bearer $SPLASH_API_KEY" \
  -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the product?\nA) drill\nB) battery\nC) charger\nD) accessory\nAnswer with the letter only."}],
  "labels": ["A", "B", "C", "D"]
}'
```

### With an image

```bash
IMG=$(base64 -i product.jpg)
curl -s http://127.0.0.1:8001/v1/score -H "Authorization: Bearer $SPLASH_API_KEY" \
  -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "Is this a cordless drill? Answer yes or no."}]}],
  "labels": ["yes", "no"]
}'
```

### Batch

`{"requests": [ ... ]}` scores up to 64 independent requests in one call. A
top-level `model` and `priority` apply to every item. All jobs are submitted
together. Prompts that share their beginning, such as our product against
1–3 candidates, are prefilled once up to the shared boundary: the engine lets
later requests wait for the first one's state there (`waiting_prefix`). The
response is `{"object": "list", "data": [ ...one score result per request ]}`.

### `POST /v1/decisions`

Several single-token questions over one shared input. Each question becomes a
score request whose user message is the input followed by the question. The
input's tokens are therefore a common prefix the cache computes once.
- `yes_no` asks for one word, yes or no.
- `choice` lists its options as letters A–H.
- `score` lists its levels as 0–9.

```bash
curl -s http://127.0.0.1:8001/v1/decisions -H "Authorization: Bearer $SPLASH_API_KEY" \
  -H 'Content-Type: application/json' -d '{
  "system": "You classify product listings.",
  "input": "Listing: Bosch GSR 12V-15 Professional cordless drill, 2x 2.0 Ah battery, charger, case",
  "reasoning_effort": "none",
  "questions": {
    "is_tool": {"type": "yes_no", "question": "Is it a power tool?"},
    "category": {"type": "choice", "question": "Which category fits best?",
                 "options": ["accessory", "drill", "battery", "charger"]},
    "completeness": {"type": "score", "question": "How complete is the listing?",
                     "levels": ["poor", "fair", "good"]}
  }
}'
```

```json
{"object": "decisions", "model": "...",
 "answers": {
  "is_tool": {"type": "yes_no", "answer": "yes", "p_yes": 0.998,
              "probabilities": {"yes": 0.998, "no": 0.002}, "label_mass": 0.97},
  "category": {"type": "choice", "answer": "B", "option": "drill",
               "probabilities": {"A": 0.01, "B": 0.98, "C": 0.005, "D": 0.005},
               "legend": {"A": "accessory", "B": "drill", "C": "battery", "D": "charger"}, "label_mass": 0.99},
  "completeness": {"type": "score", "answer": 2, "level": "good", "expected": 1.93,
                   "probabilities": {"0": 0.01, "1": 0.05, "2": 0.94},
                   "legend": {"0": "poor", "1": "fair", "2": "good"}, "label_mass": 0.98}},
 "usage": {"prompt_tokens": 410, "cached_tokens": 192}}
```

`label_mass` shows how much of the model's next-token probability falls on
the expected answers. A low value means the model would rather write
something else, for example a capitalized "Yes"; such a question deserves
rewording before its probabilities are trusted.

## Limits

- Labels must be single tokens. A multi-token answer needs a prefix that ends
  where the answers diverge, or `label_token_ids`.
- Bit-identical results across packing and cache need the prefix rows to come
  from score prefills, or from prefill chunks of more than 32 rows. A prefix
  that a Chat request computed in small chunks carries its K-split rounding,
  which a score request resuming from it inherits.
- Score requests occupy a lane only during prefill. They follow the same
  priority and decode-share rules as Chat prefill.
