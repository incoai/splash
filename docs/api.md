# API

Splash serves OpenAI's Chat Completions, Completions and Responses APIs and
Anthropic's Messages API. OpenAI SDKs take the base URL
`http://127.0.0.1:8000/v1` and Anthropic SDKs `http://127.0.0.1:8000`, on the
port `--port` sets. With [authentication](server.md#network-and-authentication) off any
API key works; with it on, clients send the server's key. `model` names the
served model ID or one of its [aliases](server.md#api-model-aliases), as `/v1/models`
lists them; another name returns 404 `model_not_found`, which Messages, whose
errors carry no code, answers as a `not_found_error`. Chat, text completion and
Responses requests that omit it, or send it empty, get the served model;
Messages requires it.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/completions` | OpenAI text completions |
| `POST /v1/responses` | OpenAI Responses |
| `GET`, `DELETE /v1/responses/{id}` | A stored response ([Responses state](#responses-state)) |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/messages/count_tokens` | A Messages request's prompt tokens, images included, as `{"input_tokens": N}`, without inference |
| `GET /v1/models`, `/v1/models/{id}` | The served model and its aliases, with the context limit and what reasoning efforts do ([default reasoning effort](server.md#default-reasoning-effort)) |
| `POST /tokenize`, `/apply-template` | A text's tokens, or a conversation's rendered prompt, without inference |
| `POST /v1/judgments`, `/v1/systemone` | Scores over fixed answers ([judgment contracts](#judgment-contracts)) |
| `POST /v1/memory/release` | Gives the weights back now, as the idle release does, when no request is in service; the next request restores them ([explicit release](internals.md#explicit-weights-release)). It takes no body (or `{}`); `200` with `{"released": true, "weights": …}`, `409 engine_busy` while the engine serves a request or restores its weights, `503 engine_busy` when the engine does not answer within 5 s, and `503 engine_recovering` or `server_shutdown` while the engine restarts or the server stops |
| `GET /status`, `/metrics` | JSON status and Prometheus metrics ([status](#status-readiness-and-metrics)) |
| `GET /health` | 200 while the server answers; public |
| `GET /ready` | 200 while the engine is ready, else 503; public |
| `GET /` | The chat page; public, and gone with `--no-webui` |

## Request fields

A field set to `null` is one left out, at a request's top level on every
endpoint and in the optional fields Splash reads in its messages, content blocks
and options, as clients that write every field they do not set as `null` mean
it; in tool arguments, schemas and template variables `null` is a value. Chat
Completions, text completions, Responses and Messages share these fields:

- `stream`: stream server-sent events. In Chat and text completions,
  `stream_options.include_usage` adds usage to the stream; their non-streaming
  responses always include it.
- `priority`: `foreground`, `normal` (the default) or `background`. A request
  waits while one of a higher priority runs, and when lanes or memory run short
  a higher priority suspends a running request of a lower one
  ([admission and eviction](internals.md#admission-and-eviction)).
- `timeout`: the seconds a request may take from its arrival; it can only
  shorten `--request-timeout`. A request that runs out returns 504
  `request_timeout`.
- `seed`: a 64-bit integer, random when omitted or `-1`; another negative seed
  is taken as its 64 bits ([sampling](#sampling)).
- `stop`, in Messages `stop_sequences`: a string or up to four strings. A stop
  string ends the output where the model writes it, inside a tool call or
  structured output too, which it cuts as the token limit does: a call it cuts
  keeps the arguments it has, and no call of the output counts as made; a cut
  structured answer is not validated.
- `return_progress`, with `stream` true: prompt progress before the output
  ([output limits and streaming](#output-limits-and-streaming)).
- `cache_salt`: a nonempty string. A request shares cached prompts, their KV and
  states, only with requests of the same salt, and a request without one only
  with others without one, as vLLM's field does. A salt no other request uses
  makes it compute its whole prompt itself, except its images' encoded rows,
  which it may take from one cache the runtime keeps for every request, keyed
  by the image and not the salt.

Chat Completions also takes `messages`, `tools`, `tool_choice` and
`parallel_tool_calls` ([tool calls](#tool-calls)), `response_format`
([structured output](#structured-output)), `reasoning_effort` and
`chat_template_kwargs` ([default reasoning effort](server.md#default-reasoning-effort)),
`thinking_token_budget` ([thinking token budget](server.md#thinking-token-budget)),
`preserve_thinking`, which reaches the template (the Qwen templates keep earlier
turns' reasoning with it), `max_completion_tokens` or `max_tokens`,
`ignore_eos` and the [sampling](#sampling) fields. The reply's reasoning comes
in `reasoning_content`, beside `content`, in the message and in each stream
delta, and an assistant message sent back may carry its `reasoning_content`
(or `reasoning` text). Message content may hold Responses' `input_text`,
`output_text` and `input_image` parts too, and a `refusal` part, an assistant's
refusal, is text. A history tool call whose `arguments` are not a JSON object,
such as one the output limit cut, is kept as text, so the conversation
continues. `n` other than 1 and `logprobs: true` return 400.

`POST /v1/completions` serves OpenAI's legacy text completions. `prompt` is one
string, which the loaded tokenizer encodes as a raw prompt, adding its own
special tokens such as a BOS and recognizing special-token strings, or one array
of token IDs from its vocabulary. No chat template, reasoning split, tools or
images apply: `text` is the generated text, decoded without special tokens.
`max_tokens` defaults to 16, OpenAI's default for the endpoint; that or a
larger value is lowered to what the context leaves, as in Chat
([output limits](#output-limits-and-streaming)). The sampling fields
below, `seed`, `stop`, `priority`, `timeout` and `stream` with
`stream_options.include_usage` work as in Chat. Batched prompts, `suffix`,
`echo`, `logprobs`, `best_of` and `n` other than 1 are rejected.

Responses takes `input` and `instructions`, function `tools` and `namespace`
groups of them, custom (freeform) `tools`, `tool_choice`,
`parallel_tool_calls`, `reasoning.effort`, `thinking_token_budget`,
`text.format` for structured output, `max_output_tokens`, the sampling fields,
Chat's `ignore_eos`, `chat_template_kwargs` and `preserve_thinking`, and `store`
and `previous_response_id` ([Responses state](#responses-state)). Reasoning
comes back as `reasoning` output items. `conversation`, `background: true` and
`truncation` other than `disabled` return 400; Splash compacts nothing, so
`context_management` leaves the input as sent.

A custom tool, such as Codex's `apply_patch`, reaches the model as a function
of one string parameter, `input`, whose description carries the tool's
`format` (its grammar, or plain text); the grammar describes the input and
does not constrain it. A call comes back as a `custom_tool_call` item with the
input as the model wrote it, streamed as one
`response.custom_tool_call_input.delta` and `.done` when the call closes;
`custom_tool_call` and `custom_tool_call_output` items are taken as input, and
`tool_choice` `{"type": "custom", "name": ...}` names one.

Messages takes `max_tokens`, `system`, `messages`, custom `tools` (every one
declared in the prompt, `defer_loading` ones too, so a `tool_result`'s
`tool_reference` blocks from a tool search render as text naming the tools
found; a system message's `tool_addition` and `tool_removal` blocks change
them in order: a `tool_definition` adds a tool or replaces the one of its name,
and a removed tool is left out until a later block adds it again),
`tool_choice` (`auto`, `any`, `tool` or `none`, with
`disable_parallel_tool_use`), `stop_sequences`, `temperature`, `top_p`, `top_k`,
`thinking`, `output_config` and Chat's `chat_template_kwargs`, whose
`enable_thinking` outranks `thinking`. Otherwise a request reasons only when
`thinking` sets it: `{"type": "enabled", "budget_tokens": N}` and
`{"type": "adaptive"}` reason at `output_config.effort`, one of `low`, `medium`,
`high` (the default), `xhigh` and `max`, while `{"type": "disabled"}` and an
omitted `thinking` do not. Enabled thinking requires `budget_tokens`, an integer
of at least 1024, which caps it ([thinking token
budget](server.md#thinking-token-budget)). Reasoning comes back in `thinking` blocks.
With `thinking.display` `omitted` (or `updates`) a block carries no text, and its
reasoning travels encrypted in its `signature` under the key in `thinking.key`
([files Splash writes](server.md#files-splash-writes)); a later request that sends the
block back has the reasoning restored. A signature Splash cannot read, such as
another provider's, keeps the block's visible text without recovering the
private reasoning. Structured output is `output_config.format` (or
`output_format`) with a JSON schema. In `context_management`, a
`clear_thinking_20251015` edit with `keep: "all"` sets `preserve_thinking`;
Splash makes no other edit, so the model reads the conversation as sent.

`POST /tokenize` accepts `{"content":"hello","add_special":false}` and returns
`{"tokens":[...]}` using the loaded tokenizer. Special-token strings are recognized;
`parse_special:false` and `with_pieces:true` are unsupported.
`POST /apply-template` accepts Chat-style `messages`, `tools` and reasoning options,
and returns `{"prompt":"..."}` using the same template as generation.
`add_generation_prompt` defaults to true. Image prompts retain textual placeholders;
raw tokenization does not account for image embeddings (use `count_tokens` for that).
Both endpoints run without inference and share bounded preparation capacity with
`count_tokens`; they can inspect prompts larger than the serving context limit.

## Request limits and timeouts

HTTP request bodies need a `Content-Length` and are limited to 128 MiB;
`serve --max-request-size 256M` overrides this. Concurrent input bytes share a
budget of at least 512 MiB (or twice the request limit), including retained
generation inputs. This is an input-byte budget, not a process RSS limit: large
ASCII/base64 strings can use roughly twice their encoded size during JSON
parsing alone. Decoded images and object-heavy JSON need additional memory.
Oversized requests return 413; exhausted ingress capacity returns 503. Image and
model context limits apply independently. Stored Responses history is charged
before decoding. Uploads allow 30 seconds of inactivity; total upload time is
limited to 30 seconds plus the body size at 512 KiB/s (286 seconds for 128 MiB),
capped by `--request-timeout` when set. Timed-out uploads return 408 and release
their input reservation. An upload refused before it is read, such as one over
the shared budget, is still received on these terms, so a client that sends its
whole body before reading the response gets the refusal. An inference request
ends when its client disconnects, and a client that shuts down its sending side
after the request, as `nc` does at the end of its input, counts as disconnected
and gets no response. A response write waits up to 30 seconds for a client that
has stopped reading, past `--request-timeout` too, and the request counts
against `--queue-size` until then. `/status` reports `http.request_body_bytes`
and `http.max_request_bytes`. A body's JSON nests at most 128 levels of arrays
and objects; deeper is refused as invalid JSON. A history tool call's arguments
nested deeper stay text, as other arguments that are not a JSON object do.

The server holds at most `--queue-size` + 64 connections at once (96 by
default), each in a connection slot. A connection waits for its request without
a thread until the request's head, its request line and headers, has arrived, or
its client has closed; one of the server's worker threads, one per slot and all
started with the server, then reads and serves the request, and the connection
keeps its slot until it has been answered. The server takes every connection the
kernel holds at each wakeup and starts no thread while connections arrive: macOS
queues at most 128 connections the server has yet to take and resets those
beyond, so the accept loop must keep up with a burst. The head is due 30 seconds
after the server accepted the connection, which is closed without a response if
the head has not arrived by then; a head that passes 64 KiB goes to its thread
unfinished, to be read with 30 seconds allowed between reads. When every slot is
taken, a connection still waiting for its head, or receiving an upload refused
unread, gives way to the new one, the longest waiting first, so stalled clients
cannot lock others out; when every slot holds a request whose head has arrived,
the new connection is refused. A connection refused, or giving way, before its
request is read gets a fixed 503 with code `frontend_overloaded` in OpenAI's
shape whatever its path ([errors](#errors)). The server then reads and drops
what its client still sends until the client closes or 2 seconds pass, so that
the client reads the 503 rather than a reset, even one that sends its request
only after the refusal. At most 1024 such connections linger at once; one more
takes the place of the one refused longest ago. The server raises its soft limit
of open files to 10240 at start (macOS starts a process at 256), so the
connections it holds cannot exhaust it.

## Sampling

Chat, text completions and Responses sample with `temperature` (default 1.0,
in [0, 2]), `top_p` (0.95), `top_k` (20, a positive integer, 0 or -1), `min_p`
(0, in [0, 1]), `presence_penalty` and `frequency_penalty` (0, in [-2, 2]) and
`repetition_penalty` (1, positive); the defaults are Qwen's generation config,
which sets no penalties, so its recommended `presence_penalty` of 1.5 for
non-thinking use must be sent explicitly. Each field that is out of range
returns 400 naming it. As in vLLM, a nonzero temperature below 0.01 samples at
0.01, and the penalties rewrite the raw logits before temperature, for greedy
requests too: repetition divides a positive logit and multiplies a negative
one for every token of the prompt or of the output so far, and presence and
frequency lower the logit of every output token by `presence_penalty` plus
`frequency_penalty` times its count. Speculative decoding stays exact: each
verified draft position counts the draft tokens before it, so a penalized
request samples as it would without a draft. Splash does not implement
`logit_bias`: a non-empty one returns 400; `null` and `{}` are accepted. Like
vLLM and HF, repetition counts every prompt token: with the Qwen templates
that includes the tool-call syntax every tool-enabled system prompt carries,
earlier tool calls and reasoning, and the `<think>` markers of the generation
prompt, so a `repetition_penalty` above 1 can delay tool calls and the end of
reasoning and change names copied from the context. Anthropic Messages defines
no penalties and no `min_p`.

`--sampling-default NAME=VALUE` (repeatable) sets the value a sampling field
takes when a request leaves it out, in place of the default above. A value
the request sends still wins, and of two pairs for one name the last wins.
The server checks each value at startup against what a request may set.

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --sampling-default temperature=0.6 --sampling-default presence_penalty=1.5
```

In vLLM's order, after temperature `min_p` first drops every token less
likely than `min_p` times the most likely one, `top_k` then keeps the most
likely of the rest, every one when it is 0 or -1 or past the vocabulary, and
`top_p` then the fewest of those whose probabilities, renormalized over them,
sum past it. All three apply to the whole vocabulary, exactly, for the first
token, the verified draft positions and their corrections alike: the sampler
sums each row's softmax denominator, finds where its distribution ends
without sorting the vocabulary, and draws over every token the distribution
keeps. vLLM itself refuses a nonzero `min_p` with speculative decoding; here
it is one more cut of the target distribution, which acceptance and
correction read as they read the others, so sampling stays exact. A request
that keeps only its most likely token, at a `min_p` of 1 or a `top_k` of 1,
decodes greedily, its draft's proposals too: it selects what sampling would
but for equally likely tokens, of which it takes the lowest id.

A lane's arithmetic can depend on the batch it decodes in and on how its prompt
is chunked. On a GGUF MoE target's staged expert tiles, an expert that a step
routes at most two rows to runs them on the column tile, which sums in another
order than the staged matmul, so a row's result also depends on the rows decoded
or prefilled beside it. Concurrent requests change both, and chunk boundaries
also come from the cached boundary a request resumes from, from a boundary
another request waits to resume from, and from one whose state, or the KV past
it, memory could not hold. With `--state-format bf16` the state is also rounded
at each state a chunk captures (a replay point, junction or checkpoint). So on a
dense target, or a MoE target under `--verify-rows all`, a greedy or seeded
request repeats its output when it runs alone with the same cached prefixes;
alongside other requests, or with other prefixes cached, it can differ. A
request's own repeat is such a case: served from the prefix cache, it computes
only the rows past the state its first run cached, so its output can differ
from that run's, which computed the whole prompt; repeats meant to match each
take a `cache_salt` of their own (below). By
default a MoE target's decode cycles verify the rows the verify budget chooses
([Drafts](models.md#drafts)), which follow the machine's timing and the requests served
before, and a row's MoE result depends on which rows share its expert tiles:
there the same greedy (temperature 0) or seeded request can give different text
from run to run even alone on an idle server, as on other engines that batch
requests, while sampled outputs keep their exact distribution. A request that
restores nothing can still read cached KV: once a command completes, each block
it filled whose prefix, tokens and identity (images and cache salt) match a
cached block is swapped for that block's page, which another command computed,
and its later commands read it. A request suspended for memory or for a higher
priority can also resume from a state another request cached past the blocks it
had computed, and its replay then reads the KV of the blocks it restores there.
The Done line of the request log counts the tokens of both, as does
`metrics.cache.deduplicated_tokens` in Chat Completions and text completions
responses (in a stream, its usage chunk), and `/status` sums them in
`cache.deduplicated_kv_tokens`. One case goes uncounted: while a request is
suspended, blocks it computed can be evicted and computed again by another
request, and when it resumes it reads those as its own. A request with a
`cache_salt` no other request uses reads no cached KV, though its images'
encoded rows may come from another request's encoding of the same image, which
gives the same rows. Alone, it computes what it would on a server that served
nothing before, with two exceptions. The [Neural Engine
split](internals.md#neural-engine-prefill) may have stopped: the split stays off for the
life of the process, so such a server prefills on the GPU alone the chunks a
fresh one splits. And where other requests' cache leaves its states no free
memory, a state's prefill command can end at the state's boundary instead of
capturing it inside a longer command, which changes the chunks. Either way the
logits differ slightly.

## Output limits and streaming

Chat's `max_completion_tokens` or `max_tokens`, Responses' `max_output_tokens`,
and text completions' and Messages' `max_tokens` bound a response's output from
above: a larger value than the context the prompt leaves generates up to the
context limit, since clients such as Claude Code ask for the same limit on every
turn. Omitted from Chat or Responses, the output may use all the context the
prompt leaves: the context limit, and `--request-timeout` when set, are the only
server bounds on a request that names no limit. A response the context limit
ends finishes as one its limit ends; Messages gives it the `stop_reason`
`model_context_window_exceeded`, not `max_tokens`, and so does one the engine
ends before its limit at what the SSD cache or memory holds of a long request's
KV ([KV on the SSD](server.md#kv-on-the-ssd)). An
invalid value returns 400 naming the field the request sent. Messages refuses
a final assistant message (prefill) with 400; `count_tokens` still counts it.

Chat and text completions accept `"ignore_eos":true` (default false), as vLLM
and llama.cpp do: the model never selects its own stop tokens, and a draft
proposal of one is rejected, so generation runs to its output budget and
finishes with `length` unless a `stop` string ends it first. Benchmarks use it
to generate a fixed number of tokens. Constrained tool calls, output under
`tool_choice: "none"` and structured output generate under a grammar, which
decides where the output ends, so combining them with `ignore_eos` returns 400.

Streaming requests accept `"return_progress":true` (default false). Before output,
`prompt_progress` reports `{total, cache, processed, time_ms}`: prompt tokens,
initial cached tokens, completed tokens including cache, and elapsed milliseconds
since prefill admission. Updates follow completed chunks and never regress during
recovery; they are not a time estimate. Chat uses empty-delta chunks, text
completions empty-text chunks, Responses `response.in_progress`, and Messages
`ping`. Queueing and prompt preparation do not advance this counter.
Non-streaming requests cannot enable it.

Usage counts cached prompt tokens and the output tokens spent reasoning in each
API's own fields: Chat `prompt_tokens_details.cached_tokens` and
`completion_tokens_details.reasoning_tokens`, Responses
`input_tokens_details.cached_tokens` and `output_tokens_details.reasoning_tokens`,
and Messages `cache_read_input_tokens` (with `input_tokens` the uncached rest)
and `output_tokens_details.thinking_tokens`, in a stream's `message_delta`.
Responses `cache_write_tokens` is 0: clients price cache writes apart from the
rest of the uncached input, and the prefix cache keeps a prompt as prefill
computes it, at no cost of its own. So uncached input is `input_tokens` less
`cached_tokens`, as in Messages, which reports no `cache_creation_input_tokens`.

Chat and text completions include a llama-server-style `timings` object, both in
non-streaming responses and in the final finish-reason chunk of a stream,
even without `include_usage`. Messages and Responses include the same object
in a complete response, and in a stream in `message_delta` and in the response
of `response.completed` or `response.incomplete`. `prompt_n` and `predicted_n` are the full prompt
and output counts; `cache_n` is the cached prompt count. `prompt_ms` measures
native start to first emission, and `predicted_ms` measures first emission to
completion. These elapsed intervals exclude the initial admission queue and
are not isolated GPU timings. `prompt_per_second` uses only uncached prompt
tokens; `predicted_per_second` excludes the entire first emission (which can
contain multiple speculative tokens). Thus rates use tokens processed in the
measured interval, not the full counts. An unavailable rate is zero, including
responses completed in one emission. A request that drafted adds
llama-server's `draft_n` and `draft_n_accepted`: the tokens its draft proposed
that its cycles verified ([Drafts](models.md#drafts)) and those verification accepted,
but for accepted proposals past the output limit or from one of the model's
stop tokens on; those a `stop` string then cuts still count. The end tokens a
[thinking token budget](server.md#thinking-token-budget) puts in place of the draft's
proposals are neither. They count that request alone, so
they hold under concurrency. `metrics.speculation` carries them with the
request's speculative `cycles`, the decode cycles in which its draft proposed
tokens, and the Done line shows what the draft yields per cycle,
`1 + accepted / cycles`; a request whose thinking budget ran out also took the
end's tokens in its cycles.

## Tool calls

With `tool_choice` `auto`, parallel calls allowed and no tool marked
`"strict": true`, nothing constrains the output: the model writes its calls in
the chat template's layout as it would without a grammar, and each call reads
back as written. A grammar constrains the output when any of these holds:

- `tool_choice` is `required` or names a tool;
- `"parallel_tool_calls": false`;
- a tool is strict;
- the request also sets a response format;
- `tool_choice` is `"none"`, which renders the tools like any other choice and
  only prevents calls.

The grammar holds each call to the template's layout and an offered tool's
name, and a strict tool's arguments to its schema; other tools' arguments stay
free. A strict tool's parameters come in schema order, as OpenAI's strict mode
writes them, which requires every field; an optional field written after a
later one cannot be placed. A Responses tool that omits `strict` is not
strict.

The model writes each argument as text, which converts by the JSON types the
tool declares for that parameter. A type name reads whatever its case and the
space around it, as `STRING` reads as `string`, and the common aliases count as
the types they name: `str` and `text` as `string`, `int` as `integer`, `float`
as `number`, `bool` as `boolean`, `dict` as `object` and `list` as `array`. A
strict tool's grammar generates the type such a name stands for where its
dialect takes the name, as Draft 3 takes any: a parameter typed `int` is
written as an integer. Any other name, such as Draft 3's `any`, stands for no
JSON type, and a type that holds one constrains none, as one the schema does
not give: `["integer", "any"]` constrains no type, and
`{"type": "integer", "allOf": [{"type": "any"}]}` is an integer.

- A parameter whose type the schema does not give stays the text as written:
  one declared without a type, or with a type that holds a name of no JSON
  type; one the tool does not declare, unless `additionalProperties` gives its
  type; and every parameter of a tool whose schema gives no types.
- A parameter that cannot be a string takes the JSON its text spells, of
  whatever type; text that is not JSON stays text. Where the tool declares a
  boolean or null, `True`, `False` and `None` also read as `true`, `false` and
  `null`, as templates that write values through Jinja's `string` filter spell
  them.
- A parameter that may be a string takes the JSON its text spells only when
  that is of another declared type (an integer counts as a number); otherwise
  it stays the text as written.

| Declared type | Text | Argument |
| --- | --- | --- |
| `integer` | `42` | `42` |
| `integer` | `4.5` | `4.5` |
| `boolean` | `True` | `true` |
| `["string", "number"]` | `12` | `12` |
| `["string", "number"]` | `12:00` | `"12:00"` |
| `["string", "integer"]` | `0800` | `"0800"` |
| `str` | `42` | `"42"` |
| `any` | `7` | `"7"` |
| not declared | `7` | `"7"` |
| not declared | `seven` | `"seven"` |

Calls are not validated against their schemas; a client reports what its tool
rejects. A `stop` string may cut a call ([request fields](#request-fields)).

The output is read as the chat template lays calls out, in the family's
syntax ([chat formats](models.md#chat-formats)). In Qwen's XML a call opens at
`<tool_call>` and the newline before `<function=`, and a value ends at
`\n</parameter>\n` where the next parameter, `</function>` or `</tool_call>`
follows; in Flint's GLM tags a call opens at `<tool_call>`, the function's name
follows it up to the first tag, and a value ends at `</arg_value>` where the
next `<arg_key>` or `</tool_call>` follows. A value may hold the tags in any
other order, as a file an agent writes may. Tags written otherwise are text.
Besides:

- names lose the space around them, and a call that names no function is
  dropped, as is a `</think>` in text;
- a repeated parameter keeps its first value, which has streamed;
- text after a call stays content, with or without its `</tool_call>`;
- a call the token limit or a stop sequence cuts keeps the unfinished
  arguments it has, which Chat and Responses (with status `incomplete`) return
  and a Messages stream has streamed, while a complete Messages response leaves
  the call out;
- JSON that would decode to a lone surrogate or nest past 122 levels keeps its
  text, which writes back as valid JSON, so that the call's arguments read back
  within the 128 levels the server reads where a request carries them deepest,
  as a `tool_use` block's `input` in a Messages history, five levels into the
  body;
- a call's opening also ends the reasoning where a call may follow, but for a
  model that opens its reasoning itself, as Flint does, which closes it before
  a call.

`dev/tests/server/test_tool_call_reading.py` holds model outputs with the calls
and content they read as, and `dev/tests/server/test_chat_formats.py` Flint's.

A strict tool's schema is checked against its JSON Schema dialect on arrival,
and every tool's arguments are framed through schema composition and local
references, which give each parameter its types. A strict tool's framed
arguments are closed, so that an object or an array takes nothing it does not
declare. Its values are written in the family's tags: raw text where any string
may be the value, whose pattern, format and length go unenforced; otherwise one
of the values each member of the value's union lists, as its text, or JSON of a
member that lists none, leaving out array item bounds and `multipleOf` above 64
and patterns the grammar cannot compile (look-around, word boundaries,
backreferences). The framed schemas of a request's strict tools, whose grammars
are built together, are limited to 16 MiB in all. Any other tool is framed when
a call first names it, for its parameters' types alone, which take no copy of
its root schema as a grammar does, and the request's such tools share another
16 MiB. A strict tool's parameter names follow the rule for a tool's name
(below). What a grammar writes reads back as written: raw text ends at its
first `\n</parameter>\n`, and an enumerated value that holds one returns 400.

A strict tool's schema must pass that check, nest at most 64 levels of JSON
objects and arrays, in which an object's `properties` map is a level of its
own, and reference no remote document, or the request returns 400. Any other
tool is declared as sent, and its parameters take the types its schema
declares, valid or not; the arguments of one whose schema nests deeper or
allows no object, whose references form a cycle, or whose framing exceeds the
budget stay text, as those of a parameter it does not declare. In framing, a
keyword that holds another kind of value than it takes, such as the `null` Go
writes for a nil slice, constrains nothing, as one its dialect does not define,
and so do a value that is not a schema where one belongs, a type name that
stands for no JSON type, such as Draft 3's `any`, and a reference to nothing in
the document.

A tool's name is written in its calls' tags and read back from them, so it
holds 1 to 256 characters in Unicode NFC, which a tokenizer such as Qwen's
keeps text in, other than `<`, `>`, line breaks and lone surrogates, and does
not start or end with whitespace; a `tool_reference` names a tool the same way.
A Chat tool or named `tool_choice` that gives no `type` is a function. A
Responses `namespace` function goes by `namespace__name`. Hosted search is
unsupported; configure client-owned tools such as MCP.

## Structured output

Chat's `response_format` takes `{"type": "json_object"}` or
`{"type": "json_schema", "json_schema": {"schema": ...}}`. Responses takes them
in `text.format`, the second without the `json_schema` object around its
fields: `{"type": "json_schema", "name": ..., "schema": ..., "strict": ...}`.
Messages takes a schema in `output_config.format` (or
`output_format`) as `{"type": "json_schema", "schema": ...}`. A schema nests at
most 64 levels of JSON objects and arrays, as a strict tool's does, cannot
reference a remote document, and cannot combine `unevaluatedProperties` with
`patternProperties`. `response_format` constrains generation and validates
final output; it does not inject formatting instructions into the prompt.
Clients should describe their output requirements in their own messages.

## Images and PDFs

Images arrive inline as base64 `data:` URLs: Chat `image_url` parts, Responses
`input_image` items and Messages `image` blocks with a `base64` source; an image
URL to fetch returns 400. JPEG, PNG, WebP and GIF decode. An image file may
hold up to 32 MiB and 33,554,432 pixels, with an aspect ratio up to 200:1, and
is resized to at most `--max-image-pixels` (4,194,304 by default). A request
carries at most 64 images, PDF pages included.

PDFs arrive inline too: Chat `file` parts with `file_data`, Responses
`input_file` items and Messages `document` blocks with a base64
`application/pdf` source. Each page becomes its text and an image of at most
1,048,576 pixels. PDF input supports base64 documents within a shared 64 MiB
source/rendering budget and the native 64-image limit (one image per page).
Model context and isolated rendering limits also apply. URL inputs, `file_id`,
opening passwords and citations are unsupported. PDFs render one at a time: a
request waits for its turn and renders without holding any of the server's
preparation capacity, so other requests prepare meanwhile.

A model served with `--language-only` refuses both with 400 ([vision](models.md#vision)).

## Responses state

With `store` true, the default, a completed or incomplete response is kept with
its input history in a store of the server process, at most 64 MiB, which
drops the least recently used responses first and is lost when the server
restarts. A response too large for it reports `"store": false`.
`previous_response_id` continues a stored response: its history comes before
the request's `input`. An ID the store no longer holds returns 404
`previous_response_not_found`, on which clients resend the full history.
`GET /v1/responses/{id}` returns a stored response and
`DELETE /v1/responses/{id}` removes it; an unknown ID returns 404.
`/status.response_store` reports the store's entries, bytes, evictions, hits and
misses. Responses automatic truncation and unsupported history edits return
errors.

## Judgment contracts

`POST /v1/systemone` accepts the [TypeSafe System One](https://docs.typesafe.ai/)
request and response shapes: `noul`, `choice` and `score` questions over a shared
state. It works with the official `typesafe-sdk` (verified with 0.7.0). Use the
actual served model ID, not a hosted Jev model name; `/v1/models` answers both
OpenAI model discovery and the SDK's `models.list()`.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8000",
    api_key="local",  # Use SPLASH_API_KEY's value if server authentication is on.
    model="mlx-community/Qwen3.8-27B-4bit",
) as client:
    result = client.system_one(
        state={"message": "I was charged twice. Please fix this today."},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": None, "technical": None, "sales": None},
            ),
            "urgency": Score(
                instructions="How urgent is the request?",
                criteria=["No urgency", "This week", "Today"],
            ),
        },
    )
    print(result.choices["department"].choice)
```

`POST /v1/judgments` scores one [SemIf](https://github.com/TheoLeeCJ/SemIf) row of
2–16 options and returns raw option logits:

```bash
curl http://127.0.0.1:8000/v1/judgments \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "approval",
    "state": "The proposal is awaiting approval.",
    "question": "What is the current approval status?",
    "options": [
      {"id": "approved", "description": "Approval was explicitly given."},
      {"id": "pending", "description": "Approval has not been given."}
    ]
  }'
```

`POST /v1/judgments` preserves SemIf's `direct-options-v1` JSON serialization,
system prompt and A–P option order. It returns the exact rendered prompt's SHA-256
(each lone surrogate read as U+FFFD, as every prompt is tokenized),
answer token IDs, raw option logits, normalized probabilities and zero completion
tokens. Every answer label must round-trip as one token, including at the actual
assistant prompt boundary. Prompts render without thinking, as effort `none`
renders them. A model that opens its reasoning itself (Flint), which no effort
turns off, would open it where the generation prompt ends: its prompt ends with
an empty think block, `<think></think>`, and the answer slots are read after it,
where its answer starts. Besides the row it takes `model`, `timeout`,
`priority` and `cache_salt` as the generation APIs do ([request
fields](#request-fields)); other generation controls return errors rather than
silently changing the scoring protocol. SemIf-derived code retains its MIT
notice in `server/judgments.py`.

`POST /v1/systemone` requires the served `model`, a string/object/array `state`,
and a nonempty `questions` map, and takes `cache_salt` as the generation APIs
do. Instructions may be omitted, null or structured;
criteria descriptions may also be structured. Noul criteria may be omitted.
Choice and score domains contain 1–255 entries. Singletons return their sole
answer without inference. Other domains use deterministic, distinct single-token
slots selected from the tokenizer. All questions are validated before any inference.
A request holds at most 64 questions and 1M total prepared prompt tokens;
larger batches are rejected before any inference.
Questions run sequentially within a request under one shared deadline, allowing
prefix reuse without filling the admission queue; independent HTTP requests still
share the scheduler. Disconnects and timeouts cancel the current question.

Preparation renders each prompt once, then enforces the context limit and the
batch token budget before the per-slot boundary checks. A label can change only
the tokens after the prompt's last message boundary (`<|im_end|>`, or Flint's
`<|assistant|>`), which the tokenizer encodes on its own, so each check encodes
that end with the option appended rather than the whole prompt. The checks
observe the request deadline and stop when the client disconnects.
Prompts that exceed the context limit are rejected, not truncated.

System One validation uses 422 `detail` arrays; successful responses contain
`model`, `answers`, and `usage`, plus an `x-typesafe-request-id` header. SDK model
discovery reports an empty `release_date` because Splash records none for a model.
The official SDK is a client only, not a server dependency. API compatibility does
not imply Jev weights, accuracy, proprietary confidence semantics or calibration.

These are local model scores, not calibrated confidence. Probabilities are a
softmax over the declared answer slots. Choice/score `confidence` is normalized
entropy concentration, `1 - H(p) / log(K)`, not an estimate of correctness.
Score answers are probability-weighted level indices. Measure accuracy and
calibrate on representative held-out data before using decision thresholds.

## Status, readiness and metrics

`GET /status` returns instance identity and the effective context limit as JSON.
Proxy consumers can use these fields; additional fields may be added:

| Field | Meaning |
| --- | --- |
| `requests.submitted`, `completed`, `cancelled`, `failed` | Native request counters since engine start |
| `active_requests` | Each request the engine holds, ordered by `id`: its `id`, `phase` (`queued`, `waiting_resources`, `waiting_prefix`, `prefill`, `decode` or `waiting_mask`, and `failed` for one whose deadline passed while a cached state it restores was being read from disk, until that read drains), `priority`, `prompt_tokens` and how many of them are encoded (`prompt_processed`, cached ones included), `generated_tokens` of `max_new_tokens`, where a block-sparse target keeps its KV (`kv_mode`: `resident` or `spill`, only while its KV can go to the SSD), and `age_ms` since it arrived |
| `memory_actual.current_bytes`, `peak_bytes` | Metal allocations, not process RSS |
| `weights.idle_release_seconds`, `released`, `restoring`, `restores`, `restore_failures` | `--idle-release` in seconds (`null` for `off`), whether the weights' memory is released now, whether a request's restore of it is under way (it stays `released` until the last image is back), the times it was restored for a request since engine start, and the restores that gave up for want of memory |
| `identity.cache.rope_scaling` | With a `--max-context` past 256K, the target's RoPE scaling, which every cached key went through: `type` `yarn`, its `factor` and `original_context_tokens`; `null` otherwise |
| `streamed_table` | While the PLE table streams, the [streamed PLE table](server.md#ple-table-streaming)'s cache, stages, reads and prefetches; `null` otherwise |
| `streamed_experts` | While routed experts stream, the [streamed experts](server.md#streamed-experts)' store: its slots and resident experts, per use its loads, reads, read-ahead and host-step round trips, and its failure; `null` otherwise |
| `kv_spill` | While a block-sparse target's KV can go to the SSD, its [KV on the SSD](server.md#kv-on-the-ssd): the requests that started in each mode, converted to `spill`, had their output bounded or ended at that bound, and were refused for the quota, the write budget or a closed tier, and the starts waiting for the quota (`requests`); the pages that hold only their index and those whose KV is on the SSD; the write-through of spilling requests' pages; page moves, rewrites and restores; the quota of KV in service; the hot block cache's memory, hits, reads and host steps (`block_cache`; `read` over `resolves` is the blocks a layer's step missed on average) and the history streams' reads, how long each layer's took (`read_ms`, whose total gives with `bytes` the streams' bandwidth) and staging, the streamed commands that ran beside decoders, and the decoders' gaps across them (`history_stream`); absent for a target whose KV stays in memory, whose plan's `memory_plan.budget.kv_spill` is `null` |
| `kv.window` | For a target with sliding-window layers, its [windows](internals.md#sliding-windows): the `window_pages` a row's window and a state's tail keep Full, whether pages past every window give up their windowed layers' K/V (`trim`, off with the native `--kv-window-trim off`), the pool's `full_pages` and `slim_pages` in use and the Slim extents' `slim_bytes`, the `pinned_blocks` states' tails hold, the `trimmed_pages` since engine start, the `upgrades` of blocks that took the page of a writer that computed them again, and the `refused_states` whose tail was not Full; absent for other targets |
| `memory_plan.budget.kv_window` | For a target whose pages give up a window's layers, a Slim page's `slim_page_bytes` and the `floor_pages` a lane keeps Full, by which one request's KV capacity is counted; absent for other targets |
| `memory_plan.budget.context_bounds` | What bounds `maximum_context_tokens` ([memory and context](server.md#memory-and-context)): the model's `window`, one request's KV in memory (`resident`), the longest context a block-sparse target's floors hold (`spilled`) and its SSD cache's room holds (`disk`), and the `bound` that sets the limit: `model`, `memory` or `disk` |
| `ane_ffn.state`, `share`, `minimum_rows`, `reason` | The prefill FFN's [Neural Engine split](internals.md#neural-engine-prefill): `split` while it serves, at `share` of the FFN's channels over chunks of `minimum_rows` rows or more; `off` when the start left the GPU alone (`share` and `minimum_rows` 0); `stopped` once it stopped while serving, until the engine restarts. `reason` is the start's outcome, or why the split stopped |
| `ane_ffn.split_commands`, `evaluations`, `ane_ms`, `reruns` | Prefill commands the split ran since engine start, the Neural Engine's evaluations in them and its milliseconds over them (each evaluation's from the GPU's signal to start it to its report); and the chunks the GPU ran again alone after the split's work for them failed |
| `verify_rows` | The rows of a draft block decode cycles verify ([drafts](models.md#drafts), `--verify-rows`): `auto` where the verify budget chooses them (a MoE target under the default), `all` where they verify every row the output limit leaves (a dense target, or `--verify-rows all`) |
| `model_timing.prefill.last_gpu_ms`, `total_gpu_ms` | GPU time of the last prefill command, and of all since engine start, warmup included: each from its first Metal command buffer's start to its last one's end, so with the Neural Engine split it spans the GPU's waits on the Neural Engine; a chunk's image encodes, a command of their own ahead of it, add theirs, as a chunk run again adds the rerun's |
| `metrics.decode_tokens_per_second` | Aggregate native decode throughput, not a request's end-to-end rate |
| `metrics.decode_wall_ms` | Total wall time of the decode commands on the GPU, from submission to completion |
| `metrics.decode_cycle_ms` | Total engine time of the decode commands, each from the previous command's completion (or its plan after idleness) to its own: the GPU command plus the host work between commands |
| `loop.max_tick_ms` | Longest control pass and engine step of the native loop. A reader thread keeps reading requests meanwhile, so a request frame whose write stalls 5 s after it started fails the engine only when the process stopped reading; while requests are pending the server asks for status every 10 s and fails an engine whose loop does not answer within 30 s. |
| `maximum_context_tokens` | Declared context limit; available memory may limit admission |
| `thermal_state` | The Mac's thermal state as macOS reports it: `nominal`, `fair`, `serious` or `critical`. As the Mac heats, its CPU and GPU clocks drop and prefill and decode slow down, on some Macs already at `fair`; readiness ignores it. The engine samples it twice a second and logs each change (`Thermal state: fair → serious`) |
| `vision`, `input_modalities` | Whether image and PDF input is accepted; `false` and `["text"]` after `--language-only` |
| `reasoning_switch` | Whether a request's reasoning effort or `enable_thinking` turns the model's reasoning on and off ([default reasoning effort](server.md#default-reasoning-effort)): `true` for the Qwen families, `false` for Flint, whose model decides each turn |
| `chat_template.later_system` | `native`, `patched` or `unsupported`: how system messages after the first render (per name for named templates) |
| `transport.recovering`, `transport.stopped`, `transport.error` | The engine is restarting, or Splash stopped restarting it after repeated failures; `error` names its failure, the last failed restart, or why restarts stopped |

`GET /ready` is 200 while the engine's own `ready` is true. While the engine
loop is busy, `/ready` keeps its last answer until the loop has left status
requests unanswered for 30 s; a loop that leaves a status refresh unanswered
that long fails the engine, which then restarts. `/ready` stays healthy under
warning pressure, and reports 503 while macOS reports critical pressure;
requests already running continue, and one that needs more memory is suspended
until the pressure lifts.

`GET /metrics` exports in Prometheus text format the `requests`,
`memory_actual` and `metrics` fields above, the status's scheduler, KV, cache,
admission and image counts, the SSD cache's quota, limit, use, volume space,
write budget, what holds its writes back and the bytes it read and wrote, and
readiness, memory pressure and thermal state; the table's other fields are in
`/status` alone.
`splash_kv_free_allocated_pages` counts free pages of allocated extents, not
remaining capacity; memory headroom is `splash_memory_headroom_bytes`. Consumers
should tolerate missing native fields while the engine is unavailable, and
counter resets after an engine restart. A proxy that displays per-request token
counts must read them from the responses' usage
([request fields](#request-fields)).

`/metrics` also exports fixed latency histograms in seconds, with a bounded
set of stages in `/status.latency`. HTTP duration includes body upload and
response writing for admitted API requests. Preparation, queue, template,
tokenization, output grammar preparation and image preparation are measured
separately; preparation includes its nested stages. Tokenization covers the
encoding call, including reuse when available. Histogram buckets are cumulative
and labeled by upper bound. HTTP TTFT (`http_ttft`) starts before upload and
ends at the first native token event; native TTFT (`metrics.ttft_ms`,
per-request `request_latency.ttft_ms`) starts when the engine receives the
request. Output intervals are between native token events, which can contain
multiple speculative tokens; they are not per-token latency. Native queue
timing is recorded from successful completions. These histograms live with the
HTTP process and survive a native engine restart.

For slow tool-bearing requests, the `latency` section of `/status` separates
preparation, tokenization, grammar preparation, native queueing and HTTP TTFT.
Grammar preparation includes construction, compilation/cache lookup and
per-request cloning; it does not include generation-time masks. The
`grammar_cache` counters show whether compiled output grammars are reused. Tool
definitions still contribute tokens to the prompt; saving their JSON alone
cannot avoid model prefill.

Exact-prefix caching reuses model work while the server runs, and across
restarts with `--persistent-cache` ([persistent cache](server.md#persistent-cache)).
Text requests also reuse tokenized history at literal message-end boundaries
when the tokenizer supports independent encoding there. This process-local cache
retains at most four prefixes and 8 MiB of text/token storage; it falls back to
full encoding for other tokenizer pipelines. `/status.tokenizer_cache` reports
its usage. It does not alter prompt text, token IDs or the GPU KV cache.

## Errors

An error is answered as the API of its path answers it, whatever the method
and whether the request was read past its request line. OpenAI's paths, and
every path no other API has, answer `{"error": {"message", "type", "code"}}`,
with `type` `server_error` for a 5xx status and `invalid_request_error`
otherwise.
`/v1/messages` and the paths under it answer as Anthropic's API does,
`{"type": "error", "error": {"type", "message"}}`, with Anthropic's type for the
status, such as `overloaded_error` for 503. `/v1/systemone` answers invalid
requests with 422 `detail` arrays and every 503 with 529. Every 503, and every
529, carries `Retry-After: 1`, unless the engine says when a retry may succeed,
as `kv_write_budget` does. A connection the server refuses before reading
any of its request, as when every connection slot is taken
([request limits](#request-limits-and-timeouts)), gets one fixed 503 whatever
its path, `/v1/systemone`'s included: OpenAI's `server_error` with code
`frontend_overloaded`. The codes a client may act on:

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request_error` | A malformed request, or a field Splash cannot honor; the message names it. |
| 400 | `context_length_exceeded` | The prompt exceeds the context limit, or leaves no room in it for output. |
| 400 | `capacity_exhausted` | The request cannot fit in memory even alone; retrying fails the same way. |
| 400 | `kv_disk_quota` | The request's KV does not fit in memory, and the SSD cache may not hold even its prompt's (`--max-cache-disk`, and the volume's free space); retrying fails the same way. |
| 401 | `authentication_error` | The API key is missing or wrong. |
| 403 | `forbidden` | A Host or Origin the server does not serve; the message names the flag that admits it. |
| 404 | `model_not_found` | `model` is neither the served ID nor an alias. |
| 404 | `previous_response_not_found` | The store no longer holds a Responses `previous_response_id`. |
| 408 | `request_timeout` | The upload stalled or ran past its time. |
| 413 | `request_too_large` | The body exceeds `--max-request-size`. |
| 415 | `invalid_request_error` | The body is not `application/json`, or is compressed. |
| 500 | `engine_failed` | The engine failed repeatedly and Splash stopped restarting it ([engine restarts](server.md#engine-restarts-and-sleep)). |
| 500 | `model_result_invalid` | The model produced a non-finite result for this request; others continue ([failures](internals.md#failures-and-crash-traces)). |
| 500 | `internal_server_error` | A fault in the server, not in the request; the console names the error's type and the server line it arose at. |
| 501 | `not_implemented` | The request's method is not one the server has. |
| 503 | `engine_recovering`, `runtime_unavailable` | The engine is restarting; retry. |
| 503 | `server_shutdown` | The server is stopping: requests that arrive or are still running get it. |
| 503 | `frontend_overloaded` | The server holds `--queue-size` requests, every connection slot holds a request ([request limits](#request-limits-and-timeouts)), or a shared input budget or request preparation is full; retry. |
| 503 | `resource_timeout` | The request waited for memory, or for the SSD cache's room for its KV, and got none in time; retry, or free memory. |
| 503 | `mask_timeout` | The server left the request's token mask unanswered for 5 s; retry. |
| 503 | `kv_unreadable` | KV of an earlier part of the request could not be read back from the SSD cache, which gave the copy up; a retry recomputes it. |
| 503 | `kv_write_budget` | The request's KV does not fit in memory and would go to the SSD cache, whose `--max-cache-disk-writes` budget is spent until it refills; `Retry-After` says when it covers what spilling requests were promised again. |
| 503 | `runtime_busy` | The server could not hand the request's token mask to the engine: its mask queue was full or the engine connection failed; retry. |
| 504 | `request_timeout` | The request's `timeout`, or `--request-timeout`, ran out. |
| 505 | `http_version_not_supported` | The request is not HTTP/1.0 or HTTP/1.1. |
