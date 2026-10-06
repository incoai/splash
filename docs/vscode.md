# VS Code

VS Code's local agent can use Splash through the `customendpoint` provider.
This setup was checked against VS Code 1.140.0 and Splash 1.2.1.

Start Splash with Qwen3.8-27B and an API model alias:

```sh
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M \
  --max-context 128K --served-model-name local-qwen \
  --default-reasoning-effort low
```

An already installed Qwen3.8 model, including
`incoai/Qwen3.8-27B-Splash`, can replace the `--model` value. Keep the alias
the same as the `id` in the configuration below.

In VS Code's model management UI, add a custom endpoint provider with this
configuration. If other providers are already configured, add this object
to the existing list:

```json
[
  {
    "name": "Splash Engine",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "local-qwen",
        "name": "Qwen3.8 Splash",
        "url": "http://127.0.0.1:8000/v1",
        "toolCalling": true,
        "vision": true,
        "thinking": true,
        "maxInputTokens": 120000,
        "maxOutputTokens": 8192,
        "supportsReasoningEffort": ["low", "medium", "high"],
        "reasoningEffortFormat": "chat-completions"
      }
    ]
  }
]
```

Select **Qwen3.8 Splash** in the local agent's model picker. Choose low,
medium or high in its reasoning selector; high maps to Qwen3.8's xhigh.
The server's low default applies when a request sends no effort.
Model aliases such as `qwen-low` and `qwen-high` identify the same loaded
model; their names do not set reasoning effort.

`thinking: true` lets VS Code preserve reasoning in later turns and send the
output limit as `max_completion_tokens`. Without it, this version of the
Chat Completions adapter removes `max_tokens`, so `maxOutputTokens` alone
does not bound generation. This behavior is visible in VS Code's
[request adapter](https://github.com/microsoft/vscode/blob/1.140.0/extensions/copilot/src/extension/byok/node/openAIEndpoint.ts#L373).

Keep `maxInputTokens + maxOutputTokens` within the server's context window.
The example reserves 8,192 output tokens within 128K. Check
`maximum_context_tokens` at `http://127.0.0.1:8000/status` and reduce the
client limits if the server advertises a smaller window. If Splash uses
another port, update the URL. A server started with `--language-only` needs
`vision: false`.

## Sampling and reasoning limits

VS Code 1.140.0's custom endpoint configuration does not read `extraBody`.
Putting `reasoning_effort`, `min_p`, penalties or stop sequences there has
no effect. Use `supportsReasoningEffort` and `reasoningEffortFormat` as
above. The supported `modelOptions` settings are temperature and top-p;
the adapter removes temperature for a model marked as thinking. These
fields are defined in the
[provider](https://github.com/microsoft/vscode/blob/1.140.0/extensions/copilot/src/extension/byok/vscode-node/customEndpointProvider.ts#L89)
and applied in the
[request adapter](https://github.com/microsoft/vscode/blob/1.140.0/extensions/copilot/src/extension/byok/node/openAIEndpoint.ts#L308).
Use a direct API client to test Splash's other
[sampling parameters](../DEVELOPMENT.md#sampling).

Reasoning effort changes the model's instructions; it is not a hard
thinking-token budget. `max_completion_tokens` bounds all generated tokens,
including reasoning. A response that reaches the limit before closing its
thinking block can therefore have `finish_reason: "length"`, reasoning
content, and no final answer. Increase both the output allowance and the
space reserved for it when a useful response needs more tokens.

## Repeated reasoning

If the model repeats paragraphs, cancel the request and retry in a fresh
conversation. Verify that the outgoing request includes the selected
`reasoning_effort` and `max_completion_tokens`. An output limit bounds a
loop's duration; it does not guarantee a final answer.

For a direct API baseline, use Qwen3.8's recommended thinking settings:
`temperature: 1.0`, `top_p: 0.95`, `top_k: 20`, `min_p: 0.0`,
`presence_penalty: 0.0`, and `repetition_penalty: 1.0`, with an explicit
output limit. See the
[model's recommendations](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices).
To turn reasoning off, send `reasoning_effort: "none"` directly, or restart
Splash with `--default-reasoning-effort none` and use a client request that
omits effort. An explicit request effort overrides the server default.

If repetition persists outside VS Code, include the Splash version, model
revision, server command, effective context limit, and a sanitized failing
request and response in the issue. Keep the finish reason and usage in the
response. A client error such as “Response contained no choices” does not
by itself establish whether the server returned no choices, the model
produced only reasoning, or the request was interrupted. See
[issue #326](https://github.com/incoai/splash/issues/326).
