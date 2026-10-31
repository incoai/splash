These unmodified upstream templates exercise the startup probe and the structural
patch for system messages after the first message (`server/chat_templates.py`,
tested by `dev/tests/server/test_chat_templates.py`). They are test fixtures,
not templates distributed with the runtime.

- `qwen36.jinja`: [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/blob/38740b847e4cb78f352aba30aa41c76e08e6eb46/chat_template.jinja),
  revision `38740b847e4cb78f352aba30aa41c76e08e6eb46`.
- `qwen38.jinja`: [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/blob/3e6447f082e89cc7f0bc6e5441afd38dfce760ff/chat_template.jinja),
  revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
- `qwen36_gguf.jinja`: embedded `tokenizer.chat_template` of
  `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` in `unsloth/Qwen3.6-35B-A3B-GGUF`,
  file SHA-256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- `qwen38_gguf.jinja`: embedded `tokenizer.chat_template` of
  `Qwen3.8-27B-UD-Q4_K_M.gguf` in `unsloth/Qwen3.8-27B-GGUF`,
  file SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
- `flint.jinja`: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint/blob/2a247be0c0aec895ccb0ac7ca92229ec6df37288/chat_template.jinja),
  revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288`, which the
  `tokenizer.chat_template` of `flint-UD-Q4_K_XL.gguf` in `unsloth/flint-GGUF`
  (revision `20468edfc38ab43d6914cc88b08cbecb05d0fc62`) holds byte for byte.
  The chat format tests (`dev/tests/server/test_chat_formats.py`) render it
  too.
- `deepseek41.jinja`: embedded `tokenizer.chat_template` of
  `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf` in
  `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`, revision
  `653a9874aa38c356233c5f3cc8c96b19a43b57eb`, which adapts Unsloth's
  DeepSeek-V4.1 template to DeepSeek's `encoding/encoding.py`.
- `deepseek41-encoding/`: the cases of DeepSeek's reference encoder,
  `encoding/tests/test_input_N.json` with the prompts it renders,
  `test_output_N.txt`, in
  [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/tests)
  (MIT license), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, which
  `deepseek41.jinja` renders alike (`dev/tests/server/test_deepseek_format.py`).
