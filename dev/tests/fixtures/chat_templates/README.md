These unmodified upstream templates exercise the startup probe and the structural
patch for system messages after the first message (`server/chat_templates.py`,
tested by `dev/tests/engine/test_chat_templates.py`). They are test fixtures,
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
