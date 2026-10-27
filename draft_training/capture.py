"""Frozen Transformers target adapter and resumable, bounded feature shards."""

import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import TrainingError, file_digest, read_json, write_json
from .data import conversations, examples


class TargetAdapter:
    def __init__(self, model, tokenizer, layer_ids, device):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.layer_ids = layer_ids
        self.device = torch.device(device)
        self.model.requires_grad_(False)
        self.embedding = model.get_input_embeddings()
        self.head = model.get_output_embeddings()
        self.decoder = model.model.language_model

    @classmethod
    def load(cls, directory, layer_ids, device):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False
        )
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(directory),
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if loading["missing_keys"] or loading.get("mismatched_keys"):
            raise TrainingError(
                "target loader reported missing or mismatched weights; refusing randomly initialized teacher parameters"
            )
        return cls(model.to(device), tokenizer, layer_ids, device)

    def release_teacher(self):
        """Keep frozen token tables, release the decoder/vision before training."""
        import gc

        self.model = None
        self.decoder = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        return self.embedding, self.head

    @torch.no_grad()
    def features(self, tokens):
        """Capture selected decoder outputs via hooks; do not retain all layers."""
        captured, handles = {}, []
        try:
            for layer in self.layer_ids:

                def capture(_module, _inputs, output, layer=layer):
                    value = output[0] if isinstance(output, tuple) else output
                    captured[layer] = value.detach()

                handles.append(
                    self.decoder.layers[layer].register_forward_hook(capture)
                )
            self.decoder(input_ids=tokens[None], use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(self.layer_ids):
            raise TrainingError("target adapter did not capture every selected layer")
        return torch.cat([captured[layer] for layer in self.layer_ids], dim=-1)[0]

    @torch.no_grad()
    def make_sample(self, messages, answer, recipe):
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        if not isinstance(prefix, torch.Tensor):
            prefix = prefix["input_ids"]
        if prefix.shape[1] + recipe.max_new_tokens > recipe.sequence_length:
            raise TrainingError(
                "prompt plus max_new_tokens exceeds sequence_length; shorten the example or increase the recipe limit"
            )
        if answer is None:
            sequence = self.model.generate(
                input_ids=prefix,
                max_new_tokens=recipe.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )[0]
        else:
            complete = self.tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": answer}],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(self.device)
            if not isinstance(complete, torch.Tensor):
                complete = complete["input_ids"]
            sequence = complete[0]
            if not torch.equal(sequence[: prefix.shape[1]], prefix[0]):
                raise TrainingError(
                    "chat template assistant prefix differs between prompt and supplied answer"
                )
            if len(sequence) > recipe.sequence_length:
                raise TrainingError("supplied answer exceeds sequence_length")
        if len(sequence) <= prefix.shape[1]:
            raise TrainingError("target produced no assistant tokens")
        mask = torch.arange(len(sequence), device=self.device) >= prefix.shape[1]
        return {
            "tokens": sequence.cpu().contiguous(),
            "features": self.features(sequence)
            .to(dtype=torch.bfloat16, device="cpu")
            .contiguous(),
            "assistant_mask": mask.cpu().contiguous(),
        }


def capture_dataset(adapter, data_path, output, recipe, identity, supervision):
    """One conversation continuation per shard bounds memory and supports restart."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    binding = {
        "identity": identity,
        "supervision": supervision,
        "data": file_digest(data_path),
        "sequence_length": recipe.sequence_length,
        "max_new_tokens": recipe.max_new_tokens,
        "generation": {"do_sample": False},
    }
    manifest = (
        read_json(manifest_path)
        if manifest_path.exists()
        else {"binding": binding, "shards": [], "complete": False}
    )
    if manifest["binding"] != binding:
        raise TrainingError(
            "cached features have incompatible target, data, or supervision provenance"
        )
    for shard in manifest["shards"]:
        if file_digest(output / shard["file"]) != shard["sha256"]:
            raise TrainingError(f"feature shard changed: {shard['file']}")
    if manifest["complete"]:
        return manifest
    start = time.monotonic()
    for i, (messages, answer) in enumerate(
        examples(conversations(data_path), supervision)
    ):
        if i < len(manifest["shards"]):
            continue
        sample = adapter.make_sample(messages, answer, recipe)
        filename = f"{i:06d}.safetensors"
        temporary = output / (filename + ".tmp")
        save_file(sample, str(temporary))
        temporary.replace(output / filename)
        manifest["shards"].append(
            {
                "file": filename,
                "sha256": file_digest(output / filename),
                "tokens": len(sample["tokens"]),
            }
        )
        write_json(manifest_path, manifest)
        print(
            f"captured {i + 1} continuations ({time.monotonic() - start:.1f}s)",
            flush=True,
        )
    if not manifest["shards"]:
        raise TrainingError("dataset has no usable assistant continuations")
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    return manifest


def load_sample(directory, shard, device="cpu"):
    path = Path(directory) / shard["file"]
    if file_digest(path) != shard["sha256"]:
        raise TrainingError("feature shard digest mismatch")
    tensors = load_file(str(path), device=str(device))
    if set(tensors) != {"tokens", "features", "assistant_mask"}:
        raise TrainingError("feature shard has unexpected tensors")
    length = len(tensors["tokens"])
    if (
        tensors["tokens"].ndim != 1
        or tensors["features"].ndim != 2
        or tensors["features"].shape[0] != length
        or tensors["assistant_mask"].shape != (length,)
    ):
        raise TrainingError("feature shard dimensions do not match tokens")
    if not torch.isfinite(tensors["features"]).all():
        raise TrainingError("feature shard contains nonfinite features")
    return tensors
