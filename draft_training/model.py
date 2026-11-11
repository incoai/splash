"""Differentiable DFlash2 with the serving tensor names and isolated blocks.

Architecture adapted from Z Lab dflash/model.py at
07ebd93db9f472af339b644bb70221ad8428328a (MIT, Copyright 2026 Z Lab).
See THIRD_PARTY_NOTICES. No inference cache or target weights are trainable here.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)


def top_candidates(logits, k):
    """Descending unary score, ascending token ID for ties, like Splash."""
    values, ids = torch.topk(logits, k, dim=-1, sorted=True)
    threshold = values[..., -1:]
    # topk does not specify which equal values survive the cutoff.
    if ((logits == threshold).sum(-1) > (values == threshold).sum(-1)).any():
        ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :k]
        values = logits.gather(-1, ids)
    by_id = ids.argsort(dim=-1, stable=True)
    ids, values = ids.gather(-1, by_id), values.gather(-1, by_id)
    by_score = values.argsort(dim=-1, descending=True, stable=True)
    return values.gather(-1, by_score), ids.gather(-1, by_score)


class DynamicConv(nn.Module):
    def __init__(self, hidden, kernel, group):
        super().__init__()
        self.group = group
        self.kernel = kernel
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel, hidden))
        self.kernel_projection = nn.Linear(
            hidden, 2 * kernel * (hidden // group), bias=False
        )
        with torch.no_grad():
            self.base_kernel[:, 0].fill_(1)
            self.kernel_projection.weight.zero_()

    def convolve(self, hidden, dynamic, base):
        batch, length, width = hidden.shape
        groups = width // self.group
        blocks = hidden.reshape(batch, length, groups, self.group)
        output = torch.zeros_like(blocks)
        for offset in range(self.kernel):
            previous = (
                blocks
                if offset == 0
                else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
            )
            fixed = base[offset].view(1, 1, groups, self.group).to(hidden.dtype)
            output = output + fixed * previous
            output = torch.addcmul(output, dynamic[:, :, offset, :, None], previous)
        return output.reshape_as(hidden)

    def prepare(self, hidden):
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel, hidden.shape[-1] // self.group
        )
        return self.convolve(
            hidden, dynamic[..., 0, :, :], self.base_kernel[0]
        ), dynamic[..., 1, :, :]

    def finish(self, hidden, dynamic):
        return self.convolve(hidden, dynamic, self.base_kernel[1])


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head_dim = c.head_dim
        self.q_heads, self.kv_heads = c.num_attention_heads, c.num_key_value_heads
        self.q_proj = nn.Linear(c.hidden_size, self.q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            c.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            c.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.q_heads * self.head_dim, c.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=c.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=c.rms_norm_eps)

    def forward(self, hidden, context, positions, visible, rotary):
        batch, rows, _ = hidden.shape
        keys_input = torch.cat((context, hidden), dim=1)
        q = self.q_norm(
            self.q_proj(hidden).view(batch, rows, self.q_heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(keys_input).view(batch, -1, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(keys_input)
            .view(batch, -1, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = rotary(hidden, positions)
        cos, sin = cos[:, None], sin[:, None]
        q = q * cos[:, :, -rows:] + rotate_half(q) * sin[:, :, -rows:]
        k = k * cos + rotate_half(k) * sin
        repeat = self.q_heads // self.kv_heads
        out = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(repeat, dim=1),
            v.repeat_interleave(repeat, dim=1),
            attn_mask=visible,
            dropout_p=0.0,
        )
        return self.o_proj(out.transpose(1, 2).reshape(batch, rows, -1))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        flash = c.dflash_config
        self.self_attn = Attention(c)
        self.mlp = Qwen3MLP(c)
        self.input_layernorm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.attention_conv = DynamicConv(
            c.hidden_size, flash["conv_kernel_size"], flash["conv_group_size"]
        )
        self.mlp_conv = DynamicConv(
            c.hidden_size, flash["conv_kernel_size"], flash["conv_group_size"]
        )

    def forward(self, hidden, context, positions, visible, rotary):
        prepared, dynamic = self.attention_conv.prepare(self.input_layernorm(hidden))
        hidden = hidden + self.attention_conv.finish(
            self.self_attn(prepared, context, positions, visible, rotary), dynamic
        )
        prepared, dynamic = self.mlp_conv.prepare(self.post_attention_layernorm(hidden))
        return hidden + self.mlp_conv.finish(self.mlp(prepared), dynamic)


class Selector(nn.Module):
    def __init__(self, c):
        super().__init__()
        rank = c.dflash_config["selector_rank"]
        self.top_k = c.dflash_config["selector_top_k"]
        self.hidden_projection = nn.Linear(c.hidden_size, rank, bias=False)
        self.predecessor_codebook = nn.Parameter(torch.empty(c.vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(c.vocab_size, rank))
        nn.init.normal_(self.predecessor_codebook, std=0.02)
        nn.init.zeros_(self.successor_codebook)

    def scores(self, hidden, unary, candidates, predecessors):
        projected = self.hidden_projection(hidden)
        # Serving stores codebooks and projected hidden rows as BF16, then
        # evaluates the three-way transition product in FP32. Round the FP32
        # master weights in this forward pass without losing their gradients.
        before = (
            F.embedding(predecessors, self.predecessor_codebook)
            .to(projected.dtype)
            .float()
        )
        after = (
            F.embedding(candidates, self.successor_codebook).to(projected.dtype).float()
        )
        context = before * projected.float()
        return unary.float() + (context.unsqueeze(-2) * after).sum(-1)

    def path(self, hidden, logits, anchor):
        unary, candidates = top_candidates(logits, self.top_k)
        paths = []
        predecessor = anchor
        for row in range(hidden.shape[1]):
            scores = self.scores(
                hidden[:, row], unary[:, row], candidates[:, row], predecessor
            )
            predecessor = (
                candidates[:, row]
                .gather(-1, scores.argmax(-1, keepdim=True))
                .squeeze(-1)
            )
            paths.append(predecessor)
        return torch.stack(paths, dim=1)


class DraftModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.raw_config = config
        self.config = c = Qwen3Config(**config)
        self.flash = config["dflash_config"]
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.num_hidden_layers))
        self.fc = nn.Linear(
            c.hidden_size * len(self.flash["target_layer_ids"]),
            c.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(c)
        self.candidate_selector = Selector(c)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Rotary frequencies are not learned or serialized. BF16-rounding them
        # before multiplication by long positions changes the phase materially.
        frequencies, _ = self.rotary_emb.compute_default_rope_parameters(self.config)
        self.rotary_emb.inv_freq = frequencies.to(device=self.fc.weight.device)
        self.rotary_emb.original_inv_freq = self.rotary_emb.inv_freq.clone()
        return result

    def forward(self, target_features, noise_embedding, positions, visible):
        context = self.hidden_norm(self.fc(target_features))
        hidden = noise_embedding
        for layer in self.layers:
            hidden = layer(hidden, context, positions, visible, self.rotary_emb)
        return self.norm(hidden)

    def block_hidden(self, tokens, features, anchor, embedding):
        """One independent query block; future captured states are never read."""
        rows, window = self.flash["block_size"], self.raw_config["sliding_window"]
        start = max(0, anchor - window + 1)
        query_ids = tokens.new_full((1, rows), self.flash["mask_token_id"])
        query_ids[0, 0] = tokens[anchor]
        noise = embedding(query_ids)
        # Slice before projection: invisible future features cannot affect even
        # intermediate computation, nor receive gradients.
        context = features[start:anchor].unsqueeze(0)
        positions = torch.arange(start, anchor + rows, device=tokens.device).unsqueeze(
            0
        )
        query_positions = torch.arange(anchor, anchor + rows, device=tokens.device)
        key_positions = positions[0]
        visible = (query_positions[:, None] - key_positions[None, :]) < window
        hidden = self(context, noise, positions, visible[None, None])
        return hidden[:, 1:]


def load_draft(directory, config, device="cpu"):
    from pathlib import Path

    from safetensors.torch import load_file

    from .config import TrainingError, checkpoint_index, expected_tensors

    model = DraftModel(config).to(dtype=torch.bfloat16)
    index = checkpoint_index(directory)
    expected = expected_tensors(config)
    if index.keys() != expected.keys():
        raise TrainingError(
            "checkpoint tensor names differ from the draft architecture"
        )
    for name, shape in expected.items():
        if index[name]["shape"] != shape or index[name]["dtype"] not in ("BF16", "F32"):
            raise TrainingError(f"checkpoint tensor type or shape mismatch: {name}")
    for filename in sorted({t["file"] for t in index.values()}):
        state = load_file(str(Path(directory) / filename))
        model.load_state_dict(state, strict=False)
    return model.to(device)
