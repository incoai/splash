"""Experimental loader-matched AF4G64 weight reconstruction with an STE.

The prepared weights match GgufPreparation.cpp, not Metal GEMM accumulation.
Optimizer masters and exported state_dict names are unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainingError


def half_away(value):
    return value.sign() * (value.abs() + 0.5).floor()


@torch.no_grad()
def affine4_groups(weight):
    if weight.ndim != 2 or weight.shape[-1] % 64:
        raise TrainingError(
            "AF4G64 projections require a matrix with whole 64-column groups"
        )
    # The loader receives the BF16 export, not the FP32 optimizer masters.
    w = weight.to(torch.bfloat16).float().reshape(*weight.shape[:-1], -1, 64)
    if not torch.isfinite(w).all():
        raise TrainingError("nonfinite AF4G64 projection")
    low, high = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True).clamp_min(0)
    low_edge = low.abs() > high.abs()
    step = ((high - low) / 15).clamp_min(1e-7)
    step = torch.where(low_edge, step, -step)
    edge = torch.where(low_edge, low, high)
    q0 = half_away(edge / step)
    # Avoid NaNs even in an unselected branch for all-zero groups.
    nonzero = q0 != 0
    step = torch.where(nonzero, edge / torch.where(nonzero, q0, 1), step)
    bias = torch.where(nonzero, edge, 0)
    codes = half_away((w - bias) / step).clamp(0, 15).to(torch.uint8)
    return codes, step.to(torch.bfloat16), bias.to(torch.bfloat16)


def fake_affine4(weight):
    codes, scale, bias = affine4_groups(weight)
    # Do not BF16-round the reconstructed weight: native scale/bias are BF16,
    # but their affine reconstruction takes place in float.
    restored = (codes.float() * scale.float() + bias.float()).reshape_as(weight)
    return restored + (weight.float() - weight.detach().float())


class Affine4Linear(nn.Linear):
    def forward(self, inputs):
        # This is a reference QAT matmul, not a fast native quantized kernel.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            result = F.linear(
                inputs.float(),
                fake_affine4(self.weight),
                None if self.bias is None else self.bias.float(),
            )
        return result.to(inputs.dtype)


def enable_affine4(model, scope):
    if scope not in ("off", "selector_context", "all"):
        raise TrainingError("affine4_scope must be off, selector_context, or all")
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    def wanted(name):
        return scope == "all" or (
            scope == "selector_context"
            and name in ("fc", "candidate_selector.hidden_projection")
        )

    # Validate the whole requested scope before mutating any module.
    for name, module in modules:
        if wanted(name) and module.in_features % 64:
            raise TrainingError(f"AF4G64 projection {name} has partial column groups")
    selected = []
    for name, module in modules:
        enabled = wanted(name)
        if enabled:
            selected.append(name)
        if enabled == isinstance(module, Affine4Linear):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        kind = Affine4Linear if enabled else nn.Linear
        # Meta construction consumes no training RNG and allocates no extra
        # full projection. Preserve masters/optimizer handles across toggles.
        replacement = kind(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
        )
        replacement.weight, replacement.bias = module.weight, module.bias
        replacement.train(module.training)
        setattr(parent, child, replacement)
    return selected
