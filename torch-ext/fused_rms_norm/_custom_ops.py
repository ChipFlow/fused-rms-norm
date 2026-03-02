import torch

from ._ops import ops


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    ops.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    ops.fused_add_rms_norm(input, residual, weight, epsilon)
