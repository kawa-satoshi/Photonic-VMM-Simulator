"""Replaceable input-feature selection, independent of the numerical core.

Policies return boolean masks of x's shape or [in_features]. Selection is detached.
The optical branch keeps physical coordinates: masked X entries are zero before
Q/DQ and again after X noise. W groups stay unchanged, including their ranges.
Zero slots participate in X ranges; there is no packing or weight-coordinate
protection. The digital branch uses unmodified FP32 X and W.
"""
from dataclasses import dataclass, asdict
import math
from typing import Protocol
import torch


class ProtectionPolicy(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class MaxAbsPolicy:
    threshold: float
    aggregation: str  # 'batch_sequence' or 'token'; explicit, no silent default

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('Protection threshold must be finite and nonnegative')
        if self.aggregation not in ('batch_sequence', 'token'):
            raise ValueError('aggregation must be batch_sequence or token')

    def __call__(self, x):
        if self.aggregation == 'token':
            return x.abs() > self.threshold
        if x.numel() == 0:
            return torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
        return x.reshape(-1, x.shape[-1]).abs().amax(0) > self.threshold

    def metadata(self):
        return dict(policy='max_abs', **asdict(self), comparison='strictly greater',
                    optical_mask='zero X before Q/DQ and after X noise',
                    x_ranges='include physical zero slots', w_ranges='unchanged',
                    digital_branch='FP32 linear(masked original X, original W)')


def select_mask(policy: ProtectionPolicy, x):
    with torch.no_grad():
        mask = policy(x.detach())
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device != x.device:
        raise ValueError('Policy must return a boolean Tensor on the input device')
    if mask.shape not in (x.shape, (x.shape[-1],)):
        raise ValueError('Policy mask must match x.shape or [in_features]')
    return mask.detach()
