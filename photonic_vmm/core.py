"""Shared FP32 inference/HAT core for tiled optical VMM."""
from dataclasses import asdict, dataclass
import hashlib
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class VMMConfig:
    # Confirmed defaults; zero remains an explicit comparison option.
    constant_range_mode: str = 'legacy_abs'  # abs(constant) fallback; zero stays zero
    output_tile_execution: str = 'spatial_parallel'  # shared distribution vs separate time slots
    weight_noise_mode: str = 'per_forward'  # confirmed: redraw per forward, shared over tokens
    tile_in: int = 32
    tile_out: int = 32
    group_size: int = 32
    x_bits: int | None = None
    w_bits: int | None = None
    y_bits: int | None = None
    x_noise: bool = False
    w_noise: bool = False
    y_noise: bool = False
    x_std: float = 0.01
    w_std: float = 0.01
    y_std: float = 0.01
    seed: int = 42
    # Optional numerical clamp of integer codes only. No analog saturation model.
    clamp_codes: bool = False

    def __post_init__(self):
        for name in ('tile_in', 'tile_out', 'group_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.group_size != self.tile_in:
            raise ValueError('Initial supported specification requires group_size == tile_in')
        if self.tile_out != self.tile_in:
            raise ValueError('Initial supported hardware is square: tile_out == tile_in')
        for name in ('x_bits', 'w_bits', 'y_bits'):
            v = getattr(self, name)
            if v is not None and (type(v) is not int or not 2 <= v <= 16):
                raise ValueError(f'{name} must be None or an integer from 2 to 16')
        for name in ('x_std', 'w_std', 'y_std'):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('x_noise', 'w_noise', 'y_noise', 'clamp_codes'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.weight_noise_mode != 'per_forward':
            raise ValueError('Confirmed specification uses weight_noise_mode=per_forward')
        if self.constant_range_mode not in ('zero', 'legacy_abs'):
            raise ValueError('Invalid constant_range_mode')
        if self.output_tile_execution not in ('spatial_parallel', 'temporal_multiplexed'):
            raise ValueError('Invalid output_tile_execution')
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')

    def metadata(self):
        return dict(config=asdict(self), accumulation='sequential FP32',
                    output_order='partial Y noise -> dynamic ADC -> dequantize -> add',
                    final_quantization=False, bias='once after all partial sums',
                    xw_range_axis='last dimension of each input tile, independently per row',
                    y_range_axis='last dimension of each output tile, independently per token',
                    tails='short tiles, no padding',
                    numerical_assumptions=['X/W last-axis row ranges', 'optional code clamp'])


def quantize(t: Tensor, bits: int | None, clamp_codes: bool = False) -> Tensor:
    """Dynamic affine Q/DQ per last-axis row. Constant rows pass unchanged."""
    if bits is None:
        return t
    lo, hi = t.amin(-1, keepdim=True), t.amax(-1, keepdim=True)
    span = hi - lo
    constant = span == 0
    # Equivalent affine formula avoids an overflowing reciprocal for tiny spans.
    levels, qmin = 2**bits - 1, -2**(bits-1)
    unit = (t-lo) / torch.where(constant, torch.ones_like(span), span)
    codes = torch.round(unit * levels + qmin)  # ties to even
    if clamp_codes:
        codes = codes.clamp(-2**(bits-1), 2**(bits-1)-1)
    return torch.where(constant, t, ((codes-qmin)/levels)*span+lo)


def draw_noise(t: Tensor, config: VMMConfig, stage: str, layer_key: str,
               forward_index: int, input_tile: int, output_tile: int) -> Tensor:
    if not getattr(config, stage+'_noise') or getattr(config, stage+'_std') == 0:
        return torch.zeros_like(t)
    lo, hi = t.amin(-1, keepdim=True), t.amax(-1, keepdim=True)
    span = hi-lo
    if config.constant_range_mode == 'legacy_abs':
        span = torch.where(span == 0, hi.abs(), span)
    step = forward_index
    out_id = -1 if stage == 'x' and config.output_tile_execution == 'spatial_parallel' else output_tile
    key = repr((config.seed, layer_key, stage, step, input_tile, out_id)).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
    generator = torch.Generator(device=t.device).manual_seed(seed)
    z = torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=generator)
    return z * (span * getattr(config, stage+'_std'))


class _IdentitySTE(torch.autograd.Function):
    """Return the exact physical value; backward ignores its range/noise graph."""
    @staticmethod
    def forward(ctx, original, physical):
        return physical

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def transform(t, bits, config, stage, layer_key, forward_index, k, o):
    # No derivative through min/max, quantization or noise amplitudes.
    with torch.no_grad():
        q = quantize(t, bits, config.clamp_codes)
        physical = q + draw_noise(q, config, stage, layer_key, forward_index, k, o)
    return _IdentitySTE.apply(t, physical)


def adc_transform(t, config):
    with torch.no_grad():
        physical = quantize(t, config.y_bits, config.clamp_codes)
    return _IdentitySTE.apply(t, physical)


def simulate_linear(x: Tensor, weight: Tensor, bias: Tensor | None, config: VMMConfig,
                    *, layer_key: str = 'linear', forward_index: int = 0,
                    trace=None, protection=None) -> Tensor:
    """Explicit K/N tiles; trace receives (K start, N start, pre-noise, ADC output)."""
    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError('Expected x[..., in_features], weight[out_features, in_features]')
    if weight.shape[0] < 1 or weight.shape[1] < 1:
        raise ValueError('Empty feature dimensions are unsupported')
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError('bias must have shape [out_features]')
    if type(forward_index) is not int or forward_index < 0:
        raise ValueError('forward_index must be a nonnegative integer')
    for t in (x, weight, bias):
        if t is not None and (t.dtype != torch.float32 or t.device != x.device):
            raise ValueError('All inputs must be FP32 on the same device')
        if t is not None and not torch.isfinite(t).all():
            raise ValueError('Inputs must be finite')
    mask = None
    if protection is not None:
        from .protection import select_mask
        mask = select_mask(protection, x)
    # Disable caller autocast; TF32 must be disabled by the evaluation driver on CUDA.
    with torch.autocast(device_type=x.device.type, enabled=False):
        y = torch.zeros((*x.shape[:-1], weight.shape[0]), dtype=torch.float32, device=x.device)
        for k in range(0, weight.shape[1], config.tile_in):
            xt = x[..., k:k+config.tile_in]
            keep = None if mask is None else ~mask[..., k:k+config.tile_in]
            optical_x = xt if keep is None else xt * keep
            shared_x = None
            if config.output_tile_execution == 'spatial_parallel':
                shared_x = transform(optical_x, config.x_bits, config, 'x', layer_key, forward_index, k, -1)
                if keep is not None:
                    shared_x = shared_x * keep
            for o in range(0, weight.shape[0], config.tile_out):
                xn = shared_x
                if xn is None:
                    xn = transform(optical_x, config.x_bits, config, 'x', layer_key, forward_index, k, o)
                    if keep is not None:
                        xn = xn * keep
                wt = weight[o:o+config.tile_out, k:k+config.tile_in]
                wn = transform(wt, config.w_bits, config, 'w', layer_key, forward_index, k, o)
                partial = torch.nn.functional.linear(xn, wn)
                noisy = transform(partial, None, config, 'y', layer_key, forward_index, k, o)
                adc = adc_transform(noisy, config)
                y[..., o:o+config.tile_out].add_(adc)
                if trace is not None:
                    trace(k, o, partial, adc)
        if mask is not None:
            y = y + torch.nn.functional.linear(x * mask, weight)
        if bias is not None:
            y = y + bias
        if not torch.isfinite(y).all():
            raise FloatingPointError('Nonfinite VMM result')
        return y


class PhotonicLinear(torch.nn.Module):
    """Inference/HAT replacement sharing parameters; reset counter for replay."""
    def __init__(self, source: torch.nn.Linear, config: VMMConfig, layer_key: str, protection=None):
        super().__init__()
        self.weight, self.bias = source.weight, source.bias
        self.in_features, self.out_features = source.in_features, source.out_features
        self.config, self.layer_key, self.forward_index = config, layer_key, 0
        self.protection = protection
        self.train(source.training)

    def forward(self, x):
        result = simulate_linear(x, self.weight, self.bias, self.config,
                                 layer_key=self.layer_key, forward_index=self.forward_index,
                                 protection=self.protection)
        self.forward_index += 1
        return result

    def reset_noise(self, forward_index=0):
        self.forward_index = forward_index
