"""FP32 next-token evaluation and small, explicit HAT training runs."""
import math
import random
import torch
from .data import validate_rows
from .core import PhotonicLinear


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def next_token_loss(model, row, device, reduction='sum'):
    tokens = torch.tensor([row], dtype=torch.long, device=device)
    logits = model(input_ids=tokens, use_cache=False).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1), reduction=reduction)


def reset_noise(model, forward_index):
    """Choose an explicit optical draw sequence, independently of train/eval mode."""
    if type(forward_index) is not int or forward_index < 0:
        raise ValueError('forward_index must be a nonnegative integer')
    for layer in model.modules():
        if isinstance(layer, PhotonicLinear):
            layer.reset_noise(forward_index)


@torch.no_grad()
def evaluate_batches(model, batches, device, *, noise_index=None):
    validate_rows(batches)
    if noise_index is not None:
        reset_noise(model, noise_index)
    model.eval()
    total_loss, count = 0., 0
    for row in batches:
        loss = next_token_loss(model, row, device)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite evaluation loss')
        total_loss += loss.double().item()
        count += len(row) - 1
    average = total_loss / count
    return dict(loss=average, perplexity=math.exp(average) if average < 709 else None,
                predicted_tokens=count, perplexity_overflow=average >= 709)


def train_hat(model, batches, device, target_paths, *, steps, learning_rate):
    """Train only selected layer weights/biases; tied Parameters remain tied.

    Each optimizer step uses one unpadded row, cycling in input order. Other
    parameters are frozen. Model dropout follows train() semantics. Validation
    should use separate data; training-row metrics measure fitting only.
    """
    validate_rows(batches)
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('steps and learning_rate must be positive and finite')
    if not target_paths:
        raise ValueError('HAT requires explicit target paths')
    selected = {}
    for path in target_paths:
        layer = model.get_submodule(path)
        for name in ('weight', 'bias'):
            parameter = getattr(layer, name, None)
            if parameter is not None:
                selected[id(parameter)] = parameter
    if not selected:
        raise ValueError('No selected trainable parameters')
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    optimizer = torch.optim.AdamW(list(selected.values()), lr=learning_rate, weight_decay=0.)
    model.train()
    losses, gradient_norms = [], []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, batches[step % len(batches)], device, reduction='mean')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        gradients = [p.grad for p in selected.values() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Missing or nonfinite HAT gradients')
        norm = sum(g.detach().double().square().sum().item() for g in gradients) ** .5
        optimizer.step()
        losses.append(loss.detach().item())
        gradient_norms.append(norm)
    return dict(steps=steps, learning_rate=learning_rate, optimizer='AdamW',
                weight_decay=0., trainable_parameters=sum(p.numel() for p in selected.values()),
                losses=losses, gradient_norms=gradient_norms,
                protocol='one row per optimizer step, cyclic input order; selected weights/biases only')
