"""Exact module-path replacement. No output path or substring selects arithmetic."""
import torch
from .core import PhotonicLinear


def convert(model, config, target_paths=None, protection=None, exclude_paths=()):
    # Standard scope includes every named nn.Linear, including a linear lm_head.
    if target_paths is None:
        target_paths = [name for name, layer in model.named_modules()
                        if name and isinstance(layer, torch.nn.Linear)]
    if len(target_paths) != len(set(target_paths)):
        raise ValueError('Duplicate target paths')
    if len(exclude_paths) != len(set(exclude_paths)):
        raise ValueError('Duplicate excluded paths')
    if set(exclude_paths) - set(target_paths):
        raise ValueError('Every excluded path must occur in the selected targets')
    target_paths = [path for path in target_paths if path not in exclude_paths]
    if not target_paths:
        raise ValueError('No named target layers remain')
    # Validate all paths before any mutation.
    layers = []
    module_paths = {}
    for name, layer in model.named_modules(remove_duplicate=False):
        module_paths.setdefault(id(layer), []).append(name)
    for path in target_paths:
        if not path:
            raise ValueError('Pass a named child module path, not the root')
        source = model.get_submodule(path)
        if not isinstance(source, torch.nn.Linear):
            raise TypeError(f'{path} is not torch.nn.Linear')
        if len(module_paths[id(source)]) > 1:
            raise ValueError(f'Aliased Linear module requires an adapter: {module_paths[id(source)]}')
        layers.append((path, source))
    manifest = []
    for path, source in layers:
        prefix, _, name = path.rpartition('.')
        parent = model.get_submodule(prefix) if prefix else model
        setattr(parent, name, PhotonicLinear(source, config, path, protection=protection))
        manifest.append(dict(path=path, in_features=source.in_features,
                             out_features=source.out_features, bias=source.bias is not None))
    return manifest
