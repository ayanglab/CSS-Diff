"""Portable, strict loading for the released auxiliary architectures."""
from pathlib import Path
from collections.abc import Mapping
import torch


def validate_auxiliary_weights(opt, modules):
    missing = []
    for module in modules:
        value = getattr(opt, module.lower() + '_weights', None)
        if not value or not Path(value).expanduser().is_file():
            missing.append('%s: %s' % (module, value))
    if missing:
        raise FileNotFoundError('Missing auxiliary checkpoint(s): ' + '; '.join(missing)
                                + '. Set --sgp_weights/--lsc_weights (or YAML keys). '
                                'See docs/CHECKPOINTS.md for acquisition and formats.')


def load_auxiliary_weights(module, path, device, name):
    path = Path(path).expanduser()
    state = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('%s checkpoint %s must contain a raw tensor state_dict' % (name, path))
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError('%s checkpoint %s does not match the architecture: %s' % (name, path, exc)) from exc

    # Auxiliary parameters were never part of the generator optimizer.
    # Freeze them explicitly while retaining gradients to generated inputs.
    module.requires_grad_(False)
    module.eval()
