from torchline.utils import Registry

LOSS_FN_REGISTRY = Registry("LOSS_FN")
LOSS_FN_REGISTRY.__doc__ = """
Registry for loss function, e.g. cross entropy loss.

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'build_loss_fn',
    'LOSS_FN_REGISTRY'
]

def build_loss_fn(cfg):
    """
    Built the loss function, defined by `cfg.loss.name`.
    """
    name = cfg.loss.name
    return LOSS_FN_REGISTRY.get(name)(cfg)
