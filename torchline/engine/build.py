
from torchline.utils import Registry, Logger

MODULE_REGISTRY = Registry('MODULE')
MODULE_REGISTRY.__doc__ = """
Registry for module template, e.g. DefaultModule.

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'MODULE_REGISTRY',
    'build_module'
]

def build_module(cfg):
    """
    Built the module template, defined by `cfg.module.name`.
    """
    name = cfg.module.name
    return MODULE_REGISTRY.get(name)(cfg)

    