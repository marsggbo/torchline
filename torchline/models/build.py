from torchline.utils import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

__all__ = [
    'build_model',
    'META_ARCH_REGISTRY'
]

def build_model(cfg):
    """
    Built the whole model, defined by `cfg.model.name`.
    """
    name = cfg.model.name
    return META_ARCH_REGISTRY.get(name)(cfg)
    