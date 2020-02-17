from torchline.utils import Registry

META_ARCH_REGISTRY = Registry("meta_arch")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Built the whole model, defined by `cfg.model.meta_arch`.
    """
    meta_arch = cfg.model.meta_arch
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)
    