
from torchline.utils import Registry, Logger

MODULE_TEMPLATE_REGISTRY = Registry('MODULE_TEMPLATE')
MODULE_TEMPLATE_REGISTRY.__doc__ = """
Registry for module template, e.g. LightningTemplateModel.

The registered object will be called with `obj(cfg)`
"""

def build_module_template(cfg):
    """
    Built the module template, defined by `cfg.MODULE_TEMPLATE.NAME`.
    """
    name = cfg.MODULE_TEMPLATE.NAME
    return MODULE_TEMPLATE_REGISTRY.get(name)(cfg)

    