
from torchline.utils import Registry, Logger

MODULE_TEMPLATE_REGISTRY = Registry('module_template')
MODULE_TEMPLATE_REGISTRY.__doc__ = """
Registry for module template, e.g. LightningTemplateModel.

The registered object will be called with `obj(cfg)`
"""

def build_module_template(cfg):
    """
    Built the module template, defined by `cfg.module_template.name`.
    """
    name = cfg.module_template.name
    return MODULE_TEMPLATE_REGISTRY.get(name)(cfg)

    