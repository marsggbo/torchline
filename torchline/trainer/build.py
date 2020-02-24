from torchline.utils import Registry

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

__all__ = [
    'TRAINER_REGISTRY',
    'build_trainer',
]

def build_trainer(cfg, hparams):
    """
    Built the whole trainer, defined by `cfg.trainer.name`.
    """
    name = cfg.trainer.name
    return TRAINER_REGISTRY.get(name)(cfg, hparams)
    