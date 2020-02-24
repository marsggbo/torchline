from torchline.utils import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset, i.e. torch.utils.data.Dataset.

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'DATASET_REGISTRY',
    'build_data'
]

def build_data(cfg):
    """
    Built the dataset, defined by `cfg.dataset.name`.
    """
    name = cfg.dataset.name
    return DATASET_REGISTRY.get(name)(cfg)
