from torchline.utils import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset, i.e.. torch.utils.data.Dataset.

The registered object will be called with `obj(cfg)`
"""


def build_data(cfg):
    """
    Built the dataset, defined by `cfg.DATASET.NAME`.
    """
    name = cfg.DATASET.NAME
    return DATASET_REGISTRY.get(name)(cfg)
