import torch
import torchvision
from torchline.utils import Registry, Logger

SAMPLER_REGISTRY = Registry('SAMPLER')
SAMPLER_REGISTRY.__doc__ = """
Registry for dataset sampler, i.e. torch.utils.data.Sampler.

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'build_sampler',
    'SAMPLER_REGISTRY'
]

def build_sampler(cfg):
    """
    Built the dataset sampler, defined by `cfg.dataset.name`.
    """
    is_train = cfg.dataset.is_train
    if is_train:
        name = cfg.dataloader.sample_train
    else:
        name = cfg.dataloader.sample_test
    if name == 'default':
        return None
    return SAMPLER_REGISTRY.get(name)(cfg)

@SAMPLER_REGISTRY.register()
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            try:
                return dataset.imgs[idx][-1]
            except Exception as e:
                print(str(e))
                raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


@SAMPLER_REGISTRY.register()
class DiffWeightedRandomSampler(torch.utils.data.sampler.Sampler):
    r"""
    Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence) : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw

    """

    def __init__(self, indices, weights, num_samples=0):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise ValueError("num_samples should be a non-negative integeral "
                             "value, but got num_samples={}".format(num_samples))
        self.indices = indices
        weights = [ weights[i] for i in self.indices ]
        self.weights = torch.DoubleTensor(weights)
        if num_samples == 0:
            self.num_samples = len(self.weights)
        else:
            self.num_samples = num_samples
        self.replacement = True

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples