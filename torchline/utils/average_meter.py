from collections import OrderedDict

__all__ = [
    'AverageMeterGroup',
    'AverageMeter'
]

class AverageMeterGroup:
    """
    Average meter group for multiple average meters.
    """

    def __init__(self, verbose_type='avg'):
        self.meters = OrderedDict()
        self.verbose_type = verbose_type

    def update(self, data):
        """
        Update the meter group with a dict of metrics.
        Non-exist average meters will be automatically created.
        """
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":.4f", self.verbose_type)
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(f"{v}" for v in self.meters.values())

    def summary(self):
        """
        Return a summary string of group data.
        """
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """
    Computes and stores the average and current value.
    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    verbose_type : str
        'all': value(avg)
        'avg': avg
    """

    def __init__(self, name, fmt=':f', verbose_type='avg'):
        self.name = name
        self.fmt = fmt
        if verbose_type not in ['all', 'avg']:
            print('Not supported verbose type, using default verbose, "avg"')
            verbose_type = 'avg'
        self.verbose_type = verbose_type
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.
        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.verbose_type=='all':
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        elif self.verbose_type=='avg':
            fmtstr = '{name} {avg' + self.fmt + '}'
        else:
            fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)