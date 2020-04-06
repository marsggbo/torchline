import torch

__all__ = [
    'generate_optimizer',
    'generate_scheduler'
]

def generate_optimizer(model, optim_name, lr, momentum=0.9, weight_decay=1e-5):
    '''
    return torch.optim.Optimizer
    '''
    if optim_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name.lower() == 'adadelta':
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == 'adam': 
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print(f"{optim_name} not implemented")
        raise NotImplementedError

def generate_scheduler(optimizer, scheduler_name, **params):
    '''
    return torch.optim.lr_scheduler
    '''
    if scheduler_name.lower() == 'CosineAnnealingLR'.lower():
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif scheduler_name.lower() == 'CosineAnnealingWarmRestarts'.lower():
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    elif scheduler_name.lower() == 'StepLR'.lower():
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif scheduler_name.lower() == 'MultiStepLR'.lower():
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        print(f"{scheduler_name} not implemented")
        raise NotImplementedError