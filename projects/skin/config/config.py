from torchline.config import CfgNode as CN

def add_skin_config(cfg):
    _C = cfg
    
    _C.LOSS.FOCAL_LOSS = CN()
    _C.LOSS.FOCAL_LOSS.alpha = 0.25
    _C.LOSS.FOCAL_LOSS.gamma = 2
    _C.LOSS.FOCAL_LOSS.size_average = True
