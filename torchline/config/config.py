from yacs.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):
    pass

global_cfg = CfgNode()

def get_cfg():
    '''
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    '''
    from .default import _C
    return _C.clone()