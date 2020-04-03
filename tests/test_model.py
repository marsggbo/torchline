import torch
import torchline as tl
x = torch.rand(1,3,64,64)
cfg = tl.config.get_cfg()
cfg.model.pretrained = False

for m in tl.models.model_list:
    cfg.model.name = m
    net = tl.models.build_model(cfg)
    try:
        y = net(x)
        print(f"{m} pass")
    except Exception as e:
        print(f"{m} fail")
        print(str(e))
        pass