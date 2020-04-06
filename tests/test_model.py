import torch
import torchline as tl
x = torch.rand(1,3,64,64)
cfg = tl.config.get_cfg()
cfg.model.pretrained = False

def cpu_test():
    print('=====cpu=====')
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

def single_gpu_test():
    print('=====single_gpu=====')
    for m in tl.models.model_list:
        cfg.model.name = m
        net = tl.models.build_model(cfg).cuda()
        x = x.cuda()
        try:
            y = net(x)
            print(f"{m} pass")
        except Exception as e:
            print(f"{m} fail")
            print(str(e))
            pass

def multi_gpus_test():
    print('=====multi_gpu=====')
    for m in tl.models.model_list:
        cfg.model.name = m
        net = tl.models.build_model(cfg).cuda()
        net = torch.nn.DataParallel(net, device_ids=[0,1])
        x = x.cuda()
        try:
            y = net(x)
            print(f"{m} pass")
        except Exception as e:
            print(f"{m} fail")
            print(str(e))
            pass

if __name__ == '__main__':
    cpu_test()
    single_gpu_test()
    multi_gpus_test()