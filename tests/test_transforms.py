import torchline as tl
cfg = tl.config.get_cfg()
cfg.dataset.is_train=True
cfg.transforms.name = 'AlbumentationsTransforms'
cfg.abtfs.hue.enable=1
cfg.abtfs.rotate.enable=1
cfg.abtfs.bright.enable=1
cfg.abtfs.noise.enable=1
tf2 = tl.data.build_transforms(cfg)