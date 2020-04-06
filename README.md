# torchline v0.2.4.3

> Easy to use Pytorch
> 
> Only one configure file is enough!
> 
> You can change anything you want just in only one configure file.

# Dependences

- Python>=3.6
- Pytorch>=1.3.1
- torchvision>=0.4.0,<0.5.0
- yacs==0.1.6
- pytorch-lightning>=0.6.0,<0.7.0


# Install

- Before you install `torchline`, please make sure you have installed the above libraries.
- Only support on Linux

```bash
pip install -U torchline
```

# Run demo

## train model with GPU0 and GPU 1
```python
cd projects/cifar10_demo
python main.py --config_file cifar10.yaml trainer.device_ids [0,1]
```

## debug，add command line `trainer.fast_dev_run True`
```python
cd projects/cifar10_demo
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml trainer.device_ids [0] trainer.fast_dev_run True
```
CIFAR demo uses ResNet50，which is trained for 72 epochs and achieved the best result (94.39% validation accuracy) at the epoch 54.

# Thanks

- [AutoML](https://zhuanlan.zhihu.com/automl)
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

