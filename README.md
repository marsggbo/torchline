# torchline
Easy to use Pytorch

# Dependences

- Python>=3.6
- Pytorch==1.3.1
- torchvision==0.4.2
- yacs==0.1.6
- pytorch-lightning==0.5.3.2


# Install

```bash
pip install torchline
```

# Structures

文件结构参照[detectron2](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiux_PXpLDmAhVOPnAKHVTjDVEQFjAAegQIBxAC&url=https%3A%2F%2Fgithub.com%2Ffacebookresearch%2Fdetectron2&usg=AOvVaw25FixXG7GH7dRKY6sOc2Oc)设计,采用注册机制来实现不同模块的灵活切换。

- `torchline`
    - `config`: 参数设置模块
        - `config.py`: 返回`CfgNode`
        - `default.py`: 默认参数设置文件，后续可以通过导入`.yaml`文件来对指定参数做修改
    - `data`: 数据集模块，返回`torch.utils.data.Dataset`
        - `build.py`: 注册数据集，提供`build_data`函数来获取不同的数据集和数据集注册器
        - 数据集相关文件：
        - `common_datasets.py`: 返回MNIST和CIFAR10数据集
        - `skin10.py`: 返回Skin10数据集
        - `skin100.py`: 返回Skin100数据集
        - transform相关文件:
        - `transforms.py`: 提供`build_transforms`函数来构建`transforms`,并且该文件中包含了默认的transform类，即`DefaultTransforms`
        - `autoaugment.py`: Google提出的自动数据增强操作
        - `data_utils.py`
    - `engine`: 
        - `lightning_module_template.py`: 提供了`LightningModule`的一个继承类模板
    - `losses`:
        - `build.py`: 提供`build_loss_fn`函数和loss注册器
        - `loss.py`: 提供一些常用的loss函数
    - `models`:
        - `build.py`: 提供`build_model`函数和模型注册器
        - `resnet_models`: 提供一些列的resnet模型
    - `utils.py`:
      - `registry.py`: 注册器模板
      - `logger.py`: 输出日志模块
    - `main.py`: 代码运行入口，后面的项目构建都可以参照这个文件写代码
- `tests`: 测试代码
- `projects`: 新的项目可以在这里面创建，避免对`torchline`干扰
    - `cifar10_demo`: 一个demo项目


# Run demo

- using 1 gpu
```python
cd projects/cifar10_demo
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1 TRAINER.LOGGER.tracking_uri ./output INPUT.SIZE "(224,224)" 
```

如果你需要调试，可以加上如下参数`TRAINER.FAST_DEV_RUN True`,即
```python
cd projects/cifar10_demo
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1 TRAINER.LOGGER.tracking_uri ./output INPUT.SIZE "(224,224)" TRAINER.FAST_DEV_RUN True
```