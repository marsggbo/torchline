# torchline
Easy to use Pytorch

# Dependences

- Python>=3.6
- Pytorch==1.3.1
- torchvision==0.4.2
- yacs==0.1.6
- pytorch-lightning==0.5.3.2


# Install

- Before you install `torchline`, please make sure you have installed the above libraries.
- Only support on Linux

```bash
pip install -U torchline
```

# Structures

文件结构参照[detectron2](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiux_PXpLDmAhVOPnAKHVTjDVEQFjAAegQIBxAC&url=https%3A%2F%2Fgithub.com%2Ffacebookresearch%2Fdetectron2&usg=AOvVaw25FixXG7GH7dRKY6sOc2Oc)设计,采用注册机制来实现不同模块的灵活切换。

- `torchline`
    - `config`: Configuration Module
        - `config.py`: return base `CfgNode`
        - `default.py`: 默认参数设置文件，后续可以通过导入`.yaml`文件来对指定参数做修改
    - `data`: 数据集模块，返回`torch.utils.data.Dataset`
        - `build.py`: 注册数据集，提供`build_data`函数来获取不同的数据集和数据集注册器
        - 数据集相关文件：
          - `common_datasets.py`: 返回MNIST和CIFAR10数据集
        - transform相关文件:
          - `transforms.py`: 提供`build_transforms`函数来构建`transforms`,并且该文件中包含了默认的transform类，即`DefaultTransforms`
          - `autoaugment.py`: Google提出的自动数据增强操作
          - `data_utils.py`
    - `engine`: 
        - `lightning_module_template.py`: 提供了`LightningModule`的一个继承类模板
    - `losses`:
        - `build.py`: 提供`build_loss_fn`函数和loss注册器
        - `loss.py`: 提供一些常用的loss函数(`CrossEntropy()`)
    - `models`:
        - `build.py`: 提供`build_model`函数和模型注册器
    - `utils.py`:
      - `registry.py`: 注册器模板
      - `logger.py`: 输出日志模块
    - `main.py`: 代码运行入口，后面的项目构建都可以参照这个文件写代码
- `tests`: 测试代码
- `projects`: You can create your own project here.
    - `cifar10_demo`: A CIFAR10 demo project


# Run demo

- using 1 gpu
```python
cd projects/cifar10_demo
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1" 
```

如果你需要调试，可以加上如下参数`trainer.fast_dev_run True`,即
```python
cd projects/cifar10_demo
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1 trainer.fast_dev_run True
```
CIFAR demo使用ResNet50，最后只训练到72epoch，在第54 epoch取得最好表现(94.39% validation accuracy)
