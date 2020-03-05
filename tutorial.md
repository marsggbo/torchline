# 代码结构


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
          - `albumentation_transforms`: 使用albumentations库做数据增广
    - `engine`: 
        - `default_module.py`: 提供了`LightningModule`的一个继承类模板
    - `losses`:
        - `build.py`: 提供`build_loss_fn`函数和loss注册器
        - `loss.py`: 提供一些常用的loss函数(`CrossEntropy()`)
    - `models`:
        - `build.py`: 提供`build_model`函数和模型注册器
    - `trainer`：
    - `utils.py`:
      - `registry.py`: 注册器模板
      - `logger.py`: 输出日志模块
    - `main.py`: 代码运行入口，后面的项目构建都可以参照这个文件写代码
- `projects`: You can create your own project here.
    - `cifar10_demo`: A CIFAR10 demo project
    - `fake_demo`: 使用随机生成的数据，方便调试和体验torchline的使用 


# 如何自定义？

待完善。。。

# 自定义参数配置

# 自定义数据集

# 自定义`engine`模板

> 可自定义 如何**读取数据**，**优化器设置**，**forward步骤**


# 自定义新的模型`model`


# 自定义损失函数

# 自定义`trainer`