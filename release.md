
# Release

## v0.1

### 2019.12.12 更新信息
- 基本框架搭建完成
- 可正常运行cifar10_demo


### 2019.12.13 更新信息
- 实现setup安装
- 完善包之间的引用关系
- 完善各模块之间的关系
  - data
    - build_data
    - build_sampler
    - build_transforms
    - build_label_transforms
  - engine
    - build_module
  - losses
    - build_loss_fn
  - models
    - build_model
    - 
### 2019.12.16更新信息

- 更新package版本至`torchline-0.1.3`
- 更新transforms类结构，可以更加方便的自定义修改

### 2019.12.17更新信息
- 可单独预测指定路径下的图片，`predict_only`模式，
- 完成`test_only`模式
- 新增topk结果显示
- 支持restore training
> 详细细节可查看[project/skin/ReadMe.md](projects/skin/ReadMe.md)

### 2019.12.19更新信息
- 修改各种路径参数设置逻辑（详见[skin/ReadMe.md](projects/skin/ReadMe.md)）


## v0.2

### 2020.02.18

- 优化代码逻辑
- 抽象化出`trainer`类
- `trainer`负责整个计算逻辑, `engine`下定义的`DefaultModule`用来指定具体的步骤包括：模型，优化器，训练、验证、测试过程，以及数据的读取等，`models`中则是定义了具体的模型，如resnet等。

## v0.2.2

- 代码逻辑更加清晰
- 修复日志bug，使用更加灵活

## v0.2.2.2

- config输出格式化

## v0.2.3.0

- 增加新的输出日志的方式（即logging），日志可保存到文件方便查看。之前默认使用tqdm，一个比较大的缺点时无法清晰的看到模型是否开始收敛。
- 引入`AverageMeterGroup`等来记录各项指标，能更清晰地看出整体收敛趋势
- 更新`fake_data` demo
- 修复`CosineAnnealingLR` 存在的bug

## v0.2.3.1
- 优化`AverageMeterGroup`输出格式，方便阅读

## v0.2.3.2
- 优化优化器代码，增加可扩展性
- 增加学习率热启动(`CosineAnnealingWarmRestarts`)

## v0.2.3.3
- 优化resume
- 增加albumentations数据增广操作
- 修改之前的resize和crop之间的逻辑关系

## v0.2.3.4
- 抽象化optimizer和scheduler定义，方便从外部直接调用
- 添加计算模型大小的函数

## v0.2.4.0
- 增加大量SOTA模型结构，如Mnasnet, mobilenet等
- 统一模型结构(features, logits, forward, last_linear)

## v0.2.4.1
- 修改单机多卡训练bug 
  - 此模式夏batch size必须是gpu的整数倍，否则汇报如下错误：
  ```Python
  ValueError: only one element tensors can be converted to Python scalars
  ```
- 规范化两种日志模式： tqdm和logging

## v0.2.4.2
- 修复单机多卡训练时的bug
- 修改和统一model forward函数： features+logits

## v0.2.4.3
- 更新module forward函数
- 增加loss函数，最小化entropy

# TODO list 


- [x] 弄清楚logging机制
- [x] save和load模型，优化器参数
- [x] skin数据集读取测试
- [x] 构建skin project
- [x] 能否预测单张图片？
- [x] 构建一个简单的API接口
- [x] 进一步完善包导入
- [x] 设置训练epoch数量
- [X] 内建更多SOTA模型
- [x] 每个epoch输出学习率大小
- [x] resume时输出checkpoint的结果
- [x] 如果resume，则自动匹配checkpoints等路径
- [x] 优化输出日志信息
- [x] 使用albumentations做数据增强
- [x] transforms resize和randomcrop逻辑关系
- [x] 从engine中抽离出optimizer和scheduler
- [x] ~~resnet结构可能有问题，resnet50应该有98MB，但是实现只有89.7~~。（没有错，只是因为计算时将classes设置成了10，所以导致了误差）
- [x] 单机多卡多GPU测试
- [x] ~~考虑是否将finetune设置内嵌到模型中~~ (取消设置，避免模型代码臃肿)
- [ ] 多GPU运行时日志会因为多线程而导致先后输出不同batch的结果，需要在结果整合后再输出结果，可以考虑将`print_log`放到`on_batch_end`里去
- [ ] 设置更多默认的数据集
- [ ] 完善使用文档
- [x] ~~评估使用hydra代替yacs的必要性~~（工作量太大）
- [ ] 增加config参数鲁棒性和兼容性
- [ ] 多机多卡测试
- [x] template project. 可快速基于torchline创建一个新项目
- [ ] 将`default_module`中的`parse_cfg_for_scheduler`解耦，放到`utils.py`文件中去
- [ ] checkpoint将scheduler参数也保存，同时添加设置可以跳过optimizer或scheduler的restore
- [ ] multi-gpus情况下日志会生成多份，打印信息也有这种情况


## v3.0 todo list
- [ ] 适配pytorchlightning 0.7.0版本 # 在大版本v0.3.0.0中更新
- [ ] 规范参数名称，尽量使用全程，如使用optimizer而不是optim # 在大版本v0.3.0.0中更新
- [ ] albumentations和torchvision读取图片使用的分别是cv2和PIL，数据格式分别是numpy和PIL.Image，后面需要考虑如何统一格式。
- [ ] 增加`Module`中`print_log`通用性