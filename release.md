
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

### todo list


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
- [ ] 设置更多默认的数据集
- [ ] 完善使用文档
- [x] 每个epoch输出学习率大小
- [x] resume时输出checkpoint的结果
- [x] 如果resume，则自动匹配checkpoints等路径
- [x] 优化输出日志信息
- [x] 使用albumentations做数据增强
- [x] transforms resize和randomcrop逻辑关系
- [ ] 规范参数名称，尽量使用全程，如使用optimizer而不是optim # 在大版本v0.3.0.0中更新
- [ ] 增加config参数鲁棒性和兼容性
- [ ] 评估使用hydra代替yacs的必要性
- [x] 抽离出optimizer和scheduler