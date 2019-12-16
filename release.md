
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
    - build_module_template
  - losses
    - build_loss_fn
  - models
    - build_model
    - 
### 2019.12.16更新信息

- 更新package版本至`torchline-0.1.3`
- 更新transforms类结构，可以更加方便的自定义修改

### todo list

- [x] 弄清楚logging机制
- [x] save和load模型，优化器参数
- [x] skin数据集读取测试
- [x] 构建skin project
- [ ] 能否预测单张图片？
- [ ] 构建一个简单的API接口
- [ ] 进一步完善包导入
- [ ] 完善使用文档
- [x] 设置训练epoch数量
- [X] 内建更多SOTA模型
- [ ] 设置更多默认的数据集