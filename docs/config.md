
# 1. 输入

## 1.1 `input`


- `size`: [112,112] # 输入图像大小

# 2. 数据集

## 2.1 `dataset`

```yaml
batch_size: 1
dir: './datasets/mydataset'
is_train: False
name: 'fakedata'
test_list: './datasets/test_list.txt'
train_list: './datasets/train_list.txt'
valid_list: './datasets/valid_list.txt'
```

## 2.2 `dataloader`

```yaml
num_workers: 0
sample_test: 'default'
sample_train: 'default'
```

## 2.3 数据增强

### 2.3.1 数据增强

#### 2.3.1.1 基于torchvision
```yaml
transforms:
    img:
        aug_cifar: False
        aug_imagenet: False
        center_crop:
            enable: 0
        color_jitter:
            brightness: 0.0
            contrast: 0.0
            enable: 0
            hue: 0.0
            saturation: 0.0
        random_crop:
            enable: 1
            padding: 4
        random_horizontal_flip:
            enable: 1
            p: 0.5
        random_resized_crop:
            enable: 0
            ratio: (0.75, 1.3333333333333333)
            scale: (0.5, 1.0)
        random_rotation:
            degrees: 10
            enable: 0
        random_vertical_flip:
            enable: 0
            p: 0.5
        resize:
            enable: 0
    name: 'DefaultTransforms'
    tensor:
        normalization:
            mean: [0.4914, 0.4822, 0.4465]
            std: [0.2023, 0.1994, 0.201]
        random_erasing:
            enable: 0
            p: 0.5
            ratio: ((0.3, 3.3),)
            scale: (0.02, 0.3)
```

#### 2.3.1.2 基于albumentations


`abtfs`: <b style="color:tomato;">al</b>bumentations <b style="color:tomato;">t</b>rans<b style="color:tomato;">f</b>orm<b style="color:tomato;">s</b>
```yaml
abtfs:
    random_grid_shuffle:
        enable: 0
        grid: 2
    channel_shuffle:
        enable: 0
    channel_dropout:
        enable: 0
        drop_range: (1, 1)
        fill_value: 127
    noise:
        enable: 1
    blur:
        enable: 0
    rotate:
        enable: 1
    bright:
        enable: 1
    distortion:
        enable: 0
    hue:
        enable: 0
    cutout:
        enable: 1
        num_holes: 10
        size: 20
        fill_value: 127
```

### 2.3.2 标签数据增强

```yaml
label_transforms:
    name: 'default'
```

# 3. 模型


```yaml
model:
    classes: 10
    name: 'FakeNet'
    pretrained: True
    # features: ['f4']
    # features_fusion: 'sum'
    # finetune: False
```

# 4. 损失函数

```yaml
loss:
    class_weight: []
    focal_loss:
        alpha: []
        gamma: 2
        size_average: True
    label_smoothing: 0.1
    name: 'CrossEntropy'
```

# 5. 优化器和步长调整器

```yaml
optim:
    base_lr: 0.1
    momentum: 0.9
    name: 'sgd'
    scheduler:
        gamma: 0.1
        milestones: [150, 250]
        mode: 'min'
        name: 'MultiStepLR'
        patience: 10
        step_size: 10
        t_max: 10
        verbose: True
    weight_decay: 0.0005
```

# 6. 引擎Module

```yaml
module:
    name: 'DefaultModule'
```

# 7. 训练控制器

```yaml
trainer:
    accumulate_grad_batches: 1
    amp_level: 'O1'
    check_val_every_n_epoch: 1
    default_root_dir: './output_fakedata'
    distributed_backend: 'dp'
    fast_dev_run: False
    gpus: []
    gradient_clip_val: 0
    log_gpu_memory: ''
    log_save_interval: 100
    logger:
        mlflow:
            experiment_name: 'torchline_logs'
            tracking_uri: './output'
        setting: 0
        test_tube:
            name: 'torchline_logs'
            save_dir: './output_fakedata'
            version: -1
        type: 'test_tube'
    max_epochs: 100
    min_epochs: 1
    name: 'DefaultTrainer'
    num_nodes: 1
    num_sanity_val_steps: 5
    overfit_pct: 0.0
    print_nan_grads: True
    process_position: 0
    resume_from_checkpoint: ''
    row_log_interval: 10
    show_progress_bar: False
    test_percent_check: 1.0
    track_grad_norm: -1
    train_percent_check: 1.0
    truncated_bptt_steps: ''
    use_amp: False
    val_check_interval: 1.0
    val_percent_check: 1.0
    weights_save_path: ''
    weights_summary: ''
```

# 其他

```yaml
VERSION: 1
DEFAULT_CUDNN_BENCHMARK: True
SEED: 666
topk: [1, 3]
```
