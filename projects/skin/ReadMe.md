
# Run

```python
CUDA_VISIBLE_DEVICES=0 python main.py --config_file skin100.yaml --gpus 1 
```

# Restore traininig

指定`TRAINER.LOGGER`信息即可，例如你在前面的实验中的日志信息(包括metrics和checkpoint等)保存如下：

```bash
|___output
    |___lightning_log # specified by TRAINER.LOGGER.TEST_TUBE.name
        |___version_0 # specified by TRAINER.LOGGER.TEST_TUBE.version
            |___metrics.csv
            |___...(other log files)
            |___checkpoint
                |___ _checkpoint_epoch_60.ckpt
```

- `TRAINER.LOGGER.SETTING `: 0 表示默认设置，1表示不用logger，2表示自定义logger
- `TRAINER.LOGGER.TEST_TUBE.name`: logger名字，如lightning_log
- `TRAINER.LOGGER.TEST_TUBE.version`: logger版本,如0


```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_file skin100.yaml --gpus 1 TRAINER.LOGGER.SETTING 2 TRAINER.LOGGER.TEST_TUBE.name lightning_log TRAINER.LOGGER.TEST_TUBE.version 0
```

# test_only

只运行验证集，参数设置同上
```
CUDA_VISIBLE_DEVICES=0 python main.py --config_file skin100.yaml --gpus 1 --test_only TRAINER.LOGGER.SETTING 2 TRAINER.LOGGER.TEST_TUBE.name lightning_log TRAINER.LOGGER.TEST_TUBE.version 0
```

# predict_only

预测指定路径下的图片,需要设置如下两个参数：

- `PREDICT_ONLY.LOAD_CKPT.checkpoint_path`: checkpoint路径
- `PREDICT_ONLY.to_pred_file_path`: 需要预测的图片路径，可以是单张图片的路径，也可以是包含多张图片的文件夹路径

```
CUDA_VISIBLE_DEVICES=0 python main.py --config_file skin100.yaml --gpus 1 --predict_only PREDICT_ONLY.LOAD_CKPT.checkpoint_path './output_skin100/lightning_logs/version_0/checkpoints/_ckpt_epoch_69.ckpt' PREDICT_ONLY.to_pred_file_path '.'
```

# 各种路径设置函数的作用和区别

## 根路径
- `TRAINER.DEFAULT_SAVE_PATH`: 指定logger和checkpoint的保存根路径

## Logger路径

代码中有两种日志模块可供选择，分别是MLFlow和test_tube,与路径相关的参数如下:
MLFLow:
- `TRAINER.LOGGER.MLFLOW.experiment_name`: 指定MLFlow的名字
- `TRAINER.LOGGER.MLFLOW.tracking_uri`: MLFlow日志模块的保存路径

Test_tube(默认使用这个模块)
- `TRAINER.LOGGER.TEST_TUBE.name`: 指定test_tube的名字，默认是`lightning_logs`
- `TRAINER.LOGGER.TEST_TUBE.save_dir`：test_tube日志模块的保存路径

## checkpoint路径
- `HOOKS.MODEL_CHECKPOINT.filepath`: checkpoints路径，不推荐设置，因为会自动在logger路径下创建一个checkpoints文件夹

## 栗子

举个栗子，假设参数设置如下(在你的`config.yaml`文件中设置)

```yaml
TRAINER:
    DEFAULT_SAVE_PATH: './output_skin100'
    LOGGER:
        type: 'test_tube'
        SETTING: 0 # 1 or 2
        TEST_TUBE:
            name: 'resnet50_logs'
            save_dir: ''
            version: 2
HOOKS:
    MODEL_CHECKPOINT:
        type: 0 # 1 or 2
        filepath: '' 
```

可以看到根路径是'./output_skin100'，那么就会在你的项目下创建这么一个文件夹。
之后再看`LOGGER`设置，可以看到`type`设置成了`test_tube`，而`SETTING`表示使用何种设置方式，下面分别详细介绍：

### logger参数解释
1) `SETTING=0`

`TEST_TUBE`下的参数会被忽略，也就是说使用默认的参数:
- `name`:`'lightning_logs'`
- `save_dir`:logger模块的根路径，如果设置了前面的`DEFAULT_SAVE_PATH`，则会与之保持一样,即使你自定义了`save_dir`也会被其覆盖。

最终生成的日志路径模板为： `save_dir/logger_name/logger_version/`,前面两个参数已经设置好了，后面的`logger_version`指明是哪一个版本的日志，默认从0开始，如果对应路径下you`version_0,version_1`，那么会自动生成`version_2`, 所以最终生成如下文件结构：

```
|___output_skin100
    |___lightning_logs
        |___version_0
```

1) `SETTING=1`

不使用logger模块，即不会保存任何信息

3) `SETTING=2`

使用上面给出的参数自定义logger，同理最后logger的保存路径为`'./output_skin100/resnet50_logs/version_2'`

```
|___output_skin100
    |___resnet50_logs
        |___version_2
```

所以可以看到`TEST_TUBE.name`可以根据你自己的需要修改，这样你可以很清楚地看到这个日志里存了什么样的信息，比如在这个例子中我就知道这个日志里存了resnet50模型的日志信息。另外，通过上面的介绍相信你也猜到，我们可以通过设置`name`和`version`参数来restore对应版本的模型和训练状态(epoch,global step, optimizer,lr_scheduler等等)，所以后面checkpoints推荐使用默认设置。


### checkpoint参数解释

`type`参数同logger的`setting`参数，推荐使用默认参数设置，即设置为0，这样checkpoint会自动保存到对应的logger版本日志中去。

```
|___output_skin100
    |___resnet50_logs
        |___version_0
            |___checkpoints
```


如果你设置为2，则最终checkpoint的保存路径可能与logger路径不一致，但是如果`filepath`参数为空字符串，则仍会自动解析成对应logger版本的保存路径，反之如果你指定了路径，则保存到指定路径。假设你设置为`filepath=./checkpoints`,则文件结构如下：
```
|___checkpoints
|___output_skin100
    |___resnet50_logs
        |___version_0
```

