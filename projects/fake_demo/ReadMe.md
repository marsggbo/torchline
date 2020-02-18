
# Run

```python
cd projects/cifar10_demo
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1 
```

# Restore traininig

指定`trainer.logger`信息即可，例如你在前面的实验中的日志信息(包括metrics和checkpoint等)保存如下：

```bash
|___output
    |___lightning_log # specified by trainer.logger.test_tube.name
        |___version_0 # specified by trainer.logger.test_tube.version
            |___metrics.csv
            |___...(other log files)
            |___checkpoint
                |___ _checkpoint_epoch_60.ckpt
```

- `trainer.logger.setting `: 0 表示默认设置，1表示不用logger，2表示自定义logger
- `trainer.logger.test_tube.name`: logger名字，如lightning_log
- `trainer.logger.test_tube.version`: logger版本,如0


```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1 trainer.logger.setting 2 trainer.logger.test_tube.name lightning_log trainer.logger.test_tube.version 0
```

# test_only

只运行验证集，参数设置同上
```
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1 --test_only trainer.logger.setting 2 trainer.logger.test_tube.name lightning_log trainer.logger.test_tube.version 0
```

# predict_only

预测指定路径下的图片,需要设置如下两个参数：

- `predict_only.load_ckpt.checkpoint_path`: checkpoint路径
- `predict_only.to_pred_file_path`: 需要预测的图片路径，可以是单张图片的路径，也可以是包含多张图片的文件夹路径

```
CUDA_VISIBLE_DEVICES=0 python main.py --config_file cifar10.yaml --gpus 1 --predict_only predict_only.load_ckpt.checkpoint_path './output_cifar10/lightning_logs/version_0/checkpoints/_ckpt_epoch_69.ckpt' predict_only.to_pred_file_path '.'
```