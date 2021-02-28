# MMDetection-运行时

## 简介

在之前的文章中，已经介绍了配置文件、数据、模型等方面的内容，在配置文件[那篇文章](https://zhouchen.blog.csdn.net/article/details/113430466)中其实简单介绍了部分运行时相关的内容，本文将详细展开。需要说明的时，官方是将runtime与schedule区分开的，不过从配置继承的角度来看，它们可以放到一起，因此本文都将其视为**运行时配置**。

## 运行时定制

### 自定义优化器

MMDetection支持所有的PyTorch定义的优化器（optimizer），如果想要使用某个优化器只需要修改配置文件中`optimizer`字段即可，比如想要使用Adam优化器则在配置文件中写入下面一行。

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

当然，往往我们需要使用自己实现的优化器，那么按照规范，需要在项目根目录下的`mmdet/core/optimizer/`下创建自定义的Python脚本，如`mmdet/core/optimizer/my_optimizer.py`文件并写入如下内容，创建自定义`MyOptimizer`的方式和PyTorch中类似，只是需要注册为MMDetection识别的OPTIMIZERS即可。

```python
from .registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):
        pass
```

然后，为了能够在配置文件中找到这个优化器，需要在`mmdet/core/optimizer/__init__.py`文件中导入该优化器，也就是添加下面的一行内容。

```python
from .my_optimizer import MyOptimizer
```

此时，就可以在配置文件中修改`optimizer`字段来使用自定义的优化器了，示例如下。

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

除了自定义优化器，我们有时候还需要对优化器进行一些额外配置，这是官方优化器并没有实现的功能，就需要通过optimizer constructor（优化器构建器）来实现，常用的一些trick如下。

- 梯度裁剪

    在配置文件中添加如下字段，其中的grad_clip参数控制梯度裁剪。
    ```python
    optimizer_config = dict(
        _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
    ```
- 动量调度

    MMDetection支持依据学习率动态调整动量使得训练收敛加快，示例如下。
    ```python
    lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
    )
    momentum_config = dict(
        policy='cyclic',
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    ```

### 自定义训练调度

默认情况下，我们使用的是`configs/_base_/schedules/schedule_1x.py`中的调度设置，这是默认的Step LR调度，在MMCV中为`StepLRHook`，同时MMDetection也支持[其他的调度器](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)，如余弦调度（`CosineAnneaing`），在配置文件中及那个`lr_config`字段修改如下即可使用余弦调度。

```python
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
```

### 自定义工作流

在MMDetection中，工作流决定整体的工作流程（包括运行顺序和轮数），它是一个元素格式为`(phase, epochs)`的列表。默认情况下其设置为`workflow = [('train', 1)]`，这表示仅仅运行一轮训练，常常我们需要在验证集上评估模型的泛化能力（通过计算某些metric），此时可以设置工作流为`[('train', 1), ('val', 1)]`，这样就会迭代进行一轮训练一轮验证。

需要注意的是，验证时不会更新模型参数，且我们是通过`total_epochs`字段控制训练总伦数的，这不会影响验证工作流的进行。

### Hook

自定义Hook的内容可以参考MMCV的教程，这里不多赘述，介绍几个常用的Hook，它们由MMCV定义，只需要在配置文件中添加对应内容即可应用。比如我们想要在配置中使用MMCV自定义的`NumClassCheckHook`来检查head中的`num_classes`是否匹配`dataset`中的`CLASSES`长度，那么可以在 `default_runtime.py`中添加如下字段。

此外，还有不少常用的hook没有注册在`custom_hooks`中，主要有如下的几种。

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

其中，`optimizer_config`、`momentum_config`和`lr_config`上文已经涉及了，这里不再介绍，介绍剩下的三个的作用。

**Checkpoint config**来源于MMCV中的`CheckpointHook`，用于控制模型参数的本地保存，可以在配置文件中添加如下字段来设置，其中常用的参数有三个，`interval`表示保存频率（多少轮进行一次保存），`save_optimizer`表示是否保存优化器参数，`max_keep_ckpts`表示最大保存数目（常常我们只需要最后的几个模型，不需要所有轮的）。

```python
checkpoint_config = dict(interval=1, save_optimizer=True, max_keep_ckpts=-1)
```

**log_config**是很多logger hooks的包装器并且统一设置保存频率（`interval`同上），用于训练记录日志的保存，目前MMCV支持了三种`WandbLoggerHook`、`MlflowLoggerHook`和`TensorboardLoggerHook`，默认MMDetection设置的日志记录配置如下，只以文本的方式记录训练日志，建议打开Tensorboard记录。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
```
**evaluation**字段用来初始化`EvalHook`控制模型的评估，除了上面的`interval`外，你还需要设置如评估指标`metric`来传递给`dataset.evaluate()`方法， 默认的评估设置如下。

```python
evaluation = dict(interval=1, metric='bbox')
```

## 总结

本文介绍了MMDetection中运行时相关的配置，[官方教程](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_runtime.html)也有对应。最后，如果我的文章对你有所帮助，欢迎点赞收藏评论一键三连，你的支持是我不懈创作的动力。


