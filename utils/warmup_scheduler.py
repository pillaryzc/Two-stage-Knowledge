from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
### 模块名称: Gradual Warmup Learning Rate Scheduler

#### 功能简介
- **主要作用**: 在训练的初始阶段逐渐增加学习率，从而实现温和的学习率预热。这种方法有助于改善模型在训练早期的稳定性和性能。
- **关键功能**:
  - 在预定的周期内逐渐增加学习率到指定的倍数。
  - 在完成预热后，可以选择接续使用其他学习率调度策略，如`ReduceLROnPlateau`。

#### 关键代码段解析
- **构造函数**:
  - 初始化时需指定优化器、学习率倍数、总预热周期，以及预热后的学习率调度器。
- **get_lr**:
  - 根据当前的训练轮次计算并返回新的学习率。如果处于预热阶段，学习率会按照线性或非线性规则递增；预热结束后，如果指定了后续调度器，将按该调度器的规则调整学习率。
- **step_ReduceLROnPlateau**:
  - 专门为与`ReduceLROnPlateau`调度器配合使用时重写的`step`方法。根据模型性能指标动态调整学习率。
- **step**:
  - 更新学习率。如果未使用`ReduceLROnPlateau`作为后续调度器，将根据预热规则或后续调度器规则更新学习率。

#### 输入输出
- **输入**: 
  - 在`step`方法中，输入为可选的当前轮次(epoch)和性能指标(metrics)。
- **输出**: 
  - 学习率更新直接影响到优化器中的学习率设置。

#### 依赖关系
- **内部依赖**: 依赖于PyTorch框架中的`_LRScheduler`和`ReduceLROnPlateau`。

#### 注意事项
- **特别注意**: 此调度器在使用时需要确保正确地配置预热周期和倍数，以避免学习率过高或过低。

#### 通俗的形象举例
将模型训练比作汽车启动和加速。直接将学习率设置得过高，就像是直接将油门踩到底，可能会导致模型训练过程中出现不稳定或是难以收敛的问题。使用渐进式预热学习率，就像是先让汽车缓慢加速，待引擎温度适宜后再逐渐加大油门，这样可以更平稳、更有效地推进训练过程。

#### 如何使用
```python
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

# 假设 model 是你的模型
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 初始化优化器
scheduler_steplr = StepLR(optimizer, step_size=30, gamma=0.1)  # 创建一个StepLR调度器作为预热后的调度器
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)  # 创建渐进预热调度器

for epoch in range(1, 50):
    # 训练模型...
    scheduler_warmup.step(epoch)  # 更新学习率
```
此示例展示了如何结合使用渐进预热调度器和StepLR调度器来优化训练过程的学习率调整，提高模型的训练效率和稳定性。

"""
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)